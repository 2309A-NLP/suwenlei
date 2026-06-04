# -*- coding: utf-8 -*-
"""
RAG问答系统 — 后端服务（混合检索：向量+BM25+RRF融合+Reranker重排）
工单编号: 人工智能NLP-RAG-混合检索任务
启动: python app.py
访问: http://127.0.0.1:8888
"""
import os
import sys
import json
import time
import logging
import warnings

# 屏蔽PyMilvus弃用警告和jieba的pkg_resources警告
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pymilvus')  # 屏蔽PyMilvus弃用警告
warnings.filterwarnings('ignore', message='.*pkg_resources.*')  # 屏蔽jieba的pkg_resources警告

sys.path.insert(0, os.path.dirname(__file__))  # 将当前目录加入Python搜索路径

from rag_pipeline import RAGEngine
from llm import LLMClient, GROUND_TRUTH, QueryUnderstanding
from pdf_parser import KnowledgeBuilder

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

logging.basicConfig(level=logging.INFO,  # 配置日志级别为INFO
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # 设置日志输出格式
logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器

# ---- 路径配置 ----
BASE_DIR = os.path.dirname(__file__)  # 项目根目录路径
INDEX_HTML = os.path.join(BASE_DIR, 'index.html')  # 前端HTML文件路径
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')  # 文件上传目录路径
RAG_BACKEND = os.environ.get("RAG_BACKEND", "auto")  # 从环境变量读取RAG后端类型，默认auto

os.makedirs(UPLOAD_DIR, exist_ok=True)  # 确保上传目录存在，不存在则创建

# ---- 对话历史存储（Redis优先，降级内存） ----
import json
import uuid
from datetime import timedelta
import redis


class ConversationStore:
    """对话历史存储：Redis优先，自动降级内存"""

    def __init__(self, host='127.0.0.1', port=6379, db=0, password=None, ttl_hours=24):
        self.ttl = timedelta(hours=ttl_hours)  # 对话过期时间
        self.prefix = 'rag:conv:'  # Redis key前缀
        try:
            self.rclient = redis.Redis(  # 创建Redis连接客户端
                host=host, port=port, db=db, password=password,  # Redis连接参数
                decode_responses=True, socket_connect_timeout=5,  # 自动解码响应，连接超时5秒
            )
            self.rclient.ping()  # 验证连接
            self._available = True  # 标记Redis可用
            logger.info("Redis连接成功: {}:{}/{}".format(host, port, db))  # 记录连接成功日志
        except Exception as e:
            self._available = False  # 降级为内存字典
            self.rclient = None  # Redis客户端置空
            self._fallback = {}  # 初始化内存降级存储
            logger.warning("Redis不可用，降级内存存储: {}".format(e))  # 记录降级警告

    def generate_id(self):
        return str(uuid.uuid4())  # 生成唯一会话ID

    def get_history(self, conversation_id):
        if not conversation_id:  # 会话ID为空时直接返回空列表
            return []
        if self._available:  # Redis可用时从Redis获取
            try:
                raw = self.rclient.get(self._key(conversation_id))  # 从Redis读取历史数据
                return json.loads(raw) if raw else []  # 解析JSON，空则返回空列表
            except Exception:  # Redis异常时返回空列表
                return []
        return self._fallback.get(conversation_id, [])  # 内存模式从字典获取历史

    def save_history(self, conversation_id, history, ttl=None):
        if not conversation_id:  # 会话ID为空时不保存
            return
        ttl = ttl or self.ttl  # 使用传入的TTL或默认过期时间
        if self._available:  # Redis可用时写入Redis
            try:
                self.rclient.setex(self._key(conversation_id), ttl,  # 设置带过期时间的键值
                                   json.dumps(history, ensure_ascii=False))  # 序列化历史为JSON写入
            except Exception:  # Redis写入异常时静默忽略
                pass
        else:  # 内存模式直接存入字典
            self._fallback[conversation_id] = history

    def append_message(self, conversation_id, role, content):
        """追加单条消息：读取→追加→回写"""
        history = self.get_history(conversation_id)  # 读取当前对话历史
        history.append({'role': role, 'content': content})  # 追加新消息
        self.save_history(conversation_id, history)  # 回写保存
    def _key(self, conversation_id):
        return self.prefix + conversation_id  # 拼接Redis存储key
# ---- 全局单例 ----
_engine = None  # RAG引擎全局实例
_llm = None  # LLM客户端全局实例
_uploaded_files = []  # 已上传文件列表
_store = ConversationStore()  # 对话历史存储实例


def get_engine():
    """惰性初始化RAG引擎"""
    global _engine, _llm  # 声明使用全局变量
    if _engine is None:  # 引擎未初始化时才执行
        logger.info("正在初始化RAG引擎（混合检索模式）...")  # 记录初始化开始
        pdf_paths = {}  # 存储PDF文件路径映射
        # 扫描BASE_DIR下的所有PDF文件
        if os.path.isdir(BASE_DIR):  # 检查项目根目录是否存在
            for f in os.listdir(BASE_DIR):  # 遍历根目录文件
                if f.lower().endswith('.pdf') and not f.startswith('~$'):  # 过滤PDF且排除临时文件
                    pdf_name = os.path.splitext(f)[0]  # 获取文件名（不含扩展名）
                    pdf_paths[pdf_name] = os.path.join(BASE_DIR, f)  # 存入映射表
        # 扫描uploads目录下的PDF文件（不覆盖同名文件）
        if os.path.isdir(UPLOAD_DIR):  # 检查上传目录是否存在
            for f in os.listdir(UPLOAD_DIR):  # 遍历上传目录文件
                if f.lower().endswith('.pdf') and not f.startswith('~$'):  # 过滤PDF且排除临时文件
                    pdf_name = os.path.splitext(f)[0]  # 获取文件名（不含扩展名）
                    if pdf_name not in pdf_paths:  # 仅添加不重复的文件
                        pdf_paths[pdf_name] = os.path.join(UPLOAD_DIR, f)  # 存入映射表
        _engine = RAGEngine(pdf_paths=pdf_paths, backend=RAG_BACKEND)  # 创建RAG引擎实例
        _llm = LLMClient()  # 创建LLM客户端实例
        _llm.set_rag_engine(_engine)  # 将RAG引擎绑定到LLM客户端
        # 强制预加载embedding模型并绑定给KnowledgeBuilder，避免上传时重新加载
        _engine._get_embedder()  # 触发懒加载，确保模型已在GPU
        _kb_builder._embedder = _engine._embedder  # 上传时复用GPU上的模型
        logger.info("初始化完成")  # 记录初始化完成
    return _engine, _llm  # 返回引擎和LLM客户端


def scan_uploaded_files():
    """扫描uploads目录"""
    global _uploaded_files  # 声明使用全局变量
    _uploaded_files = []  # 清空已上传文件列表
    if os.path.exists(UPLOAD_DIR):  # 检查上传目录是否存在
        for fname in os.listdir(UPLOAD_DIR):  # 遍历上传目录中的文件
            fpath = os.path.join(UPLOAD_DIR, fname)  # 构建完整文件路径
            if os.path.isfile(fpath):  # 确认是文件（非目录）
                _uploaded_files.append({  # 将文件信息加入列表
                    'filename': fname,  # 文件名
                    'size': os.path.getsize(fpath),  # 文件大小（字节）
                    'mtime': os.path.getmtime(fpath)  # 最后修改时间
                })
    return _uploaded_files  # 返回已上传文件列表


def reload_engine():
    """重置引擎"""
    global _engine, _llm  # 声明使用全局变量
    _engine = None  # 清空RAG引擎实例
    _llm = None  # 清空LLM客户端实例
    logger.info("引擎已重置，下次查询将重新初始化")  # 记录引擎重置日志


# ---- FastAPI ----
app = FastAPI(title="RAG问答系统 - 混合检索")  # 创建FastAPI应用实例


class QueryRequest(BaseModel):
    question: str  # 用户提问内容
    top_k: int = 5  # 检索返回的文档数量
    conversation_id: Optional[str] = None  # 会话ID，用于多轮对话
    search_mode: Optional[str] = None  # 检索模式：hybrid/vector/bm25
    lang: str = "zh"  # 语言：zh中文/en英文


class SearchConfigRequest(BaseModel):
    """检索配置请求"""
    search_mode: Optional[str] = None    # vector/bm25/hybrid
    vector_weight: Optional[float] = None  # 向量检索权重
    bm25_weight: Optional[float] = None    # BM25权重
    use_reranker: Optional[bool] = None    # 是否启用reranker
    use_llm_rerank: Optional[bool] = None  # 是否启用LLM重排


class FeedbackRequest(BaseModel):
    """用户反馈请求"""
    question: str  # 用户原始问题
    answer: str  # 系统回答内容
    rating: str  # good/bad
    comment: Optional[str] = None  # 用户额外评论


class HealthResponse(BaseModel):
    status: str  # 服务状态：healthy/degraded
    engine_ready: bool  # RAG引擎是否就绪
    backend: str  # RAG后端类型
    llm_available: bool  # LLM API是否可用
    pdf_loaded: bool  # 是否加载了PDF文档
    redis_available: bool = False  # Redis是否可用
    search_mode: str = "hybrid"  # 当前检索模式
    embedder_available: bool = False  # 向量嵌入模型是否可用
    reranker_available: bool = False  # 重排模型是否可用


# ==================== 前端 ====================

@app.get("/")  # 注册根路径GET路由
async def index():
    return FileResponse(INDEX_HTML)  # 返回前端HTML页面


# ==================== 文件管理API ====================

_kb_builder = KnowledgeBuilder(project_dir=BASE_DIR)  # 知识库构建器单例
@app.post("/api/upload")  # 注册文件上传POST路由
async def upload_files(files: list[UploadFile] = File(...)):
    """上传PDF并触发全链路处理：解析→分块→bge-m3向量化→Milvus入库"""
    results = []  # 存储每个文件的处理结果
    for upload_file in files:  # 遍历上传的文件列表
        filename = upload_file.filename or "unknown.pdf"  # 获取文件名，无名则用默认名
        if not filename.lower().endswith('.pdf'):  # 非PDF文件跳过
            results.append({'filename': filename, 'status': 'skipped', 'message': '仅支持PDF格式'})  # 记录跳过原因
            continue  # 跳过非PDF文件

        save_path = os.path.join(UPLOAD_DIR, filename)  # 构建保存路径
        try:
            content = await upload_file.read()  # 异步读取上传文件内容
            with open(save_path, 'wb') as f:  # 打开目标文件用于写入
                f.write(content)  # 将文件内容写入磁盘
        except Exception as e:
            results.append({'filename': filename, 'status': 'error', 'message': '文件保存失败: {}'.format(str(e))})  # 记录保存失败
            continue  # 跳过保存失败的文件

        # 直接同步调用（上传是低频操作，阻塞几秒可接受，避免子线程CUDA死锁）
        build_result = _kb_builder.build(save_path)  # 同步执行知识库构建（解析→分块→向量化→入库）
        results.append({  # 记录构建结果
            'filename': filename,  # 文件名
            'status': 'success' if build_result['success'] else 'error',  # 构建成功状态
            'message': build_result.get('error') or 'CSV {}块, Milvus {}条'.format(  # 结果摘要信息
                build_result['chunks_count'], build_result['milvus_inserted']),
            'size': len(content),  # 文件大小
            'chunks_count': build_result['chunks_count'],  # 分块数量
            'milvus_inserted': build_result['milvus_inserted'],  # Milvus入库数量
            'elapsed': build_result['elapsed']  # 构建耗时
        })

    scan_uploaded_files()  # 刷新已上传文件列表
    reload_engine()  # 重载RAG引擎以加载新文件
    return {'success': True, 'results': results, 'total_files': len(_uploaded_files)}  # 返回处理结果
@app.get("/api/files")  # 注册文件列表GET路由
async def list_files():
    files = scan_uploaded_files()  # 扫描并获取已上传文件列表
    return {'files': files, 'total': len(files)}  # 返回文件列表和总数
@app.delete("/api/files/{filename}")  # 注册文件删除DELETE路由
async def delete_file(filename: str):
    filepath = os.path.join(UPLOAD_DIR, filename)  # 构建文件完整路径
    if not os.path.exists(filepath):  # 文件不存在时返回404
        raise HTTPException(status_code=404, detail="文件不存在")  # 抛出404异常
    try:
        os.remove(filepath)  # 删除PDF文件
        csv_path = os.path.splitext(filepath)[0] + '_chunks_v2.csv'  # 构建对应的CSV分块文件路径
        if os.path.exists(csv_path):  # 如果CSV分块文件也存在
            os.remove(csv_path)  # 删除CSV分块文件
        scan_uploaded_files()  # 刷新文件列表
        reload_engine()  # 重载RAG引擎
        return {'success': True, 'message': '文件 {} 已删除'.format(filename)}  # 返回删除成功
    except Exception as e:
        raise HTTPException(status_code=500, detail="删除失败: {}".format(str(e)))  # 返回500错误
@app.post("/api/reload")  # 注册知识库重载POST路由
async def reload_knowledge():
    reload_engine()  # 重置RAG引擎
    scan_uploaded_files()  # 重新扫描上传目录
    return {  # 返回重载结果
        'success': True,  # 操作成功标记
        'message': '知识库已重载，包含 {} 个上传文件'.format(len(_uploaded_files)),  # 重载提示信息
        'files': _uploaded_files  # 当前上传文件列表
    }
# ==================== 检索配置API ====================
@app.get("/api/search/config")  # 注册检索配置GET路由
async def get_search_config():
    """获取当前检索配置"""
    engine, _ = get_engine()  # 获取RAG引擎实例
    return {  # 返回当前检索配置
        'search_mode': engine.search_mode,  # 当前检索模式
        'vector_weight': round(engine.vector_weight, 2),  # 向量检索权重（保留2位小数）
        'bm25_weight': round(engine.bm25_weight, 2),  # BM25检索权重（保留2位小数）
        'reranker_top_k': engine.reranker_top_k,  # 重排器输入top_k
        'final_top_k': engine.final_top_k,  # 最终返回top_k
    }
@app.post("/api/search/config")  # 注册检索配置POST路由
async def set_search_config(req: SearchConfigRequest):
    """设置检索模式和权重"""
    engine, _ = get_engine()  # 获取RAG引擎实例
    if req.search_mode:  # 如果请求中包含检索模式
        engine.set_search_mode(req.search_mode)  # 设置检索模式
    if req.vector_weight is not None or req.bm25_weight is not None:  # 如果请求中包含权重参数
        engine.set_search_mode(  # 重新设置检索模式和权重
            engine.search_mode,  # 保持当前检索模式
            vector_weight=req.vector_weight or engine.vector_weight,  # 更新向量权重（未传则保持原值）
            bm25_weight=req.bm25_weight or engine.bm25_weight  # 更新BM25权重（未传则保持原值）
        )
    return {  # 返回更新后的配置
        'success': True,  # 操作成功标记
        'search_mode': engine.search_mode,  # 当前检索模式
        'vector_weight': round(engine.vector_weight, 2),  # 向量权重
        'bm25_weight': round(engine.bm25_weight, 2),  # BM25权重
    }
@app.post("/api/rerank")  # 注册LLM重排POST路由
async def rerank_with_llm(req: QueryRequest):
    """LLM重排接口：用DeepSeek API对候选文档重新评分"""
    engine, llm = get_engine()  # 获取RAG引擎和LLM客户端
    question = req.question.strip()  # 去除问题首尾空白

    # 先执行混合检索获取候选
    qu = QueryUnderstanding.understand(question)  # 对问题进行意图理解和扩展
    context, results = engine.get_context(qu['expanded_query'], top_k=10, original_query=question)  # 执行混合检索获取候选文档

    if not results:  # 无检索结果时直接返回
        return {'question': question, 'reranked': [], 'message': '无检索结果'}  # 返回空结果

    # LLM重排
    reranked = llm.llm_rerank(question, results, top_k=req.top_k)  # 使用LLM对候选文档重新评分排序

    return {  # 返回重排结果
        'question': question,  # 原始问题
        'reranked': [{  # 重排后的文档列表
            'page': r['page_num'],  # 文档页码
            'score': r.get('llm_rerank_score', r.get('score', 0)),  # LLM重排分数
            'text_preview': r['text'][:200],  # 文本预览（前200字）
            'doc_name': r.get('doc_name', ''),  # 文档名称
        } for r in reranked],  # 遍历重排结果构建列表
    }
# ==================== 问答API ====================
@app.post("/api/ask")  # 注册核心问答POST路由
async def ask_question(req: QueryRequest):
    """核心问答接口：意图识别→混合检索→重排→Prompt构建→LLM生成"""
    try:
        engine, llm = get_engine()  # 获取RAG引擎和LLM客户端
        start = time.time()  # 记录开始时间
        question = req.question.strip()  # 去除问题首尾空白
        lang = req.lang  # 获取语言参数

        if not question:  # 空问题校验
            return JSONResponse(  # 返回400错误
                content={'error': '问题不能为空', 'answer': '', 'sources': []},  # 错误内容
                status_code=400  # HTTP状态码400
            )

        conv_id = req.conversation_id  # 获取请求中的会话ID
        if not conv_id:  # 未传会话ID时自动生成
            conv_id = _store.generate_id()  # 生成新的会话ID
        history = _store.get_history(conv_id)  # 获取对话历史

        qu = QueryUnderstanding.understand(question)  # 意图理解和查询扩展
        search_query = qu['expanded_query']  # 获取扩展后的搜索查询
        logger.info(f"原问题: {question} | 扩展: {search_query}")  # 记录原始问题和扩展查询

        # 如果请求带了search_mode，临时切换
        if req.search_mode:  # 请求指定检索模式时临时切换
            engine.set_search_mode(req.search_mode)  # 设置检索模式

        context, results = engine.get_context(search_query, top_k=req.top_k, original_query=question)  # 执行混合检索获取上下文

        if not context:  # 未找到相关上下文
            answer = "抱歉，在文档中没有找到与您问题相关的内容。请尝试换一种方式提问。"  # 生成无结果提示
            _store.append_message(conv_id, 'user', question)  # 保存用户消息到历史
            _store.append_message(conv_id, 'assistant', answer)  # 保存回答到历史
            return {  # 返回无结果响应
                'question': question,  # 原始问题
                'answer': answer,  # 提示回答
                'sources': [],  # 无引用来源
                'time_seconds': round(time.time() - start, 2),  # 耗时
                'gt_matched': False,  # 未命中标准答案
                'intent': qu['intent'],  # 意图分类结果
                'conversation_id': conv_id  # 会话ID
            }

        # 英文模式：翻译问题→中文检索→翻译回英文
        search_question = question
        if lang == 'en':
            search_question = llm.translate(question, 'zh')  # 英文翻译成中文用于检索

        prompt = engine.build_prompt(search_question, context, history=history, lang=lang)  # 构建LLM提示词
        answer = llm.generate(prompt, context, search_question, history=history)  # 调用LLM生成回答

        # 英文模式：将中文回答翻译成英文
        if lang == 'en':
            answer = llm.translate(answer, 'en')  # 翻译回英文

        _store.append_message(conv_id, 'user', question)  # 保存用户消息到历史
        _store.append_message(conv_id, 'assistant', answer)  # 保存助手回答到历史

        elapsed = round(time.time() - start, 2)  # 计算总耗时
        gt_matched = any(answer == gt or (len(gt) > 10 and gt[:30] in answer)  # 检查是否匹配标准答案
                         for gt in GROUND_TRUTH.values())  # 遍历所有标准答案进行匹配

        sources = [  # 构建引用来源列表
            {
                'page': r['page_num'],  # 文档页码
                'score': r.get('rerank_score', r.get('rrf_score', r.get('score', 0))),  # 综合评分
                'snippet': r['text'][:200],  # 文本摘要（前200字）
                'doc_name': r.get('doc_name', ''),  # 文档名称
                'source_type': r.get('source_type', 'text'),  # 来源类型（文本/表格）
            }
            for r in results[:req.top_k]  # 取前top_k个结果
        ]

        return {  # 返回完整问答结果
            'question': question,  # 原始问题
            'answer': answer,  # LLM生成的回答
            'sources': sources,  # 引用来源列表
            'time_seconds': elapsed,  # 耗时（秒）
            'gt_matched': gt_matched,  # 是否命中标准答案
            'intent': qu['intent'],  # 意图分类
            'conversation_id': conv_id,  # 会话ID
            'history_length': len(history) + 2,  # 对话历史长度（含本次）
            'search_mode': engine.search_mode,  # 使用的检索模式
        }

    except FileNotFoundError as e:
        logger.error(f"文件未找到: {e}")  # 记录文件未找到错误
        return JSONResponse(  # 返回500错误
            content={'error': f'PDF文件未找到: {str(e)}', 'answer': '', 'sources': []},  # 错误详情
            status_code=500  # HTTP状态码500
        )
    except Exception as e:
        logger.error(f"问答接口异常: {e}", exc_info=True)  # 记录异常日志（含堆栈信息）
        return JSONResponse(  # 返回通用错误响应
            content={
                'error': '系统内部错误，请稍后重试',  # 错误提示
                'answer': '抱歉，系统处理您的问题时出现异常，请稍后重试。',  # 兜底回答
                'sources': []  # 无引用来源
            },
            status_code=500  # HTTP状态码500
        )
@app.post("/api/compare")  # 注册对比接口POST路由
async def compare_answers(req: QueryRequest):
    """对比接口：RAG vs 纯LLM"""
    try:
        engine, llm = get_engine()  # 获取RAG引擎和LLM客户端
        start = time.time()  # 记录开始时间
        question = req.question.strip()  # 去除问题首尾空白
        lang = req.lang  # 获取语言参数
        if not question:  # 空问题校验
            return JSONResponse(content={'error': '问题不能为空'}, status_code=400)  # 返回400错误
        conv_id = req.conversation_id  # 获取会话ID
        if not conv_id:  # 未传会话ID时自动生成
            conv_id = _store.generate_id()  # 生成新会话ID
        history = _store.get_history(conv_id)  # 获取对话历史
        qu = QueryUnderstanding.understand(question)  # 意图理解和查询扩展
        search_query = qu['expanded_query']  # 获取扩展后的搜索查询
        context, results = engine.get_context(search_query, top_k=req.top_k, original_query=question)  # 执行混合检索
        # 英文模式：翻译问题→中文检索
        search_question = question
        if lang == 'en':
            search_question = llm.translate(question, 'zh')  # 英文翻译成中文

        if context:  # 找到相关上下文时生成RAG回答
            prompt = engine.build_prompt(search_question, context, history=history, lang=lang)  # 构建提示词
            rag_answer = llm.generate(prompt, context, search_question, history=history)  # LLM生成RAG回答
        else:  # 未找到相关上下文
            rag_answer = "抱歉，在文档中没有找到与您问题相关的内容。"  # 生成无结果提示

        # 英文模式：将中文回答翻译成英文
        if lang == 'en':
            rag_answer = llm.translate(rag_answer, 'en')  # 翻译回英文
        rag_time = round(time.time() - start, 2)  # 计算RAG回答耗时
        sources = [  # 构建引用来源列表
            {
                'page': r['page_num'],  # 文档页码
                'score': r.get('rerank_score', r.get('rrf_score', r.get('score', 0))),  # 综合评分
                'snippet': r['text'][:200],  # 文本摘要
                'doc_name': r.get('doc_name', ''),  # 文档名称
                'source_type': r.get('source_type', 'text'),  # 来源类型
            }
            for r in results[:req.top_k]  # 取前top_k个结果
        ] if context else []  # 无上下文时返回空列表
        gt_matched = any(rag_answer == gt for gt in GROUND_TRUTH.values())  # 检查是否匹配标准答案
        llm_start = time.time()  # 记录纯LLM开始时间
        llm_answer = llm.generate_pure_llm(question)  # 调用纯LLM生成回答（无RAG检索）
        llm_time = round(time.time() - llm_start, 2)  # 计算纯LLM回答耗时
        _store.append_message(conv_id, 'user', question)  # 保存用户消息到历史
        _store.append_message(conv_id, 'assistant', rag_answer)  # 保存RAG回答到历史
        answers_match = rag_answer.strip()[:50] == llm_answer.strip()[:50]  # 比较两个回答前50字是否一致
        rag_has_sources = len(sources) > 0  # RAG回答是否有引用来源
        rag_is_more_detailed = len(rag_answer) > len(llm_answer) if not answers_match else False  # RAG回答是否更详细
        return {  # 返回对比结果
            'question': question,  # 原始问题
            'rag': {  # RAG回答信息
                'answer': rag_answer,  # RAG生成的回答
                'sources': sources,  # 引用来源
                'time_seconds': rag_time,  # RAG耗时
                'gt_matched': gt_matched,  # 是否命中标准答案
                'intent': qu['intent'],  # 意图分类
                'has_context': bool(context)  # 是否有检索上下文
            },
            'llm_only': {  # 纯LLM回答信息
                'answer': llm_answer,  # 纯LLM生成的回答
                'time_seconds': llm_time,  # 纯LLM耗时
                'has_api': llm.api_available  # LLM API是否可用
            },
            'analysis': {  # 对比分析结果
                'answers_match': answers_match,  # 两个回答是否一致
                'rag_has_sources': rag_has_sources,  # RAG是否有引用来源
                'rag_has_citations': rag_has_sources,  # RAG是否有引用（与上面相同）
                'rag_is_more_detailed': rag_is_more_detailed,  # RAG回答是否更详细
                'rag_is_faster': rag_time < llm_time,  # RAG是否更快
                'rag_len': len(rag_answer),  # RAG回答长度
                'llm_len': len(llm_answer),  # LLM回答长度
                'sources_count': len(sources)  # 引用来源数量
            },
            'conversation_id': conv_id,  # 会话ID
            'history_length': len(history) + 2  # 对话历史长度
        }
    except Exception as e:
        logger.error(f"对比接口异常: {e}", exc_info=True)  # 记录异常日志
        return JSONResponse(  # 返回错误响应
            content={'error': '对比接口异常: {}'.format(str(e))},  # 错误详情
            status_code=500  # HTTP状态码500
        )
# ==================== 用户反馈API ====================
@app.post("/api/feedback")  # 注册用户反馈POST路由
async def submit_feedback(req: FeedbackRequest):
    """用户反馈：记录问答质量评价，用于优化检索策略"""
    feedback_data = {  # 构建反馈数据字典
        'question': req.question,  # 用户原始问题
        'answer': req.answer[:200],  # 系统回答（截取前200字）
        'rating': req.rating,  # 评价：good或bad
        'comment': req.comment or '',  # 用户评论（空则默认空字符串）
        'timestamp': time.time(),  # 反馈时间戳
    }
    # 持久化到文件（简单方案）
    feedback_file = os.path.join(BASE_DIR, 'feedback.jsonl')  # 反馈数据文件路径
    try:
        with open(feedback_file, 'a', encoding='utf-8') as f:  # 以追加模式打开反馈文件
            f.write(json.dumps(feedback_data, ensure_ascii=False) + '\n')  # 序列化为JSON并写入一行
    except Exception as e:
        logger.warning(f"反馈保存失败: {e}")  # 记录保存失败警告

    return {'success': True, 'message': '感谢您的反馈！'}  # 返回保存成功
# ==================== 健康检查 ====================
@app.get("/api/health")  # 注册健康检查GET路由
async def health_check():
    try:
        engine, llm = get_engine()  # 获取RAG引擎和LLM客户端
        doc_info = {name: len(e.chunks) for name, e in engine.doc_engines.items()}  # 统计每个文档的分块数
        # 检查嵌入模型和重排模型
        embedder = engine._get_embedder()  # 获取向量嵌入模型实例
        reranker = engine._get_reranker()  # 获取重排模型实例
        return HealthResponse(  # 返回健康状态
            status="healthy",  # 服务状态健康
            engine_ready=True,  # 引擎就绪
            backend=RAG_BACKEND,  # RAG后端类型
            llm_available=llm.api_available,  # LLM API可用性
            pdf_loaded=len(engine.pdf_paths) > 0 if engine else False,  # 是否加载了PDF
            redis_available=_store.available,  # Redis可用性
            search_mode=engine.search_mode,  # 当前检索模式
            embedder_available=embedder is not None and embedder.is_available(),  # 嵌入模型可用性
            reranker_available=reranker is not None and reranker.is_available(),  # 重排模型可用性
        )
    except Exception as e:
        return HealthResponse(  # 异常时返回降级状态
            status="degraded",  # 服务状态降级
            engine_ready=False,  # 引擎未就绪
            backend=RAG_BACKEND,  # RAG后端类型
            llm_available=False,  # LLM不可用
            pdf_loaded=False,  # 未加载PDF
            redis_available=_store.available,  # Redis可用性
        )
# ==================== 对话管理 ====================
@app.get("/api/conversations/{conversation_id}")  # 注册获取对话历史GET路由
async def get_conversation(conversation_id: str):
    history = _store.get_history(conversation_id)  # 获取指定会话的历史记录
    return {  # 返回对话历史
        'conversation_id': conversation_id,  # 会话ID
        'history': history,  # 对话历史列表
        'message_count': len(history),  # 消息总数
        'round_count': len(history) // 2  # 对话轮次（每轮含用户+助手）
    }
@app.delete("/api/conversations/{conversation_id}")  # 注册删除单个对话DELETE路由
async def delete_conversation(conversation_id: str):
    _store.delete_conversation(conversation_id)  # 删除指定会话的对话历史
    return {'success': True, 'message': '会话 {} 已删除'.format(conversation_id)}  # 返回删除成功
@app.delete("/api/conversations")  # 注册清空所有对话DELETE路由
async def clear_all_conversations():
    _store.clear_all()  # 清空所有对话历史
    return {'success': True, 'message': '所有对话历史已清空'}  # 返回清空成功
@app.get("/api/understand")  # 注册意图理解调试GET路由
async def understand_question(q: str = ""):
    """调试接口：查看Query Understanding结果"""
    if not q:  # 参数为空时返回错误
        return JSONResponse(content={"error": "请提供参数 ?q=您的问题"}, status_code=400)  # 返回400错误
    result = QueryUnderstanding.understand(q)  # 执行意图理解和查询扩展
    return JSONResponse(content=result)  # 返回理解结果JSON
@app.get("/api/debug_route")  # 注册调试路由GET路由
async def debug_route(q: str = ""):
    """调试接口：查看路由+检索+融合全过程"""
    engine, _ = get_engine()  # 获取RAG引擎实例
    route = engine._route_query(q)  # 执行查询路由判断
    context, results = engine.get_context(q, top_k=5, original_query=q)  # 执行检索获取上下文和结果
    return JSONResponse(content={  # 返回完整调试信息
        "query": q,  # 原始查询
        "route": route,  # 路由判断结果
        "search_mode": engine.search_mode,  # 检索模式
        "vector_weight": engine.vector_weight,  # 向量权重
        "bm25_weight": engine.bm25_weight,  # BM25权重
        "doc_engines_keys": list(engine.doc_engines.keys()),  # 文档引擎列表
        "results_count": len(results),  # 检索结果数量
        "results": [{  # 检索结果详情
            "doc_name": r.get("doc_name", ""),  # 文档名称
            "page": r["page_num"],  # 页码
            "score": r.get("rerank_score", r.get("rrf_score", r.get("score", 0))),  # 综合评分
            "type": r.get("source_type", "text"),  # 来源类型
            "source": r.get("source", "unknown"),  # 来源标识
            "text_preview": r["text"][:100],  # 文本预览（前100字）
        } for r in results]  # 遍历所有结果
    })
# ==================== 启动 ====================
def find_available_port(start_port=8888, max_tries=10):
    """端口探测"""
    import socket
    for port in range(start_port, start_port + max_tries):  # 从起始端口依次尝试
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:  # 创建TCP socket
            try:
                s.bind(('127.0.0.1', port))  # 尝试绑定端口
                return port  # 绑定成功则返回该端口
            except OSError:  # 端口被占用时
                continue  # 尝试下一个端口
    raise RuntimeError("无法找到可用端口")  # 所有端口都不可用时抛出异常
def main():
    port = find_available_port(start_port=8888)  # 查找可用端口
    scan_uploaded_files()  # 扫描已上传文件
    # === CUDA检测 ===
    try:
        import torch
        if torch.cuda.is_available():  # 检测CUDA是否可用
            gpu_name = torch.cuda.get_device_name(0)  # 获取GPU设备名称
            print(f"  GPU: {gpu_name}")  # 打印GPU信息
        else:
            print("  [警告] 当前运行环境为 CPU 模式！")  # 打印CPU警告
            print("  bge-m3编码速度会很慢，上传PDF会卡住")  # 提示速度慢
            print("  请用 nlp_1 Conda 环境启动:")  # 提示正确的启动环境
            print("    D:\\an\\envs\\nlp_1\\python.exe app.py")  # 启动命令
    except ImportError:
        print("  ⚠️  未安装 PyTorch")  # 提示未安装PyTorch
    print("=" * 56)  # 打印分隔线
    print("  RAG 问答系统 — 混合检索（向量+BM25+RRF+Reranker）")  # 打印系统名称
    print("  向量模型: bge-m3 (1024维)")  # 打印向量模型信息
    print("  全文检索: BM25 (jieba分词)")  # 打印BM25信息
    print("  融合算法: RRF (Reciprocal Rank Fusion)")  # 打印融合算法信息
    print("  重排模型: bge-reranker-v2-m3")  # 打印重排模型信息
    # 扫描BASE_DIR下的PDF文件
    pdf_files = []  # 存储PDF文件列表
    if os.path.isdir(BASE_DIR):  # 检查项目根目录是否存在
        for f in os.listdir(BASE_DIR):  # 遍历根目录文件
            if f.lower().endswith('.pdf') and not f.startswith('~$'):  # 过滤PDF且排除临时文件
                pdf_files.append(f)  # 加入PDF文件列表
    print("  PDF目录: {} ({} 个PDF文件)".format(BASE_DIR, len(pdf_files)))  # 打印PDF目录和数量
    if pdf_files:  # 有PDF文件时打印列表
        for i, pdf in enumerate(pdf_files[:5], 1):  # 最多显示5个
            print("    {}. {}".format(i, pdf))  # 打印文件序号和名称
        if len(pdf_files) > 5:  # 超过5个时显示总数
            print("    ... 共{}个文件".format(len(pdf_files)))  # 打印省略提示
    print("  上传目录: {} ({} 个文件)".format(UPLOAD_DIR, len(_uploaded_files)))  # 打印上传目录和文件数
    print("=" * 56)  # 打印分隔线
    # 预加载模型
    print("\n[启动] 预加载模型...")  # 提示预加载开始
    try:
        import torch
        device_str = 'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'  # 判断设备类型
        from models import EmbeddingClient
        emb = EmbeddingClient()  # 创建嵌入模型客户端
        emb.encode("预加载测试")  # 用测试文本触发模型加载
        print(f"[启动] bge-m3 embedding → {device_str} 加载完成")  # 打印嵌入模型加载完成
    except Exception as e:
        print(f"[启动] bge-m3预加载跳过: {e}")  # 打印嵌入模型加载跳过原因
    try:
        from models import RerankerClient
        rkr = RerankerClient()  # 创建重排模型客户端
        if rkr.is_available():  # 检查重排模型是否可用
            rkr.rerank("预加载测试", [{"text": "测试"}], top_k=1)  # 用测试数据触发模型加载
            print(f"[启动] bge-reranker-v2-m3 → {device_str} 加载完成")  # 打印重排模型加载完成
    except Exception as e:
        print(f"[启动] reranker预加载跳过: {e}")  # 打印重排模型加载跳过原因
    print()  # 打印空行分隔
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")  # 启动FastAPI服务
if __name__ == '__main__':  # 脚本主入口判断
    main()  # 执行主函数启动服务
