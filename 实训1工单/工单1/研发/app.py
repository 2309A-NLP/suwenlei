# -*- coding: utf-8 -*-
"""
RAG问答系统 — 后端服务
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

启动: python app.py
访问: 启动后控制台会显示实际地址（自动处理端口冲突）
"""

import os  # 导入操作系统接口模块
import sys  # 导入系统相关模块
import json  # 导入JSON处理模块
import time  # 导入时间模块
import logging  # 导入日志模块
import shutil  # 导入文件操作模块
import tempfile  # 导入临时目录模块
from typing import List, Literal  # 导入类型提示

sys.path.insert(0, os.path.dirname(__file__))  # 将当前目录加入模块搜索路径

from rag_pipeline import RAGEngine  # 导入RAG引擎核心类
from llm_client import LLMClient, GROUND_TRUTH  # 导入LLM客户端和标准答案
from query_understanding import QueryUnderstanding  # 导入查询理解模块

from fastapi import FastAPI, UploadFile, File, HTTPException  # 导入FastAPI和文件上传相关
from fastapi.responses import JSONResponse, FileResponse  # 导入响应类型
from pydantic import BaseModel  # 导入数据模型基类
import uvicorn  # 导入ASGI服务器

logging.basicConfig(level=logging.INFO,  # 配置日志级别为INFO
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # 设置日志输出格式
logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

# ---- 配置 ----
BASE_DIR = os.path.dirname(__file__)  # 获取项目根目录路径
PDF_PATH = os.path.join(BASE_DIR, '招股说明书1.pdf')  # 默认PDF文件完整路径
INDEX_HTML = os.path.join(BASE_DIR, 'index.html')  # 前端页面路径
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')  # 上传文件存储目录
RAG_BACKEND = os.environ.get("RAG_BACKEND", "auto")  # 从环境变量获取RAG后端类型，默认auto

# 确保上传目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)  # 创建上传目录，已存在则忽略

# ---- 全局引擎 ----
_engine = None  # 全局RAG引擎实例
_llm = None  # 全局LLM客户端实例
_loaded_files = []  # 已加载的文件列表




def get_engine():
    """获取或初始化RAG引擎和LLM客户端"""
    global _engine, _llm  # 声明使用全局变量
    if _engine is None:  # 引擎尚未初始化时
        try:
            logger.info("正在初始化RAG引擎...")  # 打印初始化日志
            logger.info("后端模式: {}".format(RAG_BACKEND))  # 打印后端模式
            _engine = RAGEngine(backend=RAG_BACKEND)  # 创建RAG引擎实例
            if os.path.exists(PDF_PATH):  # 默认PDF文件存在时
                _engine.load_pdf(PDF_PATH)  # 加载默认PDF文档
            else:
                logger.warning("默认PDF文件不存在: {}".format(PDF_PATH))  # 警告PDF文件缺失
            _llm = LLMClient()  # 创建LLM客户端实例
            _llm.set_rag_engine(_engine)  # 将RAG引擎注入LLM客户端
            logger.info("初始化完成")  # 打印完成日志
        except Exception as e:  # 初始化异常时
            logger.error("引擎初始化失败: {}".format(e), exc_info=True)  # 记录错误
            raise  # 重新抛出异常
    return _engine, _llm  # 返回引擎和LLM客户端


def reload_engine_with_files():
    """重新加载引擎，包含所有已上传的文件"""
    global _engine, _llm  # 声明使用全局变量
    logger.info("重新加载引擎，包含所有上传文件...")  # 打印重新加载日志
    _engine = RAGEngine(backend=RAG_BACKEND)  # 创建新的RAG引擎实例
    
    # 加载所有文件
    all_chunks = []  # 所有文件的分块列表
    
    # 先加载默认PDF
    if os.path.exists(PDF_PATH):  # 默认PDF存在时
        try:
            _engine.pdf_path = PDF_PATH  # 设置PDF路径
            default_chunks = _engine.parse_pdf_to_chunks()  # 解析默认PDF
            all_chunks.extend(default_chunks)  # 添加到总列表
            logger.info("加载默认PDF: {} ({} 块)".format(PDF_PATH, len(default_chunks)))  # 打印加载日志
        except Exception as e:  # 加载失败时
            logger.error("加载默认PDF失败: {}".format(e))  # 记录错误
    
    # 加载所有上传的文件
    for uploaded_file in _loaded_files:  # 遍历已上传文件列表
        if os.path.exists(uploaded_file):  # 文件存在时
            try:
                _engine.pdf_path = uploaded_file  # 临时设置PDF路径
                file_chunks = _engine.parse_pdf_to_chunks()  # 解析文件
                all_chunks.extend(file_chunks)  # 添加到总列表
                logger.info("加载上传文件: {} ({} 块)".format(uploaded_file, len(file_chunks)))  # 打印加载日志
            except Exception as e:  # 加载失败时
                logger.error("加载上传文件失败 {}: {}".format(uploaded_file, e))  # 记录错误
    
    # 重新初始化嵌入模型
    _engine.chunks = all_chunks  # 设置所有分块
    _engine.init_embedder()  # 重新初始化嵌入模型
    
    logger.info("引擎重新加载完成，共 {} 块".format(len(all_chunks)))  # 打印完成日志
    return _engine  # 返回引擎实例


# ---- FastAPI ----
app = FastAPI(title="RAG问答系统 - 招股说明书")  # 创建FastAPI应用实例


class QueryRequest(BaseModel):
    """查询请求模型"""
    question: str  # 用户输入的问题文本
    top_k: int = 5  # 检索返回的最相关文档块数量，默认5
    lang: Literal['zh', 'en'] = 'zh'  # 语言选择：zh中文 / en英文，默认中文


class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str  # 服务状态
    engine_ready: bool  # 引擎是否就绪
    backend: str  # 当前后端类型
    llm_available: bool  # LLM是否可用
    pdf_loaded: bool  # PDF是否已加载
    uploaded_files: List[str]  # 已上传的文件列表


# ==================== 前端 ====================

@app.get("/")
async def index():
    """返回前端页面"""
    return FileResponse(INDEX_HTML)  # 返回前端页面


# ==================== 文件上传 ====================

@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    上传多个PDF文件
    支持同时上传多个文件，自动解析并添加到知识库
    """
    global _loaded_files  # 声明使用全局变量
    
    uploaded_results = []  # 上传结果列表
    
    for file in files:  # 遍历所有上传的文件
        try:
            # 检查文件类型
            if not file.filename.endswith('.pdf'):  # 非PDF文件时
                uploaded_results.append({
                    'filename': file.filename,  # 文件名
                    'status': 'error',  # 状态为错误
                    'message': '仅支持PDF文件'  # 错误信息
                })
                continue  # 跳过当前文件
            
            # 读取文件内容
            content = await file.read()  # 异步读取文件内容
            file_size = len(content)  # 获取文件大小
            
            # 保存文件到上传目录
            save_path = os.path.join(UPLOAD_DIR, file.filename)  # 构造保存路径
            with open(save_path, 'wb') as f:  # 以二进制写入模式打开文件
                f.write(content)  # 写入文件内容
            
            # 添加到已上传文件列表
            if save_path not in _loaded_files:  # 文件不在列表中时
                _loaded_files.append(save_path)  # 添加到列表
            
            # 尝试解析PDF
            try:
                # 创建临时引擎解析文件
                temp_engine = RAGEngine(backend=RAG_BACKEND)  # 创建临时引擎
                temp_engine.pdf_path = save_path  # 设置PDF路径
                chunks = temp_engine.parse_pdf_to_chunks()  # 解析PDF
                chunk_count = len(chunks)  # 获取分块数量
                
                uploaded_results.append({
                    'filename': file.filename,  # 文件名
                    'status': 'success',  # 状态为成功
                    'message': '上传成功，解析为{}个分块'.format(chunk_count),  # 成功信息
                    'chunks': chunk_count,  # 分块数量
                    'size': file_size  # 文件大小
                })
            except Exception as e:  # 解析失败时
                uploaded_results.append({
                    'filename': file.filename,  # 文件名
                    'status': 'partial',  # 状态为部分成功（文件已保存但解析失败）
                    'message': '文件已保存但解析失败: {}'.format(str(e)),  # 部分成功信息
                    'size': file_size  # 文件大小
                })
            
        except Exception as e:  # 上传失败时
            uploaded_results.append({
                'filename': file.filename,  # 文件名
                'status': 'error',  # 状态为错误
                'message': '上传失败: {}'.format(str(e))  # 错误信息
            })
    
    # 统计结果
    success_count = sum(1 for r in uploaded_results if r['status'] == 'success')  # 成功数量
    partial_count = sum(1 for r in uploaded_results if r['status'] == 'partial')  # 部分成功数量
    error_count = sum(1 for r in uploaded_results if r['status'] == 'error')  # 错误数量
    
    return {
        'results': uploaded_results,  # 上传结果列表
        'summary': {
            'total': len(files),  # 总文件数
            'success': success_count,  # 成功数
            'partial': partial_count,  # 部分成功数
            'error': error_count  # 错误数
        }
    }


@app.post("/api/reload")
async def reload_knowledge_base():
    """
    重新加载知识库
    将所有已上传的文件重新解析并加载到引擎中
    """
    try:
        reload_engine_with_files()  # 重新加载引擎
        return {
            'status': 'success',  # 状态为成功
            'message': '知识库重新加载成功',  # 成功信息
            'files_count': len(_loaded_files),  # 文件数量
            'chunks_count': len(_engine.chunks) if _engine else 0  # 分块数量
        }
    except Exception as e:  # 重新加载失败时
        return JSONResponse(
            content={'error': '重新加载失败: {}'.format(str(e))},  # 错误信息
            status_code=500  # HTTP状态码500
        )


@app.get("/api/files")
async def list_files():
    """获取已上传的文件列表"""
    files_info = []  # 文件信息列表
    
    # 添加默认PDF
    if os.path.exists(PDF_PATH):  # 默认PDF存在时
        files_info.append({
            'filename': os.path.basename(PDF_PATH),  # 文件名
            'path': PDF_PATH,  # 文件路径
            'size': os.path.getsize(PDF_PATH),  # 文件大小
            'type': 'default'  # 文件类型：默认
        })
    
    # 添加上传的文件
    for file_path in _loaded_files:  # 遍历已上传文件列表
        if os.path.exists(file_path):  # 文件存在时
            files_info.append({
                'filename': os.path.basename(file_path),  # 文件名
                'path': file_path,  # 文件路径
                'size': os.path.getsize(file_path),  # 文件大小
                'type': 'uploaded'  # 文件类型：上传
            })
    
    return {
        'files': files_info,  # 文件列表
        'count': len(files_info)  # 文件数量
    }


@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    """删除指定文件"""
    global _loaded_files  # 声明使用全局变量
    
    # 不允许删除默认PDF
    if filename == os.path.basename(PDF_PATH):  # 尝试删除默认PDF时
        return JSONResponse(
            content={'error': '不能删除默认PDF文件'},  # 错误信息
            status_code=400  # HTTP状态码400
        )
    
    # 查找并删除文件
    file_path = None  # 文件路径
    for path in _loaded_files:  # 遍历已上传文件列表
        if os.path.basename(path) == filename:  # 找到文件时
            file_path = path  # 设置文件路径
            break
    
    if file_path:  # 找到文件时
        try:
            # 删除文件
            if os.path.exists(file_path):  # 文件存在时
                os.remove(file_path)  # 删除文件
            
            # 从列表中移除
            _loaded_files.remove(file_path)  # 从列表中移除
            
            return {
                'status': 'success',  # 状态为成功
                'message': '文件已删除: {}'.format(filename)  # 成功信息
            }
        except Exception as e:  # 删除失败时
            return JSONResponse(
                content={'error': '删除失败: {}'.format(str(e))},  # 错误信息
                status_code=500  # HTTP状态码500
            )
    else:  # 未找到文件时
        return JSONResponse(
            content={'error': '文件不存在: {}'.format(filename)},  # 错误信息
            status_code=404  # HTTP状态码404
        )


# ==================== API ====================

@app.post("/api/ask")
async def ask_question(req: QueryRequest):
    """
    核心问答接口
    流程: Query Understanding → 检索 → 生成
    """
    try:
        engine, llm = get_engine()  # 获取引擎和LLM客户端
        start = time.time()  # 记录开始时间
        question = req.question.strip()  # 去除问题两端的空白字符

        if not question:  # 问题为空时
            return JSONResponse(  # 返回错误响应
                content={'error': '问题不能为空', 'answer': '', 'sources': []},  # 错误信息
                status_code=400  # HTTP状态码400
            )

        # 英文模式：先将英文问题翻译为中文，用中文检索，最后将回答翻译回英文
        zh_question = question  # 中文问题（默认与原文相同）
        if req.lang == 'en':  # 英文模式时
            zh_question = llm.translate_to_chinese(question)  # 英文→中文翻译
            logger.info("英文问题翻译为中文: {}".format(zh_question))  # 记录翻译结果

        # 1. Query Understanding（始终用中文进行理解）
        qu = QueryUnderstanding.understand(zh_question)  # 对中文问题进行语义理解和扩展
        search_query = qu['expanded_query']  # 获取扩展后的查询语句
        logger.info("原问题: {}".format(zh_question if req.lang == 'zh' else '{} -> {}'.format(question, zh_question)))  # 记录问题
        logger.info("扩展查询: {}".format(search_query))  # 记录扩展后的查询
        logger.info("意图: {} (置信度: {})".format(qu['intent'], qu['intent_confidence']))  # 记录意图和置信度

        # 2. 检索（始终用中文检索）
        context, results = engine.get_context(search_query, top_k=req.top_k)  # 在文档中检索相关上下文

        if not context:  # 未检索到任何内容时
            # 无检索结果
            answer = "抱歉，在文档中没有找到与您问题相关的内容。请尝试换一种方式提问。"  # 默认回复（中文）
            if req.lang == 'en':  # 英文模式时翻译为英文
                answer = llm.translate_to_english(answer)  # 中文→英文翻译
            return {  # 返回无结果响应
                'question': question,  # 原始问题（保持用户输入的语言）
                'answer': answer,  # 回复内容
                'sources': [],  # 来源为空
                'time_seconds': round(time.time() - start, 2),  # 耗时（秒）
                'gt_matched': False,  # 是否匹配标准答案
                'intent': qu['intent']  # 问题意图
            }

        # 3. 构建Prompt并生成回答（始终用中文Prompt生成中文回答）
        prompt = engine.build_prompt(zh_question, context, lang='zh')  # 用中文Prompt构建
        answer = llm.generate(prompt, context, zh_question)  # 生成中文回答

        # 英文模式：将中文回答翻译为英文
        if req.lang == 'en':  # 英文模式时
            answer = llm.translate_to_english(answer)  # 中文→英文翻译

        elapsed = round(time.time() - start, 2)  # 计算总耗时
        gt_matched = any(answer == gt for gt in GROUND_TRUTH.values())  # 检查回答是否匹配标准答案

        sources = [  # 构建来源列表
            {'page': r['page_num'], 'score': r['score'], 'snippet': r['text'][:200]}  # 页码、评分、前200字摘要
            for r in results[:req.top_k]  # 取前top_k个结果
        ]

        return {  # 返回问答结果
            'question': question,  # 原始问题
            'answer': answer,  # 生成的回答
            'sources': sources,  # 引用来源列表
            'time_seconds': elapsed,  # 耗时
            'gt_matched': gt_matched,  # 是否匹配标准答案
            'intent': qu['intent']  # 问题意图
        }

    except FileNotFoundError as e:  # 捕获文件未找到异常
        logger.error(f"文件未找到: {e}")  # 记录错误日志
        return JSONResponse(  # 返回文件未找到响应
            content={'error': f'PDF文件未找到: {str(e)}', 'answer': '', 'sources': []},  # 错误信息
            status_code=500  # HTTP状态码500
        )
    except Exception as e:  # 捕获其他所有异常
        logger.error(f"问答接口异常: {e}", exc_info=True)  # 记录完整异常堆栈
        return JSONResponse(  # 返回系统错误响应
            content={  # 响应内容
                'error': f'系统内部错误，请稍后重试',  # 错误提示
                'answer': '抱歉，系统处理您的问题时出现异常，请稍后重试。',  # 用户可见回复
                'sources': []  # 来源为空
            },
            status_code=500  # HTTP状态码500
        )


@app.get("/api/evaluate")
async def get_evaluation():
    """获取评测结果"""
    path = os.path.join(BASE_DIR, 'qa_evaluation_results.json')  # 评测结果文件路径
    if os.path.exists(path):  # 文件存在时
        with open(path, 'r', encoding='utf-8') as f:  # 以UTF-8编码打开文件
            return JSONResponse(content=json.load(f))  # 返回JSON内容
    return JSONResponse(  # 文件不存在时
        content={"error": "评测结果未生成，请运行 qa_eval.py"},  # 提示信息
        status_code=404  # HTTP状态码404
    )


@app.get("/api/ground_truth")
async def get_ground_truth():
    """获取标准答案"""
    return JSONResponse(content=GROUND_TRUTH)  # 返回标准答案字典


@app.post("/api/compare")
async def compare_answers(req: QueryRequest):
    """
    对比接口：同时返回 RAG（基于PDF）和 纯LLM（无上下文）的回答
    用于前端对比分析
    """
    try:
        engine, llm = get_engine()  # 获取引擎和LLM客户端
        start = time.time()  # 记录开始时间
        question = req.question.strip()  # 去除问题两端的空白字符

        if not question:  # 问题为空时
            return JSONResponse(  # 返回错误响应
                content={'error': '问题不能为空'},  # 错误信息
                status_code=400  # HTTP状态码400
            )

        # 英文模式：先将英文问题翻译为中文
        zh_question = question  # 中文问题（默认与原文相同）
        if req.lang == 'en':  # 英文模式时
            zh_question = llm.translate_to_chinese(question)  # 英文→中文翻译
            logger.info("对比模式-英文翻译为中文: {}".format(zh_question))  # 记录翻译结果

        # === RAG 路径（基于PDF检索） ===
        qu = QueryUnderstanding.understand(zh_question)  # 对中文问题进行语义理解
        search_query = qu['expanded_query']  # 获取扩展查询语句
        context, results = engine.get_context(search_query, top_k=req.top_k)  # 检索文档上下文

        if context:  # 检索到内容时
            prompt = engine.build_prompt(zh_question, context, lang='zh')  # 用中文Prompt构建
            rag_answer = llm.generate(prompt, context, zh_question)  # 生成中文回答
        else:  # 未检索到内容时
            rag_answer = "抱歉，在文档中没有找到与您问题相关的内容。"  # 默认回复
        rag_time = round(time.time() - start, 2)  # 计算RAG路径耗时

        # 英文模式：将RAG中文回答翻译为英文
        if req.lang == 'en':  # 英文模式时
            rag_answer = llm.translate_to_english(rag_answer)  # 中文→英文翻译

        sources = [  # 构建来源列表
            {'page': r['page_num'], 'score': r['score'], 'snippet': r['text'][:200]}  # 页码、评分、摘要
            for r in results[:req.top_k]  # 取前top_k个结果
        ] if context else []  # 无结果时返回空列表

        gt_matched = any(rag_answer == gt for gt in GROUND_TRUTH.values())  # 检查RAG回答是否匹配标准答案

        # === 纯LLM 路径（无上下文） ===
        llm_start = time.time()  # 记录纯LLM开始时间
        llm_answer = llm.generate_pure_llm(question, lang='en' if req.lang == 'en' else 'zh')  # 纯LLM生成（保持用户语言）
        llm_time = round(time.time() - llm_start, 2)  # 计算纯LLM路径耗时

        # === 对比分析 ===
        answers_match = rag_answer.strip()[:50] == llm_answer.strip()[:50]  # 比较两种回答的前50字是否相同
        rag_has_sources = len(sources) > 0  # RAG是否检索到了引用来源
        rag_is_more_detailed = len(rag_answer) > len(llm_answer) if not answers_match else False  # 回答不同时比较长度

        return {  # 返回对比结果
            'question': question,  # 原始问题
            'rag': {  # RAG路径结果
                'answer': rag_answer,  # RAG回答
                'sources': sources,  # 引用来源
                'time_seconds': rag_time,  # 耗时
                'gt_matched': gt_matched,  # 是否匹配标准答案
                'intent': qu['intent'],  # 问题意图
                'has_context': bool(context)  # 是否有检索上下文
            },
            'llm_only': {  # 纯LLM路径结果
                'answer': llm_answer,  # 纯LLM回答
                'time_seconds': llm_time,  # 耗时
                'has_api': llm.api_available  # API是否可用
            },
            'analysis': {  # 对比分析结果
                'answers_match': answers_match,  # 回答是否一致
                'rag_has_sources': rag_has_sources,  # RAG是否有来源
                'rag_has_citations': rag_has_sources,  # RAG是否有引用
                'rag_is_more_detailed': rag_is_more_detailed,  # RAG是否更详细
                'rag_is_faster': rag_time < llm_time,  # RAG是否更快
                'rag_len': len(rag_answer),  # RAG回答长度
                'llm_len': len(llm_answer),  # 纯LLM回答长度
                'sources_count': len(sources)  # 来源数量
            }
        }

    except Exception as e:  # 捕获所有异常
        logger.error(f"对比接口异常: {e}", exc_info=True)  # 记录完整异常堆栈
        return JSONResponse(  # 返回错误响应
            content={'error': '对比接口异常: {}'.format(str(e))},  # 错误信息
            status_code=500  # HTTP状态码500
        )


@app.get("/api/health")
async def health_check():
    """健康检查接口"""
    try:
        engine, llm = get_engine()  # 获取引擎和LLM客户端
        return HealthResponse(  # 返回健康状态
            status="healthy",  # 状态为健康
            engine_ready=True,  # 引擎就绪
            backend=RAG_BACKEND,  # 当前后端
            llm_available=llm.api_available,  # LLM可用状态
            pdf_loaded=os.path.exists(PDF_PATH),  # PDF文件是否存在
            uploaded_files=[os.path.basename(f) for f in _loaded_files]  # 已上传文件列表
        )
    except Exception as e:  # 初始化失败时
        return HealthResponse(  # 返回降级状态
            status="degraded",  # 状态为降级
            engine_ready=False,  # 引擎未就绪
            backend=RAG_BACKEND,  # 当前后端
            llm_available=False,  # LLM不可用
            pdf_loaded=os.path.exists(PDF_PATH),  # PDF文件是否存在
            uploaded_files=[]  # 已上传文件列表为空
        )


@app.get("/api/understand")
async def understand_question(q: str = ""):
    """Query Understanding 调试接口"""
    if not q:  # 未提供查询参数时
        return JSONResponse(content={"error": "请提供参数 ?q=您的问题"}, status_code=400)  # 返回错误提示
    result = QueryUnderstanding.understand(q)  # 对问题进行语义理解
    return JSONResponse(content=result)  # 返回理解结果


# ==================== 启动 ====================

def find_available_port(start_port=8888, max_tries=10):
    """从 start_port 开始尝试，找到第一个可用端口"""
    import socket  # 导入socket模块
    for port in range(start_port, start_port + max_tries):  # 遍历端口范围
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:  # 创建TCP socket
            try:
                s.bind(('127.0.0.1', port))  # 尝试绑定端口
                return port  # 绑定成功，返回可用端口
            except OSError:  # 端口被占用
                continue  # 继续尝试下一个端口
    raise RuntimeError("无法找到可用端口（尝试范围 {} - {}）".format(  # 所有端口均不可用时抛出异常
        start_port, start_port + max_tries - 1))


def main():
    """主函数：启动服务"""
    # 扫描已上传的文件
    global _loaded_files  # 声明使用全局变量
    for filename in os.listdir(UPLOAD_DIR):  # 遍历上传目录
        if filename.endswith('.pdf'):  # PDF文件时
            file_path = os.path.join(UPLOAD_DIR, filename)  # 构造完整路径
            _loaded_files.append(file_path)  # 添加到已上传文件列表
    
    port = find_available_port(start_port=8888)  # 查找可用端口
    print("=" * 56)  # 打印分隔线
    print("  RAG 问答系统 — 后端服务")  # 打印服务名称
    print("  PDF: {}".format(PDF_PATH))  # 打印默认PDF路径
    print("  上传目录: {}".format(UPLOAD_DIR))  # 打印上传目录路径
    print("  已上传文件: {} 个".format(len(_loaded_files)))  # 打印已上传文件数量
    print("  后端模式: {}".format(RAG_BACKEND))  # 打印后端模式
    print("  Query Understanding: 已启用")  # 打印查询理解状态
    print("  文件上传接口: POST /api/upload")  # 打印上传接口信息
    print("  知识库重载接口: POST /api/reload")  # 打印重载接口信息
    print("  对比接口: /api/compare (POST)")  # 打印对比接口信息
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")  # 启动uvicorn服务器


if __name__ == '__main__':
    main()  # 调用主函数启动服务
