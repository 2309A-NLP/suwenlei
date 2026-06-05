# -*- coding: utf-8 -*-
"""
RAG问答系统 — 后端服务（多文档路由检索）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化

启动: python app.py
前提: CSV分块文件已生成（uploads目录中）
访问: 启动后控制台会显示实际地址（自动处理端口冲突）
核心功能：
1. 多文档独立向量库：每个PDF生成独立CSV和独立Milvus集合
2. 查询路由：根据问题内容自动判断搜索哪个文档的向量库
3. PDF解析增强：水印去除 + Qwen-VL多模态图片识别 + 文本清理
"""

import os  # 导入操作系统接口模块
import sys  # 导入系统相关模块
import json  # 导入JSON处理模块
import time  # 导入时间模块
import logging  # 导入日志模块
import shutil  # 导入文件操作模块
from typing import Literal  # 导入Literal类型注解（用于lang参数）

sys.path.insert(0, os.path.dirname(__file__))  # 将当前目录加入模块搜索路径

from rag_pipeline import RAGEngine  # 导入RAG引擎核心类
from llm_client import LLMClient  # 导入LLM客户端
from query_understanding import QueryUnderstanding  # 导入查询理解模块
from pdf_parser import KnowledgeBuilder  # 从pdf_parser模块导入知识库构建器

from fastapi import FastAPI, UploadFile, File, HTTPException  # 导入FastAPI和文件上传相关
from fastapi.responses import JSONResponse, FileResponse  # 导入响应类型
from pydantic import BaseModel  # 导入数据模型基类
import uvicorn  # 导入ASGI服务器

logging.basicConfig(level=logging.INFO,  # 配置日志级别为INFO
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # 设置日志输出格式
logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

# ---- 配置 ----
BASE_DIR = os.path.dirname(__file__)  # 获取项目根目录路径
# 优化：支持双PDF文件加载（招股说明书1.pdf + 招股说明书2.pdf）
PDF_PATH = os.path.join(BASE_DIR, '招股说明书1.pdf')  # 主PDF文件路径（用于初始化）
PDF_PATH2 = os.path.join(BASE_DIR, '招股说明书2.pdf')  # 第二个PDF文件路径（力源信息）
INDEX_HTML = os.path.join(BASE_DIR, 'index.html')  # 前端页面路径
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')  # 上传文件存储目录
RAG_BACKEND = os.environ.get("RAG_BACKEND", "auto")  # 从环境变量获取RAG后端类型，默认auto

# 确保上传目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)  # 创建上传目录，已存在则忽略

# ---- 全局引擎 ----
_engine = None  # 全局RAG引擎实例
_llm = None  # 全局LLM客户端实例
_uploaded_files = []  # 已上传文件列表


def get_engine():
    """
    获取或初始化RAG引擎和LLM客户端
    优化：为每个PDF创建独立的检索引擎，查询时按文档路由
    """
    global _engine, _llm  # 声明使用全局变量
    if _engine is None:  # 引擎尚未初始化时
        logger.info("正在初始化RAG引擎（多文档路由模式）...")  # 打印初始化日志
        logger.info("后端模式: {}".format(RAG_BACKEND))  # 打印后端模式
        # 优化：构建文档名→PDF路径的映射字典
        pdf_paths = {}  # 文档路径字典
        # 优化：先检查根目录，再检查uploads目录（PDF实际在uploads中）
        for pdf_path in [PDF_PATH, PDF_PATH2]:  # 遍历两个PDF路径
            if os.path.exists(pdf_path):  # 根目录存在时
                doc_name = os.path.splitext(os.path.basename(pdf_path))[0]  # 获取文档名
                pdf_paths[doc_name] = pdf_path  # 添加到字典
        # 优化：扫描uploads目录中的PDF文件
        if os.path.isdir(UPLOAD_DIR):  # uploads目录存在时
            for f in os.listdir(UPLOAD_DIR):  # 遍历uploads目录
                if f.endswith('.pdf'):  # 是PDF文件时
                    pdf_name = os.path.splitext(f)[0]  # 去除扩展名作为文档名
                    if pdf_name not in pdf_paths:  # 避免重复添加
                        pdf_paths[pdf_name] = os.path.join(UPLOAD_DIR, f)  # 添加到字典
        _engine = RAGEngine(pdf_paths=pdf_paths, backend=RAG_BACKEND)  # 创建RAG引擎实例
        _llm = LLMClient()  # 创建LLM客户端实例
        _llm.set_rag_engine(_engine)  # 将RAG引擎注入LLM客户端
        logger.info("初始化完成")  # 打印完成日志
    return _engine, _llm  # 返回引擎和LLM客户端


def scan_uploaded_files():
    """
    扫描uploads目录，获取已上传的文件列表
    """
    global _uploaded_files  # 声明使用全局变量
    _uploaded_files = []  # 清空列表
    if os.path.exists(UPLOAD_DIR):  # 上传目录存在时
        for fname in os.listdir(UPLOAD_DIR):  # 遍历目录中的文件
            fpath = os.path.join(UPLOAD_DIR, fname)  # 获取完整路径
            if os.path.isfile(fpath):  # 确认是文件（非目录）
                _uploaded_files.append({  # 添加到列表
                    'filename': fname,  # 文件名
                    'size': os.path.getsize(fpath),  # 文件大小（字节）
                    'mtime': os.path.getmtime(fpath)  # 最后修改时间
                })
    return _uploaded_files  # 返回文件列表


def reload_engine():
    """
    重新加载RAG引擎（上传新文件后调用）
    """
    global _engine, _llm  # 声明使用全局变量
    _engine = None  # 清空旧引擎
    _llm = None  # 清空旧LLM客户端
    logger.info("引擎已重置，下次查询将重新初始化")  # 打印日志


# ---- FastAPI ----
app = FastAPI(title="RAG问答系统 - 多文档路由检索")  # 创建FastAPI应用实例


class QueryRequest(BaseModel):
    question: str  # 用户输入的问题文本
    top_k: int = 25  # 检索返回的最相关文档块数量，默认25
    lang: Literal['zh', 'en'] = 'zh'  # 语言选择：zh中文 / en英文，默认中文


class HealthResponse(BaseModel):
    status: str  # 服务状态
    engine_ready: bool  # 引擎是否就绪
    backend: str  # 当前使用的后端类型
    llm_available: bool  # LLM是否可用
    pdf_loaded: bool  # PDF是否已加载


# ==================== 前端 ====================

@app.get("/")
async def index():
    return FileResponse(INDEX_HTML)  # 返回前端页面


# ==================== 文件管理API ====================

# 全局知识库构建器实例
_kb_builder = KnowledgeBuilder(project_dir=BASE_DIR)  # 创建知识库构建器实例


@app.post("/api/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    """
    上传多个PDF文件 — 自动完成解析+分块+向量+Milvus入库
    流程：保存文件 → kb_builder.build() → CSV分块 + Milvus向量入库
    """
    results = []  # 存储每个文件的处理结果
    for upload_file in files:  # 遍历上传的文件
        filename = upload_file.filename or "unknown.pdf"  # 获取文件名
        # 安全检查：只允许PDF文件
        if not filename.lower().endswith('.pdf'):  # 非PDF文件时跳过
            results.append({  # 记录跳过信息
                'filename': filename,  # 文件名
                'status': 'skipped',  # 状态为跳过
                'message': '仅支持PDF格式'  # 提示信息
            })
            continue  # 跳过当前文件

        # 保存上传文件到uploads目录
        save_path = os.path.join(UPLOAD_DIR, filename)  # 生成保存路径
        try:
            content = await upload_file.read()  # 读取文件内容
            with open(save_path, 'wb') as f:  # 以二进制写入模式打开目标文件
                f.write(content)  # 写入文件内容
        except Exception as e:  # 捕获写入异常
            results.append({  # 记录写入失败信息
                'filename': filename,  # 文件名
                'status': 'error',  # 状态为错误
                'message': '文件保存失败: {}'.format(str(e))  # 错误信息
            })
            continue  # 跳过后续处理

        # 使用kb_builder完成：PDF解析 → CSV分块 → TF-IDF嵌入 → Milvus入库
        build_result = _kb_builder.build(save_path)  # 一键构建知识库
        results.append({  # 记录处理结果
            'filename': filename,  # 文件名
            'status': 'success' if build_result['success'] else 'error',  # 状态
            'message': build_result.get('error') or 'CSV {}块, Milvus {}条'.format(  # 处理信息
                build_result['chunks_count'], build_result['milvus_inserted']
            ),
            'size': len(content),  # 文件大小
            'chunks_count': build_result['chunks_count'],  # 分块数量
            'milvus_inserted': build_result['milvus_inserted'],  # Milvus入库数量
            'elapsed': build_result['elapsed']  # 耗时
        })

    # 更新文件列表
    scan_uploaded_files()  # 扫描上传目录

    # 重新加载引擎以包含新文件
    reload_engine()  # 重置引擎

    return {  # 返回上传结果
        'success': True,  # 操作成功
        'results': results,  # 各文件处理结果
        'total_files': len(_uploaded_files)  # 当前总文件数
    }


@app.get("/api/files")
async def list_files():
    """获取已上传的文件列表"""
    files = scan_uploaded_files()  # 扫描上传目录
    return {  # 返回文件列表
        'files': files,  # 文件信息数组
        'total': len(files)  # 文件总数
    }


@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    """删除指定的上传文件"""
    filepath = os.path.join(UPLOAD_DIR, filename)  # 生成文件完整路径
    if not os.path.exists(filepath):  # 文件不存在时
        raise HTTPException(status_code=404, detail="文件不存在")  # 抛出404异常
    try:
        os.remove(filepath)  # 删除文件
        # 同时删除对应的CSV分块文件（如果存在）
        csv_path = os.path.splitext(filepath)[0] + '_chunks_v2.csv'  # 生成CSV路径
        if os.path.exists(csv_path):  # CSV文件存在时
            os.remove(csv_path)  # 删除CSV文件
        scan_uploaded_files()  # 更新文件列表
        reload_engine()  # 重置引擎
        return {'success': True, 'message': '文件 {} 已删除'.format(filename)}  # 返回成功信息
    except Exception as e:  # 捕获删除异常
        raise HTTPException(status_code=500, detail="删除失败: {}".format(str(e)))  # 抛出500异常


@app.post("/api/reload")
async def reload_knowledge():
    """重新加载知识库（从CSV重新加载，不重新解析PDF）"""
    reload_engine()  # 重置引擎
    scan_uploaded_files()  # 更新文件列表
    return {  # 返回重载结果
        'success': True,  # 操作成功
        'message': '知识库已重载，包含 {} 个上传文件'.format(len(_uploaded_files)),  # 提示信息
        'files': _uploaded_files  # 文件列表
    }


# ==================== 问答API ====================

@app.post("/api/ask")
async def ask_question(req: QueryRequest):
    """
    核心问答接口
    流程: Query Understanding → 文档路由 → 检索 → 生成
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

        # 1. Query Understanding
        qu = QueryUnderstanding.understand(zh_question)  # 对中文问题进行语义理解和扩展
        search_query = qu['expanded_query']  # 获取扩展后的查询语句
        logger.info("原问题: {}".format(zh_question if req.lang == 'zh' else '{} -> {}'.format(question, zh_question)))  # 记录问题
        logger.info(f"扩展查询: {search_query}")  # 记录扩展后的查询
        logger.info(f"意图: {qu['intent']} (置信度: {qu['intent_confidence']})")  # 记录意图和置信度

        # 2. 检索（自动路由到目标文档）
        context, results = engine.get_context(search_query, top_k=req.top_k, original_query=zh_question)  # 在文档中检索相关上下文（路由用中文查询）

        if not context:  # 未检索到任何内容时
            # 无检索结果
            answer = "抱歉，在文档中没有找到与您问题相关的内容。请尝试换一种方式提问。"  # 默认回复
            if req.lang == 'en':  # 英文模式时翻译为英文
                answer = llm.translate_to_english(answer)  # 中文→英文翻译
            return {  # 返回无结果响应
                'question': question,  # 原始问题
                'answer': answer,  # 回复内容
                'sources': [],  # 来源为空
                'time_seconds': round(time.time() - start, 2),  # 耗时（秒）
                'gt_matched': False,  # 是否匹配标准答案
                'intent': qu['intent']  # 问题意图
            }

        # 3. 构建Prompt并生成回答
        prompt = engine.build_prompt(zh_question, context, lang='zh')  # 用中文Prompt构建
        answer = llm.generate(prompt, context, zh_question, lang='zh')  # 用中文生成回答

        # 英文模式：将中文回答翻译为英文
        if req.lang == 'en':  # 英文模式时
            answer = llm.translate_to_english(answer)  # 中文→英文翻译

        elapsed = round(time.time() - start, 2)  # 计算总耗时

        sources = [  # 构建来源列表
            {
                'page': r['page_num'],  # 页码
                'score': r['score'],  # 评分
                'snippet': r['text'][:200],  # 前200字摘要
                'doc_name': r.get('doc_name', ''),  # 来源文档名（新增）
                'source_type': r.get('source_type', 'text'),  # 来源类型（新增）
            }
            for r in results[:req.top_k]  # 取前top_k个结果
        ]

        return {  # 返回问答结果
            'question': question,  # 原始问题
            'answer': answer,  # 生成的回答
            'sources': sources,  # 引用来源列表
            'time_seconds': elapsed,  # 耗时
            'gt_matched': False,  # 是否匹配标准答案
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


@app.post("/api/compare")
async def compare_answers(req: QueryRequest):
    """
    对比接口：同时返回 RAG（基于PDF检索）和 纯LLM（无上下文）的回答
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
        context, results = engine.get_context(search_query, top_k=req.top_k, original_query=zh_question)  # 检索文档上下文（路由用中文查询）

        if context:  # 检索到内容时
            prompt = engine.build_prompt(zh_question, context, lang='zh')  # 用中文Prompt构建
            rag_answer = llm.generate(prompt, context, zh_question, lang='zh')  # 生成中文回答
        else:  # 未检索到内容时
            rag_answer = "抱歉，在文档中没有找到与您问题相关的内容。"  # 默认回复
        rag_time = round(time.time() - start, 2)  # 计算RAG路径耗时

        # 英文模式：将RAG中文回答翻译为英文
        if req.lang == 'en':  # 英文模式时
            rag_answer = llm.translate_to_english(rag_answer)  # 中文→英文翻译

        sources = [  # 构建来源列表
            {
                'page': r['page_num'],  # 页码
                'score': r['score'],  # 评分
                'snippet': r['text'][:200],  # 摘要
                'doc_name': r.get('doc_name', ''),  # 来源文档名
                'source_type': r.get('source_type', 'text'),  # 来源类型
            }
            for r in results[:req.top_k]  # 取前top_k个结果
        ] if context else []  # 无结果时返回空列表

        gt_matched = False  # 已移除预定义精确答案功能

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
        # 优化：获取文档引擎信息
        doc_info = {name: len(e.chunks) for name, e in engine.doc_engines.items()}  # 各文档分块数
        return HealthResponse(  # 返回健康状态
            status="healthy",  # 状态为健康
            engine_ready=True,  # 引擎就绪
            backend=RAG_BACKEND,  # 当前后端
            llm_available=llm.api_available,  # LLM可用状态
            pdf_loaded=os.path.exists(PDF_PATH) and os.path.exists(PDF_PATH2)  # 优化：检查两个PDF文件是否存在
        )
    except Exception as e:  # 初始化失败时
        return HealthResponse(  # 返回降级状态
            status="degraded",  # 状态为降级
            engine_ready=False,  # 引擎未就绪
            backend=RAG_BACKEND,  # 当前后端
            llm_available=False,  # LLM不可用
            pdf_loaded=os.path.exists(PDF_PATH) and os.path.exists(PDF_PATH2)  # 优化：检查两个PDF文件是否存在
        )


@app.get("/api/understand")
async def understand_question(q: str = ""):
    """Query Understanding 调试接口"""
    if not q:  # 未提供查询参数时
        return JSONResponse(content={"error": "请提供参数 ?q=您的问题"}, status_code=400)  # 返回错误提示
    result = QueryUnderstanding.understand(q)  # 对问题进行语义理解
    return JSONResponse(content=result)  # 返回理解结果


@app.get("/api/debug_route")
async def debug_route(q: str = ""):
    """路由调试接口：直接返回路由结果和检索来源"""  # 调试路由的API端点
    engine, _ = get_engine()  # 获取RAG引擎
    route = engine._route_query(q)  # 测试路由（用原始查询）
    context, results = engine.get_context(q, top_k=5, original_query=q)  # 测试完整检索
    return JSONResponse(content={  # 返回调试信息
        "query": q,  # 原始查询
        "route": route,  # 路由结果
        "doc_engines_keys": list(engine.doc_engines.keys()),  # 可用文档列表
        "results_count": len(results),  # 结果数量
        "results": [{  # 结果详情
            "doc_name": r.get("doc_name", ""),  # 文档名
            "page": r["page_num"],  # 页码
            "score": r["score"],  # 分数
            "type": r.get("source_type", "text"),  # 类型
            "text_preview": r["text"][:100],  # 内容预览
        } for r in results]  # 遍历所有结果
    })


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
    port = find_available_port(start_port=8888)  # 查找可用端口
    scan_uploaded_files()  # 启动时扫描上传文件列表
    print("=" * 56)  # 打印分隔线
    print("  RAG 问答系统 — 多文档路由检索")  # 打印服务名称
    print("  PDF1: {}".format(PDF_PATH))  # 打印第一个PDF路径
    print("  PDF2: {}".format(PDF_PATH2))  # 打印第二个PDF路径
    print("  上传目录: {} ({} 个文件)".format(UPLOAD_DIR, len(_uploaded_files)))  # 打印上传目录信息
    print("  后端模式: {}".format(RAG_BACKEND))  # 打印后端模式
    print("  查询路由: 已启用（根据问题自动选择文档）")  # 打印路由状态
    print("  文件上传: /api/upload (POST)")  # 打印上传接口信息
    print("  对比接口: /api/compare (POST)")  # 打印对比接口信息
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")  # 启动uvicorn服务器


if __name__ == '__main__':
    main()  # 调用主函数启动服务
