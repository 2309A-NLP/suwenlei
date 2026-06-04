# -*- coding: utf-8 -*-
"""
RAG问答系统 — 后端服务（多文档路由检索）

启动: python app.py
前提: CSV分块文件已生成（uploads目录中）
访问: 启动后控制台会显示实际地址（自动处理端口冲突）
"""
import os
import sys
import json
import time
import logging

sys.path.insert(0, os.path.dirname(__file__))

from rag_pipeline import RAGEngine
from llm_client import LLMClient, GROUND_TRUTH
from query_understanding import QueryUnderstanding
from pdf_parser import KnowledgeBuilder
from redis_store import ConversationStore

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional
from typing import Literal
import uvicorn

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---- 路径与后端配置 ----
BASE_DIR = os.path.dirname(__file__)
PDF_PATH = os.path.join(BASE_DIR, '招股说明书1.pdf')
PDF_PATH2 = os.path.join(BASE_DIR, '招股说明书2.pdf')
INDEX_HTML = os.path.join(BASE_DIR, 'index.html')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
RAG_BACKEND = os.environ.get("RAG_BACKEND", "auto")  # 支持auto/milvus/tfidf切换

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---- 全局单例：惰性初始化，首次查询时加载 ----
_engine = None
_llm = None
_uploaded_files = []
_store = ConversationStore()  # Redis对话存储，自动降级内存


def get_engine():
    """惰性初始化RAG引擎：首次调用时扫描所有PDF，后续复用"""
    global _engine, _llm
    if _engine is None:
        logger.info("正在初始化RAG引擎（多文档路由模式）...")
        pdf_paths = {}
        # 优先加载项目根目录下的固定PDF
        for pdf_path in [PDF_PATH, PDF_PATH2]:
            if os.path.exists(pdf_path):
                doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
                pdf_paths[doc_name] = pdf_path
        # 补充uploads目录下用户上传的PDF（排除已加载的）
        if os.path.isdir(UPLOAD_DIR):
            for f in os.listdir(UPLOAD_DIR):
                if f.endswith('.pdf'):
                    pdf_name = os.path.splitext(f)[0]
                    if pdf_name not in pdf_paths:
                        pdf_paths[pdf_name] = os.path.join(UPLOAD_DIR, f)
        _engine = RAGEngine(pdf_paths=pdf_paths, backend=RAG_BACKEND)
        _llm = LLMClient()
        _llm.set_rag_engine(_engine)
        logger.info("初始化完成")
    return _engine, _llm


def scan_uploaded_files():
    """扫描uploads目录，构建前端文件管理所需的元数据列表"""
    global _uploaded_files
    _uploaded_files = []
    if os.path.exists(UPLOAD_DIR):
        for fname in os.listdir(UPLOAD_DIR):
            fpath = os.path.join(UPLOAD_DIR, fname)
            if os.path.isfile(fpath):
                _uploaded_files.append({
                    'filename': fname,
                    'size': os.path.getsize(fpath),
                    'mtime': os.path.getmtime(fpath)
                })
    return _uploaded_files


def reload_engine():
    """重置引擎为None，下次查询触发重新加载（上传/删除文件后调用）"""
    global _engine, _llm
    _engine = None
    _llm = None
    logger.info("引擎已重置，下次查询将重新初始化")


# ---- FastAPI ----
app = FastAPI(title="RAG问答系统 - 多文档路由检索")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    conversation_id: Optional[str] = None
    lang: Literal['zh', 'en'] = 'zh'


class HealthResponse(BaseModel):
    status: str
    engine_ready: bool
    backend: str
    llm_available: bool
    pdf_loaded: bool
    redis_available: bool = False


# ==================== 前端 ====================

@app.get("/")
async def index():
    return FileResponse(INDEX_HTML)


# ==================== 文件管理API ====================

_kb_builder = KnowledgeBuilder(project_dir=BASE_DIR)


@app.post("/api/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    """上传PDF并触发全链路处理：解析→分块→TF-IDF训练→Milvus入库"""
    results = []
    for upload_file in files:
        filename = upload_file.filename or "unknown.pdf"
        if not filename.lower().endswith('.pdf'):
            results.append({'filename': filename, 'status': 'skipped', 'message': '仅支持PDF格式'})
            continue

        save_path = os.path.join(UPLOAD_DIR, filename)
        try:
            content = await upload_file.read()
            with open(save_path, 'wb') as f:
                f.write(content)
        except Exception as e:
            results.append({'filename': filename, 'status': 'error', 'message': '文件保存失败: {}'.format(str(e))})
            continue

        # 构建知识库：PDF解析→CSV分块→Milvus向量入库
        build_result = _kb_builder.build(save_path)
        results.append({
            'filename': filename,
            'status': 'success' if build_result['success'] else 'error',
            'message': build_result.get('error') or 'CSV {}块, Milvus {}条'.format(
                build_result['chunks_count'], build_result['milvus_inserted']),
            'size': len(content),
            'chunks_count': build_result['chunks_count'],
            'milvus_inserted': build_result['milvus_inserted'],
            'elapsed': build_result['elapsed']
        })

    scan_uploaded_files()  # 刷新文件列表
    reload_engine()        # 触发下次查询时重新加载引擎
    return {'success': True, 'results': results, 'total_files': len(_uploaded_files)}


@app.get("/api/files")
async def list_files():
    files = scan_uploaded_files()
    return {'files': files, 'total': len(files)}


@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    filepath = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="文件不存在")
    try:
        os.remove(filepath)
        # 同时清理对应的分块CSV
        csv_path = os.path.splitext(filepath)[0] + '_chunks_v2.csv'
        if os.path.exists(csv_path):
            os.remove(csv_path)
        scan_uploaded_files()
        reload_engine()
        return {'success': True, 'message': '文件 {} 已删除'.format(filename)}
    except Exception as e:
        raise HTTPException(status_code=500, detail="删除失败: {}".format(str(e)))


@app.post("/api/reload")
async def reload_knowledge():
    """手动触发知识库重载（用于调试或外部数据变更）"""
    reload_engine()
    scan_uploaded_files()
    return {
        'success': True,
        'message': '知识库已重载，包含 {} 个上传文件'.format(len(_uploaded_files)),
        'files': _uploaded_files
    }


# ==================== 问答API ====================

@app.post("/api/ask")
async def ask_question(req: QueryRequest):
    """核心问答接口：意图识别→多文档路由检索→Prompt构建→LLM生成"""
    try:
        engine, llm = get_engine()
        start = time.time()
        question = req.question.strip()

        if not question:
            return JSONResponse(
                content={'error': '问题不能为空', 'answer': '', 'sources': []},
                status_code=400
            )

        # 英文模式：先将英文问题翻译为中文，用中文检索，最后将回答翻译回英文
        zh_question = question
        if req.lang == 'en':
            zh_question = llm.translate_to_chinese(question)
            logger.info("英文问题翻译为中文: {}".format(zh_question))

        conv_id = req.conversation_id
        if not conv_id:
            conv_id = _store.generate_id()
        history = _store.get_history(conv_id)

        # Query Understanding：意图识别+实体消歧+查询扩展
        qu = QueryUnderstanding.understand(zh_question)
        search_query = qu['expanded_query']
        logger.info("原问题: {} | 扩展: {}".format(
            zh_question if req.lang == 'zh' else '{} -> {}'.format(question, zh_question),
            search_query))

        # 多文档路由检索：根据问题自动选择目标文档
        context, results = engine.get_context(search_query, top_k=req.top_k, original_query=zh_question)

        if not context:
            answer = "抱歉，在文档中没有找到与您问题相关的内容。请尝试换一种方式提问。"
            if req.lang == 'en':
                answer = llm.translate_to_english(answer)
            _store.append_message(conv_id, 'user', question)
            _store.append_message(conv_id, 'assistant', answer)
            return {
                'question': question,
                'answer': answer,
                'sources': [],
                'time_seconds': round(time.time() - start, 2),
                'gt_matched': False,
                'intent': qu['intent'],
                'conversation_id': conv_id
            }

        prompt = engine.build_prompt(zh_question, context, history=history)
        answer = llm.generate(prompt, context, zh_question, history=history)

        # 英文模式：将中文回答翻译为英文
        if req.lang == 'en':
            answer = llm.translate_to_english(answer)

        # 持久化对话历史
        _store.append_message(conv_id, 'user', question)
        _store.append_message(conv_id, 'assistant', answer)

        elapsed = round(time.time() - start, 2)
        # 评估：检查答案是否命中预定义标准答案（用于准确率评测）
        gt_matched = any(answer == gt or (len(gt) > 10 and gt[:30] in answer)
                         for gt in GROUND_TRUTH.values())

        sources = [
            {
                'page': r['page_num'],
                'score': r['score'],
                'snippet': r['text'][:200],
                'doc_name': r.get('doc_name', ''),
                'source_type': r.get('source_type', 'text'),
            }
            for r in results[:req.top_k]
        ]

        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'time_seconds': elapsed,
            'gt_matched': gt_matched,
            'intent': qu['intent'],
            'conversation_id': conv_id,
            'history_length': len(history) + 2
        }

    except FileNotFoundError as e:
        logger.error(f"文件未找到: {e}")
        return JSONResponse(
            content={'error': f'PDF文件未找到: {str(e)}', 'answer': '', 'sources': []},
            status_code=500
        )
    except Exception as e:
        logger.error(f"问答接口异常: {e}", exc_info=True)
        return JSONResponse(
            content={
                'error': f'系统内部错误，请稍后重试',
                'answer': '抱歉，系统处理您的问题时出现异常，请稍后重试。',
                'sources': []
            },
            status_code=500
        )


@app.post("/api/compare")
async def compare_answers(req: QueryRequest):
    """对比接口：同一问题分别用RAG和纯LLM回答，用于效果对比评测"""
    try:
        engine, llm = get_engine()
        start = time.time()
        question = req.question.strip()

        if not question:
            return JSONResponse(content={'error': '问题不能为空'}, status_code=400)

        # 英文模式：先将英文问题翻译为中文
        zh_question = question
        if req.lang == 'en':
            zh_question = llm.translate_to_chinese(question)
            logger.info("对比模式-英文翻译为中文: {}".format(zh_question))

        conv_id = req.conversation_id
        if not conv_id:
            conv_id = _store.generate_id()
        history = _store.get_history(conv_id)

        # RAG回答：意图识别→检索→生成（使用中文问题）
        qu = QueryUnderstanding.understand(zh_question)
        search_query = qu['expanded_query']
        context, results = engine.get_context(search_query, top_k=req.top_k, original_query=zh_question)

        if context:
            prompt = engine.build_prompt(zh_question, context, history=history)
            rag_answer = llm.generate(prompt, context, zh_question, history=history)
        else:
            rag_answer = "抱歉，在文档中没有找到与您问题相关的内容。"
        rag_time = round(time.time() - start, 2)

        # 英文模式：将RAG中文回答翻译为英文
        if req.lang == 'en':
            rag_answer = llm.translate_to_english(rag_answer)

        sources = [
            {
                'page': r['page_num'],
                'score': r['score'],
                'snippet': r['text'][:200],
                'doc_name': r.get('doc_name', ''),
                'source_type': r.get('source_type', 'text'),
            }
            for r in results[:req.top_k]
        ] if context else []

        gt_matched = any(rag_answer == gt for gt in GROUND_TRUTH.values())

        # 纯LLM回答：无RAG上下文，仅凭模型自身知识
        llm_start = time.time()
        llm_answer = llm.generate_pure_llm(question)
        llm_time = round(time.time() - llm_start, 2)

        _store.append_message(conv_id, 'user', question)
        _store.append_message(conv_id, 'assistant', rag_answer)

        # 对比分析指标
        answers_match = rag_answer.strip()[:50] == llm_answer.strip()[:50]  # 前50字是否一致
        rag_has_sources = len(sources) > 0
        rag_is_more_detailed = len(rag_answer) > len(llm_answer) if not answers_match else False

        return {
            'question': question,
            'rag': {
                'answer': rag_answer,
                'sources': sources,
                'time_seconds': rag_time,
                'gt_matched': gt_matched,
                'intent': qu['intent'],
                'has_context': bool(context)
            },
            'llm_only': {
                'answer': llm_answer,
                'time_seconds': llm_time,
                'has_api': llm.api_available
            },
            'analysis': {
                'answers_match': answers_match,
                'rag_has_sources': rag_has_sources,
                'rag_has_citations': rag_has_sources,
                'rag_is_more_detailed': rag_is_more_detailed,
                'rag_is_faster': rag_time < llm_time,
                'rag_len': len(rag_answer),
                'llm_len': len(llm_answer),
                'sources_count': len(sources)
            },
            'conversation_id': conv_id,
            'history_length': len(history) + 2
        }

    except Exception as e:
        logger.error(f"对比接口异常: {e}", exc_info=True)
        return JSONResponse(
            content={'error': '对比接口异常: {}'.format(str(e))},
            status_code=500
        )


@app.get("/api/health")
async def health_check():
    try:
        engine, llm = get_engine()
        doc_info = {name: len(e.chunks) for name, e in engine.doc_engines.items()}
        return HealthResponse(
            status="healthy",
            engine_ready=True,
            backend=RAG_BACKEND,
            llm_available=llm.api_available,
            pdf_loaded=os.path.exists(PDF_PATH) and os.path.exists(PDF_PATH2),
            redis_available=_store.available
        )
    except Exception as e:
        return HealthResponse(
            status="degraded",
            engine_ready=False,
            backend=RAG_BACKEND,
            llm_available=False,
            pdf_loaded=os.path.exists(PDF_PATH) and os.path.exists(PDF_PATH2),
            redis_available=_store.available
        )


# ==================== 多轮对话管理API ====================

@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    history = _store.get_history(conversation_id)
    return {
        'conversation_id': conversation_id,
        'history': history,
        'message_count': len(history),
        'round_count': len(history) // 2
    }


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    _store.delete_conversation(conversation_id)
    return {'success': True, 'message': '会话 {} 已删除'.format(conversation_id)}


@app.delete("/api/conversations")
async def clear_all_conversations():
    _store.clear_all()
    return {'success': True, 'message': '所有对话历史已清空'}


@app.get("/api/understand")
async def understand_question(q: str = ""):
    """调试接口：查看Query Understanding的完整处理结果"""
    if not q:
        return JSONResponse(content={"error": "请提供参数 ?q=您的问题"}, status_code=400)
    result = QueryUnderstanding.understand(q)
    return JSONResponse(content=result)


@app.get("/api/debug_route")
async def debug_route(q: str = ""):
    """调试接口：查看查询路由决策过程和检索结果"""
    engine, _ = get_engine()
    route = engine._route_query(q)
    context, results = engine.get_context(q, top_k=5, original_query=q)
    return JSONResponse(content={
        "query": q,
        "route": route,
        "doc_engines_keys": list(engine.doc_engines.keys()),
        "results_count": len(results),
        "results": [{
            "doc_name": r.get("doc_name", ""),
            "page": r["page_num"],
            "score": r["score"],
            "type": r.get("source_type", "text"),
            "text_preview": r["text"][:100],
        } for r in results]
    })


# ==================== 启动 ====================

def find_available_port(start_port=8888, max_tries=10):
    """端口探测：避免多实例启动时端口冲突"""
    import socket
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    raise RuntimeError("无法找到可用端口（尝试范围 {} - {}）".format(
        start_port, start_port + max_tries - 1))


def main():
    port = find_available_port(start_port=8888)
    scan_uploaded_files()
    print("=" * 56)
    print("  RAG 问答系统 — 多文档路由检索")
    print("  PDF1: {}".format(PDF_PATH))
    print("  PDF2: {}".format(PDF_PATH2))
    print("  上传目录: {} ({} 个文件)".format(UPLOAD_DIR, len(_uploaded_files)))
    print("  后端模式: {}".format(RAG_BACKEND))
    print("  查询路由: 已启用（根据问题自动选择文档）")
    print("  文件上传: /api/upload (POST)")
    print("  对比接口: /api/compare (POST)")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == '__main__':
    main()
