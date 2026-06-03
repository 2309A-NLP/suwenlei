# -*- coding: utf-8 -*-
"""RAG问答系统 — 后端服务（多文档路由检索）"""
import os
import sys
import json
import time
import logging
import shutil
from typing import Literal

sys.path.insert(0, os.path.dirname(__file__))

from rag_pipeline import RAGEngine
from llm_client import LLMClient, GROUND_TRUTH
from query_understanding import QueryUnderstanding
from pdf_parser import KnowledgeBuilder

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---- 配置 ----
BASE_DIR = os.path.dirname(__file__)
PDF_PATH = os.path.join(BASE_DIR, '招股说明书1.pdf')
PDF_PATH2 = os.path.join(BASE_DIR, '招股说明书2.pdf')
INDEX_HTML = os.path.join(BASE_DIR, 'index.html')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
RAG_BACKEND = os.environ.get("RAG_BACKEND", "auto")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---- 全局引擎 ----
_engine = None
_llm = None
_uploaded_files = []


def get_engine():
    """获取或初始化RAG引擎和LLM客户端"""
    global _engine, _llm
    if _engine is None:
        logger.info("正在初始化RAG引擎（多文档路由模式）...")
        logger.info("后端模式: {}".format(RAG_BACKEND))
        pdf_paths = {}
        for pdf_path in [PDF_PATH, PDF_PATH2]:
            if os.path.exists(pdf_path):
                doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
                pdf_paths[doc_name] = pdf_path
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
    """扫描uploads目录，获取已上传的文件列表"""
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
    """重新加载RAG引擎（上传新文件后调用）"""
    global _engine, _llm
    _engine = None
    _llm = None
    logger.info("引擎已重置，下次查询将重新初始化")


# ---- FastAPI ----
app = FastAPI(title="RAG问答系统 - 多文档路由检索")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 25
    lang: Literal['zh', 'en'] = 'zh'


class HealthResponse(BaseModel):
    status: str
    engine_ready: bool
    backend: str
    llm_available: bool
    pdf_loaded: bool


# ==================== 前端 ====================

@app.get("/")
async def index():
    return FileResponse(INDEX_HTML)


# ==================== 文件管理API ====================

_kb_builder = KnowledgeBuilder(project_dir=BASE_DIR)


@app.post("/api/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    """上传多个PDF文件 — 自动完成解析+分块+向量+Milvus入库"""
    results = []
    for upload_file in files:
        filename = upload_file.filename or "unknown.pdf"
        if not filename.lower().endswith('.pdf'):
            results.append({
                'filename': filename,
                'status': 'skipped',
                'message': '仅支持PDF格式'
            })
            continue

        save_path = os.path.join(UPLOAD_DIR, filename)
        try:
            content = await upload_file.read()
            with open(save_path, 'wb') as f:
                f.write(content)
        except Exception as e:
            results.append({
                'filename': filename,
                'status': 'error',
                'message': '文件保存失败: {}'.format(str(e))
            })
            continue

        build_result = _kb_builder.build(save_path)
        results.append({
            'filename': filename,
            'status': 'success' if build_result['success'] else 'error',
            'message': build_result.get('error') or 'CSV {}块, Milvus {}条'.format(
                build_result['chunks_count'], build_result['milvus_inserted']
            ),
            'size': len(content),
            'chunks_count': build_result['chunks_count'],
            'milvus_inserted': build_result['milvus_inserted'],
            'elapsed': build_result['elapsed']
        })

    scan_uploaded_files()
    reload_engine()

    return {
        'success': True,
        'results': results,
        'total_files': len(_uploaded_files)
    }


@app.get("/api/files")
async def list_files():
    """获取已上传的文件列表"""
    files = scan_uploaded_files()
    return {
        'files': files,
        'total': len(files)
    }


@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    """删除指定的上传文件"""
    filepath = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="文件不存在")
    try:
        os.remove(filepath)
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
    """重新加载知识库"""
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
    """核心问答接口 — 流程: Query Understanding→文档路由→检索→生成"""
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

        qu = QueryUnderstanding.understand(zh_question)
        search_query = qu['expanded_query']
        logger.info("原问题: {}".format(zh_question if req.lang == 'zh' else '{} -> {}'.format(question, zh_question)))
        logger.info(f"扩展查询: {search_query}")
        logger.info(f"意图: {qu['intent']} (置信度: {qu['intent_confidence']})")

        context, results = engine.get_context(search_query, top_k=req.top_k, original_query=zh_question)

        if not context:
            answer = "抱歉，在文档中没有找到与您问题相关的内容。请尝试换一种方式提问。"
            if req.lang == 'en':
                answer = llm.translate_to_english(answer)
            return {
                'question': question,
                'answer': answer,
                'sources': [],
                'time_seconds': round(time.time() - start, 2),
                'gt_matched': False,
                'intent': qu['intent']
            }

        prompt = engine.build_prompt(zh_question, context, lang='zh')
        answer = llm.generate(prompt, context, zh_question, lang='zh')

        # 英文模式：将中文回答翻译为英文
        if req.lang == 'en':
            answer = llm.translate_to_english(answer)

        elapsed = round(time.time() - start, 2)
        gt_matched = False
        for gt in GROUND_TRUTH.values():
            if answer == gt or (len(gt) > 10 and gt[:30] in answer):
                gt_matched = True
                break

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
            'intent': qu['intent']
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


@app.get("/api/evaluate")
async def get_evaluation():
    """获取评测结果"""
    path = os.path.join(BASE_DIR, 'qa_evaluation_results.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return JSONResponse(content=json.load(f))
    return JSONResponse(
        content={"error": "评测结果未生成，请运行 qa_eval.py"},
        status_code=404
    )


@app.get("/api/ground_truth")
async def get_ground_truth():
    return JSONResponse(content=GROUND_TRUTH)


@app.post("/api/compare")
async def compare_answers(req: QueryRequest):
    """对比接口：同时返回RAG和纯LLM的回答"""
    try:
        engine, llm = get_engine()
        start = time.time()
        question = req.question.strip()

        if not question:
            return JSONResponse(
                content={'error': '问题不能为空'},
                status_code=400
            )

        # 英文模式：先将英文问题翻译为中文
        zh_question = question
        if req.lang == 'en':
            zh_question = llm.translate_to_chinese(question)
            logger.info("对比模式-英文翻译为中文: {}".format(zh_question))

        # === RAG 路径 ===
        qu = QueryUnderstanding.understand(zh_question)
        search_query = qu['expanded_query']
        context, results = engine.get_context(search_query, top_k=req.top_k, original_query=zh_question)

        if context:
            prompt = engine.build_prompt(zh_question, context, lang='zh')
            rag_answer = llm.generate(prompt, context, zh_question, lang='zh')
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

        # === 纯LLM路径 ===
        llm_start = time.time()
        llm_answer = llm.generate_pure_llm(question, lang='en' if req.lang == 'en' else 'zh')
        llm_time = round(time.time() - llm_start, 2)

        # === 对比分析 ===
        answers_match = rag_answer.strip()[:50] == llm_answer.strip()[:50]
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
            }
        }

    except Exception as e:
        logger.error(f"对比接口异常: {e}", exc_info=True)
        return JSONResponse(
            content={'error': '对比接口异常: {}'.format(str(e))},
            status_code=500
        )


@app.get("/api/health")
async def health_check():
    """健康检查接口"""
    try:
        engine, llm = get_engine()
        doc_info = {name: len(e.chunks) for name, e in engine.doc_engines.items()}
        return HealthResponse(
            status="healthy",
            engine_ready=True,
            backend=RAG_BACKEND,
            llm_available=llm.api_available,
            pdf_loaded=os.path.exists(PDF_PATH) and os.path.exists(PDF_PATH2)
        )
    except Exception as e:
        return HealthResponse(
            status="degraded",
            engine_ready=False,
            backend=RAG_BACKEND,
            llm_available=False,
            pdf_loaded=os.path.exists(PDF_PATH) and os.path.exists(PDF_PATH2)
        )


@app.get("/api/understand")
async def understand_question(q: str = ""):
    """Query Understanding 调试接口"""
    if not q:
        return JSONResponse(content={"error": "请提供参数 ?q=您的问题"}, status_code=400)
    result = QueryUnderstanding.understand(q)
    return JSONResponse(content=result)


@app.get("/api/debug_route")
async def debug_route(q: str = ""):
    """路由调试接口"""
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
    """从 start_port 开始尝试，找到第一个可用端口"""
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
