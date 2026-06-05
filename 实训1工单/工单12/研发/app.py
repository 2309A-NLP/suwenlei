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

sys.path.insert(0, os.path.dirname(__file__))

from rag_pipeline import RAGEngine
from llm import LLMClient, GROUND_TRUTH, QueryUnderstanding
from pdf_parser import KnowledgeBuilder

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import re as _re
from pydantic import BaseModel
from typing import Optional


def _is_chinese_text(text: str) -> bool:
    if not text:
        return False
    return bool(_re.search(r'[\u4e00-\u9fff]', text[:500]))
import uvicorn

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---- 路径配置 ----
BASE_DIR = os.path.dirname(__file__)
PDF_PATH = os.path.join(BASE_DIR, '招股说明书1.pdf')
PDF_PATH2 = os.path.join(BASE_DIR, '招股说明书2.pdf')
INDEX_HTML = os.path.join(BASE_DIR, 'index.html')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
RAG_BACKEND = os.environ.get("RAG_BACKEND", "auto")

os.makedirs(UPLOAD_DIR, exist_ok=True)

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
            self.rclient = redis.Redis(
                host=host, port=port, db=db, password=password,
                decode_responses=True, socket_connect_timeout=5,
            )
            self.rclient.ping()  # 验证连接
            self._available = True
            logger.info("Redis连接成功: {}:{}/{}".format(host, port, db))
        except Exception as e:
            self._available = False  # 降级为内存字典
            self.rclient = None
            self._fallback = {}
            logger.warning("Redis不可用，降级内存存储: {}".format(e))

    def generate_id(self):
        return str(uuid.uuid4())

    def get_history(self, conversation_id):
        if not conversation_id:
            return []
        if self._available:
            try:
                raw = self.rclient.get(self._key(conversation_id))
                return json.loads(raw) if raw else []
            except Exception:
                return []
        return self._fallback.get(conversation_id, [])

    def save_history(self, conversation_id, history, ttl=None):
        if not conversation_id:
            return
        ttl = ttl or self.ttl
        if self._available:
            try:
                self.rclient.setex(self._key(conversation_id), ttl,
                                   json.dumps(history, ensure_ascii=False))
            except Exception:
                pass
        else:
            self._fallback[conversation_id] = history

    def append_message(self, conversation_id, role, content):
        """追加单条消息：读取→追加→回写"""
        history = self.get_history(conversation_id)
        history.append({'role': role, 'content': content})
        self.save_history(conversation_id, history)

    def _key(self, conversation_id):
        return self.prefix + conversation_id


# ---- 全局单例 ----
_engine = None
_llm = None
_uploaded_files = []
_store = ConversationStore()


def get_engine():
    """惰性初始化RAG引擎"""
    global _engine, _llm
    if _engine is None:
        logger.info("正在初始化RAG引擎（混合检索模式）...")
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
    """扫描uploads目录"""
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
    """重置引擎"""
    global _engine, _llm
    _engine = None
    _llm = None
    logger.info("引擎已重置，下次查询将重新初始化")


# ---- FastAPI ----
app = FastAPI(title="RAG问答系统 - 混合检索")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    conversation_id: Optional[str] = None
    search_mode: Optional[str] = None  # 检索模式：hybrid/vector/bm25
    lang: Optional[str] = None  # 语言模式：zh/en


class SearchConfigRequest(BaseModel):
    """检索配置请求"""
    search_mode: Optional[str] = None    # vector/bm25/hybrid
    vector_weight: Optional[float] = None  # 向量检索权重
    bm25_weight: Optional[float] = None    # BM25权重
    use_reranker: Optional[bool] = None    # 是否启用reranker
    use_llm_rerank: Optional[bool] = None  # 是否启用LLM重排


class FeedbackRequest(BaseModel):
    """用户反馈请求"""
    question: str
    answer: str
    rating: str  # good/bad
    comment: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    engine_ready: bool
    backend: str
    llm_available: bool
    pdf_loaded: bool
    redis_available: bool = False
    search_mode: str = "hybrid"
    embedder_available: bool = False
    reranker_available: bool = False


# ==================== 前端 ====================

@app.get("/")
async def index():
    return FileResponse(INDEX_HTML)


# ==================== 文件管理API ====================

_kb_builder = KnowledgeBuilder(project_dir=BASE_DIR)


@app.post("/api/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    """上传PDF并触发全链路处理：解析→分块→bge-m3向量化→Milvus入库"""
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

    scan_uploaded_files()
    reload_engine()
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
    reload_engine()
    scan_uploaded_files()
    return {
        'success': True,
        'message': '知识库已重载，包含 {} 个上传文件'.format(len(_uploaded_files)),
        'files': _uploaded_files
    }


# ==================== 检索配置API ====================

@app.get("/api/search/config")
async def get_search_config():
    """获取当前检索配置"""
    engine, _ = get_engine()
    return {
        'search_mode': engine.search_mode,
        'vector_weight': round(engine.vector_weight, 2),
        'bm25_weight': round(engine.bm25_weight, 2),
        'reranker_top_k': engine.reranker_top_k,
        'final_top_k': engine.final_top_k,
    }


@app.post("/api/search/config")
async def set_search_config(req: SearchConfigRequest):
    """设置检索模式和权重"""
    engine, _ = get_engine()
    if req.search_mode:
        engine.set_search_mode(req.search_mode)
    if req.vector_weight is not None or req.bm25_weight is not None:
        engine.set_search_mode(
            engine.search_mode,
            vector_weight=req.vector_weight or engine.vector_weight,
            bm25_weight=req.bm25_weight or engine.bm25_weight
        )
    return {
        'success': True,
        'search_mode': engine.search_mode,
        'vector_weight': round(engine.vector_weight, 2),
        'bm25_weight': round(engine.bm25_weight, 2),
    }


@app.post("/api/rerank")
async def rerank_with_llm(req: QueryRequest):
    """LLM重排接口：用DeepSeek API对候选文档重新评分"""
    engine, llm = get_engine()
    question = req.question.strip()

    # 先执行混合检索获取候选
    qu = QueryUnderstanding.understand(question)
    context, results = engine.get_context(qu['expanded_query'], top_k=10, original_query=question)

    if not results:
        return {'question': question, 'reranked': [], 'message': '无检索结果'}

    # LLM重排
    reranked = llm.llm_rerank(question, results, top_k=req.top_k)

    return {
        'question': question,
        'reranked': [{
            'page': r['page_num'],
            'score': r.get('llm_rerank_score', r.get('score', 0)),
            'text_preview': r['text'][:200],
            'doc_name': r.get('doc_name', ''),
        } for r in reranked],
    }


# ==================== 问答API ====================

@app.post("/api/ask")
async def ask_question(req: QueryRequest):
    """核心问答接口：意图识别→混合检索→重排→Prompt构建→LLM生成"""
    try:
        engine, llm = get_engine()
        start = time.time()
        question = req.question.strip()

        if not question:
            return JSONResponse(
                content={'error': '问题不能为空', 'answer': '', 'sources': []},
                status_code=400
            )

        conv_id = req.conversation_id
        if not conv_id:
            conv_id = _store.generate_id()
        history = _store.get_history(conv_id)

        qu = QueryUnderstanding.understand(question)
        search_query = qu['expanded_query']
        logger.info(f"原问题: {question} | 扩展: {search_query}")

        # 如果请求带了search_mode，临时切换
        if req.search_mode:
            engine.set_search_mode(req.search_mode)

        # 多语言处理：英文问题先翻译为中文用于检索
        target_lang = req.lang if req.lang in ('zh', 'en') else 'zh'
        question_for_index = llm.translate(question, 'zh') if target_lang == 'en' else question
        qu_index = QueryUnderstanding.understand(question_for_index)
        search_query = qu_index['expanded_query']

        context, results = engine.get_context(question_for_index, top_k=req.top_k, original_query=question_for_index)

        if not context:
            answer = "抱歉，在文档中没有找到与您问题相关的内容。请尝试换一种方式提问。"
            if target_lang == 'en':
                answer = llm.translate(answer, 'en')
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

        prompt = engine.build_prompt(question_for_index, context, history=history, lang=target_lang)
        answer = llm.generate(prompt, context, question_for_index, history=history)
        # 英文模式下如果返回中文则翻译回英文
        if target_lang == 'en' and _is_chinese_text(answer):
            answer = llm.translate(answer, 'en')

        _store.append_message(conv_id, 'user', question)
        _store.append_message(conv_id, 'assistant', answer)

        elapsed = round(time.time() - start, 2)
        gt_matched = any(answer == gt or (len(gt) > 10 and gt[:30] in answer)
                         for gt in GROUND_TRUTH.values())

        sources = [
            {
                'page': r['page_num'],
                'score': r.get('rerank_score', r.get('rrf_score', r.get('score', 0))),
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
            'history_length': len(history) + 2,
            'search_mode': engine.search_mode,
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
                'error': '系统内部错误，请稍后重试',
                'answer': '抱歉，系统处理您的问题时出现异常，请稍后重试。',
                'sources': []
            },
            status_code=500
        )


@app.post("/api/compare")
async def compare_answers(req: QueryRequest):
    """对比接口：RAG vs 纯LLM"""
    try:
        engine, llm = get_engine()
        start = time.time()
        question = req.question.strip()

        if not question:
            return JSONResponse(content={'error': '问题不能为空'}, status_code=400)

        conv_id = req.conversation_id
        if not conv_id:
            conv_id = _store.generate_id()
        history = _store.get_history(conv_id)

        # 多语言处理：英文问题先翻译为中文用于检索
        target_lang = req.lang if req.lang in ('zh', 'en') else 'zh'
        question_for_index = llm.translate(question, 'zh') if target_lang == 'en' else question
        qu = QueryUnderstanding.understand(question_for_index)
        search_query = qu['expanded_query']
        context, results = engine.get_context(question_for_index, top_k=req.top_k, original_query=question_for_index)

        if context:
            prompt = engine.build_prompt(question_for_index, context, history=history, lang=target_lang)
            rag_answer = llm.generate(prompt, context, question_for_index, history=history)
            if target_lang == 'en' and _is_chinese_text(rag_answer):
                rag_answer = llm.translate(rag_answer, 'en')
        else:
            rag_answer = "抱歉，在文档中没有找到与您问题相关的内容。"
            if target_lang == 'en':
                rag_answer = llm.translate(rag_answer, 'en')
        rag_time = round(time.time() - start, 2)

        sources = [
            {
                'page': r['page_num'],
                'score': r.get('rerank_score', r.get('rrf_score', r.get('score', 0))),
                'snippet': r['text'][:200],
                'doc_name': r.get('doc_name', ''),
                'source_type': r.get('source_type', 'text'),
            }
            for r in results[:req.top_k]
        ] if context else []

        gt_matched = any(rag_answer == gt for gt in GROUND_TRUTH.values())

        llm_question = question if target_lang == 'zh' else llm.translate(question, 'zh')
        llm_start = time.time()
        llm_answer = llm.generate_pure_llm(llm_question)
        if target_lang == 'en' and _is_chinese_text(llm_answer):
            llm_answer = llm.translate(llm_answer, 'en')
        llm_time = round(time.time() - llm_start, 2)

        _store.append_message(conv_id, 'user', question)
        _store.append_message(conv_id, 'assistant', rag_answer)

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


# ==================== 用户反馈API ====================

@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    """用户反馈：记录问答质量评价，用于优化检索策略"""
    feedback_data = {
        'question': req.question,
        'answer': req.answer[:200],
        'rating': req.rating,
        'comment': req.comment or '',
        'timestamp': time.time(),
    }
    # 持久化到文件（简单方案）
    feedback_file = os.path.join(BASE_DIR, 'feedback.jsonl')
    try:
        with open(feedback_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback_data, ensure_ascii=False) + '\n')
    except Exception as e:
        logger.warning(f"反馈保存失败: {e}")

    return {'success': True, 'message': '感谢您的反馈！'}


# ==================== LightRAG 知识图谱API ====================

@app.post("/api/lightrag/build")
async def build_lightrag():
    """构建LightRAG知识图谱"""
    try:
        from lightrag_engine import build_knowledge_graph
        import nest_asyncio
        try:
            nest_asyncio.apply()
        except Exception:
            pass
        result = build_knowledge_graph()
        return result
    except Exception as e:
        logger.error(f"LightRAG构建失败: {e}", exc_info=True)
        return JSONResponse(content={'error': str(e)}, status_code=500)


@app.post("/api/lightrag/query")
async def query_lightrag_api(req: QueryRequest):
    """LightRAG查询"""
    try:
        from lightrag_engine import query_lightrag
        import nest_asyncio
        try:
            nest_asyncio.apply()
        except Exception:
            pass
        mode = req.search_mode or 'hybrid'
        result = query_lightrag(req.question, mode=mode)
        return result
    except Exception as e:
        logger.error(f"LightRAG查询失败: {e}", exc_info=True)
        return JSONResponse(content={'error': str(e)}, status_code=500)


@app.post("/api/lightrag/compare")
async def compare_rag_lightrag(req: QueryRequest):
    """RAG vs LightRAG 对比"""
    try:
        import time as _time
        import nest_asyncio
        try:
            nest_asyncio.apply()
        except Exception:
            pass
        from lightrag_engine import query_lightrag

        question = req.question.strip()
        engine, llm = get_engine()

        # RAG检索
        t1 = _time.time()
        qu = QueryUnderstanding.understand(question)
        context, rag_results = engine.get_context(qu['expanded_query'], top_k=req.top_k, original_query=question)
        rag_answer = ''
        if context:
            prompt = engine.build_prompt(question, context)
            rag_answer = llm.generate(prompt, context, question)
        rag_time = round(_time.time() - t1, 2)

        # LightRAG检索
        t2 = _time.time()
        lr_result = query_lightrag(question, mode='hybrid')
        lr_time = round(_time.time() - t2, 2)

        return {
            'question': question,
            'rag': {
                'answer': rag_answer,
                'time_seconds': rag_time,
                'sources_count': len(rag_results),
                'sources': [{'page': r['page_num'], 'score': r.get('rerank_score', r.get('rrf_score', 0)), 'snippet': r['text'][:200]} for r in rag_results[:5]],
            },
            'lightrag': {
                'answer': lr_result.get('answer', ''),
                'time_seconds': lr_time,
                'mode': 'hybrid',
            },
        }
    except Exception as e:
        logger.error(f"对比查询失败: {e}", exc_info=True)
        return JSONResponse(content={'error': str(e)}, status_code=500)


@app.get("/api/lightrag/graph")
async def get_lightrag_graph():
    """导出LightRAG知识图谱"""
    try:
        from lightrag_engine import export_knowledge_graph
        return export_knowledge_graph()
    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)


# ==================== 健康检查 ====================

@app.get("/api/health")
async def health_check():
    try:
        engine, llm = get_engine()
        doc_info = {name: len(e.chunks) for name, e in engine.doc_engines.items()}
        # 检查嵌入模型和重排模型
        embedder = engine._get_embedder()
        reranker = engine._get_reranker()
        return HealthResponse(
            status="healthy",
            engine_ready=True,
            backend=RAG_BACKEND,
            llm_available=llm.api_available,
            pdf_loaded=os.path.exists(PDF_PATH) and os.path.exists(PDF_PATH2),
            redis_available=_store.available,
            search_mode=engine.search_mode,
            embedder_available=embedder is not None and embedder.is_available(),
            reranker_available=reranker is not None and reranker.is_available(),
        )
    except Exception as e:
        return HealthResponse(
            status="degraded",
            engine_ready=False,
            backend=RAG_BACKEND,
            llm_available=False,
            pdf_loaded=os.path.exists(PDF_PATH) and os.path.exists(PDF_PATH2),
            redis_available=_store.available,
        )


# ==================== 对话管理 ====================

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
    """调试接口：查看Query Understanding结果"""
    if not q:
        return JSONResponse(content={"error": "请提供参数 ?q=您的问题"}, status_code=400)
    result = QueryUnderstanding.understand(q)
    return JSONResponse(content=result)


@app.get("/api/debug_route")
async def debug_route(q: str = ""):
    """调试接口：查看路由+检索+融合全过程"""
    engine, _ = get_engine()
    route = engine._route_query(q)
    context, results = engine.get_context(q, top_k=5, original_query=q)
    return JSONResponse(content={
        "query": q,
        "route": route,
        "search_mode": engine.search_mode,
        "vector_weight": engine.vector_weight,
        "bm25_weight": engine.bm25_weight,
        "doc_engines_keys": list(engine.doc_engines.keys()),
        "results_count": len(results),
        "results": [{
            "doc_name": r.get("doc_name", ""),
            "page": r["page_num"],
            "score": r.get("rerank_score", r.get("rrf_score", r.get("score", 0))),
            "type": r.get("source_type", "text"),
            "source": r.get("source", "unknown"),
            "text_preview": r["text"][:100],
        } for r in results]
    })


# ==================== 启动 ====================

def find_available_port(start_port=8888, max_tries=10):
    """端口探测"""
    import socket
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    raise RuntimeError("无法找到可用端口")


def main():
    port = find_available_port(start_port=8888)
    scan_uploaded_files()
    try:
        import jieba as _jieba
        _jieba.initialize()
    except Exception:
        pass
    print("=" * 56)
    print("  RAG 问答系统 — 混合检索（向量+BM25+RRF+Reranker）")
    print("  向量模型: bge-m3 (1024维)")
    print("  全文检索: BM25 (jieba分词)")
    print("  融合算法: RRF (Reciprocal Rank Fusion)")
    print("  重排模型: bge-reranker-v2-m3")
    print("  PDF1: {}".format(PDF_PATH))
    print("  PDF2: {}".format(PDF_PATH2))
    print("  上传目录: {} ({} 个文件)".format(UPLOAD_DIR, len(_uploaded_files)))
    print("=" * 56)
    # 预加载模型到GPU
    print("\n[启动] 预加载模型到GPU...")
    try:
        from models import EmbeddingClient
        emb = EmbeddingClient()
        emb.encode("预加载测试")
        print("[启动] bge-m3 embedding → GPU 加载完成")
    except Exception as e:
        print(f"[启动] bge-m3预加载跳过: {e}")
    try:
        from models import RerankerClient
        rkr = RerankerClient()
        if rkr.is_available():
            rkr.rerank("预加载测试", [{"text": "测试"}], top_k=1)
            print("[启动] bge-reranker-v2-m3 → GPU 加载完成")
    except Exception as e:
        print(f"[启动] reranker预加载跳过: {e}")
    print()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == '__main__':
    main()
