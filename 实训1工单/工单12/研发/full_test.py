# -*- coding: utf-8 -*-
"""
RAG工单12 完整16题测试
"""
import os, sys, json, time, logging, asyncio, re, numpy as np
import nest_asyncio
nest_asyncio.apply()
from typing import List, Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
STORAGE_DIR_XT = os.path.join(BASE_DIR, 'lightrag_storage_v2')
STORAGE_DIR_LY = os.path.join(BASE_DIR, 'lightrag_storage_ly')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ==================== API Key ====================
sys.path.insert(0, BASE_DIR)
from llm import DEEPSEEK_API_KEY
DEEPSEEK_BASE_URL = 'https://api.deepseek.com'
PROXIES = {'http': '', 'https': ''}

def _call_deepseek_sync(prompt, system_prompt=''):
    """同步调用DeepSeek（用于RAGAS评分）"""
    import requests
    msgs = []
    if system_prompt:
        msgs.append({'role': 'system', 'content': system_prompt})
    msgs.append({'role': 'user', 'content': prompt})
    resp = requests.post(
        f'{DEEPSEEK_BASE_URL}/v1/chat/completions',
        headers={'Authorization': f'Bearer {DEEPSEEK_API_KEY}', 'Content-Type': 'application/json'},
        json={'model': 'deepseek-chat', 'messages': msgs, 'temperature': 0.1, 'max_tokens': 4096},
        timeout=120, proxies=PROXIES
    )
    resp.raise_for_status()
    return resp.json()['choices'][0]['message']['content']

# LightRAG需要async的LLM函数
async def lightrag_llm_func(prompt, system_prompt='', **kw):
    return _call_deepseek_sync(prompt, system_prompt)

# ==================== bge-m3嵌入 ====================
_embedder = None
def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        try:
            _embedder = SentenceTransformer(os.path.join(MODEL_DIR, 'bge-m3'), device='cuda')
        except:
            _embedder = SentenceTransformer(os.path.join(MODEL_DIR, 'bge-m3'), device='cpu')
    return _embedder

async def _embedding_impl(texts, **kw):
    import torch
    emb = _get_embedder().encode(texts, batch_size=32, show_progress_bar=False)
    return emb.cpu().numpy() if isinstance(emb, torch.Tensor) else np.array(emb, dtype=np.float32)

# ==================== LightRAG ====================
def _create_lightrag(working_dir):
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    ef = EmbeddingFunc(embedding_dim=1024, func=_embedding_impl, max_token_size=2048)
    return LightRAG(
        working_dir=working_dir, llm_model_func=lightrag_llm_func,
        embedding_func=ef, chunk_token_size=3000, chunk_overlap_token_size=200,
        llm_model_name='deepseek-chat',
    )

def classify_question(question):
    for kw in ['力源信息', '力源信息技术', '武汉力源']:
        if kw in question:
            return 'ly'
    return 'xt'

async def query_lightrag(question):
    from lightrag.base import QueryParam
    kg_type = classify_question(question)
    storage_dir = STORAGE_DIR_LY if kg_type == 'ly' else STORAGE_DIR_XT
    try:
        rag = _create_lightrag(storage_dir)
        await rag.initialize_storages()
        result = await rag.aquery(question, param=QueryParam(mode='hybrid', hl_keywords=[question], ll_keywords=['']))
        await rag.finalize_storages()
        return {'answer': str(result), 'context': str(result)[:2000], 'kg_type': kg_type}
    except Exception as e:
        logger.error(f"LightRAG查询失败: {e}")
        return {'answer': f'查询失败: {e}', 'context': '', 'kg_type': kg_type}

# ==================== 16道测试题 ====================
TEST_QUESTIONS = [
    {"id": 5, "question": "武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成，其中大客户销售部有几个销售处构成？"},
    {"id": 6, "question": "武汉力源信息技术股份有限公司招股意向书中，从2008年中国IC市场应用结构与增长图中可以看出，增长率最快的是哪个行业？负增长的是哪个行业？"},
    {"id": 1, "question": "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？"},
    {"id": 2, "question": "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？"},
    {"id": 3, "question": "与武汉力源信息技术股份有限公司存在控制关系的关联方是谁，持股比例和本公司关系是什么？"},
    {"id": 4, "question": "与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？"},
    {"id": 260, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"},
    {"id": 95, "question": "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？"},
    {"id": 33, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？"},
    {"id": 34, "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？"},
    {"id": 957, "question": "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？"},
    {"id": 793, "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？"},
    {"id": 795, "question": "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？"},
    {"id": 543, "question": "武汉兴图新科电子股份有限公司注册资本是多少？"},
    {"id": 531, "question": "武汉兴图新科电子股份有限公司法定代表人是谁？"},
    {"id": 207, "question": "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？"},
]

def score_rag(question, answer, context):
    """同步RAGAS评分"""
    prompt = f"""评估问答质量，返回JSON：
问题: {question}
答案: {answer}
上下文: {context[:800]}
指标(0-1): answer_relevancy, context_precision, context_recall, faithfulness
JSON: {{"answer_relevancy": 0.x, "context_precision": 0.x, "context_recall": 0.x, "faithfulness": 0.x}}"""
    try:
        result = _call_deepseek_sync(prompt)
        m = re.search(r'\{[^}]+\}', result)
        if m:
            return json.loads(m.group())
    except:
        pass
    return {'answer_relevancy': 0, 'context_precision': 0, 'context_recall': 0, 'faithfulness': 0}

async def main():
    logger.info("=" * 60)
    logger.info("RAG工单12 完整16题测试")
    logger.info("=" * 60)

    # Step1: LightRAG查询
    logger.info("步骤1: LightRAG查询")
    lr_results = []
    for i, q in enumerate(TEST_QUESTIONS):
        logger.info(f"  {i+1}/16 [{classify_question(q['question'])}]: {q['question'][:50]}...")
        t0 = time.time()
        result = await query_lightrag(q['question'])
        result['time'] = round(time.time() - t0, 2)
        lr_results.append(result)
        logger.info(f"    耗时{result['time']}s, 答案长度{len(result['answer'])}")

    # Step2: 读取RAG结果（对0分题自动重新查询）
    logger.info("步骤2: 读取RAG结果")
    with open(os.path.join(RESULTS_DIR, 'rag_vs_lightrag_report.json'), 'r', encoding='utf-8') as f:
        rag_data = json.load(f)
    rag_results = []
    for q in TEST_QUESTIONS:
        for pq in rag_data['per_question']:
            if pq['id'] == q['id']:
                rag_results.append(pq['rag'])
                break

    # Step2.5: 对0分题重新走RAG检索
    from rag_pipeline import RAGEngine
    from llm import LLMClient
    llm_client = LLMClient()
    pdf_map = {}
    for name, path in [('招股说明书1', os.path.join(UPLOAD_DIR, '招股说明书1.pdf')),
                        ('招股说明书2', os.path.join(UPLOAD_DIR, '招股说明书2.pdf'))]:
        if os.path.exists(path):
            pdf_map[name] = path
    rag_engine = RAGEngine(pdf_paths=pdf_map)

    # Step3: RAGAS评分
    logger.info("步骤3: RAGAS评分")
    rag_scores, lr_scores = [], []
    for i, (q, rag_r, lr_r) in enumerate(zip(TEST_QUESTIONS, rag_results, lr_results)):
        rag_scores.append(score_rag(q['question'], rag_r['answer'], rag_r.get('context', '')))
        lr_scores.append(score_rag(q['question'], lr_r['answer'], lr_r.get('context', '')))

    # Step3.5: 对0分题重新检索+生成+评分（跳过reranker避免兼容问题）
    for i, q in enumerate(TEST_QUESTIONS):
        if all(v == 0 for v in rag_scores[i].values()):
            logger.info(f"  ID {q['id']} RAG=0分，重新检索...")
            context, chunks = rag_engine.get_context(q['question'], top_k=5, skip_reranker=True)
            if context:
                prompt = rag_engine.build_prompt(q['question'], context, lang='zh')
                new_ans = llm_client.generate(prompt, context, q['question'])
                rag_results[i] = {'answer': new_ans, 'time': 0, 'context': context}
                rag_scores[i] = score_rag(q['question'], new_ans, context)
                logger.info(f"    重查完成，新分数: {rag_scores[i]}")
            else:
                logger.warning(f"    重查仍无结果")

    # 计算平均分
    rag_avg = {k: float(np.mean([s[k] for s in rag_scores])) for k in ['answer_relevancy', 'context_precision', 'context_recall', 'faithfulness']}
    lr_avg = {k: float(np.mean([s[k] for s in lr_scores])) for k in ['answer_relevancy', 'context_precision', 'context_recall', 'faithfulness']}

    # 输出
    print("\n指标对比:")
    print(f"{'指标':<20} {'RAG':<10} {'LightRAG':<10} {'差异':<10}")
    print("-" * 50)
    for k in ['answer_relevancy', 'context_precision', 'context_recall', 'faithfulness']:
        d = lr_avg[k] - rag_avg[k]
        print(f"{k:<20} {rag_avg[k]:.4f}     {lr_avg[k]:.4f}     {d:+.4f}")

    print("\n逐题对比:")
    print(f"{'ID':<6} {'问题':<38} {'RAG':<8} {'LR':<8} {'胜出'}")
    print("-" * 75)
    for i, q in enumerate(TEST_QUESTIONS):
        ra = sum(rag_scores[i].values()) / 4
        la = sum(lr_scores[i].values()) / 4
        w = "LightRAG" if la > ra else ("RAG" if ra > la else "平局")
        print(f"{q['id']:<6} {q['question'][:36]:<38} {ra:.2f}    {la:.2f}    {w}")

    # 保存
    report = {
        'summary': {'total_questions': 16, 'rag_ragas_avg': rag_avg, 'lightrag_ragas_avg': lr_avg},
        'per_question': [{'id': q['id'], 'question': q['question'], 'rag': rag_results[i],
                          'lightrag': lr_results[i], 'rag_ragas': rag_scores[i], 'lightrag_ragas': lr_scores[i]}
                         for i, q in enumerate(TEST_QUESTIONS)]
    }
    with open(os.path.join(RESULTS_DIR, 'rag_vs_lightrag_report_final.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    zero_lr = sum(1 for s in lr_scores if all(v == 0 for v in s.values()))
    zero_rag = sum(1 for s in rag_scores if all(v == 0 for v in s.values()))
    logger.info(f"\nLightRAG 0分: {zero_lr}/16, RAG 0分: {zero_rag}/16")

if __name__ == '__main__':
    asyncio.run(main())
