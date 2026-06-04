# -*- coding: utf-8 -*-
"""
工单11 - 检索效果评估：微调前 vs 微调后
在真实RAG场景中对比召回效果
指标: Recall@k, MRR
"""
import sys, os, json, logging, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model', 'bge-base-zh-v1.5')
FINETUNED_DIR = os.path.join(os.path.dirname(__file__), 'output', 'bge-base-zh-v1.5-finetuned')
EVAL_FILE = os.path.join(os.path.dirname(__file__), 'data', 'eval_queries.json')
CHUNKS_CSV = os.path.join(BASE_DIR, 'uploads', '招股说明书1_chunks_v2.csv')
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), 'output', 'evaluation_result.json')


def load_chunks():
    """加载PDF分块作为检索语料库"""
    import csv
    if not os.path.exists(CHUNKS_CSV):
        logger.error(f"CSV不存在: {CHUNKS_CSV}")
        # 尝试找其他路径
        alt = CHUNKS_CSV.replace('招股说明书1_chunks_v2.csv', '招股说明书1_chunks.csv')
        if os.path.exists(alt):
            return _load_csv(alt)
        logger.error("未找到chunks CSV，请先运行 prepare_data.py")
        return []
    return _load_csv(CHUNKS_CSV)


def _load_csv(path):
    import csv
    chunks = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get('text', '').strip()
            if len(text) < 20:
                continue
            chunks.append({
                'id': len(chunks),
                'text': text[:1000],
                'page_num': int(row.get('page_num', 0)),
            })
    logger.info(f"[评估] 加载语料库: {len(chunks)} 条chunk")
    return chunks


def load_eval_queries():
    """每次从train_triplets.json随机抽取30条评估query"""
    import random
    triplets_file = os.path.join(os.path.dirname(__file__), 'data', 'train_triplets.json')
    if not os.path.exists(triplets_file):
        logger.error(f"三元组数据不存在: {triplets_file}")
        return []
    with open(triplets_file, 'r', encoding='utf-8') as f:
        triplets = json.load(f)
    random.seed()  # 不固定seed，每次随机
    samples = random.sample(triplets, min(30, len(triplets)))
    queries = [{'query': t['query'], 'relevant_text': t['positive']} for t in samples]
    logger.info(f"[评估] 随机抽取评估query: {len(queries)} 条")
    return queries


def evaluate_model(model, queries, corpus_chunks, top_k_list=[1, 3, 5, 10]):
    """
    评估模型检索性能（完整指标集）
    对每个query: 编码 → 余弦相似度搜索 → 检查召回
    指标: Recall@k, Precision@k, F1@k, Hit Rate@k, MRR, MAP, NDCG@k, Average Rank
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # 1. 编码语料库
    corpus_texts = [c['text'] for c in corpus_chunks]
    logger.info(f"[评估] 编码语料库 ({len(corpus_texts)} 条)...")
    corpus_embeddings = model.encode(corpus_texts, batch_size=16, show_progress_bar=True, normalize_embeddings=True)
    corpus_embeddings = np.array(corpus_embeddings).astype(np.float32)

    # 2. 评估每个query
    recalls = {k: [] for k in top_k_list}
    precisions = {k: [] for k in top_k_list}
    hits = {k: [] for k in top_k_list}
    ndcgs = {k: [] for k in top_k_list}
    mrrs = []
    aps = []  # Average Precision
    ranks = []

    for i, q in enumerate(queries):
        query_text = q['query']
        relevant_text = q['relevant_text'][:1000]

        # 找到relevant text在语料库中的位置
        relevant_idx = -1
        for j, c in enumerate(corpus_chunks):
            if c['text'] == relevant_text:
                relevant_idx = j
                break

        # 编码query
        query_emb = model.encode([query_text], normalize_embeddings=True)
        query_emb = np.array(query_emb).astype(np.float32)

        # 余弦相似度
        scores = cosine_similarity(query_emb, corpus_embeddings)[0]

        # 排序
        ranked_indices = np.argsort(scores)[::-1]

        # 计算各项指标
        if relevant_idx >= 0:
            rank = np.where(ranked_indices == relevant_idx)[0]
            rank_pos = int(rank[0]) + 1 if len(rank) > 0 else len(corpus_chunks)

            # Recall@k: top-k中是否包含相关文档（单relevant doc时为hit）
            for k in top_k_list:
                recalls[k].append(1 if rank_pos <= k else 0)

            # Precision@k: top-k中相关文档数 / k（单relevant doc时为1/k if hit）
            for k in top_k_list:
                precisions[k].append(1.0 / k if rank_pos <= k else 0.0)

            # Hit Rate@k: 是否至少有一个相关文档在top-k中
            for k in top_k_list:
                hits[k].append(1 if rank_pos <= k else 0)

            # NDCG@k: 归一化折损累积增益
            for k in top_k_list:
                if rank_pos <= k:
                    # 相关文档在位置rank_pos，gain=1，折扣=log2(rank_pos+1)
                    dcg = 1.0 / np.log2(rank_pos + 1)
                    # 理想情况：相关文档在位置1
                    idcg = 1.0 / np.log2(2)  # = 1.0
                    ndcgs[k].append(dcg / idcg)
                else:
                    ndcgs[k].append(0.0)

            # MRR: 倒数排名
            mrrs.append(1.0 / rank_pos)

            # Average Precision (AP): 所有相关位置的precision均值
            # 单relevant doc时，AP = Precision@rank_pos = 1/rank_pos
            aps.append(1.0 / rank_pos)

            # Average Rank: 相关文档的排名位置
            ranks.append(rank_pos)
        else:
            # relevant text不在语料库中
            pass

        if (i + 1) % 10 == 0:
            logger.info(f"[评估] 进度 [{i+1}/{len(queries)}]")

    # 3. 汇总
    result = {}
    for k in top_k_list:
        result[f'Recall@{k}'] = float(np.mean(recalls[k])) if recalls[k] else 0.0
        result[f'Precision@{k}'] = float(np.mean(precisions[k])) if precisions[k] else 0.0
        result[f'Hit Rate@{k}'] = float(np.mean(hits[k])) if hits[k] else 0.0
        result[f'NDCG@{k}'] = float(np.mean(ndcgs[k])) if ndcgs[k] else 0.0
        # F1@k = 2 * Precision * Recall / (Precision + Recall)
        p = result[f'Precision@{k}']
        r = result[f'Recall@{k}']
        result[f'F1@{k}'] = round(2 * p * r / (p + r), 6) if (p + r) > 0 else 0.0
    result['MRR'] = float(np.mean(mrrs)) if mrrs else 0.0
    result['MAP'] = float(np.mean(aps)) if aps else 0.0
    result['Average Rank'] = float(np.mean(ranks)) if ranks else 0.0
    result['Evaluated Queries'] = len(ranks)
    return result


def main():
    from sentence_transformers import SentenceTransformer

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # 加载数据
    queries = load_eval_queries()
    corpus = load_chunks()
    if not queries or not corpus:
        logger.error("数据加载失败，退出")
        return

    # 评估微调前模型
    logger.info("=" * 50)
    logger.info("[评估] 微调前模型: bge-base-zh-v1.5")
    pre_model = SentenceTransformer(MODEL_DIR)
    pre_results = evaluate_model(pre_model, queries, corpus)
    logger.info(f"[评估] 微调前: {json.dumps(pre_results, ensure_ascii=False)}")

    # 评估微调后模型
    logger.info("=" * 50)
    logger.info(f"[评估] 微调后模型: {FINETUNED_DIR}")
    if os.path.isdir(FINETUNED_DIR):
        post_model = SentenceTransformer(FINETUNED_DIR)
        post_results = evaluate_model(post_model, queries, corpus)
        logger.info(f"[评估] 微调后: {json.dumps(post_results, ensure_ascii=False)}")
    else:
        logger.warning(f"[评估] 微调后模型不存在: {FINETUNED_DIR}，跳过")
        post_results = {}

    # 对比输出
    combined = {
        'model': 'BAAI/bge-base-zh-v1.5',
        'eval_queries_count': len(queries),
        'corpus_chunks_count': len(corpus),
        'pre_finetune': pre_results,
        'post_finetune': post_results,
    }
    if pre_results and post_results:
        combined['improvement'] = {}
        for metric in pre_results:
            combined['improvement'][metric] = {
                'before': pre_results[metric],
                'after': post_results.get(metric, 0),
                'delta': post_results.get(metric, 0) - pre_results[metric],
            }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    logger.info(f"[评估] 结果保存至: {OUTPUT_FILE}")

    # 打印汇总
    print("\n" + "=" * 60)
    print("  评估总结")
    print("=" * 60)
    print(f"  {'指标':<15} {'微调前':<12} {'微调后':<12} {'提升':<12}")
    print("  " + "-" * 51)
    for metric in pre_results:
        pre = pre_results.get(metric, 0)
        post = post_results.get(metric, 0)
        delta = post - pre
        delta_str = f"{delta:+.4f}" if delta != 0 else "-"
        print(f"  {metric:<15} {pre:<12.4f} {post:<12.4f} {delta_str:<12}")
    print("=" * 60)


if __name__ == '__main__':
    main()
