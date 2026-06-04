# -*- coding: utf-8 -*-
"""
Milvus入库脚本 — 使用bge-m3向量化，将PDF分块入库到Milvus
用法：python milvus_init.py
"""
import os
import csv
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(PROJECT_DIR, 'uploads')

DOCUMENTS = [
    '招股说明书1',
    '招股说明书2',
]


def load_csv_chunks(csv_path):
    """从CSV文件加载分块数据"""
    chunks = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            chunks.append({
                'content': row.get('text', ''),
                'page_num': int(row.get('page_num', 0)),
                'chunk_idx': int(row.get('id', row.get('chunk_index', 0))),
                'source_type': row.get('source_type', 'text'),
            })
    return chunks


def process_document(doc_name):
    """处理单个文档：加载CSV → bge-m3向量化 → Milvus入库"""
    from embedder import get_embedder
    from milvus_store import MilvusStore

    csv_path = os.path.join(UPLOADS_DIR, f'{doc_name}_chunks_v2.csv')
    if not os.path.exists(csv_path):
        logger.error(f"CSV文件不存在: {csv_path}")
        return 0

    start_time = time.time()
    logger.info(f"========== 开始处理: {doc_name} ==========")

    chunks = load_csv_chunks(csv_path)
    logger.info(f"[{doc_name}] 加载了 {len(chunks)} 个分块")
    if not chunks:
        logger.warning(f"[{doc_name}] 无分块数据，跳过")
        return 0

    # bge-m3向量化
    texts = [c['content'] for c in chunks]
    logger.info(f"[{doc_name}] bge-m3向量化 {len(texts)} 条文本...")
    embedder = get_embedder()
    embeddings = embedder.embed_texts(texts)
    logger.info(f"[{doc_name}] 向量化完成, 维度: {len(embeddings[0]) if embeddings else 0}")

    # Milvus入库
    milvus = MilvusStore()
    if not milvus.is_available():
        logger.warning(f"[{doc_name}] Milvus不可用，跳过入库")
        return 0

    logger.info(f"[{doc_name}] 重建Milvus集合...")
    milvus.create_collection(doc_name, force_recreate=True)

    logger.info(f"[{doc_name}] 向Milvus插入 {len(chunks)} 条分块...")
    milvus.insert_chunks(chunks, embeddings, doc_name)

    elapsed = round(time.time() - start_time, 2)
    logger.info(f"[{doc_name}] Milvus入库完成: {len(chunks)} 条, 耗时{elapsed}s")
    return len(chunks)


if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info("RAG工单5 — bge-m3 Milvus批量入库脚本")
    logger.info("=" * 50)

    total_inserted = 0
    for doc_name in DOCUMENTS:
        try:
            count = process_document(doc_name)
            total_inserted += count
        except Exception as e:
            logger.error(f"[{doc_name}] 入库异常: {e}", exc_info=True)

    logger.info("=" * 50)
    logger.info(f"全部完成! 共入库 {total_inserted} 条记录")
    logger.info("=" * 50)
