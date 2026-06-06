# -*- coding: utf-8 -*-
"""
Milvus入库脚本 — 独立运行，将两个PDF的CSV分块分别入库到各自的Milvus集合
用法：python milvus_init.py
"""
import os
import csv
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 项目目录
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(PROJECT_DIR, 'uploads')

# 需要入库的文档列表
DOCUMENTS = [
    '招股说明书1',
    '招股说明书2',
]

# 停用词表
STOP_WORDS = {
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
    '都', '一', '上', '也', '很', '到', '说', '要', '去', '你',
    '会', '着', '没有', '看', '好', '自己', '这', '他', '她',
    '它', '们', '那', '些', '与', '及', '或', '对', '被', '把',
    '从', '向', '以', '为', '由', '于', '而', '但', '且', '之',
    '其', '所', '者', '过', '将', '让', '使', '能', '可', '得',
    '已', '还', '又', '再', '才', '则', '等', '如', '若', '虽',
    '因', '故', '并', '非', '即', '既', '各', '每', '某', '该',
    '本', '哪', '何', '么', '吗', '呢', '吧', '啊', '哦', '嗯',
    '涉及', '包括', '通过', '进行', '实现', '提供', '取得',
    '分别', '相关', '上述', '其中', '以及', '报告', '期内', '来自',
}


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
    """处理单个文档：加载CSV → bge-m3嵌入 → Milvus入库"""
    from milvus_store import MilvusStore
    from embedder import embed_texts

    csv_path = os.path.join(UPLOADS_DIR, f'{doc_name}_chunks_v2.csv')
    if not os.path.exists(csv_path):
        logger.error(f"CSV文件不存在: {csv_path}")
        return 0

    start_time = time.time()
    logger.info(f"========== 开始处理: {doc_name} ==========")

    # 1. 加载CSV分块
    chunks = load_csv_chunks(csv_path)
    logger.info(f"[{doc_name}] 加载了 {len(chunks)} 个分块")
    if not chunks:
        logger.warning(f"[{doc_name}] 无分块数据，跳过")
        return 0

    # 2. 使用bge-m3生成4096维向量
    texts = [c['content'] for c in chunks]
    logger.info(f"[{doc_name}] 使用bge-m3生成嵌入向量...")
    embeddings = embed_texts(texts)
    if not embeddings:
        logger.warning(f"[{doc_name}] 嵌入向量生成失败，跳过")
        return 0
    logger.info(f"[{doc_name}] bge-m3嵌入完成: {len(texts)}条, 维度={len(embeddings[0])}")

    # 3. Milvus入库
    milvus = MilvusStore()
    if not milvus.is_available():
        logger.warning(f"[{doc_name}] Milvus不可用，跳过入库")
        return 0

    # 重建集合（清空旧数据）
    logger.info(f"[{doc_name}] 重建Milvus集合...")
    milvus.create_collection(doc_name, force_recreate=True)

    # 插入分块和向量
    logger.info(f"[{doc_name}] 向Milvus插入 {len(chunks)} 条分块...")
    milvus.insert_chunks(chunks, embeddings, doc_name)

    elapsed = round(time.time() - start_time, 2)
    logger.info(f"[{doc_name}] Milvus入库完成: {len(chunks)} 条, 耗时{elapsed}s")
    return len(chunks)


if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info("RAG工单3 — Milvus批量入库脚本")
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
