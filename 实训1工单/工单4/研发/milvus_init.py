# -*- coding: utf-8 -*-
"""Milvus入库脚本 — 独立运行，将两个PDF的CSV分块分别入库到各自的Milvus集合
用法：python milvus_init.py
"""
import os
import csv
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # 配置日志格式
logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))  # 项目根目录
UPLOADS_DIR = os.path.join(PROJECT_DIR, 'uploads')  # 上传文件目录

DOCUMENTS = [  # 需要入库的文档列表
    '招股说明书1',  # 文档1
    '招股说明书2',  # 文档2
]


def load_csv_chunks(csv_path):
    """从CSV文件加载分块数据"""
    chunks = []  # 初始化分块列表
    with open(csv_path, 'r', encoding='utf-8-sig') as f:  # 以UTF-8 BOM格式打开
        reader = csv.DictReader(f)  # 创建字典读取器
        for row in reader:  # 遍历每一行
            chunks.append({  # 添加分块字典
                'content': row.get('text', ''),  # 文本内容
                'page_num': int(row.get('page_num', 0)),  # 页码
                'chunk_idx': int(row.get('id', row.get('chunk_index', 0))),  # 分块索引
                'source_type': row.get('source_type', 'text'),  # 来源类型
            })
    return chunks  # 返回分块列表


def process_document(doc_name):
    """处理单个文档：加载CSV → BGE-M3向量化 → Milvus入库"""
    from embedder import get_embedder  # 导入嵌入器工厂函数
    from milvus_store import MilvusStore  # 导入Milvus存储类

    csv_path = os.path.join(UPLOADS_DIR, f'{doc_name}_chunks_v2.csv')  # 推导CSV路径
    if not os.path.exists(csv_path):  # CSV不存在时
        logger.error(f"CSV文件不存在: {csv_path}")  # 记录错误
        return 0  # 返回0

    start_time = time.time()  # 记录开始时间
    logger.info(f"========== 开始处理: {doc_name} ==========")  # 打印标题

    chunks = load_csv_chunks(csv_path)  # 加载CSV分块
    logger.info(f"[{doc_name}] 加载了 {len(chunks)} 个分块")  # 记录分块数量
    if not chunks:  # 无分块时
        logger.warning(f"[{doc_name}] 无分块数据，跳过")  # 记录警告
        return 0  # 返回0

    texts = [c['content'] for c in chunks]  # 提取所有分块的文本内容

    embedder = get_embedder()  # 获取BGE-M3嵌入器单例
    logger.info(f"[{doc_name}] 用BGE-M3生成4096维向量...")  # 记录向量化开始
    embeddings = embedder.embed_texts(texts)  # 生成4096维向量
    logger.info(f"[{doc_name}] BGE-M3向量生成完成: {len(embeddings)}条 x {len(embeddings[0])}维")  # 记录维度信息

    milvus = MilvusStore()  # 创建Milvus存储实例
    if not milvus.is_available():  # Milvus不可用时
        logger.warning(f"[{doc_name}] Milvus不可用，跳过入库")  # 记录警告
        return 0  # 返回0

    logger.info(f"[{doc_name}] 重建Milvus集合...")  # 记录重建日志
    milvus.create_collection(doc_name, force_recreate=True)  # 强制重建集合

    logger.info(f"[{doc_name}] 向Milvus插入 {len(chunks)} 条分块...")  # 记录插入日志
    milvus.insert_chunks(chunks, embeddings, doc_name)  # 批量插入分块和向量

    elapsed = round(time.time() - start_time, 2)  # 计算耗时
    logger.info(f"[{doc_name}] Milvus入库完成: {len(chunks)} 条, 耗时{elapsed}s")  # 记录完成
    return len(chunks)  # 返回入库数量


if __name__ == '__main__':  # 主入口
    logger.info("=" * 50)  # 打印分隔线
    logger.info("RAG工单4 — Milvus批量入库脚本（BGE-M3 4096维）")  # 打印标题
    logger.info("=" * 50)  # 打印分隔线

    total_inserted = 0  # 总入库计数
    for doc_name in DOCUMENTS:  # 遍历文档列表
        try:
            count = process_document(doc_name)  # 处理单个文档
            total_inserted += count  # 累加入库数量
        except Exception as e:  # 捕获异常
            logger.error(f"[{doc_name}] 入库异常: {e}", exc_info=True)  # 记录异常堆栈

    logger.info("=" * 50)  # 打印分隔线
    logger.info(f"全部完成! 共入库 {total_inserted} 条记录")  # 打印完成统计
    logger.info("=" * 50)  # 打印分隔线
