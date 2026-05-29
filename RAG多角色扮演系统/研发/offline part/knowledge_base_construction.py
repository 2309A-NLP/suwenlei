# -*- coding: utf-8 -*-
# knowledge_base_construction.py

"""
高血压防治指南知识库向量化脚本（使用 MilvusClient）
功能：将预处理好的指南文本块（CSV格式）导入 Milvus 向量数据库
依赖：pymilvus, sentence-transformers, pandas, torch
"""

# ===================== 导入依赖库 =====================
# Milvus 客户端，用于向量数据库连接、建表、插入、查询等操作
from pymilvus import MilvusClient, DataType
# 文本嵌入模型，用于将文本转为向量
from sentence_transformers import SentenceTransformer
# 数据处理库，读取CSV文本块
import pandas as pd
# 日志模块，记录程序运行状态
import logging
# 时间模块，用于控制插入间隔、避免请求过快
import time
# PyTorch，用于判断是否使用GPU加速
import torch
# 数值计算库，辅助向量格式转换
import numpy as np
import os  # 路径处理

# ===================== 日志配置 =====================
# 设置日志输出格式：时间 - 日志级别 - 日志信息
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===================== 全局配置常量 =====================
# Milvus 服务连接地址
MILVUS_URI = "http://localhost:19530"
# Milvus 集合（表）名称
COLLECTION_NAME = "hypertension_guideline_kb"
# 本地文本嵌入模型路径（bge-m3）
EMBEDDING_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "bge-m3")
# 每批次插入数据库的数据量
BATCH_SIZE = 10

# 自动判断使用GPU（cuda）还是CPU运行模型
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"使用设备: {DEVICE}")


def init_milvus_collection():
    """
    初始化 Milvus 向量数据库集合
    功能：连接Milvus → 删除旧表 → 创建新表 → 建立向量索引 → 加载集合
    返回：MilvusClient 客户端实例
    """
    logging.info("开始初始化指南知识库 Milvus 集合...")
    try:
        # 1. 创建 Milvus 客户端连接
        client = MilvusClient(uri=MILVUS_URI)
        logging.info(f"已连接到 Milvus: {MILVUS_URI}")

        # 2. 如果已存在同名集合，先删除（保证每次导入都是全新数据）
        if client.has_collection(COLLECTION_NAME):
            client.drop_collection(COLLECTION_NAME)
            logging.info(f"已删除旧 Collection: {COLLECTION_NAME}")

        # 3. 定义表结构（Schema）：关闭自动ID、关闭动态字段
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)

        # 文本块唯一ID（主键）
        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, is_primary=True, max_length=50)
        # 文本向量（bge-m3 输出维度 1024）
        schema.add_field(field_name="chunk_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        # 文本内容
        schema.add_field(field_name="chunk_text", datatype=DataType.VARCHAR, max_length=2000)
        # 页码
        schema.add_field(field_name="page_num", datatype=DataType.INT64)
        # 章节标题
        schema.add_field(field_name="section_title", datatype=DataType.VARCHAR, max_length=200)

        # 4. 配置向量索引
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="chunk_vector",  # 对向量字段建立索引
            index_type="IVF_FLAT",  # 索引类型（适合中小规模数据）
            metric_type="L2",  # 相似度计算方式：欧氏距离
            params={"nlist": 128}  # 索引聚类数量
        )

        # 5. 创建集合并绑定结构与索引
        client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params
        )
        logging.info(f"Collection '{COLLECTION_NAME}' 已创建，索引已配置")

        # 6. 加载集合到内存（必须加载才能插入/查询）
        client.load_collection(COLLECTION_NAME)
        logging.info("指南知识库 Collection 初始化完成")
        return client

    except Exception as e:
        logging.error(f"初始化失败：{str(e)}")
        # 异常抛出，终止程序
        raise


def embed_guideline_chunks(csv_path, client):
    """
    读取CSV文本块 → 生成向量 → 批量插入 Milvus
    :param csv_path: 预处理好的文本块CSV路径
    :param client: Milvus 客户端
    """
    logging.info(f"开始向量化指南文本块，CSV: {csv_path}")
    try:
        # 1. 读取CSV文件
        df = pd.read_csv(csv_path)
        logging.info(f"加载文本块数据，共 {len(df)} 条记录")

        # 2. 数据清洗：去除空值、截断超长文本（避免数据库报错）
        # 必须字段去空
        df = df.dropna(subset=['chunk_id', 'chunk_text'])
        # 文本内容截断到 1990 字符以内
        df['chunk_text'] = df['chunk_text'].astype(str).str.slice(0, 1990)
        # 章节标题填充空值并截断
        df['section_title'] = df.get('section_title', '').fillna('').astype(str).str.slice(0, 190)
        # 页码转为整数，缺失值补0
        df['page_num'] = df['page_num'].fillna(0).astype(int)

        # 3. 加载文本嵌入模型
        embedder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
        # 设置模型最大输入长度
        embedder.max_seq_length = 512
        logging.info("嵌入模型加载完成")

        # 4. 分批向量化 + 插入数据库
        total = len(df)
        for i in range(0, total, BATCH_SIZE):
            # 取当前批次数据
            batch = df.iloc[i:i + BATCH_SIZE]
            # 提取文本列表
            texts = batch['chunk_text'].tolist()
            # 生成向量：不归一化（L2距离不需要归一化）
            vectors = embedder.encode(texts, normalize_embeddings=False)
            # 转为列表格式（Milvus 要求）
            vectors_list = [v.tolist() for v in vectors]

            # 组装插入数据
            insert_data = []
            for idx, row in batch.iterrows():
                insert_data.append({
                    "chunk_id": row['chunk_id'],
                    "chunk_vector": vectors_list[idx - i],
                    "chunk_text": row['chunk_text'],
                    "page_num": int(row['page_num']),
                    "section_title": row['section_title'],
                })

            # 批量插入 Milvus
            client.insert(COLLECTION_NAME, insert_data)
            logging.info(f"批次 {i // BATCH_SIZE + 1} 插入完成")

            # 每100批次刷盘一次，保证数据持久化
            if (i + BATCH_SIZE) % 100 == 0 and i > 0:
                client.flush(COLLECTION_NAME)
                logging.info("定期刷盘完成")

            # 轻微延迟，减轻服务压力
            time.sleep(0.3)

        # 全部插入完成后刷盘
        client.flush(COLLECTION_NAME)
        # 输出最终数据量
        stats = client.get_collection_stats(COLLECTION_NAME)
        logging.info(f"知识库构建完成，共 {stats['row_count']} 条记录")

    except Exception as e:
        logging.error(f"文本块向量化失败：{str(e)}")
        raise


if __name__ == "__main__":
    # 预处理好的文本块CSV路径
    CHUNKS_CSV = "dataset/guideline_chunks_advanced.csv"
    logging.info("程序开始执行...")

    # 初始化数据库
    milvus_client = init_milvus_collection()
    # 向量化并入库
    embed_guideline_chunks(CHUNKS_CSV, milvus_client)
    # 关闭连接
    milvus_client.close()

    logging.info("程序执行完毕")