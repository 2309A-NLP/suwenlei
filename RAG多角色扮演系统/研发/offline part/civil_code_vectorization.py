# -*- coding: utf-8 -*-
# civil_code_vectorization.py

"""
民法典知识库向量化脚本（使用 MilvusClient）
功能：将预处理好的民法典文本块（CSV格式）导入 Milvus 向量数据库
依赖：pymilvus, sentence-transformers, pandas, torch
"""

# ===================== 1. 导入依赖库 =====================
from pymilvus import MilvusClient, DataType  # Milvus客户端与数据类型
from sentence_transformers import SentenceTransformer  # 文本嵌入模型
import pandas as pd  # 数据读取与清洗
import logging  # 日志输出
import time  # 批次间隔休眠
import torch  # 判断GPU/CPU设备
import os   # 路径处理

# ===================== 2. 日志配置 =====================
# 配置日志格式与输出级别，方便查看运行状态
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===================== 3. 全局常量配置 =====================
# Milvus服务连接地址
MILVUS_URI = "http://localhost:19530"
# 向量数据库集合名称
COLLECTION_NAME = "civil_code_kb"
# 本地文本嵌入模型路径（bge-m3）
EMBEDDING_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "bge-m3")
# 每批次插入向量数据的行数（防止一次性插入过大）
BATCH_SIZE = 20

# 自动判断使用GPU（cuda）还是CPU运行模型
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"使用设备: {DEVICE}")


# ===================== 4. 初始化Milvus集合 =====================
def init_milvus_collection_civil():
    """
    初始化民法典向量库：
    1. 连接Milvus
    2. 如果旧集合已存在则删除
    3. 创建新集合与字段结构
    4. 创建向量索引
    5. 加载集合到内存
    """
    logging.info("开始初始化 Milvus 集合（民法典）...")
    try:
        # 连接Milvus服务
        client = MilvusClient(uri=MILVUS_URI)
        logging.info(f"已连接到 Milvus: {MILVUS_URI}")

        # 如果已存在同名集合，先删除（保证每次都是全新库）
        if client.has_collection(COLLECTION_NAME):
            client.drop_collection(COLLECTION_NAME)
            logging.info(f"已删除旧 Collection: {COLLECTION_NAME}")

        # 创建数据结构schema：关闭自动ID、关闭动态字段
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)

        # 文本块唯一ID（主键）
        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
        # 文本向量（bge-m3输出1024维）
        schema.add_field(field_name="chunk_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        # 民法典文本内容
        schema.add_field(field_name="chunk_text", datatype=DataType.VARCHAR, max_length=65535)
        # 法条编号
        schema.add_field(field_name="article_num", datatype=DataType.VARCHAR, max_length=100)
        # 民法典·编
        schema.add_field(field_name="book", datatype=DataType.VARCHAR, max_length=500)
        # 民法典·章
        schema.add_field(field_name="chapter", datatype=DataType.VARCHAR, max_length=500)
        # 民法典·节
        schema.add_field(field_name="section", datatype=DataType.VARCHAR, max_length=500)

        # 配置向量索引参数
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="chunk_vector",
            index_type="IVF_FLAT",  # 向量索引类型
            metric_type="L2",  # 相似度计算方式：欧氏距离
            params={"nlist": 128}  # 索引聚类数量
        )

        # 创建集合与索引
        client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params
        )
        logging.info(f"Collection '{COLLECTION_NAME}' 已创建，索引已配置")

        # 将集合加载到内存（必须执行，否则无法查询）
        client.load_collection(COLLECTION_NAME)
        logging.info(f"民法典知识库 Collection {COLLECTION_NAME} 初始化完成")
        return client

    except Exception as e:
        logging.error(f"初始化失败：{str(e)}")
        raise  # 抛出异常终止程序


# ===================== 5. 读取CSV并向量化入库 =====================
def embed_civil_code_chunks(csv_path, client):
    """
    读取预处理好的民法典文本CSV，生成向量并批量插入Milvus
    :param csv_path: 文本块CSV路径
    :param client: Milvus客户端实例
    """
    logging.info(f"开始向量化民法典文本块，CSV: {csv_path}")
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        logging.info(f"读取 CSV 成功，原始行数: {len(df)}")

        # 剔除关键字段为空的行
        df = df.dropna(subset=['chunk_text', 'chunk_id'])
        logging.info(f"去除缺失值后剩余: {len(df)} 行")

        # 统一转为字符串并去除首尾空格
        str_cols = ['chunk_id', 'chunk_text', 'article_num', 'book', 'chapter', 'section']
        for col in str_cols:
            df[col] = df[col].astype(str).str.strip()

        # 字段长度截断（防止超过Milvus字段最大长度）
        df['chunk_text'] = df['chunk_text'].str.slice(0, 65530)
        df['article_num'] = df['article_num'].str.slice(0, 90)
        df['book'] = df['book'].str.slice(0, 450)
        df['chapter'] = df['chapter'].str.slice(0, 450)
        df['section'] = df['section'].str.slice(0, 450)
        logging.info("数据清洗与截断完成")

        # 加载本地bge-m3嵌入模型
        embedder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
        embedder.max_seq_length = 512  # 模型最大输入token长度
        logging.info("嵌入模型加载完成")

        # 按批次生成向量并插入Milvus
        total = len(df)
        for i in range(0, total, BATCH_SIZE):
            # 取出当前批次数据
            batch = df.iloc[i:i + BATCH_SIZE]
            # 提取文本列表
            texts = batch['chunk_text'].tolist()
            # 生成向量：不归一化（适配L2欧氏距离）
            vectors = embedder.encode(texts, normalize_embeddings=False).tolist()

            # 组装插入数据格式
            insert_data = []
            for idx, row in batch.iterrows():
                insert_data.append({
                    "chunk_id": row['chunk_id'],
                    "chunk_vector": vectors[idx - i],
                    "chunk_text": row['chunk_text'],
                    "article_num": row['article_num'],
                    "book": row['book'],
                    "chapter": row['chapter'],
                    "section": row['section'],
                })

            # 批量插入数据库
            client.insert(COLLECTION_NAME, insert_data)
            logging.info(f"批次 {i // BATCH_SIZE + 1}/{(total + BATCH_SIZE - 1) // BATCH_SIZE} 完成")
            time.sleep(0.2)  # 轻微间隔，减轻服务压力

        # 刷新数据写入磁盘
        client.flush(COLLECTION_NAME)
        # 输出最终数据量统计
        stats = client.get_collection_stats(COLLECTION_NAME)
        logging.info(f"民法典知识库构建完成，共 {stats['row_count']} 条向量记录")

    except Exception as e:
        logging.error(f"向量化失败：{str(e)}")
        raise


# ===================== 6. 主程序入口 =====================
if __name__ == "__main__":
    # 文本块CSV路径
    CHUNKS_CSV = "dataset/civil_code_chunks.csv"

    # 初始化Milvus集合
    milvus_client = init_milvus_collection_civil()

    # 读取CSV → 生成向量 → 入库
    embed_civil_code_chunks(CHUNKS_CSV, milvus_client)

    # 关闭客户端连接
    milvus_client.close()

    print("\n✅ 民法典知识库构建完成！")