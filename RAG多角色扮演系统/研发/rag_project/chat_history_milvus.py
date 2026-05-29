# -*- coding: utf-8 -*-
# chat_history_milvus.py
"""
聊天记录存储管理器
功能：使用 Milvus 数据库持久化存储用户对话历史（问答记录）
说明：Milvus 强制要求至少一个向量字段，因此使用 dummy_vector 作为占位向量，无实际业务意义
"""

import logging
import time
import uuid  # 生成唯一记录ID
from pymilvus import MilvusClient, DataType

# 日志对象
logger = logging.getLogger(__name__)


class ChatHistoryMilvus:
    def __init__(self, client: MilvusClient, collection_name: str = "chat_records"):
        """
        初始化聊天记录管理器
        :param client: 已连接的 Milvus 客户端实例
        :param collection_name: 聊天记录表名，默认 chat_records
        """
        # 注入 Milvus 客户端
        self.client = client
        # 聊天记录集合名称
        self.collection_name = collection_name
        # 初始化集合（不存在则创建）
        self._init_collection()
        logger.info("ChatHistoryMilvus 初始化完成")

    def _init_collection(self):
        """
        初始化 Milvus 聊天记录集合
        检查集合是否存在 → 不存在则创建表结构 + 索引
        """
        start_time = time.time()
        # 如果集合已存在，直接返回
        if self.client.has_collection(self.collection_name):
            logger.info(f"聊天记录集合 '{self.collection_name}' 已存在")
            elapsed = time.time() - start_time
            logger.info(f"_init_collection 耗时: {elapsed:.4f} 秒")
            return

        # 创建表结构：关闭自动ID、关闭动态扩展字段
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)

        # ===================== 定义表字段 =====================
        # 聊天记录唯一ID（主键）
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
        # 用户ID，用于区分不同用户
        schema.add_field(field_name="user_id", datatype=DataType.VARCHAR, max_length=50)
        # 角色：user / assistant
        schema.add_field(field_name="role", datatype=DataType.VARCHAR, max_length=50)
        # 用户问题
        schema.add_field(field_name="question", datatype=DataType.VARCHAR, max_length=5000)
        # 系统回答
        schema.add_field(field_name="answer", datatype=DataType.VARCHAR, max_length=5000)
        # 时间戳字符串（如 2025-05-20 10:30:00）
        schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)
        # Milvus 强制要求必须有向量字段，使用2维占位向量（无实际计算意义）
        schema.add_field(field_name="dummy_vector", datatype=DataType.FLOAT_VECTOR, dim=2)

        # ===================== 配置索引 =====================
        index_params = self.client.prepare_index_params()
        # 给占位向量创建简易索引（满足 Milvus 强制要求，不影响业务）
        index_params.add_index(field_name="dummy_vector", metric_type="COSINE", index_type="FLAT")

        # ===================== 创建集合 =====================
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )
        elapsed = time.time() - start_time
        logger.info(f"成功创建聊天记录集合: {self.collection_name}, 耗时: {elapsed:.4f} 秒")

    def store_message(self, user_id: str, role: str, question: str, answer: str, timestamp: str):
        """
        存储单条用户聊天记录到 Milvus
        :param user_id: 用户唯一标识
        :param role: 消息角色（user / assistant）
        :param question: 用户问题内容
        :param answer: 系统回答内容
        :param timestamp: 时间戳字符串
        """
        start_time = time.time()
        # 生成全局唯一的记录ID
        record_id = str(uuid.uuid4())

        # 组装插入数据
        data = [{
            "id": record_id,
            "user_id": user_id,
            "role": role,
            "question": question,
            "answer": answer,
            "timestamp": timestamp,
            "dummy_vector": [0.0, 0.0]  # 固定占位向量
        }]

        try:
            # 插入数据（Milvus 会自动刷盘，无需每次单独 flush）
            self.client.insert(collection_name=self.collection_name, data=data)
            elapsed = time.time() - start_time
            logger.info(f"聊天记录保存成功: user_id={user_id}, record_id={record_id}, 耗时: {elapsed:.4f} 秒")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"保存聊天记录失败: {e}, 耗时: {elapsed:.4f} 秒", exc_info=True)

    def close(self):
        """
        关闭连接（预留接口）
        因 client 由外部传入，此处无需实际关闭
        """
        start_time = time.time()
        # 预留关闭逻辑（当前为空）
        elapsed = time.time() - start_time
        logger.debug(f"close 方法耗时: {elapsed:.4f} 秒")
        pass