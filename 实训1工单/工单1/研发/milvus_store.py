# -*- coding: utf-8 -*-
"""
Milvus向量存储 — pymilvus未安装时自动降级
"""
import logging  # 导入日志模块

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

# 尝试导入pymilvus，失败则使用桩类
try:
    from pymilvus import (  # 尝试从pymilvus导入所需组件
        connections,  # Milvus连接管理
        FieldSchema, CollectionSchema, DataType,  # 字段、集合模式与数据类型定义
        Collection, utility  # 集合对象与工具函数
    )
    HAS_MILVUS = True  # 标记pymilvus库可用
except ImportError:
    HAS_MILVUS = False  # 标记pymilvus库不可用
    logger.warning("pymilvus 未安装，MilvusStore 将被禁用。请执行: pip install pymilvus")  # 提示用户安装缺失依赖


class MilvusStore:
    """Milvus向量存储 — 无pymilvus时自动降级"""
    _instance = None  # 单例模式的类级实例缓存

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:  # 如果尚未创建实例
            cls._instance = super().__new__(cls)  # 调用父类的__new__创建实例
        return cls._instance  # 返回缓存的单例实例

    def __init__(self, host="localhost", port="19530", collection_name="rag_document", dim=128):
        if hasattr(self, 'initialized'):  # 防止重复初始化
            return
        self.host = host  # 保存Milvus主机地址
        self.port = port  # 保存Milvus服务端口
        self.collection_name = collection_name  # 保存集合名称
        self.dim = dim  # 保存向量维度
        self.client = None  # 初始化客户端引用为None
        self.collection = None  # 初始化集合引用为None
        self._available = HAS_MILVUS  # 根据pymilvus导入状态设置可用标记
        if self._available:
            self.connect()  # 可用时尝试连接Milvus
        else:
            logger.info("[MilvusStore] pymilvus未安装，已降级为非可用状态（TF-IDF将作为默认检索后端）")  # 记录降级信息
        self.initialized = True  # 标记初始化已完成

    def is_available(self):
        return self._available  # 返回Milvus是否可用

    def connect(self):
        if not self._available:  # 不可用时直接返回
            return
        try:
            connections.connect(  # 建立Milvus连接
                alias="default",  # 连接别名设为default
                host=self.host,  # 指定Milvus主机
                port=self.port  # 指定Milvus端口
            )
            logger.info(f"[Milvus] 连接成功: {self.host}:{self.port}")  # 记录连接成功日志
        except Exception as e:
            logger.warning(f"[Milvus] 连接失败，降级为TF-IDF检索: {e}")  # 记录连接失败并降级
            self._available = False  # 将可用标记设为False

    def is_collection_exists(self):
        if not self._available:  # 不可用时返回False
            return False
        return utility.has_collection(self.collection_name)  # 检查指定集合是否已存在

    def create_collection(self, force_recreate=False):
        if not self._available:  # 不可用时跳过创建
            logger.warning("[MilvusStore] Milvus不可用，跳过集合创建")  # 警告跳过
            return
        if self.is_collection_exists() and not force_recreate:  # 集合已存在且不强制重建
            self.collection = Collection(self.collection_name)  # 直接加载已有集合
            logger.info(f"[Milvus] 集合已存在: {self.collection_name}")  # 记录集合已存在
            return
        if utility.has_collection(self.collection_name):  # 集合存在但需重建
            Collection(self.collection_name).drop()  # 删除旧集合
        fields = [  # 定义集合的字段列表
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # 自增主键ID
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),  # 浮点向量嵌入字段
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),  # 文本内容字段
            FieldSchema(name="page_num", dtype=DataType.INT64),  # 页码字段
            FieldSchema(name="chunk_idx", dtype=DataType.INT64)  # 文本块索引字段
        ]
        schema = CollectionSchema(fields, description="RAG文档向量库")  # 创建集合模式
        self.collection = Collection(self.collection_name, schema)  # 用模式创建新集合
        index_params = {"metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 128}}  # 配置索引参数（内积+IVF_FLAT）
        self.collection.create_index("embedding", index_params)  # 为embedding字段创建索引
        logger.info(f"[Milvus] 集合创建成功")  # 记录集合创建成功

    def insert_chunks(self, chunks, embeddings):
        if not self._available:  # 不可用时直接返回
            return
        if not self.collection:  # 如果集合尚未加载
            self.create_collection()  # 自动创建集合
        total = len(chunks)  # 获取待插入总块数
        batch_size = 100  # 每批插入100条
        for i in range(0, total, batch_size):  # 按批次循环插入
            batch = chunks[i:i+batch_size]  # 截取当前批次的文本块
            batch_emb = embeddings[i:i+batch_size]  # 截取当前批次的向量
            entities = [  # 组装Milvus插入实体
                batch_emb,  # 向量数据
                [c["content"] for c in batch],  # 提取文本内容列表
                [c["page_num"] for c in batch],  # 提取页码列表
                [c["chunk_idx"] for c in batch]  # 提取块索引列表
            ]
            self.collection.insert(entities)  # 将实体插入Milvus集合
            logger.info(f"[Milvus] 插入进度: {i}-{min(i+batch_size, total)} / {total}")  # 记录插入进度
        self.collection.flush()  # 将数据刷入磁盘持久化
        self.collection.load()  # 将集合加载到内存以支持搜索

    def search(self, query_emb, top_k=5):
        if not self._available:  # 不可用时返回空列表
            return []
        if not self.collection:  # 如果集合未加载
            self.collection = Collection(self.collection_name)  # 加载已有集合
        self.collection.load()  # 确保集合已加载到内存
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}  # 搜索参数（内积度量，探针数10）
        results = self.collection.search(  # 执行向量相似度搜索
            data=[query_emb],  # 查询向量
            anns_field="embedding",  # 指定搜索的向量字段
            param=search_params,  # 搜索参数
            limit=top_k,  # 返回前top_k条结果
            output_fields=["content", "page_num", "chunk_idx"]  # 返回的额外字段
        )
        return results[0]  # 返回第一个查询的结果列表
