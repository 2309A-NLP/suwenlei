# -*- coding: utf-8 -*-
"""
Milvus向量存储 — 支持按文档分集合的多文档检索
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化

核心设计：
- 每个PDF文档对应一个独立的Milvus集合（如 rag_doc_招股说明书1）
- 支持单文档搜索和跨文档搜索
- pymilvus未安装时自动降级
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
    """Milvus向量存储 — 支持按文档分集合，无pymilvus时自动降级"""
    _instance = None  # 单例模式的类级实例缓存

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:  # 如果尚未创建实例
            cls._instance = super().__new__(cls)  # 调用父类的__new__创建实例
        return cls._instance  # 返回缓存的单例实例

    def __init__(self, host="localhost", port="19530", dim=4096):
        if hasattr(self, 'initialized'):  # 防止重复初始化
            return
        self.host = host  # 保存Milvus主机地址
        self.port = port  # 保存Milvus服务端口
        self.dim = dim  # 保存向量维度
        self.client = None  # 初始化客户端引用为None
        self.collections = {}  # 按文档名缓存的集合字典 {doc_name: Collection}
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

    # ==================== 按文档管理集合 ====================

    def _get_collection_name(self, doc_name: str) -> str:
        """根据文档名生成Milvus集合名（Milvus只允许字母、数字、下划线）"""
        # 将中文名映射为英文标识，确保集合名不含中文
        name_map = {
            '招股说明书1': 'zgsm1',  # 招股说明书1 → zgsm1
            '招股说明书2': 'zgsm2',  # 招股说明书2 → zgsm2
            '招股说明书3': 'zgsm3',  # 招股说明书3 → zgsm3
        }
        safe_name = name_map.get(doc_name, '')  # 先查映射表
        if not safe_name:  # 映射表中没有时
            # 用哈希值生成唯一标识（取后8位十六进制）
            safe_name = f"doc_{abs(hash(doc_name)) % (10**8):08d}"  # 如 doc_38472910
        return f"rag_doc_{safe_name}"  # 返回集合名

    def get_collection(self, doc_name: str):
        """获取指定文档的Milvus集合（不存在则创建）"""
        if not self._available:  # 不可用时返回None
            return None
        if doc_name in self.collections:  # 已缓存时直接返回
            return self.collections[doc_name]
        # 尝试加载已有集合
        col_name = self._get_collection_name(doc_name)  # 生成集合名
        try:
            if utility.has_collection(col_name):  # 集合已存在时
                col = Collection(col_name)  # 加载集合
                self.collections[doc_name] = col  # 缓存集合
                logger.info(f"[Milvus] 加载集合: {col_name}")  # 记录日志
                return col  # 返回集合
        except Exception as e:  # 加载失败时
            logger.warning(f"[Milvus] 加载集合失败: {e}")  # 记录警告
        return None  # 集合不存在时返回None

    def create_collection(self, doc_name: str = "default", force_recreate=False):
        """兼容方法：create_collection_for_doc的别名，供knowledge_builder调用"""
        return self.create_collection_for_doc(doc_name, force_recreate)  # 委托给按文档创建方法

    def create_collection_for_doc(self, doc_name: str, force_recreate=False):
        """为指定文档创建Milvus集合"""
        if not self._available:  # 不可用时跳过创建
            logger.warning("[MilvusStore] Milvus不可用，跳过集合创建")  # 警告跳过
            return None  # 返回None
        col_name = self._get_collection_name(doc_name)  # 生成集合名
        # 如果已存在且不强制重建，直接返回已有集合
        if utility.has_collection(col_name) and not force_recreate:  # 集合已存在且不强制重建
            col = Collection(col_name)  # 加载已有集合
            self.collections[doc_name] = col  # 缓存集合
            logger.info(f"[Milvus] 集合已存在: {col_name}")  # 记录集合已存在
            return col  # 返回集合
        # 删除旧集合（如果存在）
        if utility.has_collection(col_name):  # 集合存在但需重建
            Collection(col_name).drop()  # 删除旧集合
        # 定义集合字段
        fields = [  # 定义集合的字段列表
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # 自增主键ID
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),  # 浮点向量嵌入字段
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),  # 文本内容字段
            FieldSchema(name="page_num", dtype=DataType.INT64),  # 页码字段
            FieldSchema(name="chunk_idx", dtype=DataType.INT64),  # 文本块索引字段
            FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=20)  # 来源类型字段（新增）
        ]
        schema = CollectionSchema(fields, description=f"RAG文档向量库 - {doc_name}")  # 创建集合模式
        col = Collection(col_name, schema)  # 用模式创建新集合
        index_params = {"metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 128}}  # 配置索引参数
        col.create_index("embedding", index_params)  # 为embedding字段创建索引
        self.collections[doc_name] = col  # 缓存集合
        logger.info(f"[Milvus] 集合创建成功: {col_name}")  # 记录集合创建成功
        return col  # 返回新集合

    def insert_chunks(self, chunks, embeddings, doc_name: str = "default"):
        """将分块和向量插入指定文档的集合"""
        if not self._available:  # 不可用时直接返回
            return
        col = self.get_collection(doc_name)  # 获取文档集合
        if not col:  # 集合不存在时
            col = self.create_collection_for_doc(doc_name)  # 自动创建集合
        if not col:  # 创建仍失败时
            return
        total = len(chunks)  # 获取待插入总块数
        batch_size = 100  # 每批插入100条
        for i in range(0, total, batch_size):  # 按批次循环插入
            batch = chunks[i:i+batch_size]  # 截取当前批次的文本块
            batch_emb = embeddings[i:i+batch_size]  # 截取当前批次的向量
            entities = [  # 组装Milvus插入实体
                batch_emb,  # 向量数据
                [c["content"] for c in batch],  # 提取文本内容列表
                [c["page_num"] for c in batch],  # 提取页码列表
                [c["chunk_idx"] for c in batch],  # 提取块索引列表
                [c.get("source_type", "text") for c in batch]  # 提取来源类型列表（新增）
            ]
            col.insert(entities)  # 将实体插入Milvus集合
            logger.info(f"[Milvus] 插入进度: {i}-{min(i+batch_size, total)} / {total}")  # 记录插入进度
        col.flush()  # 将数据刷入磁盘持久化
        col.load()  # 将集合加载到内存以支持搜索
        logger.info(f"[Milvus] 插入完成: {doc_name}, {total}条")  # 记录完成日志

    def search(self, query_emb, top_k=5, doc_name: str = None):
        """在指定文档的集合中搜索（doc_name=None时搜索所有集合）"""
        if not self._available:  # 不可用时返回空列表
            return []
        # 指定文档搜索
        if doc_name:  # 指定了文档名
            col = self.get_collection(doc_name)  # 获取文档集合
            if not col:  # 集合不存在时
                return []  # 返回空列表
            col.load()  # 确保集合已加载到内存
            search_params = {"metric_type": "IP", "params": {"nprobe": 10}}  # 搜索参数
            results = col.search(  # 执行向量相似度搜索
                data=[query_emb],  # 查询向量
                anns_field="embedding",  # 指定搜索的向量字段
                param=search_params,  # 搜索参数
                limit=top_k,  # 返回前top_k条结果
                output_fields=["content", "page_num", "chunk_idx", "source_type"]  # 返回的额外字段
            )
            return results[0]  # 返回第一个查询的结果列表
        # 跨文档搜索：搜索所有集合并合并结果
        all_hits = []  # 所有命中的结果列表
        for dname, col in self.collections.items():  # 遍历所有已缓存的集合
            try:
                col.load()  # 确保集合已加载到内存
                search_params = {"metric_type": "IP", "params": {"nprobe": 10}}  # 搜索参数
                results = col.search(  # 执行向量相似度搜索
                    data=[query_emb],  # 查询向量
                    anns_field="embedding",  # 指定搜索的向量字段
                    param=search_params,  # 搜索参数
                    limit=top_k,  # 每个集合取top_k条结果
                    output_fields=["content", "page_num", "chunk_idx", "source_type"]  # 返回的额外字段
                )
                if results and results[0]:  # 有结果时
                    all_hits.extend(results[0])  # 添加到总结果列表
            except Exception as e:  # 搜索失败时
                logger.warning(f"[Milvus] 搜索集合 {dname} 失败: {e}")  # 记录警告
        # 按分数排序并取top_k
        all_hits.sort(key=lambda h: h.score, reverse=True)  # 按分数降序排列
        return all_hits[:top_k]  # 返回前top_k条结果

    def drop_collection(self, doc_name: str):
        """删除指定文档的集合"""
        if not self._available:  # 不可用时直接返回
            return
        col_name = self._get_collection_name(doc_name)  # 生成集合名
        try:
            if utility.has_collection(col_name):  # 集合存在时
                Collection(col_name).drop()  # 删除集合
                if doc_name in self.collections:  # 从缓存中移除
                    del self.collections[doc_name]
                logger.info(f"[Milvus] 已删除集合: {col_name}")  # 记录日志
        except Exception as e:  # 删除失败时
            logger.warning(f"[Milvus] 删除集合失败: {e}")  # 记录警告

    def get_collection_info(self, doc_name: str = None):
        """获取集合信息（单文档或全部）"""
        if doc_name:  # 指定文档
            col = self.get_collection(doc_name)  # 获取集合
            if col:  # 集合存在时
                return {
                    'doc_name': doc_name,  # 文档名
                    'collection_name': self._get_collection_name(doc_name),  # 集合名
                    'num_entities': col.num_entities,  # 实体数量
                }
            return None  # 集合不存在时返回None
        # 获取所有集合信息
        infos = []  # 集合信息列表
        for dname in self.collections:  # 遍历所有已缓存的集合
            info = self.get_collection_info(dname)  # 递归获取信息
            if info:  # 信息有效时
                infos.append(info)  # 添加到列表
        return infos  # 返回所有集合信息
