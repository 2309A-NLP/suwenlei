# -*- coding: utf-8 -*-
"""
Milvus向量存储 — 支持按文档分集合的多文档检索

核心设计：
- 每个PDF文档对应一个独立的Milvus集合（隔离不同文档的向量空间）
- pymilvus未安装时自动禁用，降级为TF-IDF本地检索
"""
import logging

logger = logging.getLogger(__name__)

try:
    from pymilvus import (
        connections, FieldSchema, CollectionSchema, DataType,
        Collection, utility
    )
    HAS_MILVUS = True
except ImportError:
    HAS_MILVUS = False
    logger.warning("pymilvus 未安装，MilvusStore 将被禁用。请执行: pip install pymilvus")


class MilvusStore:
    """Milvus向量存储（单例模式）：无pymilvus时自动降级"""
    _instance = None

    def __new__(cls, *args, **kwargs):
        # 单例：避免多处import导致重复连接
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, host="localhost", port="19530", dim=1024):
        if hasattr(self, 'initialized'):
            return
        self.host = host
        self.port = port
        self.dim = dim  # 向量维度，与bge-m3模型输出一致(1024)
        self.client = None
        self.collections = {}  # doc_name → Collection 映射缓存
        self._available = HAS_MILVUS
        if self._available:
            self.connect()
        else:
            logger.info("[MilvusStore] pymilvus未安装，已降级")
        self.initialized = True

    def is_available(self):
        return self._available

    def connect(self):
        if not self._available:
            return
        try:
            connections.connect(alias="default", host=self.host, port=self.port)
            logger.info(f"[Milvus] 连接成功: {self.host}:{self.port}")
        except Exception as e:
            logger.warning(f"[Milvus] 连接失败，降级TF-IDF: {e}")
            self._available = False

    def _get_collection_name(self, doc_name: str) -> str:
        """文档名→Milvus集合名映射：已知文档用短名，未知文档用哈希"""
        name_map = {
            '招股说明书1': 'zgsm1',
            '招股说明书2': 'zgsm2',
            '招股说明书3': 'zgsm3',
        }
        safe_name = name_map.get(doc_name, '')
        if not safe_name:
            safe_name = f"doc_{abs(hash(doc_name)) % (10**8):08d}"
        return f"rag_doc_{safe_name}"

    def get_collection(self, doc_name: str):
        """获取已有集合：优先从缓存读取，避免重复加载"""
        if not self._available:
            return None
        if doc_name in self.collections:
            return self.collections[doc_name]
        col_name = self._get_collection_name(doc_name)
        try:
            if utility.has_collection(col_name):
                col = Collection(col_name)
                self.collections[doc_name] = col
                logger.info(f"[Milvus] 加载集合: {col_name}")
                return col
        except Exception as e:
            logger.warning(f"[Milvus] 加载集合失败: {e}")
        return None

    def create_collection(self, doc_name: str = "default", force_recreate=False):
        """兼容接口：供knowledge_builder调用"""
        return self.create_collection_for_doc(doc_name, force_recreate)

    def create_collection_for_doc(self, doc_name: str, force_recreate=False):
        """创建或重建文档的Milvus集合：force_recreate=True时清空旧数据"""
        if not self._available:
            return None
        col_name = self._get_collection_name(doc_name)
        if utility.has_collection(col_name) and not force_recreate:
            col = Collection(col_name)
            self.collections[doc_name] = col
            return col
        if utility.has_collection(col_name):
            Collection(col_name).drop()
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="page_num", dtype=DataType.INT64),
            FieldSchema(name="chunk_idx", dtype=DataType.INT64),
            FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=20)
        ]
        schema = CollectionSchema(fields, description=f"RAG文档向量库 - {doc_name}")
        col = Collection(col_name, schema)
        # IVF_FLAT索引：内积(IP)度量，nlist=128控制聚类数，平衡精度和速度
        index_params = {"metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
        col.create_index("embedding", index_params)
        self.collections[doc_name] = col
        logger.info(f"[Milvus] 集合创建成功: {col_name}")
        return col

    def insert_chunks(self, chunks, embeddings, doc_name: str = "default"):
        """批量插入分块和向量：分批写入避免单次过大"""
        if not self._available:
            return
        col = self.get_collection(doc_name)
        if not col:
            col = self.create_collection_for_doc(doc_name)
        if not col:
            return
        total = len(chunks)
        batch_size = 100  # 控制单次写入量，避免Milvus内存压力
        for i in range(0, total, batch_size):
            batch = chunks[i:i+batch_size]
            batch_emb = embeddings[i:i+batch_size]
            entities = [
                batch_emb,
                [c["content"] for c in batch],
                [c["page_num"] for c in batch],
                [c["chunk_idx"] for c in batch],
                [c.get("source_type", "text") for c in batch]
            ]
            col.insert(entities)
        col.flush()  # 刷写磁盘确保持久化
        col.load()   # 加载到内存供检索使用
        logger.info(f"[Milvus] 插入完成: {doc_name}, {total}条")

    def search(self, query_emb, top_k=5, doc_name: str = None):
        """向量检索：指定doc_name则单集合搜索，否则遍历所有集合合并结果"""
        if not self._available:
            return []
        if doc_name:
            col = self.get_collection(doc_name)
            if not col:
                return []
            col.load()
            # nprobe=10：搜索10个聚类中心，精度和速度的平衡点
            search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
            results = col.search(
                data=[query_emb], anns_field="embedding",
                param=search_params, limit=top_k,
                output_fields=["content", "page_num", "chunk_idx", "source_type"]
            )
            return results[0]

        # 多集合搜索：遍历所有已知集合，合并后按分数排序
        all_hits = []
        for dname, col in self.collections.items():
            try:
                col.load()
                search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
                results = col.search(
                    data=[query_emb], anns_field="embedding",
                    param=search_params, limit=top_k,
                    output_fields=["content", "page_num", "chunk_idx", "source_type"]
                )
                if results and results[0]:
                    all_hits.extend(results[0])
            except Exception as e:
                logger.warning(f"[Milvus] 搜索 {dname} 失败: {e}")
        all_hits.sort(key=lambda h: h.score, reverse=True)
        return all_hits[:top_k]

    def drop_collection(self, doc_name: str):
        """删除指定文档的Milvus集合（文件删除时调用）"""
        if not self._available:
            return
        col_name = self._get_collection_name(doc_name)
        try:
            if utility.has_collection(col_name):
                Collection(col_name).drop()
                self.collections.pop(doc_name, None)
                logger.info(f"[Milvus] 已删除集合: {col_name}")
        except Exception as e:
            logger.warning(f"[Milvus] 删除集合失败: {e}")

    def get_collection_info(self, doc_name: str = None):
        """获取集合元信息：用于健康检查和调试"""
        if doc_name:
            col = self.get_collection(doc_name)
            if col:
                return {
                    'doc_name': doc_name,
                    'collection_name': self._get_collection_name(doc_name),
                    'num_entities': col.num_entities,
                }
            return None
        infos = []
        for dname in self.collections:
            info = self.get_collection_info(dname)
            if info:
                infos.append(info)
        return infos
