# -*- coding: utf-8 -*-
"""
合并检索引擎 — BM25全文检索 + Milvus向量检索
工单编号: 人工智能NLP-RAG-混合检索任务 / RAG工单6

BM25Index: jieba分词 + 倒排索引 + Okapi BM25评分
MilvusStore: 单例模式, 多集合支持, 1024维向量(bge-m3), pymilvus未安装时自动降级
"""
import re           # 正则表达式：jieba不可用时的降级分词
import math         # 数学运算：log用于IDF计算
import logging      # 日志模块
from typing import List, Dict  # 类型提示

logger = logging.getLogger(__name__)  # 模块级日志器

# ==================== BM25部分 ====================

# 中文停用词表：过滤虚词，提升BM25对实义词的权重
STOP_WORDS = set([
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
    # --- 英文停用词 ---
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'do', 'does', 'did', 'have', 'has', 'had', 'having',
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as',
    'and', 'or', 'but', 'not', 'no', 'nor', 'so', 'yet',
    'this', 'that', 'these', 'those', 'it', 'its',
    'what', 'which', 'who', 'whom', 'whose', 'when', 'where', 'how',
    'if', 'then', 'else', 'than', 'too', 'very', 'can', 'could',
    'will', 'would', 'shall', 'should', 'may', 'might', 'must',
    'about', 'above', 'after', 'before', 'between', 'into', 'through',
    'during', 'without', 'again', 'further', 'once', 'here', 'there',
    'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
    'some', 'such', 'only', 'own', 'same', 'also', 'just', 'over',
])


def _tokenize(text: str) -> List[str]:
    """中英混合分词：jieba中文分词 + 英文按空格/标点切分"""
    # 先提取英文单词（正则匹配连续英文字母）
    en_words = re.findall(r'[a-zA-Z]{2,}', text)
    en_words = [w.lower() for w in en_words]  # 英文统一小写
    try:
        import jieba                        # 尝试导入jieba分词库
        words = list(jieba.cut(text))       # jieba精确模式分词（主要处理中文）
    except ImportError:
        # jieba不可用时降级为正则切分（中文词或英文数字串）
        words = re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z0-9]+', text)
    # 合并中文分词结果和英文单词，过滤停用词和单字
    all_tokens = [w for w in words if len(w) >= 1 and w not in STOP_WORDS]
    all_tokens += [w for w in en_words if w not in STOP_WORDS]
    return all_tokens


class BM25Index:
    """BM25检索索引：维护倒排索引和文档统计量

    BM25公式: score(q,d) = Σ IDF(qi) * tf(qi,d) * (k1+1) / (tf(qi,d) + k1*(1-b+b*dl/avgdl))
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1          # 词频饱和参数：越大对高频词越宽容
        self.b = b            # 文档长度归一化参数：越大对长文档惩罚越大
        self.documents = []       # 原始分块列表
        self.doc_tokens = []      # 每个文档的分词结果列表
        self.doc_lens = []        # 每个文档的分词数量
        self.avg_dl = 0.0         # 所有文档的平均分词长度
        self.df = {}              # 词项→出现在多少个文档中（文档频率）
        self.postings = {}        # 词项→{文档索引: 词频} 倒排索引
        self.n_docs = 0           # 文档总数N

    def build(self, chunks: List[Dict]):
        """从分块列表构建BM25倒排索引"""
        self.documents = chunks              # 存储原始文档
        self.n_docs = len(chunks)            # 记录文档总数
        if self.n_docs == 0:
            return                           # 空文档列表直接返回
        total_len = 0                        # 累计所有文档分词长度
        for idx, chunk in enumerate(chunks):
            text = chunk.get('text', chunk.get('content', ''))  # 取文本内容
            tokens = _tokenize(text)         # 对文本分词
            self.doc_tokens.append(tokens)   # 记录分词结果
            self.doc_lens.append(len(tokens)) # 记录文档长度
            total_len += len(tokens)         # 累加总长度
            # 统计本文档中每个词的词频
            tf_map = {}
            for token in tokens:
                tf_map[token] = tf_map.get(token, 0) + 1
            # 更新倒排索引：记录词项在哪些文档中出现及词频
            for token, count in tf_map.items():
                if token not in self.postings:
                    self.postings[token] = {}   # 首次出现，初始化posting列表
                    self.df[token] = 0          # 初始化文档频率
                self.postings[token][idx] = count  # 记录该文档中的词频
                self.df[token] += 1              # 文档频率+1
        self.avg_dl = total_len / self.n_docs  # 计算平均文档长度
        logger.info(f"[BM25] 索引构建完成: {self.n_docs}个文档, {len(self.postings)}个词汇, 平均长度{self.avg_dl:.1f}")

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """BM25检索：对查询分词后计算每个文档的BM25分数"""
        query_tokens = _tokenize(query)  # 对查询文本分词
        if not query_tokens or self.n_docs == 0:
            return []                    # 无有效词或无文档则返回空
        scores = [0.0] * self.n_docs     # 初始化每个文档的BM25得分为0
        for token in query_tokens:
            if token not in self.postings:
                continue                 # 查询词不在索引中，跳过
            # 计算IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            df_t = self.df[token]        # 该词的文档频率
            idf = math.log((self.n_docs - df_t + 0.5) / (df_t + 0.5) + 1)
            # 遍历包含该词的所有文档，累加BM25得分
            for doc_idx, tf in self.postings[token].items():
                dl = self.doc_lens[doc_idx]  # 目标文档长度
                # BM25核心公式：tf归一化 + 文档长度惩罚
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                scores[doc_idx] += idf * numerator / denominator
        # 按得分降序排列，取前top_k
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for doc_idx, score in ranked:
            if score <= 0:
                continue                 # 得分<=0的结果过滤掉
            chunk = self.documents[doc_idx]
            results.append({
                'text': chunk.get('text', chunk.get('content', '')),  # 文本内容
                'page_num': chunk.get('page_num', 0),                 # 页码
                'score': round(score, 4),                             # 统一分数字段
                'source_type': chunk.get('source_type', 'text'),     # 来源类型
                'doc_name': chunk.get('doc_name', ''),                # 文档名
                'bm25_score': round(score, 4),                        # BM25专用分数
            })
        return results


# ==================== Milvus向量存储部分 ====================

try:
    from pymilvus import (
        connections, FieldSchema, CollectionSchema, DataType,
        Collection, utility
    )                                      # 尝试导入pymilvus客户端
    HAS_MILVUS = True                      # 标记pymilvus可用
except ImportError:
    HAS_MILVUS = False                     # pymilvus未安装
    logger.warning("pymilvus 未安装，MilvusStore 将被禁用。请执行: pip install pymilvus")


class MilvusStore:
    """Milvus向量存储（单例模式）：按文档分集合，1024维向量，无pymilvus时自动降级"""
    _instance = None                       # 单例实例引用

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)  # 首次调用创建实例
        return cls._instance                       # 后续调用返回同一实例

    def __init__(self, host=None, port="19530", dim=1024):
        if hasattr(self, 'initialized'):
            return                         # 单例已初始化则跳过
        import os
        self.host = host or os.environ.get("MILVUS_HOST", "localhost")
        self.port = port                   # Milvus服务端口
        self.dim = dim                     # 向量维度：bge-m3模型输出1024维
        self.client = None                 # 客户端引用（预留）
        self.collections = {}              # doc_name → Collection 缓存映射
        self._available = HAS_MILVUS       # 记录pymilvus是否可用
        if self._available:
            self.connect()                 # 可用则连接Milvus
            if not self._available and host is None:
                self._try_wsl_connect()    # localhost失败则尝试WSL IP
        else:
            logger.info("[MilvusStore] pymilvus未安装，已降级")  # 降级提示
        self.initialized = True            # 标记初始化完成

    def is_available(self):
        """检查Milvus是否可用"""
        return self._available             # 返回可用状态

    def connect(self):
        """连接Milvus服务"""
        if not self._available:
            return                         # 不可用则跳过
        try:
            connections.connect(alias="default", host=self.host, port=self.port)  # 建立连接
            logger.info(f"[Milvus] 连接成功: {self.host}:{self.port}")
        except Exception as e:
            logger.warning(f"[Milvus] 连接失败: {self.host}:{self.port} - {e}")
            self._available = False        # 连接失败则禁用Milvus

    def _try_wsl_connect(self):
        """自动探测WSL2 IP并尝试连接"""
        import subprocess
        try:
            result = subprocess.run(
                ["wsl", "--", "bash", "-c", "hostname -I | awk '{print $1}'"],
                capture_output=True, text=True, timeout=5
            )
            wsl_ip = result.stdout.strip()
            if wsl_ip:
                logger.info(f"[Milvus] 尝试通过WSL IP连接: {wsl_ip}:{self.port}")
                connections.connect(alias="default", host=wsl_ip, port=self.port)
                logger.info(f"[Milvus] WSL连接成功: {wsl_ip}:{self.port}")
                self.host = wsl_ip
                self._available = True
        except Exception as e:
            logger.warning(f"[Milvus] WSL连接失败: {e}")

    def _get_collection_name(self, doc_name: str) -> str:
        """文档名→Milvus集合名映射（特殊文档用短名，其他用哈希）"""
        name_map = {
            '招股说明书1': 'zgsm1',        # 招股说明书1映射
            '招股说明书2': 'zgsm2',        # 招股说明书2映射
            '招股说明书3': 'zgsm3',        # 招股说明书3映射
        }
        safe_name = name_map.get(doc_name, '')  # 查特殊映射表
        if not safe_name:
            # 未知文档名：用哈希值生成8位短标识
            safe_name = f"doc_{abs(hash(doc_name)) % (10**8):08d}"
        return f"rag_doc_{safe_name}"      # 统一前缀格式

    def get_collection(self, doc_name: str):
        """获取已有集合：优先从内存缓存读取"""
        if not self._available:
            return None                    # Milvus不可用返回None
        if doc_name in self.collections:
            return self.collections[doc_name]  # 缓存命中直接返回
        col_name = self._get_collection_name(doc_name)  # 计算集合名
        try:
            if utility.has_collection(col_name):         # 检查集合是否存在
                col = Collection(col_name)               # 加载集合对象
                self.collections[doc_name] = col         # 写入缓存
                logger.info(f"[Milvus] 加载集合: {col_name}")
                return col
        except Exception as e:
            logger.warning(f"[Milvus] 加载集合失败: {e}")
        return None                        # 集合不存在或加载失败返回None

    def create_collection(self, doc_name: str = "default", force_recreate=False):
        """兼容接口：供knowledge_builder调用，转发到create_collection_for_doc"""
        return self.create_collection_for_doc(doc_name, force_recreate)

    def create_collection_for_doc(self, doc_name: str, force_recreate=False):
        """创建或重建指定文档的Milvus集合"""
        if not self._available:
            return None                    # Milvus不可用返回None
        col_name = self._get_collection_name(doc_name)  # 计算集合名
        if utility.has_collection(col_name) and not force_recreate:
            # 集合已存在且不需要重建，直接加载
            col = Collection(col_name)
            self.collections[doc_name] = col
            return col
        if utility.has_collection(col_name):
            Collection(col_name).drop()    # 需要重建，先删除旧集合
        # 定义集合Schema：6个字段
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),       # 自增主键
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),           # 1024维向量
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),             # 文本内容
            FieldSchema(name="page_num", dtype=DataType.INT64),                                # 页码
            FieldSchema(name="chunk_idx", dtype=DataType.INT64),                               # 分块索引
            FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=20)             # 来源类型
        ]
        schema = CollectionSchema(fields, description=f"RAG文档向量库 - {doc_name}")  # 创建Schema
        col = Collection(col_name, schema)                   # 创建集合
        # IVF_FLAT索引：内积(IP)度量，nlist=128控制聚类中心数
        index_params = {"metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
        col.create_index("embedding", index_params)          # 创建向量索引
        self.collections[doc_name] = col                     # 写入缓存
        logger.info(f"[Milvus] 集合创建成功: {col_name} (dim={self.dim})")
        return col

    def insert_chunks(self, chunks, embeddings, doc_name: str = "default"):
        """批量插入分块文本和对应向量到Milvus"""
        if not self._available:
            return                         # Milvus不可用则跳过
        col = self.get_collection(doc_name)  # 获取集合
        if not col:
            col = self.create_collection_for_doc(doc_name)  # 不存在则创建
        if not col:
            return                         # 创建失败则退出
        total = len(chunks)                # 总分块数
        batch_size = 100                   # 每批插入100条
        for i in range(0, total, batch_size):
            batch = chunks[i:i+batch_size]          # 取当前批次分块
            batch_emb = embeddings[i:i+batch_size]  # 取当前批次向量
            entities = [
                batch_emb,                                     # 向量列表
                [c["content"] for c in batch],                 # 文本内容列表
                [c["page_num"] for c in batch],                # 页码列表
                [c["chunk_idx"] for c in batch],               # 分块索引列表
                [c.get("source_type", "text") for c in batch]  # 来源类型列表
            ]
            col.insert(entities)           # 批量插入
        col.flush()                        # 刷新写入磁盘
        col.load()                         # 加载到内存
        logger.info(f"[Milvus] 插入完成: {doc_name}, {total}条")

    def search(self, query_emb, top_k=5, doc_name: str = None):
        """向量检索：指定doc_name则搜索单集合，否则遍历所有集合"""
        if not self._available:
            return []                      # Milvus不可用返回空
        if doc_name:
            # 单集合搜索模式
            col = self.get_collection(doc_name)
            if not col:
                return []                  # 集合不存在返回空
            col.load()                     # 加载集合到内存
            search_params = {"metric_type": "IP", "params": {"nprobe": 10}}  # 搜索参数：探测10个聚类
            results = col.search(
                data=[query_emb],          # 查询向量
                anns_field="embedding",    # 搜索的向量字段
                param=search_params,       # 搜索参数
                limit=top_k,               # 返回数量
                output_fields=["content", "page_num", "chunk_idx", "source_type"]  # 返回字段
            )
            return results[0]              # 返回第一个查询的结果列表
        # 全集合搜索模式：遍历所有已缓存的集合
        all_hits = []
        for dname, col in self.collections.items():
            try:
                col.load()                 # 加载集合
                search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
                results = col.search(
                    data=[query_emb], anns_field="embedding",
                    param=search_params, limit=top_k,
                    output_fields=["content", "page_num", "chunk_idx", "source_type"]
                )
                if results and results[0]:
                    all_hits.extend(results[0])  # 合并结果
            except Exception as e:
                logger.warning(f"[Milvus] 搜索 {dname} 失败: {e}")
        all_hits.sort(key=lambda h: h.score, reverse=True)  # 按得分降序排序
        return all_hits[:top_k]            # 返回全局top_k

    def drop_collection(self, doc_name: str):
        """删除指定文档的Milvus集合"""
        if not self._available:
            return                         # Milvus不可用则跳过
        col_name = self._get_collection_name(doc_name)  # 计算集合名
        try:
            if utility.has_collection(col_name):
                Collection(col_name).drop()       # 删除Milvus集合
                self.collections.pop(doc_name, None)  # 移除缓存
                logger.info(f"[Milvus] 已删除集合: {col_name}")
        except Exception as e:
            logger.warning(f"[Milvus] 删除集合失败: {e}")

    def get_collection_info(self, doc_name: str = None):
        """获取集合元信息：指定doc_name返回单个，否则返回所有"""
        if doc_name:
            col = self.get_collection(doc_name)
            if col:
                return {
                    'doc_name': doc_name,                                # 文档名
                    'collection_name': self._get_collection_name(doc_name),  # 集合名
                    'num_entities': col.num_entities,                     # 向量数量
                }
            return None                    # 集合不存在返回None
        # 无参数：返回所有已缓存集合的信息
        infos = []
        for dname in self.collections:
            info = self.get_collection_info(dname)
            if info:
                infos.append(info)
        return infos
