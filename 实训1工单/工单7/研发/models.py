"""
RAG工单6 模型客户端合并文件
合并自: embedding_client.py（EmbeddingClient）+ reranker.py（RerankerClient）
"""

import os
import time
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)  # 模块级日志
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder


# ==================== EmbeddingClient ====================

class EmbeddingClient:
    """Embedding 客户端 —— 单例 + 懒加载，基于 bge-m3 模型"""

    _instance = None          # 单例实例
    _model = None             # SentenceTransformer 模型对象

    # bge-m3 模型本地目录（相对于本文件）
    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'bge-m3')
    EMBEDDING_DIM = 1024     # 输出向量维度

    def __new__(cls):         # 单例模式
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load_model(self):    # 懒加载：首次调用时加载模型
        if self._model is not None:
            return
        device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 自动检测 GPU
        print(f"[EmbeddingClient] 加载模型: {self.MODEL_DIR}, device={device}")
        self._model = SentenceTransformer(self.MODEL_DIR, device=device)  # 加载本地 bge-m3
        print("[EmbeddingClient] 模型加载完成")

    def is_available(self) -> bool:  # 检查模型目录是否存在
        return os.path.isdir(self.MODEL_DIR)

    def encode(self, texts: list[str], batch_size: int = 4) -> np.ndarray:  # 批量编码文本列表
        """编码文本列表，返回 shape=(N, EMBEDDING_DIM) 的 numpy 数组"""
        self._load_model()                      # 确保模型已加载
        truncated = [t[:1000] if len(t) > 1000 else t for t in texts]  # 截断过长文本
        logger.info(f"[Embedding] 编码{len(truncated)}条文本, batch_size={batch_size}")
        embeddings = self._model.encode(        # 标准encode调用
            truncated, batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True            # L2归一化
        )
        logger.info(f"[Embedding] 编码完成, shape={np.array(embeddings).shape}")
        return np.array(embeddings, dtype=np.float32)  # 转为float32

    def encode_query(self, query: str) -> np.ndarray:  # 单条查询编码
        """编码单条查询文本，返回 1D numpy 数组"""
        return self.encode([query])[0]          # 复用批量接口，取第一行


# ==================== RerankerClient ====================

class RerankerClient:
    """Reranker 客户端 —— 单例 + 懒加载，基于 bge-reranker-v2-m3 CrossEncoder"""

    _instance = None          # 单例实例
    _model = None             # CrossEncoder 模型对象

    # bge-reranker-v2-m3 模型本地目录
    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'bge-reranker-v2-m3')

    def __new__(cls):         # 单例模式
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load_model(self):    # 懒加载：首次调用时加载模型
        if self._model is not None:
            return
        device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 自动检测 GPU
        print(f"[RerankerClient] 加载模型: {self.MODEL_DIR}, device={device}")
        self._model = CrossEncoder(self.MODEL_DIR)  # 加载本地 CrossEncoder
        # CrossEncoder 的 device 参数不可靠，手动将内部模型移到目标设备
        self._model.model = self._model.model.to(device)
        print("[RerankerClient] 模型加载完成")

    def is_available(self) -> bool:  # 检查模型目录是否存在
        return os.path.isdir(self.MODEL_DIR)

    def rerank(self, query: str, documents: list, top_k: int = 10, max_length: int = 256) -> list[dict]:  # 重排序
        """对候选文档进行打分排序，返回 top_k 个结果（含 score 字段，sigmoid 归一化到 [0,1]）"""
        self._load_model()                      # 确保模型已加载
        if not documents:                       # 空文档列表直接返回
            return []
        # 从dict或str中提取文本，构建 query-document 对
        t0 = time.time()
        def extract_text(doc):                  # 统一提取文本
            if isinstance(doc, dict):           # dict→取text字段
                return doc.get('text', str(doc))
            return str(doc)                     # 直接是字符串
        texts = [extract_text(doc)[:max_length] for doc in documents]  # 截断过长文本
        pairs = [[query[:128], text] for text in texts]  # query也截断，减少tokenize量
        t1 = time.time()
        scores = self._model.predict(pairs, batch_size=len(pairs))  # 单batch避免多次forward
        t2 = time.time()
        # sigmoid 归一化到 [0, 1]
        normalized_scores = 1 / (1 + np.exp(-np.array(scores)))
        # 组装结果：保留原始dict的所有字段，追加score
        results = []
        for doc, score in zip(documents, normalized_scores):
            if isinstance(doc, dict):           # dict→保留原字段+追加score
                item = {**doc, 'score': float(score)}
            else:                              # str→只含text+score
                item = {'text': doc, 'score': float(score)}
            results.append(item)
        results.sort(key=lambda x: x['score'], reverse=True)  # 分数从高到低
        logger.info(f"[reranker] 准备={t1-t0:.2f}s predict={t2-t1:.2f}s 后处理={time.time()-t2:.2f}s 共{len(pairs)}对")
        return results[:top_k]                 # 返回前 top_k 条
