# -*- coding: utf-8 -*-
"""BGE-M3嵌入模型懒加载单例模块"""
import os
import logging

logger = logging.getLogger(__name__)

_instance = None


def get_embedder():
    """获取BGE-M3嵌入器单例"""
    global _instance
    if _instance is not None:
        return _instance
    _instance = _BGEM3Embedder()
    return _instance


class _BGEM3Embedder:
    """BGE-M3嵌入器（懒加载，1024维向量）"""

    def __init__(self):
        self._model = None

    def _load(self):
        """懒加载模型"""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            local_path = os.path.join(os.path.dirname(__file__), 'model', 'bge-m3')
            if os.path.isdir(local_path):
                model_path = local_path
                logger.info(f"[Embedder] 加载本地BGE-M3模型: {model_path}")
            else:
                model_path = "BAAI/bge-m3"
                logger.info(f"[Embedder] 从HuggingFace加载: {model_path}")
            self._model = SentenceTransformer(model_path)
            logger.info(f"[Embedder] BGE-M3加载成功, 向量维度: {self._model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"[Embedder] BGE-M3加载失败: {e}")
            raise

    def embed_texts(self, texts: list) -> list:
        """批量编码文本，返回1024维向量列表"""
        self._load()
        if not texts:
            return []
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list:
        """编码单条查询文本，返回1024维向量"""
        self._load()
        embedding = self._model.encode([query], normalize_embeddings=True)
        return embedding[0].tolist()
