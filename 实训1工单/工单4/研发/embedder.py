# -*- coding: utf-8 -*-
"""BGE-M3嵌入模型懒加载单例模块"""
import os
import logging

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

_instance = None  # 全局单例缓存


def get_embedder():
    """获取BGE-M3嵌入器单例"""
    global _instance
    if _instance is not None:  # 已初始化时直接返回
        return _instance
    _instance = _BGEM3Embedder()  # 创建单例实例
    return _instance


class _BGEM3Embedder:
    """BGE-M3嵌入器（懒加载，1024维向量）"""

    def __init__(self):
        self._model = None  # 模型缓存
        self._tokenizer = None  # 分词器缓存

    def _load(self):
        """懒加载模型和分词器"""
        if self._model is not None:  # 已加载时跳过
            return
        try:
            from sentence_transformers import SentenceTransformer  # 导入模型加载器
            # 优先加载本地模型路径
            local_path = os.path.join(os.path.dirname(__file__), 'model', 'bge-m3')  # 本地模型目录
            if os.path.isdir(local_path):  # 本地路径存在时
                model_path = local_path  # 使用本地路径
                logger.info(f"[Embedder] 加载本地BGE-M3模型: {model_path}")  # 记录本地加载
            else:
                model_path = "BAAI/bge-m3"  # 回退到在线模型
                logger.info(f"[Embedder] 本地模型不存在，从HuggingFace加载: {model_path}")  # 记录在线加载
            self._model = SentenceTransformer(model_path)  # 加载模型
            logger.info(f"[Embedder] BGE-M3加载成功, 向量维度: {self._model.get_sentence_embedding_dimension()}")  # 记录维度
        except Exception as e:  # 加载失败时
            logger.error(f"[Embedder] BGE-M3加载失败: {e}")  # 记录错误
            raise  # 抛出异常

    def embed_texts(self, texts: list) -> list:
        """批量编码文本，返回1024维向量列表"""
        self._load()  # 确保模型已加载
        if not texts:  # 空文本列表时
            return []  # 返回空列表
        embeddings = self._model.encode(texts, normalize_embeddings=True)  # 编码文本为归一化向量
        return embeddings.tolist()  # 转为Python列表

    def embed_query(self, query: str) -> list:
        """编码单条查询文本，返回1024维向量"""
        self._load()  # 确保模型已加载
        embedding = self._model.encode([query], normalize_embeddings=True)  # 编码查询为归一化向量
        return embedding[0].tolist()  # 返回第一条的列表形式
