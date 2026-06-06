# -*- coding: utf-8 -*-
import os
import logging

logger = logging.getLogger(__name__)

_model = None
_MODEL_DIM = 4096

def get_embedding_model():
    global _model
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
        model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
        # 优先使用本地模型目录（避免HuggingFace连接失败）
        local_model_dir = os.path.join(os.path.dirname(__file__), "model", "bge-m3")
        if os.path.exists(local_model_dir):
            model_name = local_model_dir
        _model = SentenceTransformer(model_name)
        logger.info(f"[Embedder] 加载嵌入模型: {model_name}, 维度={_MODEL_DIM}")
    except Exception as e:
        logger.error(f"[Embedder] 嵌入模型加载失败: {e}")
        _model = False
    return _model

def embed_texts(texts: list) -> list:
    model = get_embedding_model()
    if not model:
        return []
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()

def embed_query(query: str) -> list:
    model = get_embedding_model()
    if not model:
        return []
    embedding = model.encode([query], normalize_embeddings=True)
    return embedding[0].tolist()
