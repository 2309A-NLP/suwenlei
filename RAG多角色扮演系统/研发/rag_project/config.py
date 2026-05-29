# -*- coding: utf-8 -*-
# config.py
import os
import torch

# ================= DeepSeek API =================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-7687dfb9b95f4a69b2e2683f3c105d6b")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

# ================= Milvus =================
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

ROLE_TO_COLLECTION = {
    "hypertension": "hypertension_guideline_kb",
    "tcm": "hypertension_guideline_kb",
    "patient_edu": "hypertension_guideline_kb",
    "lawyer": "civil_code_kb"
}
DEFAULT_COLLECTION = "hypertension_guideline_kb"

# ================= Redis =================
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_EXPIRE = int(os.getenv("REDIS_EXPIRE", 7200))

# ================= 模型路径 =================
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", os.path.join(MODEL_DIR, "bge-m3"))
RERANK_MODEL = os.getenv("RERANK_MODEL", os.path.join(MODEL_DIR, "bge-reranker-v2-m3"))

# ================= 检索参数 =================
TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", 5))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", 3))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.65))
RRF_K = int(os.getenv("RRF_K", 60))
BM25_CACHE_DIR = os.getenv("BM25_CACHE_DIR", "bm25_cache")

# 新增：重排序分数阈值，低于此分数视为检索结果与问题不相关
RERANK_THRESHOLD = float(os.getenv("RERANK_THRESHOLD", 0.5))

# ================= 设备 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= BM25 可用性标志 =================
BM25_AVAILABLE = False
try:
    import jieba
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    pass