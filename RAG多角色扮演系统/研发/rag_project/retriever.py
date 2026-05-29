# -*- coding: utf-8 -*-
# retriever.py
"""
混合检索模块：实现向量检索 + BM25 关键词检索 + RRF 融合 + 重排序
核心流程：
1. 向量检索（语义匹配）
2. BM25 关键词检索（精确匹配）
3. RRF 算法融合两路结果
4. 交叉编码器重排序（精排）
5. 阈值过滤 → 返回最终知识文本
"""
import logging
import time
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from pymilvus import MilvusClient
from rag_project.config import (
    EMBEDDING_MODEL, RERANK_MODEL, DEVICE,
    TOP_K_RETRIEVE, TOP_K_RERANK, SIMILARITY_THRESHOLD, RRF_K,
    ROLE_TO_COLLECTION, DEFAULT_COLLECTION, RERANK_THRESHOLD
)
from rag_project.utils import preprocess_query
from rag_project.bm25_handler import BM25Handler

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    混合检索器（向量 + BM25 + RRF 融合 + 重排序）
    作用：结合语义匹配与关键词匹配优势，大幅提升检索准确率
    """

    def __init__(self, milvus_client: MilvusClient, bm25_handler: Optional[BM25Handler]):
        """初始化检索器"""
        start_time = time.time()
        logger.info("HybridRetriever 初始化开始")

        self.milvus_client = milvus_client
        self.bm25 = bm25_handler

        # 向量嵌入模型（生成查询向量）
        self.embedder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)

        # 交叉编码器重排序模型
        self.reranker = CrossEncoder(RERANK_MODEL, device=DEVICE)
        # CUDA 可用时转为 FP16 减少显存占用 + 加速推理
        if DEVICE.startswith("cuda"):
            self.reranker.model.half()

        # 持久化线程池，避免每次检索创建/销毁线程的开销
        self._executor = ThreadPoolExecutor(max_workers=2)

        logger.info(f"HybridRetriever 初始化完成，耗时：{time.time() - start_time:.3f}s")

    def _vector_search(self, query: str, coll_name: str) -> List[Dict]:
        """向量检索：语义匹配"""
        start = time.time()
        clean_query = " ".join(query.split())
        q_vector = self.embedder.encode(clean_query, normalize_embeddings=False).tolist()
        # nprobe=8 已足够，比 10 略快（Milvus 默认 nprobe=8）
        search_params = {"metric_type": "L2", "params": {"nprobe": 8}}
        results = self.milvus_client.search(
            collection_name=coll_name,
            data=[q_vector],
            anns_field="chunk_vector",
            search_params=search_params,
            limit=TOP_K_RETRIEVE,
            output_fields=["chunk_id", "chunk_text"]
        )
        candidates = []
        if results and len(results) > 0:
            for rank, hit in enumerate(results[0]):
                entity = hit.get("entity", {})
                chunk_id = entity.get("chunk_id")
                text = entity.get("chunk_text")
                if not text or not chunk_id:
                    continue
                distance = hit.get("distance", 1.0)
                similarity = 1 / (1 + distance)
                candidates.append({
                    "chunk_id": chunk_id,
                    "text": text,
                    "vector_score": similarity,
                    "vector_rank": rank + 1
                })
        logger.info(f"向量召回: {len(candidates)} 条，耗时: {time.time() - start:.3f}s")
        return candidates

    def _bm25_search(self, query: str, coll_name: str) -> List[Dict]:
        """BM25 关键词检索：字面匹配"""
        start = time.time()
        if self.bm25 is None:
            return []
        processed_query = preprocess_query(query)
        bm25_results = self.bm25.search(processed_query, coll_name)
        candidates = []
        for rank, (doc, score) in enumerate(bm25_results):
            candidates.append({
                "chunk_id": doc["chunk_id"],
                "text": doc["chunk_text"],
                "bm25_score": score,
                "bm25_rank": rank + 1
            })
        logger.info(f"BM25 召回: {len(candidates)} 条，耗时: {time.time() - start:.3f}s")
        return candidates

    def _rrf_fusion(self, vector_candidates: List[Dict], bm25_candidates: List[Dict]) -> List[Dict]:
        """RRF 召回融合（Reciprocal Rank Fusion）"""
        start = time.time()
        fusion = {}
        for cand in vector_candidates:
            cid = cand["chunk_id"]
            fusion[cid] = {
                "chunk_id": cid,
                "text": cand["text"],
                "rrf_score": 1.0 / (RRF_K + cand["vector_rank"]),
                "vector_score": cand["vector_score"],
                "bm25_score": 0.0
            }
        for cand in bm25_candidates:
            cid = cand["chunk_id"]
            rrf = 1.0 / (RRF_K + cand["bm25_rank"])
            if cid in fusion:
                fusion[cid]["rrf_score"] += rrf
                fusion[cid]["bm25_score"] = cand["bm25_score"]
            else:
                fusion[cid] = {
                    "chunk_id": cid,
                    "text": cand["text"],
                    "rrf_score": rrf,
                    "vector_score": 0.0,
                    "bm25_score": cand["bm25_score"]
                }
        if not fusion:
            return []
        result = sorted(fusion.values(), key=lambda x: x["rrf_score"], reverse=True)[:TOP_K_RETRIEVE]
        logger.info(f"RRF 融合后 {len(result)} 条，耗时: {time.time() - start:.3f}s")
        return result

    def _rerank(self, question: str, candidates: List[Dict]) -> List[Dict]:
        """交叉编码器重排序（精排）"""
        start = time.time()
        if not candidates:
            return []
        pairs = [(question, cand["text"]) for cand in candidates]
        # 显式设置 show_progress_bar=False 消除 tqdm 输出开销
        rerank_scores = self.reranker.predict(pairs, show_progress_bar=False)
        for i, score in enumerate(rerank_scores):
            candidates[i]["rerank_score"] = score
        final = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:TOP_K_RERANK]
        logger.info(f"重排序后 {len(final)} 条，耗时: {time.time() - start:.3f}s")
        return final

    def retrieve(self, question: str, role: str) -> str:
        """对外暴露的统一检索入口"""
        total_start = time.time()
        coll_name = ROLE_TO_COLLECTION.get(role, DEFAULT_COLLECTION)

        # 多线程并发：向量 + BM25 同时检索（复用持久线程池）
        future_vector = self._executor.submit(self._vector_search, question, coll_name)
        future_bm25 = self._executor.submit(self._bm25_search, question, coll_name)
        vector_candidates = future_vector.result()
        bm25_candidates = future_bm25.result()

        fused = self._rrf_fusion(vector_candidates, bm25_candidates)
        if not fused:
            return ""
        final = self._rerank(question, fused)
        if not final:
            return ""

        max_score = max(cand["rerank_score"] for cand in final)
        if max_score < RERANK_THRESHOLD:
            return ""

        knowledge_str = "\n---\n".join([c["text"] for c in final])
        logger.info(f"检索完成，{len(final)} 条，总耗时: {time.time() - total_start:.3f}s")
        return knowledge_str
