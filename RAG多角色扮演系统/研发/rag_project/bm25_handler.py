# -*- coding: utf-8 -*-
# bm25_handler.py
"""
BM25 索引处理器 —— 使用独立 Milvus 集合持久化分词结果
已升级为 MilvusClient
功能：
1. 从 Milvus 向量库读取高血压指南文本
2. 对文本进行分词，构建 BM25 全文检索索引
3. 将分词结果持久化存入独立 Milvus 集合，避免重复构建
4. 支持根据用户问题进行 BM25 关键词检索
"""

import logging
import time
from typing import Dict, List, Tuple
from rank_bm25 import BM25Okapi
from pymilvus import MilvusClient
from rag_project.config import BM25_AVAILABLE, TOP_K_RETRIEVE, MILVUS_HOST, MILVUS_PORT
from rag_project.utils import tokenize_text

logger = logging.getLogger(__name__)


class BM25Handler:

    def __init__(self, collection_names: set):
        start_time = time.time()
        logger.info("初始化 BM25Handler...")
        self.collection_names = collection_names
        self.bm25_indexes: Dict[str, BM25Okapi] = {}
        self.bm25_docs: Dict[str, List[Dict]] = {}
        self.client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")

        if BM25_AVAILABLE:
            self._load_or_build_all()
        else:
            logger.warning("BM25 不可用，只使用向量检索。")

        logger.info(f"BM25Handler 初始化完成，总耗时: {time.time() - start_time:.3f}s")

    def _get_index_collection_name(self, coll_name: str) -> str:
        return f"{coll_name}_bm25_index"

    def _load_or_build_all(self):
        start_total = time.time()
        for coll_name in self.collection_names:
            index_coll_name = self._get_index_collection_name(coll_name)
            if self.client.has_collection(index_coll_name):
                try:
                    self._load_from_milvus(coll_name, index_coll_name)
                    continue
                except Exception as e:
                    logger.warning(f"加载索引失败: {e}，将重建")
                    self.client.drop_collection(index_coll_name)
            self._build_index(coll_name)
        logger.info(f"所有 BM25 索引加载/构建完成，耗时: {time.time() - start_total:.3f}s")

    def _load_from_milvus(self, coll_name: str, index_coll_name: str):
        start = time.time()
        tokenized_corpus, doc_list = [], []
        offset, limit = 0, 1000

        while True:
            results = self.client.query(
                collection_name=index_coll_name,
                filter="chunk_id != ''",
                output_fields=["chunk_id", "tokens", "chunk_text"],
                limit=limit, offset=offset
            )
            if not results:
                break
            for rec in results:
                tokens_str = rec.get("tokens", "")
                if not tokens_str:
                    continue
                tokenized_corpus.append(tokens_str.split(" "))
                doc_list.append({
                    "chunk_id": rec["chunk_id"],
                    "chunk_text": rec["chunk_text"]
                })
            offset += len(results)
            if len(results) < limit:
                break

        if not tokenized_corpus:
            raise ValueError("索引集合为空")

        self.bm25_indexes[coll_name] = BM25Okapi(tokenized_corpus)
        self.bm25_docs[coll_name] = doc_list
        logger.info(f"从 {index_coll_name} 加载索引完成，文档数: {len(doc_list)}，耗时: {time.time() - start:.3f}s")

    def _build_index(self, coll_name: str):
        start = time.time()
        logger.info(f"构建 BM25 索引: {coll_name}")

        all_data = []
        offset, limit = 0, 1000
        while True:
            results = self.client.query(
                collection_name=coll_name,
                filter="chunk_id != ''",
                output_fields=["chunk_id", "chunk_text"],
                limit=limit, offset=offset
            )
            if not results:
                break
            all_data.extend(results)
            offset += len(results)
            if len(results) < limit:
                break

        if not all_data:
            logger.warning(f"集合 {coll_name} 为空，跳过")
            return

        tokenized_corpus, doc_list = [], []
        chunk_ids, tokens_list, chunk_texts = [], [], []

        for item in all_data:
            tokens = tokenize_text(item["chunk_text"], BM25_AVAILABLE)
            if not tokens:
                continue
            tokenized_corpus.append(tokens)
            doc_list.append({
                "chunk_id": item["chunk_id"],
                "chunk_text": item["chunk_text"]
            })
            chunk_ids.append(item["chunk_id"])
            tokens_list.append(" ".join(tokens))
            chunk_texts.append(item["chunk_text"])

        if not tokenized_corpus:
            logger.warning("无有效分词，跳过")
            return

        self.bm25_indexes[coll_name] = BM25Okapi(tokenized_corpus)
        self.bm25_docs[coll_name] = doc_list
        self._create_index_collection(
            self._get_index_collection_name(coll_name),
            chunk_ids, tokens_list, chunk_texts
        )
        logger.info(f"构建索引 {coll_name} 完成，耗时: {time.time() - start:.3f}s")

    def _create_index_collection(self, index_coll_name, chunk_ids, tokens_list, chunk_texts):
        start = time.time()
        if self.client.has_collection(index_coll_name):
            self.client.drop_collection(index_coll_name)

        schema = self.client.create_schema(enable_dynamic_field=False, auto_id=False)
        schema.add_field(field_name="chunk_id", datatype=MilvusClient.DataType.VARCHAR, is_primary=True, max_length=100)
        schema.add_field(field_name="tokens", datatype=MilvusClient.DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="chunk_text", datatype=MilvusClient.DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dummy_vector", datatype=MilvusClient.DataType.FLOAT_VECTOR, dim=2)
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="dummy_vector", metric_type="COSINE", index_type="FLAT")
        self.client.create_collection(collection_name=index_coll_name, schema=schema, index_params=index_params)

        data = [{
            "chunk_id": chunk_ids[i],
            "tokens": tokens_list[i],
            "chunk_text": chunk_texts[i],
            "dummy_vector": [0.0, 0.0]
        } for i in range(len(chunk_ids))]
        self.client.insert(collection_name=index_coll_name, data=data)
        self.client.flush(collection_name=index_coll_name)
        logger.info(f"索引集合 {index_coll_name} 创建完成，{len(data)} 条，耗时: {time.time() - start:.3f}s")

    def search(self, query: str, coll_name: str) -> List[Tuple[Dict, float]]:
        if not BM25_AVAILABLE or coll_name not in self.bm25_indexes:
            return []

        bm25 = self.bm25_indexes[coll_name]
        docs = self.bm25_docs[coll_name]
        tokenized_query = tokenize_text(query, BM25_AVAILABLE)
        if not tokenized_query:
            return []

        scores = bm25.get_scores(tokenized_query)
        scored = [(docs[i], scores[i]) for i in range(len(docs)) if scores[i] > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:TOP_K_RETRIEVE]
