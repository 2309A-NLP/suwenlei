# -*- coding: utf-8 -*-
"""
知识库构建器 — 从PDF文件自动构建完整知识库

流程：PDF解析 → CSV分块 → TF-IDF训练 → Milvus向量入库
数据来源：前端文件上传的PDF文件
"""

import os
import csv
import logging
import time
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
logger = logging.getLogger(__name__)


class KnowledgeBuilder:
    """知识库构建器：一站式完成PDF到Milvus的全流程"""

    def __init__(self, project_dir: str = '.'):
        self.project_dir = os.path.abspath(project_dir)
        self.uploads_dir = os.path.join(self.project_dir, 'uploads')
        os.makedirs(self.uploads_dir, exist_ok=True)

    def build(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 150):
        """构建知识库主流程：PDF解析→CSV分块→Milvus向量入库"""
        start_time = time.time()
        result = {
            'success': False,
            'csv_path': '',
            'chunks_count': 0,
            'milvus_inserted': 0,
            'milvus_available': False,
            'error': None,
            'elapsed': 0,
        }

        try:
            # 第1步：PDF解析生成CSV分块文件
            logger.info(f"[KB] 第1步：解析PDF → CSV分块: {os.path.basename(pdf_path)}")
            csv_path = self._parse_pdf_to_csv(pdf_path, chunk_size, chunk_overlap)
            if not csv_path:
                result['error'] = 'PDF解析失败，未能生成CSV分块文件'
                return result
            result['csv_path'] = csv_path

            # 第2步：从CSV加载分块数据
            logger.info("[KB] 第2步：加载CSV分块数据")
            chunks = self._load_csv_chunks(csv_path)
            if not chunks:
                result['error'] = 'CSV文件加载失败或无有效分块'
                return result
            result['chunks_count'] = len(chunks)
            logger.info(f"[KB] 加载完成: {len(chunks)} 个分块")

            # 第3步：TF-IDF嵌入→Milvus向量入库
            logger.info("[KB] 第3步：构建TF-IDF嵌入 → Milvus入库")
            doc_name = os.path.splitext(os.path.basename(pdf_path))[0]  # 如"招股说明书1"
            milvus_inserted = self._insert_to_milvus(chunks, doc_name)
            result['milvus_inserted'] = milvus_inserted

            result['milvus_available'] = self._check_milvus_available()

            elapsed = round(time.time() - start_time, 2)
            result['elapsed'] = elapsed
            result['success'] = True
            logger.info(f"[KB] 知识库构建完成: {len(chunks)}块, Milvus入库{milvus_inserted}条, 耗时{elapsed}s")

        except Exception as e:
            logger.error(f"[KB] 知识库构建异常: {e}", exc_info=True)
            result['error'] = str(e)

        return result

    def _parse_pdf_to_csv(self, pdf_path: str, chunk_size: int, chunk_overlap: int) -> str:
        """调用PDFProcessor解析PDF并生成CSV分块文件"""
        try:
            from .pdf_processor import PDFProcessor

            base = os.path.splitext(pdf_path)[0]
            csv_path = f"{base}_chunks_v2.csv"

            # 一键处理：解析→去噪→分块→CSV导出
            result = PDFProcessor.process_pdf(
                pdf_path=pdf_path,
                output_csv=csv_path,
                extract_images=True
            )

            if result and os.path.exists(csv_path):
                logger.info(f"[KB] PDF解析完成: {os.path.basename(csv_path)}")
                return csv_path

            logger.error("[KB] PDF解析未生成有效CSV")
            return ''

        except Exception as e:
            logger.error(f"[KB] PDF解析失败: {e}", exc_info=True)
            return ''

    def _load_csv_chunks(self, csv_path: str) -> list:
        """从CSV加载分块数据为字典列表"""
        chunks = []
        try:
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    chunks.append({
                        'content': row.get('text', ''),
                        'page_num': int(row.get('page_num', 0)),
                        'chunk_idx': int(row.get('id', row.get('chunk_index', 0))),
                        'source_type': row.get('source_type', 'text'),
                    })
            return chunks
        except Exception as e:
            logger.error(f"[KB] CSV加载失败: {e}")
            return []

    def _insert_to_milvus(self, chunks: list, doc_name: str = "default") -> int:
        """训练TF-IDF嵌入模型，将分块向量写入Milvus"""
        try:
            from milvus_store import MilvusStore
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            import jieba

            texts = [c['content'] for c in chunks]
            if not texts:
                logger.warning("[KB] 无文本内容，跳过Milvus入库")
                return 0

            # 中文分词+停用词过滤（与rag_pipeline保持一致）
            stop_words = {
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
            }

            def jieba_tokenizer(text):
                words = list(jieba.cut(text))
                return [w for w in words if len(w) > 1 and w not in stop_words]

            # TF-IDF提取词频特征 → SVD降维到128维（适配Milvus向量维度）
            logger.info("[KB] 训练TF-IDF嵌入模型...")
            tfidf = TfidfVectorizer(max_features=5000, tokenizer=jieba_tokenizer)
            tfidf_mat = tfidf.fit_transform(texts)
            svd = TruncatedSVD(n_components=128, random_state=42)
            embeddings = svd.fit_transform(tfidf_mat).tolist()
            logger.info(f"[KB] TF-IDF({len(tfidf.vocabulary_)}词) → SVD(128维) 完成")

            milvus = MilvusStore()
            if not milvus.is_available():
                logger.warning("[KB] Milvus不可用，跳过向量入库（已生成CSV，TF-IDF本地检索可用）")
                return 0

            # 强制重建集合：确保数据最新
            logger.info("[KB] 重建Milvus集合...")
            milvus.create_collection(doc_name, force_recreate=True)

            logger.info(f"[KB] 向Milvus插入 {len(chunks)} 条分块...")
            milvus.insert_chunks(chunks, embeddings, doc_name)

            logger.info(f"[KB] Milvus入库完成: {len(chunks)} 条")
            return len(chunks)

        except ImportError as e:
            logger.warning(f"[KB] 依赖缺失，跳过Milvus入库: {e}")
            return 0
        except Exception as e:
            logger.warning(f"[KB] Milvus入库失败（CSV已生成，本地检索可用）: {e}")
            return 0

    def _check_milvus_available(self) -> bool:
        """检查Milvus服务是否可用"""
        try:
            from milvus_store import MilvusStore
            milvus = MilvusStore()
            return milvus.is_available()
        except Exception:
            return False

    def get_collection_info(self) -> dict:
        """获取Milvus集合信息：用于调试和状态检查"""
        info = {
            'available': False,
            'collection_exists': False,
            'collection_name': 'rag_document',
            'num_entities': 0,
        }

        try:
            from milvus_store import MilvusStore
            milvus = MilvusStore()
            info['available'] = milvus.is_available()
            info['collection_name'] = milvus.collection_name

            if info['available']:
                info['collection_exists'] = milvus.is_collection_exists()
                if info['collection_exists'] and milvus.collection:
                    info['num_entities'] = milvus.collection.num_entities

        except Exception as e:
            logger.warning(f"[KB] 获取Milvus集合信息失败: {e}")

        return info

    def delete_file_chunks(self, pdf_filename: str) -> bool:
        """删除指定PDF对应的CSV分块文件"""
        try:
            base = os.path.splitext(pdf_filename)[0]
            csv_path = os.path.join(self.uploads_dir, f"{base}_chunks_v2.csv")

            if os.path.exists(csv_path):
                os.remove(csv_path)
                logger.info(f"[KB] 已删除CSV分块: {os.path.basename(csv_path)}")
                return True

            return False

        except Exception as e:
            logger.error(f"[KB] 删除CSV分块失败: {e}")
            return False
