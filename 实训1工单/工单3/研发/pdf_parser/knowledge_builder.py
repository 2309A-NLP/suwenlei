# -*- coding: utf-8 -*-
"""
知识库构建器 — 从PDF文件自动构建完整知识库

流程：PDF解析 → CSV分块 → TF-IDF训练 → Milvus向量入库
数据来源：前端文件上传的PDF文件
"""

import os      # 导入操作系统接口模块
import csv     # 导入CSV文件处理模块
import logging # 导入日志模块
import time    # 导入时间模块
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class KnowledgeBuilder:
    """知识库构建器 — 一站式完成PDF到Milvus的全流程"""

    def __init__(self, project_dir: str = '.'):
        """
        初始化知识库构建器

        参数:
            project_dir: 项目根目录路径（包含 pdf_parser/ 和 milvus_store.py 的目录）
        """
        self.project_dir = os.path.abspath(project_dir)  # 保存项目根目录的绝对路径
        self.uploads_dir = os.path.join(self.project_dir, 'uploads')  # 上传文件存储目录

        # 确保 uploads 目录存在
        os.makedirs(self.uploads_dir, exist_ok=True)  # 创建上传目录，已存在则忽略

    def build(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 150):
        """
        构建知识库：PDF → CSV → TF-IDF → Milvus

        参数:
            pdf_path: PDF文件的绝对路径
            chunk_size: 分块大小（字符数），默认1000
            chunk_overlap: 分块重叠大小（字符数），默认150

        返回:
            dict: {
                'success': bool,           # 是否成功
                'csv_path': str,           # 生成的CSV路径
                'chunks_count': int,       # 分块数量
                'milvus_inserted': int,    # Milvus入库数量
                'milvus_available': bool,  # Milvus是否可用
                'error': str or None,      # 错误信息（失败时）
                'elapsed': float,          # 耗时（秒）
            }
        """
        start_time = time.time()  # 记录开始时间
        result = {  # 初始化结果字典
            'success': False,           # 成功标记
            'csv_path': '',             # CSV路径
            'chunks_count': 0,          # 分块数量
            'milvus_inserted': 0,       # Milvus入库数量
            'milvus_available': False,  # Milvus可用标记
            'error': None,              # 错误信息
            'elapsed': 0,               # 耗时
        }

        try:
            # ---- 第1步：PDF解析 → CSV分块 ----
            logger.info(f"[KB] 第1步：解析PDF → CSV分块: {os.path.basename(pdf_path)}")  # 打印步骤日志
            csv_path = self._parse_pdf_to_csv(pdf_path, chunk_size, chunk_overlap)  # 调用PDF解析
            if not csv_path:  # 解析失败时
                result['error'] = 'PDF解析失败，未能生成CSV分块文件'  # 设置错误信息
                return result  # 返回失败结果
            result['csv_path'] = csv_path  # 保存CSV路径

            # ---- 第2步：从CSV加载分块 ----
            logger.info("[KB] 第2步：加载CSV分块数据")  # 打印步骤日志
            chunks = self._load_csv_chunks(csv_path)  # 从CSV加载分块
            if not chunks:  # 加载失败时
                result['error'] = 'CSV文件加载失败或无有效分块'  # 设置错误信息
                return result  # 返回失败结果
            result['chunks_count'] = len(chunks)  # 记录分块数量
            logger.info(f"[KB] 加载完成: {len(chunks)} 个分块")  # 打印加载统计

            # ---- 第3步：Milvus入库（如果可用） ----
            logger.info("[KB] 第3步：构建TF-IDF嵌入 → Milvus入库")  # 打印步骤日志
            # 从PDF文件名提取文档名（去掉扩展名），用于Milvus集合命名
            doc_name = os.path.splitext(os.path.basename(pdf_path))[0]  # 如"招股说明书1"
            milvus_inserted = self._insert_to_milvus(chunks, doc_name)  # 传入文档名调用Milvus入库
            result['milvus_inserted'] = milvus_inserted  # 记录入库数量

            # ---- 检查Milvus状态 ----
            result['milvus_available'] = self._check_milvus_available()  # 检查Milvus是否可用

            # ---- 完成 ----
            elapsed = round(time.time() - start_time, 2)  # 计算总耗时
            result['elapsed'] = elapsed  # 保存耗时
            result['success'] = True  # 标记成功
            logger.info(f"[KB] 知识库构建完成: {len(chunks)}块, Milvus入库{milvus_inserted}条, 耗时{elapsed}s")  # 打印完成日志

        except Exception as e:  # 捕获所有异常
            logger.error(f"[KB] 知识库构建异常: {e}", exc_info=True)  # 记录异常堆栈
            result['error'] = str(e)  # 保存错误信息

        return result  # 返回构建结果

    def _parse_pdf_to_csv(self, pdf_path: str, chunk_size: int, chunk_overlap: int) -> str:
        """
        第1步：调用 pdf_parser 解析PDF并生成CSV分块文件

        参数:
            pdf_path: PDF文件路径
            chunk_size: 分块大小
            chunk_overlap: 分块重叠

        返回:
            str: 生成的CSV文件路径，失败返回空字符串
        """
        try:
            from .pdf_processor import PDFProcessor  # 从同包导入PDF处理器

            # 推导输出CSV路径（与PDF同目录，同名但后缀为 _chunks_v2.csv）
            base = os.path.splitext(pdf_path)[0]  # 去除PDF扩展名
            csv_path = f"{base}_chunks_v2.csv"  # 生成CSV文件路径

            # 使用静态方法一键处理：解析 → 去噪 → 分块 → CSV导出
            result = PDFProcessor.process_pdf(  # 调用PDFProcessor的静态方法
                pdf_path=pdf_path,              # PDF文件路径
                output_csv=csv_path,            # 输出CSV路径
                extract_images=True             # 启用图片提取（新增）
            )

            if result and os.path.exists(csv_path):  # 处理成功且CSV文件存在时
                logger.info(f"[KB] PDF解析完成: {os.path.basename(csv_path)}")  # 打印成功日志
                return csv_path  # 返回CSV路径

            logger.error("[KB] PDF解析未生成有效CSV")  # 记录失败日志
            return ''  # 返回空字符串表示失败

        except Exception as e:  # 捕获异常
            logger.error(f"[KB] PDF解析失败: {e}", exc_info=True)  # 记录异常
            return ''  # 返回空字符串

    def _load_csv_chunks(self, csv_path: str) -> list:
        """
        第2步：从CSV文件加载分块数据

        参数:
            csv_path: CSV文件路径

        返回:
            list: 分块字典列表 [{content, page_num, chunk_idx, source_type}, ...]
        """
        chunks = []  # 初始化分块列表
        try:
            with open(csv_path, 'r', encoding='utf-8-sig') as f:  # 以UTF-8带BOM格式打开CSV
                reader = csv.DictReader(f)  # 创建字典读取器
                for row in reader:  # 遍历每一行
                    chunks.append({  # 添加分块字典
                        'content': row.get('text', ''),          # 文本内容
                        'page_num': int(row.get('page_num', 0)), # 页码
                        'chunk_idx': int(row.get('id', row.get('chunk_index', 0))),  # 分块索引
                        'source_type': row.get('source_type', 'text'),  # 来源类型
                    })
            return chunks  # 返回分块列表
        except Exception as e:  # 捕获异常
            logger.error(f"[KB] CSV加载失败: {e}")  # 记录错误
            return []  # 返回空列表

    def _insert_to_milvus(self, chunks: list, doc_name: str = "default") -> int:
        """
        第3步：训练TF-IDF嵌入模型，将分块向量写入Milvus

        参数:
            chunks: 分块字典列表
            doc_name: 文档名称（用于Milvus集合命名）

        返回:
            int: 成功入库的数量
        """
        try:
            # 导入Milvus存储模块
            from milvus_store import MilvusStore
            from embedder import embed_texts

            # 获取文本列表
            texts = [c['content'] for c in chunks]  # 提取所有分块的文本内容
            if not texts:  # 文本列表为空时
                logger.warning("[KB] 无文本内容，跳过Milvus入库")
                return 0  # 返回0

            # 使用bge-m3生成4096维向量
            logger.info("[KB] 使用bge-m3生成嵌入向量...")
            embeddings = embed_texts(texts)
            if not embeddings:
                logger.warning("[KB] 嵌入向量生成失败，跳过Milvus入库")
                return 0
            logger.info(f"[KB] bge-m3嵌入完成: {len(texts)}条, 维度={len(embeddings[0])}")

            # 初始化Milvus并插入数据
            milvus = MilvusStore()  # 创建Milvus存储实例
            if not milvus.is_available():  # Milvus不可用时
                logger.warning("[KB] Milvus不可用，跳过向量入库（已生成CSV，TF-IDF本地检索可用）")  # 记录警告
                return 0  # 返回0

            # 重建集合（确保数据最新，清空旧数据）
            logger.info("[KB] 重建Milvus集合...")  # 打印重建日志
            milvus.create_collection(doc_name, force_recreate=True)  # 传入文档名强制重建集合

            # 插入分块和向量
            logger.info(f"[KB] 向Milvus插入 {len(chunks)} 条分块...")  # 打印插入日志
            milvus.insert_chunks(chunks, embeddings, doc_name)  # 传入文档名批量插入分块和向量

            logger.info(f"[KB] Milvus入库完成: {len(chunks)} 条")  # 打印完成日志
            return len(chunks)  # 返回入库数量

        except ImportError as e:  # 导入错误（缺少依赖）
            logger.warning(f"[KB] 依赖缺失，跳过Milvus入库: {e}")  # 记录警告
            return 0  # 返回0
        except Exception as e:  # 其他异常
            logger.warning(f"[KB] Milvus入库失败（CSV已生成，本地检索可用）: {e}")  # 记录警告
            return 0  # 返回0

    def _check_milvus_available(self) -> bool:
        """
        检查Milvus是否可用

        返回:
            bool: Milvus是否可用
        """
        try:
            from milvus_store import MilvusStore  # 导入Milvus存储类
            milvus = MilvusStore()  # 创建实例
            return milvus.is_available()  # 返回可用状态
        except Exception:  # 异常时
            return False  # 返回不可用

    def get_collection_info(self) -> dict:
        """
        获取Milvus集合信息

        返回:
            dict: {
                'available': bool,       # Milvus是否可用
                'collection_exists': bool, # 集合是否存在
                'collection_name': str,   # 集合名称
                'num_entities': int,      # 集合中实体数量
            }
        """
        info = {  # 初始化信息字典
            'available': False,          # Milvus可用标记
            'collection_exists': False,  # 集合是否存在
            'collection_name': 'rag_document',  # 默认集合名称
            'num_entities': 0,           # 实体数量
        }

        try:
            from milvus_store import MilvusStore  # 导入Milvus存储类
            milvus = MilvusStore()  # 创建实例
            info['available'] = milvus.is_available()  # 记录可用状态
            info['collection_name'] = milvus.collection_name  # 记录集合名称

            if info['available']:  # Milvus可用时
                info['collection_exists'] = milvus.is_collection_exists()  # 检查集合是否存在
                if info['collection_exists'] and milvus.collection:  # 集合存在且已加载时
                    info['num_entities'] = milvus.collection.num_entities  # 获取实体数量

        except Exception as e:  # 捕获异常
            logger.warning(f"[KB] 获取Milvus集合信息失败: {e}")  # 记录警告

        return info  # 返回集合信息

    def delete_file_chunks(self, pdf_filename: str) -> bool:
        """
        删除指定PDF对应的CSV分块文件

        参数:
            pdf_filename: PDF文件名（如 "report.pdf"）

        返回:
            bool: 是否成功删除
        """
        try:
            # 推导CSV文件路径
            base = os.path.splitext(pdf_filename)[0]  # 去除PDF扩展名
            csv_path = os.path.join(self.uploads_dir, f"{base}_chunks_v2.csv")  # 构建CSV路径

            if os.path.exists(csv_path):  # CSV文件存在时
                os.remove(csv_path)  # 删除CSV文件
                logger.info(f"[KB] 已删除CSV分块: {os.path.basename(csv_path)}")  # 打印删除日志
                return True  # 返回成功

            return False  # CSV不存在，返回False

        except Exception as e:  # 捕获异常
            logger.error(f"[KB] 删除CSV分块失败: {e}")  # 记录错误
            return False  # 返回失败
