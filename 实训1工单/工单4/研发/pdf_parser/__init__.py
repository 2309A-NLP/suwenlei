# -*- coding: utf-8 -*-
"""
pdf_parser 包 — 独立的PDF文档预处理模块
工单编号：人工智能NLP-RAG-图像内容解析及检索优化

提供 PDFProcessor 类用于：
1. 解析PDF文件，去除水印/页眉/页脚
2. 提取表格内容并转为结构化文本
3. 提取图片并通过多模态API生成图片语义描述（新增）
4. 文本清理：去除换行符、特殊符号、多余空格（新增）
5. 文本分块（自适应段落分块，带重叠）
6. 输出CSV文件

用法（命令行）：
    python -m pdf_parser.pdf_processor 招股说明书1.pdf -o output.csv

或独立运行：
    cd pdf_parser && python pdf_processor.py 招股说明书1.pdf
"""
from .pdf_processor import PDFProcessor, Chunk, PageContent, ExtractedTable, CHUNK_SIZE, CHUNK_OVERLAP, ENABLE_IMAGE_EXTRACT  # 从pdf_processor导入核心类和常量
from .knowledge_builder import KnowledgeBuilder  # 从knowledge_builder导入知识库构建器

__all__ = ['PDFProcessor', 'Chunk', 'PageContent', 'ExtractedTable', 'CHUNK_SIZE', 'CHUNK_OVERLAP', 'ENABLE_IMAGE_EXTRACT', 'KnowledgeBuilder']  # 模块公开导出列表
