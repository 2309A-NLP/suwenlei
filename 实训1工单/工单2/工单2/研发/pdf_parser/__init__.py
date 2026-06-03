# -*- coding: utf-8 -*-
"""
pdf_parser 包 — 独立的PDF文档预处理模块

提供 PDFProcessor 类用于：
1. 解析PDF文件，去除水印/页眉/页脚
2. 提取表格内容并转为结构化文本
3. 提取图片及关联文本（可选，默认关闭）
4. 文本分块（自适应段落分块，带重叠）
5. 输出CSV文件

用法（命令行）：
    python -m pdf_parser.pdf_processor 招股说明书1.pdf -o output.csv

或独立运行：
    cd pdf_parser && python pdf_processor.py 招股说明书1.pdf
"""
from .pdf_processor import PDFProcessor, Chunk, PageContent, ExtractedTable, CHUNK_SIZE, CHUNK_OVERLAP

__all__ = ['PDFProcessor', 'Chunk', 'PageContent', 'ExtractedTable', 'CHUNK_SIZE', 'CHUNK_OVERLAP']
