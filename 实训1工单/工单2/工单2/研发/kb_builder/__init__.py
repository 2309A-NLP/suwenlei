# -*- coding: utf-8 -*-
"""
知识库构建模块 — PDF解析 + CSV分块 + 嵌入向量 + Milvus入库

提供 KnowledgeBuilder 类用于：
1. 解析PDF文件（调用 pdf_parser）
2. 生成CSV分块文件
3. 训练TF-IDF嵌入模型
4. 将分块向量写入Milvus向量数据库

用法：
    from kb_builder import KnowledgeBuilder
    builder = KnowledgeBuilder(project_dir='.')
    result = builder.build('/path/to/uploaded.pdf')
"""

from .builder import KnowledgeBuilder  # 导入知识库构建核心类

__all__ = ['KnowledgeBuilder']  # 导出列表
