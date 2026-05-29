# -*- coding: utf-8 -*-
# utils.py 通用工具函数
import re
import logging
from typing import List

logger = logging.getLogger(__name__)

# 预编译正则表达式，避免每次调用都重新编译
_RE_PUNCT = re.compile(r'[^\u4e00-\u9fa5a-zA-Z0-9 ]')
_RE_SPACE = re.compile(r'\s+')


def preprocess_query(query: str) -> str:
    """轻量预处理：去标点、去多余空格"""
    query = _RE_PUNCT.sub(' ', query)
    return _RE_SPACE.sub(' ', query).strip()


def is_empty_knowledge(knowledge_str: str) -> bool:
    """判断知识字符串是否有效（长度<10视为空）"""
    return len(knowledge_str.strip()) < 10


def tokenize_text(text: str, bm25_available: bool) -> List[str]:
    """中文分词，若 jieba 不可用则使用简单字符切分"""
    if not bm25_available:
        return list(text.lower())
    try:
        import jieba
        return list(jieba.cut_for_search(text))
    except ImportError:
        return list(text.lower())
