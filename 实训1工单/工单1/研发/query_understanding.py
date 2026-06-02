# -*- coding: utf-8 -*-
"""
Query Understanding — 意图识别、消歧、分解、扩展
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

功能：
1. 意图识别（Intent Recognition）：判断问题类型（实体查询、数值查询、列举、验证等）
2. 消歧（Disambiguation）：处理多义词和模糊表述
3. 分解（Decomposition）：将复杂问题拆解为子问题
4. 查询扩展（Query Expansion）：提取关键词，丰富查询词汇
"""

import re  # 正则表达式模块，用于文本匹配和分割
import logging  # 日志模块，用于输出调试和运行信息
from typing import List, Dict, Optional, Set, Tuple  # 类型提示工具，增强代码可读性

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


# ==================== 问句类型定义 ====================

QUESTION_TYPES = {  # 定义8种问句类型的模式字典，用于意图识别
    'entity': ['谁', '是谁', '哪家公司', '什么公司', '哪个公司', '是谁做的', '法定代表人'],  # 实体查询类问句模式
    'numeric': ['多少', '几个', '多少万', '多少钱', '多少元', '占比', '比重', '比例'],  # 数值查询类问句模式
    'list': ['哪些', '分别', '有哪些', '包括哪些', '分为哪些', '列举'],  # 列举查询类问句模式
    'confirm': ['是否', '是不是', '有没有', '是否属于', '是否包含', '有没有提到'],  # 确认验证类问句模式
    'definition': ['什么是', '是什么', '什么叫', '如何理解'],  # 定义解释类问句模式
    'comparison': ['有什么区别', '有何不同', '比较', '对比', '与...相比', '相较于'],  # 比较对比类问句模式
    'procedure': ['如何', '怎么', '怎样', '步骤', '流程'],  # 流程步骤类问句模式
    'time': ['什么时候', '何时', '哪一年', '报告期内', '最近'],  # 时间查询类问句模式
}

# 招股说明书领域实体映射
_ENTITY_ALIASES = {  # 定义领域实体别名映射，用于消歧处理
    '公司': '武汉兴图新科电子股份有限公司',  # 简称映射为全称
    '兴图新科': '武汉兴图新科电子股份有限公司',  # 简称映射为全称
    '武汉兴图': '武汉兴图新科电子股份有限公司',  # 简称映射为全称
    '招股书': '招股意向书',  # 招股书简称映射为标准名称
    '招股说明书': '招股意向书',  # 招股说明书全称映射为标准名称
    '军用的收入': '来自军用领域的收入',  # 模糊表述映射为准确表述
    '军用领域收入': '来自军用领域的收入',  # 同义表述映射为标准表述
    '军用收入': '来自军用领域的收入',  # 简写映射为完整表述
    '国防客户': '国防客户',  # 国防客户保持原样
    '视频指挥': '视频指挥系统',  # 简称映射为完整系统名称
    '技术规范': '某视频技术规范1.0',  # 泛指映射为具体规范名称
    'C4ISR': '某情报、指挥、控制与通信网络一体化工程',  # 英文缩写映射为中文全称
    '上游企业': '电子信息行业的上游企业',  # 泛指映射为领域特定表述
    '下游行业': '电子信息行业的下游行业',  # 泛指映射为领域特定表述
    '供应商': '国防军队视频指挥领域重要供应商',  # 泛指映射为具体描述
    '法人': '法定代表人',  # 简称映射为标准法律术语
    '法定代表': '法定代表人',  # 不完整表述映射为完整术语
    '资本': '注册资本',  # 泛指映射为具体财务术语
    '资金': '补充流动资金',  # 泛指映射为募资用途具体表述
    '募集资金': '发行募集资金',  # 简称映射为完整表述
}


class QueryUnderstanding:
    """Query Understanding — 理解用户问题意图"""

    @staticmethod  # 声明为静态方法，无需实例即可调用
    def recognize_intent(query: str) -> Tuple[str, float]:
        """
        识别问题意图类型
        返回: (intent_type, confidence_score)
        """
        for intent, patterns in QUESTION_TYPES.items():  # 遍历所有意图类型及其模式列表
            for pattern in patterns:  # 遍历当前意图的每个匹配模式
                if pattern in query:  # 检查模式是否出现在问题字符串中
                    return intent, 0.9  # 匹配成功，返回意图类型和较高置信度0.9
        # 默认
        return 'entity', 0.5  # 无匹配时默认返回实体查询类型和较低置信度0.5

    @staticmethod  # 声明为静态方法
    def disambiguate(query):  # 消歧方法：将模糊表述替换为准确表述
        resolved = query  # 初始化消歧结果为原始问题
        # 按长度降序匹配，防止短词误匹配长词子串
        sorted_aliases = sorted(_ENTITY_ALIASES.keys(), key=len, reverse=True)  # 按别名长度降序排序，长词优先匹配
        replaced_ranges = []  # 记录已替换位置的起止索引列表，防止重叠替换
        for alias in sorted_aliases:  # 遍历按长度排序后的别名列表
            replacement = _ENTITY_ALIASES[alias]  # 获取别名对应的替换文本
            idx = resolved.find(alias)  # 在消歧结果中查找别名的位置
            if idx == -1:  # 如果别名未出现在文本中
                continue  # 跳过当前别名，处理下一个
            # 检查该匹配是否与已替换的位置重叠
            overlap = any(s <= idx < e for (s, e) in replaced_ranges)  # 判断当前匹配位置是否已被替换过
            if overlap:  # 如果存在重叠
                continue  # 跳过本次替换，避免重复处理
            # 通用检查：alias周围的文本是否已经包含完整的replacement
            # 避免"法定代表人"中替换"法定代表"、"注册资本"中替换"资本"等产生重复
            if QueryUnderstanding._already_contains_replacement(resolved, idx, alias, replacement):  # 检查文本是否已包含完整替换文本
                continue  # 已包含则跳过，防止重复替换
            resolved = resolved[:idx] + replacement + resolved[idx + len(alias):]  # 执行替换：将别名替换为完整表述
            replaced_ranges.append((idx, idx + len(alias)))  # 记录本次替换的位置范围
        if resolved != query:  # 如果消歧后结果与原问题不同
            logger.info(f"[QueryUnderstanding] 消歧: '{query}' -> '{resolved}'")  # 记录消歧日志
        return resolved  # 返回消歧后的结果

    @staticmethod  # 声明为静态方法
    def _already_contains_replacement(text, idx, alias, replacement):  # 检查text中alias位置是否已包含完整的replacement
        """检查text中alias位置是否已经包含完整的replacement"""
        # 情况1: alias在replacement开头 (如 法定代表 → 法定代表人)
        if replacement.startswith(alias):  # 如果别名位于替换文本的开头
            remaining = replacement[len(alias):]  # 提取别名之后的部分
            if text[idx + len(alias):].startswith(remaining):  # 检查文本中别名之后是否跟有该部分
                return True  # 已包含完整替换文本，无需替换
        # 情况2: alias在replacement结尾 (如 资本 → 注册资本)
        if replacement.endswith(alias):  # 如果别名位于替换文本的结尾
            prefix = replacement[:-len(alias)]  # 提取别名之前的部分
            if idx >= len(prefix) and text[idx - len(prefix):idx] == prefix:  # 检查文本中别名之前是否有该部分
                return True  # 已包含完整替换文本，无需替换
        # 情况3: alias在replacement中间
        pos = replacement.find(alias)  # 查找别名在替换文本中的位置
        if 0 < pos < len(replacement) - len(alias):  # 如果别名位于替换文本中间
            before = replacement[:pos]  # 提取别名之前的部分
            after = replacement[pos + len(alias):]  # 提取别名之后的部分
            if (idx >= len(before) and text[idx - len(before):idx] == before  # 检查别名前后是否都有对应文本
                    and text[idx + len(alias):idx + len(alias) + len(after)] == after):  # 检查别名前后文本均匹配
                return True  # 已包含完整替换文本，无需替换
        return False  # 未包含完整替换文本，可以执行替换

    @staticmethod  # 声明为静态方法
    def decompose(query: str) -> List[str]:  # 分解复杂问题为多个子问题
        """
        分解复杂问题为子问题
        返回子问题列表（如果不能分解则返回原问题列表）
        """
        # 包含"分别"的问题拆解
        if '分别' in query and '、' in query or '和' in query:  # 检查是否包含并列标志词
            # 尝试提取并列项
            # 例: "报告期内各年收入分别是多少？" -> 单问题，不分解
            pass  # 暂不处理，保留后续扩展空间

        # 多问句拆分（用分号、问号分割）
        parts = re.split(r'[;；。]', query)  # 用分号或句号分割多问句
        parts = [p.strip() + ('？' if not p.strip().endswith('？') else '')  # 为每个子句补全问号
                 for p in parts if p.strip() and len(p.strip()) > 3]  # 过滤掉空串和过短的片段

        # 如果分割后只有一条，尝试内容拆分
        if len(parts) <= 1:  # 如果分割后只有一条或零条
            # 包含"和"、"以及"可能表示复合问题
            conj_patterns = [  # 定义并列连接词的正则模式列表
                (r'(.*?)(?:和|与|及)(.*?)(?:是多少|是什么|有哪些|分别是)', 2),  # 匹配"A和B是多少"类复合问句
            ]
            for pattern, expected_groups in conj_patterns:  # 遍历所有并列模式
                m = re.match(pattern, query)  # 用正则匹配当前模式
                if m:  # 如果匹配成功
                    parts = [f'{m.group(1).strip()}是多少？',  # 提取并列项A构造子问题
                             f'{m.group(2).strip()}是多少？']  # 提取并列项B构造子问题
                    logger.info(f"[QueryUnderstanding] 分解: '{query}' -> {parts}")  # 记录分解日志
                    break  # 只使用第一个匹配模式

        return parts if len(parts) > 1 else [query]  # 有多个子问题则返回列表，否则返回原问题的单元素列表

    @staticmethod  # 声明为静态方法
    def extract_keywords(query: str) -> List[str]:  # 从查询中提取关键词用于扩展
        """
        提取查询关键词（用于查询扩展）
        """
        # 去除标点和停用词
        clean = re.sub(r'[？?，。！、：；""''（）!?,.:;\'\"()]', ' ', query)  # 用正则替换所有标点符号为空格
        words = clean.split()  # 按空格分割为单词列表
        keywords = [w for w in words if len(w) >= 2 and w not in [  # 过滤停用词和单字词
            '一个', '这个', '那个', '什么', '怎么', '如何', '哪些',  # 常见疑问词停用列表
            '多少', '分别', '根据', '报告', '期内', '过来', '一下',  # 常见功能词停用列表
            '我们', '你们', '他们', '自己', '可以', '没有',  # 常见代词和否定词停用列表
        ]]  # 只保留长度>=2且不在停用词表中的词
        return keywords  # 返回过滤后的关键词列表

    @staticmethod  # 声明为静态方法
    def expand_query(query: str) -> str:  # 通过添加同义词扩展查询以提升检索召回率
        """
        查询扩展 — 添加同义词和关联词以提升检索召回率
        """
        expansions = {  # 定义同义扩展映射字典：原词 -> 扩展词串
            '收入': '收入 销售额 营业收入 营收',  # 收入相关近义词扩展
            '比重': '比重 占比 比例 百分比',  # 比重相关近义词扩展
            '军用': '军用 国防 军队 军事',  # 军用领域相关近义词扩展
            '技术标准': '技术标准 规范 标准 视频技术规范',  # 技术标准相关近义词扩展
            '下游': '下游 客户 应用领域 行业',  # 下游相关近义词扩展
            '上游': '上游 供应商 原材料 采购',  # 上游相关近义词扩展
            '注册资本': '注册资本 资本 股本 出资额',  # 注册资本相关近义词扩展
            '法人': '法人 法定代表人 代表人',  # 法人相关近义词扩展
            '募集资金': '募集资金 募资 融资 发行',  # 募集资金相关近义词扩展
        }

        expanded = query  # 初始化扩展结果为原查询
        for term, replacement in expansions.items():  # 遍历所有扩展映射项
            if term in query:  # 如果原查询包含扩展词
                expanded = expanded.replace(term, replacement)  # 将原词替换为扩展词串
                break  # 只扩展一个匹配项，避免过度扩展

        if expanded != query:  # 如果扩展后的查询不同于原查询
            logger.info(f"[QueryUnderstanding] 查询扩展: '{query}' -> '{expanded}'")  # 记录扩展日志

        return expanded  # 返回扩展后的查询字符串

    @staticmethod  # 声明为静态方法
    def understand(query: str) -> Dict:  # 完整Query Understanding流程编排：意图识别+消歧+分解+扩展
        """
        完整的Query Understanding流程
        返回包含意图、消歧后问题、子问题、关键词的字典
        """
        intent, confidence = QueryUnderstanding.recognize_intent(query)  # 第一步：识别问题意图类型
        disambiguated = QueryUnderstanding.disambiguate(query)  # 第二步：执行消歧处理
        sub_questions = QueryUnderstanding.decompose(query)  # 第三步：分解复杂问题为子问题
        keywords = QueryUnderstanding.extract_keywords(disambiguated)  # 第四步：从消歧后的查询中提取关键词
        expanded_query = QueryUnderstanding.expand_query(disambiguated)  # 第五步：扩展消歧后的查询

        result = {  # 组装理解结果字典
            'original': query,  # 原始查询文本
            'disambiguated': disambiguated,  # 消歧后的查询文本
            'expanded_query': expanded_query,  # 扩展后的查询文本
            'intent': intent,  # 识别出的意图类型
            'intent_confidence': confidence,  # 意图识别的置信度分数
            'sub_questions': sub_questions,  # 分解后的子问题列表
            'keywords': keywords,  # 提取的关键词列表
            'is_complex': len(sub_questions) > 1,  # 是否为复杂问题（有多个子问题）
        }

        logger.info(f"[QueryUnderstanding] 理解结果: {result}")  # 记录完整理解结果的日志
        return result  # 返回理解结果字典
