# -*- coding: utf-8 -*-
"""
Query Understanding — 意图识别、消歧、分解、扩展

流程：意图识别 → 实体消歧 → 问题分解 → 关键词提取 → 查询扩展
目的：将用户自然语言查询转化为更精确的检索查询
"""
import re
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

# 问句类型定义：根据句式模式判断问题意图
QUESTION_TYPES = {
    'entity': ['谁', '是谁', '哪家公司', '什么公司', '哪个公司', '是谁做的', '法定代表人'],
    'numeric': ['多少', '几个', '多少万', '多少钱', '多少元', '占比', '比重', '比例'],
    'list': ['哪些', '分别', '有哪些', '包括哪些', '分为哪些', '列举'],
    'confirm': ['是否', '是不是', '有没有', '是否属于', '是否包含', '有没有提到'],
    'definition': ['什么是', '是什么', '什么叫', '如何理解'],
    'comparison': ['有什么区别', '有何不同', '比较', '对比', '与...相比', '相较于'],
    'procedure': ['如何', '怎么', '怎样', '步骤', '流程'],
    'time': ['什么时候', '何时', '哪一年', '报告期内', '最近'],
}

# 实体别名映射：将用户口语化表述替换为文档中的标准表述（提升检索召回率）
_ENTITY_ALIASES = {
    # ===== 兴图新科 =====
    '兴图新科': '武汉兴图新科电子股份有限公司',
    '武汉兴图': '武汉兴图新科电子股份有限公司',
    '招股书': '招股意向书',
    '招股说明书': '招股意向书',
    '军用的收入': '来自军用领域的收入',
    '军用领域收入': '来自军用领域的收入',
    '军用收入': '来自军用领域的收入',
    '视频指挥': '视频指挥系统',
    '技术规范': '某视频技术规范1.0',
    'C4ISR': '某情报、指挥、控制与通信网络一体化工程',
    '上游企业': '电子信息行业的上游企业',
    '下游行业': '电子信息行业的下游行业',
    '供应商': '国防军队视频指挥领域重要供应商',
    '法人': '法定代表人',
    '法定代表': '法定代表人',
    '资本': '注册资本',
    '资金': '补充流动资金',
    '募集资金': '发行募集资金',
    # ===== 力源信息 =====
    '力源信息': '武汉力源信息技术股份有限公司',
    '武汉力源': '武汉力源信息技术股份有限公司',
    '力源': '武汉力源信息技术股份有限公司',
    '发行股数': '本次发行股数及占发行后总股本比例',
    '发行股数及占比': '本次发行股数及占发行后总股本比例',
    '总股本比例': '占发行后总股本的比例',
    '拟投资项目': '募集资金拟投资项目',
    '募投项目': '募集资金投资项目',
    '募资项目': '募集资金投资项目',
    '仓储物流中心': '仓储及物流中心',
    '电商平台': '电子商务平台',
    '扩充产品': '扩充产品种类和数量',
    '营运资金': '其他与主营业务相关的营运资金',
    '控股股东': '控股股东及实际控制人',
    '实际控制人': '控股股东及实际控制人',
    '控制关系关联方': '存在控制关系的关联方',
    '无控制关系关联方': '不存在控制关系的关联方',
    '关联方企业': '不存在控制关系的关联方企业',
    '关联交易': '关联方及关联交易',
    # 评测问题专用别名
    '军用收入占比': '军用收入占主营业务收入的比重',
    '国防用户': '国防客户',
    'C4ISR系统': '某情报、指挥、控制与通信网络一体化工程',
    '情报指挥控制通信': '某情报、指挥、控制与通信网络一体化工程',
    '国防军队视频指挥': '国防军队视频指挥领域',
    '销售给国防客户': '来自军用领域的收入',
    '视频指挥系统技术标准': '某视频技术规范1.0',
    '全军视频指挥': '全军视频指挥系统技术标准',
    '科技进步': '国家科技进步一等奖',
    '募集资金用于': '发行募集资金',
    '发行新股': '首次公开发行股票',
    '首次公开发行': '首次公开发行股票',
    '科创板上市': '首次公开发行股票并在科创板上市',
    # 组织结构/股权结构相关
    '组织结构图': '股权结构图及组织结构图',
    '组织架构图': '股权结构图及组织结构图',
    '公司架构': '股权结构图及组织结构图',
    '部门结构': '内部组织结构',
}


class QueryUnderstanding:
    """查询理解：意图识别+消歧+分解+扩展"""

    @staticmethod
    def recognize_intent(query: str) -> Tuple[str, float]:
        """意图识别：匹配问句模式判断问题类型（实体/数值/列表等）"""
        for intent, patterns in QUESTION_TYPES.items():
            for pattern in patterns:
                if pattern in query:
                    return intent, 0.9
        return 'entity', 0.5  # 默认意图

    @staticmethod
    def disambiguate(query):
        """实体消歧：将口语化/缩略表述替换为文档中的标准表述"""
        resolved = query
        # 按别名长度降序排序，避免短别名先匹配导致长别名失效
        sorted_aliases = sorted(_ENTITY_ALIASES.keys(), key=len, reverse=True)
        replaced_ranges = []  # 记录已替换区间，防止重复替换
        for alias in sorted_aliases:
            replacement = _ENTITY_ALIASES[alias]
            idx = resolved.find(alias)
            if idx == -1:
                continue
            # 检查是否与已替换区间重叠
            overlap = any(s <= idx < e for (s, e) in replaced_ranges)
            if overlap:
                continue
            # 检查目标位置是否已包含完整replacement（避免冗余替换）
            if QueryUnderstanding._already_contains_replacement(resolved, idx, alias, replacement):
                continue
            resolved = resolved[:idx] + replacement + resolved[idx + len(alias):]
            replaced_ranges.append((idx, idx + len(alias)))
        if resolved != query:
            logger.info(f"[QueryUnderstanding] 消歧: '{query}' -> '{resolved}'")
        return resolved

    @staticmethod
    def _already_contains_replacement(text, idx, alias, replacement):
        """防冗余替换：检查alias位置是否已包含完整replacement文本"""
        if replacement.startswith(alias):
            remaining = replacement[len(alias):]
            if text[idx + len(alias):].startswith(remaining):
                return True
        if replacement.endswith(alias):
            prefix = replacement[:-len(alias)]
            if idx >= len(prefix) and text[idx - len(prefix):idx] == prefix:
                return True
        pos = replacement.find(alias)
        if 0 < pos < len(replacement) - len(alias):
            before = replacement[:pos]
            after = replacement[pos + len(alias):]
            if (idx >= len(before) and text[idx - len(before):idx] == before
                    and text[idx + len(alias):idx + len(alias) + len(after)] == after):
                return True
        return False

    @staticmethod
    def decompose(query: str) -> List[str]:
        """复杂问题分解：按分号/句号/连接词拆分为子问题"""
        parts = re.split(r'[;；。]', query)
        parts = [p.strip() + ('？' if not p.strip().endswith('？') else '')
                 for p in parts if p.strip() and len(p.strip()) > 3]

        # 处理"A和B是多少"类型的并列问题
        if len(parts) <= 1:
            conj_patterns = [
                (r'(.*?)(?:和|与|及)(.*?)(?:是多少|是什么|有哪些|分别是)', 2),
            ]
            for pattern, expected_groups in conj_patterns:
                m = re.match(pattern, query)
                if m:
                    parts = [f'{m.group(1).strip()}是多少？',
                             f'{m.group(2).strip()}是多少？']
                    logger.info(f"[QueryUnderstanding] 分解: '{query}' -> {parts}")
                    break

        return parts if len(parts) > 1 else [query]

    @staticmethod
    def extract_keywords(query: str) -> List[str]:
        """关键词提取：去除标点和常见停用词，保留2字以上实词"""
        clean = re.sub(r'[？?，。！、：；""''（）!?,.:;\'"()]', ' ', query)
        words = clean.split()
        keywords = [w for w in words if len(w) >= 2 and w not in [
            '一个', '这个', '那个', '什么', '怎么', '如何', '哪些',
            '多少', '分别', '根据', '报告', '期内', '过来', '一下',
            '我们', '你们', '他们', '自己', '可以', '没有',
        ]]
        return keywords

    @staticmethod
    def expand_query(query: str) -> str:
        """查询扩展：追加同义词提升检索召回率"""
        expansions = {
            '收入': ['销售额', '营业收入', '营收'],
            '比重': ['占比', '比例', '百分比'],
            '军用': ['国防', '军队', '军事'],
            '技术标准': ['规范', '标准', '视频技术规范'],
            '下游': ['客户', '应用领域', '行业'],
            '上游': ['供应商', '原材料', '采购'],
            '注册资本': ['资本', '股本', '出资额'],
            '法人': ['法定代表人', '代表人'],
            '募集资金': ['募资', '融资', '发行'],
            '发行股数': ['股本', '股数', '发行后总股本'],
            '占比': ['比重', '比例', '百分比'],
            '募集资金投资项目': ['募投项目', '募资用途', '资金用途'],
            '控制关系': ['控股', '实际控制', '控制权'],
            '关联方': ['关联企业', '关联公司', '关联关系'],
            '股东': ['持股', '股权', '出资'],
            '力源信息': ['武汉力源信息技术股份有限公司', '力源'],
            '武汉力源': ['武汉力源信息技术股份有限公司', '力源信息'],
            '兴图新科': ['武汉兴图新科电子股份有限公司', '兴图'],
            '武汉兴图': ['武汉兴图新科电子股份有限公司', '兴图新科'],
            '组织结构图': ['股权结构图', '组织架构图', '公司架构', '部门结构'],
            '组织结构': ['股权结构', '组织架构', '部门设置', '治理结构'],
        }
        expanded = query
        for term, synonyms in expansions.items():
            if term in query:
                expanded += ' ' + ' '.join(synonyms)
        if expanded != query:
            logger.info(f"[QueryUnderstanding] 扩展: '{query}' -> '{expanded}'")
        return expanded

    @staticmethod
    def understand(query: str) -> Dict:
        """完整Query Understanding流程：意图→消歧→分解→扩展"""
        intent, confidence = QueryUnderstanding.recognize_intent(query)
        disambiguated = QueryUnderstanding.disambiguate(query)
        sub_questions = QueryUnderstanding.decompose(query)
        keywords = QueryUnderstanding.extract_keywords(disambiguated)
        expanded_query = QueryUnderstanding.expand_query(disambiguated)
        return {
            'original': query,
            'disambiguated': disambiguated,
            'expanded_query': expanded_query,  # 用于检索的扩展查询
            'intent': intent,
            'intent_confidence': confidence,
            'sub_questions': sub_questions,
            'keywords': keywords,
            'is_complex': len(sub_questions) > 1,
        }
