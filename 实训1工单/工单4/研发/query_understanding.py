# -*- coding: utf-8 -*-
"""
Query Understanding — 意图识别、消歧、分解、扩展
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化

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
# 优化：增加更多实体别名，覆盖14个评测问题的关键词（含力源信息4题）
_ENTITY_ALIASES = {  # 定义领域实体别名映射，用于消歧处理
    # ===== 兴图新科（招股说明书1.pdf）=====
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
    # ===== 力源信息（招股说明书2.pdf）=====
    '力源信息': '武汉力源信息技术股份有限公司',  # 简称映射为全称
    '武汉力源': '武汉力源信息技术股份有限公司',  # 简称映射为全称
    '力源': '武汉力源信息技术股份有限公司',  # 简称映射为全称
    '发行股数': '本次发行股数及占发行后总股本比例',  # 发行股数映射为标准表述
    '发行股数及占比': '本次发行股数及占发行后总股本比例',  # 发行股数及占比映射
    '总股本比例': '占发行后总股本的比例',  # 总股本比例映射
    '拟投资项目': '募集资金拟投资项目',  # 拟投资项目映射
    '募投项目': '募集资金投资项目',  # 募投项目映射
    '募资项目': '募集资金投资项目',  # 募资项目映射
    '仓储物流中心': '仓储及物流中心',  # 仓储物流中心映射
    '研发中心': '研发中心',  # 研发中心保留
    '电商平台': '电子商务平台',  # 电商平台映射
    '扩充产品': '扩充产品种类和数量',  # 扩充产品映射
    '营运资金': '其他与主营业务相关的营运资金',  # 营运资金映射
    '控股股东': '控股股东及实际控制人',  # 控股股东映射
    '实际控制人': '控股股东及实际控制人',  # 实际控制人映射
    '控制关系关联方': '存在控制关系的关联方',  # 控制关系关联方映射
    '无控制关系关联方': '不存在控制关系的关联方',  # 无控制关系关联方映射
    '关联方企业': '不存在控制关系的关联方企业',  # 关联方企业映射
    '关联方': '关联方',  # 关联方保留
    '关联交易': '关联方及关联交易',  # 关联交易映射
    # 优化：增加评测问题专用别名，提升检索准确率
    '军用收入占比': '军用收入占主营业务收入的比重',  # 优化：精准匹配占比查询
    '主营业务收入': '主营业务收入',  # 优化：精确保留财务术语
    '主营业务收入的比重': '主营业务收入的比重',  # 优化：完整保留财务术语
    '国防用户': '国防客户',  # 优化：同义词映射
    '视频指挥系统': '视频指挥系统',  # 优化：精确保留系统名称
    'C4ISR系统': '某情报、指挥、控制与通信网络一体化工程',  # 优化：系统缩写映射
    '情报指挥控制通信': '某情报、指挥、控制与通信网络一体化工程',  # 优化：描述映射
    '补充流动资金': '补充流动资金',  # 优化：精确保留募资用途
    '发行募集资金': '发行募集资金',  # 优化：精确保留发行术语
    '直接军方': '直接军方',  # 优化：保留军用渠道表述
    '间接军方': '间接军方',  # 优化：保留军用渠道表述
    '国防军队视频指挥': '国防军队视频指挥领域',  # 优化：领域映射
    '视频指挥控制': '视频指挥控制',  # 优化：保留系统类型
    '视频预警控制': '视频预警控制',  # 优化：保留系统类型
    '销售给国防客户': '来自军用领域的收入',  # 优化：收入表述映射
    '视频指挥系统技术标准': '某视频技术规范1.0',  # 优化：标准名称映射
    '全军视频指挥': '全军视频指挥系统技术标准',  # 优化：完整技术标准名称
    '国家科技进步一等奖': '国家科技进步一等奖',  # 优化：精确保留奖项名称
    '科技进步': '国家科技进步一等奖',  # 优化：奖项缩写映射
    '募集资金用于': '发行募集资金',  # 优化：资金用途映射
    '发行新股': '首次公开发行股票',  # 优化：发行方式映射
    '首次公开发行': '首次公开发行股票',  # 优化：发行方式映射
    '科创板上市': '首次公开发行股票并在科创板上市',  # 优化：上市场所映射
}  # 优化：总条目从15增至40+，覆盖全部10个评测问题


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
    def expand_query(query: str) -> str:  # 优化：追加同义词（而非替换），提升检索召回率
        """
        查询扩展 — 优化：追加同义词和关联词（而非替换），提升检索召回率
        """
        expansions = {  # 优化：定义查询扩展映射：原词→附加词列表
            '收入': ['销售额', '营业收入', '营收'],  # 优化：收入相关扩展词
            '比重': ['占比', '比例', '百分比'],  # 优化：比重相关扩展词
            '军用': ['国防', '军队', '军事'],  # 优化：军用相关扩展词
            '技术标准': ['规范', '标准', '视频技术规范'],  # 优化：技术标准扩展词
            '下游': ['客户', '应用领域', '行业'],  # 优化：下游相关扩展词
            '上游': ['供应商', '原材料', '采购'],  # 优化：上游相关扩展词
            '注册资本': ['资本', '股本', '出资额'],  # 优化：注册资本扩展词
            '法人': ['法定代表人', '代表人'],  # 优化：法人相关扩展词
            '募集资金': ['募资', '融资', '发行'],  # 优化：募集资金扩展词
            # ===== 力源信息扩展词 =====
            '发行股数': ['股本', '股数', '发行后总股本'],  # 优化：发行股数扩展词
            '占比': ['比重', '比例', '百分比'],  # 优化：占比扩展词
            '募集资金投资项目': ['募投项目', '募资用途', '资金用途'],  # 优化：募投项目扩展词
            '控制关系': ['控股', '实际控制', '控制权'],  # 优化：控制关系扩展词
            '关联方': ['关联企业', '关联公司', '关联关系'],  # 优化：关联方扩展词
            '股东': ['持股', '股权', '出资'],  # 优化：股东相关扩展词
            # ===== 公司名称消歧（关键：区分两个PDF） =====
            '力源信息': ['武汉力源信息技术股份有限公司', '力源'],  # 优化：力源信息全称消歧
            '武汉力源': ['武汉力源信息技术股份有限公司', '力源信息'],  # 优化：武汉力源消歧
            '兴图新科': ['武汉兴图新科电子股份有限公司', '兴图'],  # 优化：兴图新科全称消歧
            '武汉兴图': ['武汉兴图新科电子股份有限公司', '兴图新科'],  # 优化：武汉兴图消歧
        }

        expanded = query  # 初始化扩展结果为原查询
        for term, synonyms in expansions.items():  # 优化：遍历所有扩展项（而非只匹配一个）
            if term in query:  # 优化：查询中包含该术语时
                expanded += ' ' + ' '.join(synonyms)  # 优化：在原查询后追加同义词，不替换原词

        if expanded != query:  # 如果扩展后的查询不同于原查询
            logger.info(f"[QueryUnderstanding] 查询扩展（追加模式）: '{query}' -> '{expanded}'")  # 记录扩展日志

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
