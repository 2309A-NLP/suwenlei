# -*- coding: utf-8 -*-
"""
RAG引擎核心模块 — 支持多文档路由检索
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化

架构说明：
- 每个PDF文档维护独立的：CSV分块、TF-IDF模型、Milvus集合
- 查询路由：分析问题内容 → 判断在哪个文档中检索 → 返回结果
- 查询路由规则：
  1. 包含"力源"相关词 → 搜索招股说明书2
  2. 包含"兴图"或军用/国防/视频指挥相关词 → 搜索招股说明书1
  3. 无明确指向 → 搜索所有文档并合并结果
"""
import os  # 导入操作系统接口模块
import re  # 导入正则表达式模块
import csv  # 导入CSV文件处理模块
import logging  # 导入日志模块
from typing import List, Dict, Tuple, Optional  # 导入类型提示

from llm_client import LLMClient  # 导入LLM客户端
from query_understanding import QueryUnderstanding  # 导入查询理解模块

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

# ==================== 中文停用词表 ====================
CHINESE_STOP_WORDS = set([  # 优化：添加停用词过滤，减少无意义词对TF-IDF检索的干扰
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
    '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
    '没有', '看', '好', '自己', '这', '他', '她', '它', '们', '那', '些',
    '与', '及', '或', '对', '被', '把', '从', '向', '以', '为', '由',
    '于', '而', '但', '且', '之', '其', '所', '者', '了', '过', '将',
    '让', '使', '能', '可', '得', '已', '还', '又', '再', '才', '则',
    '等', '如', '若', '虽', '因', '故', '并', '非', '即', '既', '各',
    '每', '某', '该', '本', '哪', '何', '么', '吗', '呢', '吧', '啊',
    '哦', '嗯', '呀', '哈', '嘛', '样', '种', '些', '点', '些', '多',
    '少', '个', '只', '第', '年', '月', '日', '元', '万', '亿',
    '涉及', '包括', '通过', '进行', '实现', '提供', '取得', '分别',
    '相关', '上述', '其中', '以及', '报告', '期内', '来自',
])  # 优化：增加中文停用词表，提升TF-IDF检索精度

# ==================== 文档路由关键词 ====================
# 定义每个文档的特征关键词，用于查询路由
DOC_KEYWORDS = {  # 文档路由关键词映射
    '招股说明书1': [  # 兴图新科相关关键词
        '兴图', '兴图新科', '武汉兴图',  # 公司名称关键词
        '军用', '国防', '军队', '军事', '军方',  # 军事相关关键词
        '视频指挥', 'C4ISR', '情报指挥', '技术规范',  # 技术系统关键词
        '程家明', '法定代表人', '注册资本',  # 公司基本信息关键词
        '上游', '下游', '行业',  # 行业链关键词（偏向兴图新科语境）
        '补充流动资金', '募集资金',  # 募资用途关键词（偏向兴图新科）
        '国家科技进步',  # 奖项关键词
    ],
    '招股说明书2': [  # 力源信息相关关键词
        '力源', '力源信息', '武汉力源',  # 公司名称关键词
        '赵马克', 'Mark Zhao',  # 实际控制人关键词
        '仓储物流', '电子商务平台', '扩充产品',  # 募投项目关键词
        '关联方', '关联交易', '控制关系',  # 关联交易关键词
        '融冰投资', '博润', '听音投资',  # 股东关键词
    ],
}


class DocumentEngine:
    """单文档检索引擎 — 维护一个PDF的分块数据和TF-IDF模型"""

    def __init__(self, doc_name: str, csv_path: str):
        self.doc_name = doc_name  # 文档名称（如"招股说明书1"）
        self.csv_path = csv_path  # CSV文件路径
        self.chunks = []  # 文档分块列表
        self.embedder = None  # BGE-M3嵌入器引用
        self._load_csv()  # 加载CSV分块数据
        self._init_embedder()  # 初始化TF-IDF嵌入模型

    def _load_csv(self):
        """从CSV文件加载分块数据"""
        if not os.path.exists(self.csv_path):  # CSV文件不存在时
            logger.warning(f"[DocEngine] CSV不存在: {self.csv_path}")  # 记录警告
            return  # 直接返回
        try:
            with open(self.csv_path, 'r', encoding='utf-8-sig') as f:  # 以UTF-8 BOM格式打开
                reader = csv.DictReader(f)  # 创建字典读取器
                for row in reader:  # 遍历每一行
                    self.chunks.append({  # 添加分块字典
                        "content": row.get('text', ''),  # 文本内容
                        "page_num": int(row.get('page_num', 0)),  # 页码
                        "chunk_idx": int(row.get('id', row.get('chunk_index', 0))),  # 分块索引
                        "source_type": row.get('source_type', 'text'),  # 来源类型
                    })
            logger.info(f"[DocEngine] {self.doc_name}: 加载{len(self.chunks)}个分块")  # 记录日志
        except Exception as e:  # 加载失败时
            logger.error(f"[DocEngine] CSV加载失败: {e}")  # 记录错误

    def _init_embedder(self):
        """初始化BGE-M3嵌入模型"""
        if not self.chunks:  # 无分块数据时
            return  # 直接返回
        try:
            from embedder import get_embedder  # 导入嵌入器工厂函数
            self.embedder = get_embedder()  # 获取BGE-M3嵌入器单例
            logger.info(f"[DocEngine] {self.doc_name}: BGE-M3嵌入器加载完成（4096维）")  # 记录初始化成功
        except Exception as e:  # 加载失败时
            logger.error(f"[DocEngine] {self.doc_name} BGE-M3嵌入器加载失败: {e}")  # 记录错误

    def generate_embeddings(self, texts: List[str]):
        """生成文本的4096维向量嵌入（使用BGE-M3）"""
        if self.embedder is None:  # 嵌入器未初始化时
            return []  # 返回空列表
        return self.embedder.embed_texts(texts)  # 调用BGE-M3生成4096维向量

    def tfidf_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """使用TF-IDF + 余弦相似度进行本地检索"""
        # 优化：提取查询关键词用于重排序
        query_keywords = [w for w in re.findall(r'[\u4e00-\u9fff]{2,}', query)
                          if w not in CHINESE_STOP_WORDS]  # 从查询中提取有效关键词
        if not self.chunks or getattr(self, 'tfidf', None) is None:  # 分块或TF-IDF未就绪时
            return []  # 返回空结果

        from sklearn.metrics.pairwise import cosine_similarity  # 导入余弦相似度计算函数
        q_vec = self.tfidf.transform([query])  # 将查询转为TF-IDF向量
        scores = cosine_similarity(q_vec, self.tfidf_mat).flatten()  # 计算与所有文档的余弦相似度

        # 优化：取更多候选用于重排序
        candidate_count = min(top_k * 5, len(scores))  # 候选数量扩大5倍
        candidate_indices = scores.argsort()[::-1][:candidate_count]  # 先按TF-IDF初筛候选

        # 优化：对候选结果进行重排序（关键词密度 + 精确匹配加分）
        scored_candidates = []  # 重排序候选列表
        for idx in candidate_indices:  # 遍历每个候选
            if scores[idx] <= 0:  # 评分为0时跳过
                continue
            c = self.chunks[idx]  # 获取对应的分块
            text = c["content"]  # 分块文本
            base_score = float(scores[idx])  # TF-IDF基础分
            boost = 0.0  # 重排序加分

            # 精确短语匹配加分
            for kw in query_keywords:  # 遍历每个关键词
                if kw in text:  # 关键词在文本中出现
                    boost += 0.08  # 每个命中关键词加0.08
                count = text.count(kw)  # 统计关键词出现次数
                if count >= 3:  # 出现3次以上
                    boost += 0.15 * min(count, 5)  # 高频词额外加分

            # 完整匹配大幅加分
            query_clean = ''.join(query_keywords)  # 合并关键词
            if len(query_clean) > 2 and query_clean in text:  # 完整匹配
                boost += 0.3  # 完整匹配大幅加分

            scored_candidates.append((idx, base_score + boost))  # 记录总分

        # 按重排序后的总分排序
        scored_candidates.sort(key=lambda x: x[1], reverse=True)  # 按总分降序排列

        results = []  # 结果列表
        for idx, final_score in scored_candidates[:top_k]:  # 遍历重排序后的结果
            c = self.chunks[idx]  # 获取对应的分块
            source_type = c.get("source_type", "text")  # 获取来源类型
            # 图片降权（仅TF-IDF路径，Milvus路径在get_context中处理）
            if source_type == "image":  # 图片类型时
                final_score *= 0.05  # 降权到5%
            elif source_type == "table":  # 表格类型时
                final_score *= 0.7  # 降权到70%
            results.append({  # 添加结果字典
                "text": c["content"],  # 文本内容
                "page_num": c["page_num"],  # 页码
                "score": round(final_score, 4),  # 使用重排序后的综合评分
                "source_type": source_type,  # 来源类型
                "doc_name": self.doc_name,  # 来源文档名（新增）
            })

        return results  # 返回结果列表


class RAGEngine:
    """RAG引擎 — 支持多文档路由检索"""

    def __init__(self, pdf_paths: Dict[str, str] = None, backend="auto"):
        """
        初始化RAG引擎

        参数:
            pdf_paths: {doc_name: pdf_path} 字典，如 {"招股说明书1": "/path/to/1.pdf"}
            backend: RAG后端类型
        """
        self.pdf_paths = pdf_paths or {}  # PDF路径字典
        self.backend = backend  # RAG后端类型
        self.doc_engines = {}  # 每个文档的检索引擎 {doc_name: DocumentEngine}
        self.milvus_store = self._init_milvus()  # 初始化Milvus向量数据库（共享）
        self.llm_client = LLMClient()  # 创建LLM客户端实例
        self.query_understand = QueryUnderstanding()  # 创建查询理解实例
        if pdf_paths:  # 如果提供了PDF路径
            self.init_rag()  # 立即初始化RAG引擎

    @staticmethod
    def _init_milvus():
        """初始化Milvus存储"""
        try:
            from milvus_store import MilvusStore  # 导入Milvus存储模块
            return MilvusStore()  # 创建Milvus存储实例
        except Exception as e:  # 导入或初始化失败时
            logger.warning(f"MilvusStore 初始化失败: {e}")  # 记录警告日志
            return None  # 返回None，降级到本地检索

    @staticmethod
    def _csv_path_for(pdf_path: str) -> str:
        """根据PDF路径推导CSV文件名"""
        base = os.path.splitext(pdf_path)[0]  # 去除PDF扩展名
        if base.endswith('_chunks'):  # 路径已以_chunks结尾
            return f"{base}_v2.csv"  # 直接追加_v2.csv
        return f"{base}_chunks_v2.csv"  # 返回标准CSV文件名

    def init_rag(self):
        """初始化RAG引擎 — 为每个PDF创建独立的检索引擎"""
        logger.info("正在初始化RAG引擎（多文档模式）...")  # 打印初始化日志
        for doc_name, pdf_path in self.pdf_paths.items():  # 遍历每个文档
            csv_path = self._csv_path_for(pdf_path)  # 推导CSV路径
            # 也检查uploads目录中的CSV
            if not os.path.exists(csv_path):  # 根目录CSV不存在时
                uploads_dir = os.path.join(os.path.dirname(pdf_path), 'uploads')  # 构建uploads路径
                if os.path.isdir(uploads_dir):  # uploads目录存在时
                    for f in os.listdir(uploads_dir):  # 遍历uploads目录
                        if f.endswith('_chunks_v2.csv') and doc_name.replace('招股说明书', '') in f:  # 匹配文档
                            csv_path = os.path.join(uploads_dir, f)  # 使用uploads中的CSV
                            break
            if os.path.exists(csv_path):  # CSV文件存在时
                engine = DocumentEngine(doc_name, csv_path)  # 创建文档引擎
                if engine.chunks:  # 有分块数据时
                    self.doc_engines[doc_name] = engine  # 缓存文档引擎
                    logger.info(f"文档引擎初始化完成: {doc_name} ({len(engine.chunks)}块)")
            else:  # CSV不存在时
                logger.warning(f"文档 {doc_name} 的CSV不存在: {csv_path}")  # 记录警告

        logger.info(f"RAG引擎初始化完成: {len(self.doc_engines)}个文档")  # 记录完成日志

    def _route_query(self, query: str) -> List[str]:
        """
        查询路由：分析问题内容，判断应该搜索哪个文档
        优化：区分强匹配（公司名）和弱匹配（通用词），强匹配优先

        返回: 应该搜索的文档名列表
        """
        # 定义强匹配关键词（公司名/品牌名），出现即锁定文档
        STRONG_KEYWORDS = {
            '招股说明书1': ['兴图', '兴图新科', '武汉兴图', '程家明', 'C4ISR', '视频指挥'],
            '招股说明书2': ['力源', '力源信息', '武汉力源', '赵马克', 'Mark Zhao', '融冰投资'],
        }
        # 定义弱匹配关键词（通用行业/财务词），仅在无强匹配时使用
        WEAK_KEYWORDS = {
            '招股说明书1': ['军用', '国防', '军队', '军事', '军方', '技术规范',
                          '法定代表人', '注册资本', '上游', '下游', '行业',
                          '补充流动资金', '募集资金', '国家科技进步'],
            '招股说明书2': ['仓储物流', '电子商务平台', '扩充产品', '关联方',
                          '关联交易', '控制关系', '博润', '听音投资'],
        }

        # 1. 先检查强匹配（公司名等明确标识）
        strong_matches = []  # 强匹配的文档列表
        for doc_name, keywords in STRONG_KEYWORDS.items():  # 遍历强关键词
            if doc_name not in self.doc_engines:  # 文档未加载时跳过
                continue
            for kw in keywords:  # 遍历关键词
                if kw in query:  # 查询中包含该关键词
                    strong_matches.append(doc_name)  # 添加到强匹配列表
                    break  # 找到一个即可

        if strong_matches:  # 有强匹配时，只搜强匹配的文档
            logger.info(f"[路由] 查询 '{query[:30]}...' → 强匹配 {strong_matches}")
            return strong_matches  # 返回强匹配的文档列表

        # 2. 无强匹配时，检查弱匹配（通用行业/财务词）
        weak_matches = []  # 弱匹配的文档列表
        for doc_name, keywords in WEAK_KEYWORDS.items():  # 遍历弱关键词
            if doc_name not in self.doc_engines:  # 文档未加载时跳过
                continue
            for kw in keywords:  # 遍历关键词
                if kw in query:  # 查询中包含该关键词
                    weak_matches.append(doc_name)  # 添加到弱匹配列表
                    break  # 找到一个即可

        if weak_matches:  # 有弱匹配时
            logger.info(f"[路由] 查询 '{query[:30]}...' → 弱匹配 {weak_matches}")
            return weak_matches  # 返回弱匹配的文档列表

        # 3. 无任何匹配时，搜索所有文档
        all_docs = list(self.doc_engines.keys())  # 获取所有已加载的文档
        logger.info(f"[路由] 查询 '{query[:30]}...' → 搜索全部 {all_docs}")
        return all_docs  # 返回所有文档

    def get_context(self, query: str, top_k=25, original_query: str = None) -> Tuple[str, List[Dict]]:
        """
        检索最相关的top_k个文档块（多文档路由）

        流程：
        1. 路由判断搜索哪个文档（使用原始查询，避免消歧干扰）
        2. 优先Milvus搜索，降级到TF-IDF本地检索
        3. 合并结果并排序

        返回: (context_str, results_list)
        """
        # 1. 路由判断（用原始查询，消歧/扩展后的查询可能引入干扰词）
        route_query = original_query if original_query else query  # 优先用原始查询路由
        target_docs = self._route_query(route_query)  # 获取目标文档列表

        all_results = []  # 所有结果列表

        for doc_name in target_docs:  # 遍历目标文档
            engine = self.doc_engines.get(doc_name)  # 获取文档引擎
            if not engine:  # 引擎不存在时
                continue  # 跳过

            # 2. 尝试Milvus搜索
            if self.milvus_store and self.milvus_store.is_available():  # Milvus可用时
                try:
                    embeddings = engine.generate_embeddings([query])  # 生成查询向量
                    if embeddings:  # 向量生成成功时
                        hits = self.milvus_store.search(embeddings[0], top_k * 3, doc_name=doc_name)  # 搜索Milvus（候选数3倍）
                        if hits:  # 检索到结果时
                            for h in hits:  # 遍历每个命中结果
                                source_type = h.entity.get("source_type", "text")  # 获取来源类型
                                score = h.score  # 获取分数
                                # Milvus路径：Qwen-VL处理后的图片描述包含结构化图表数据，不做降权
                                # 仅对纯装饰性图片（无数据价值）降权，由prompt引导LLM区分
                                all_results.append({  # 添加结果字典
                                    "text": h.entity.get("content", ""),  # 文本内容
                                    "page_num": h.entity.get("page_num", 0),  # 页码
                                    "score": round(score, 4),  # 相似度评分
                                    "source_type": source_type,  # 来源类型
                                    "doc_name": doc_name,  # 来源文档名
                                })
                            continue  # Milvus搜索成功，跳过TF-IDF
                except Exception as e:  # Milvus检索失败时
                    logger.warning(f"[{doc_name}] Milvus检索失败，降级到TF-IDF: {e}")  # 记录警告

            # 3. 降级到TF-IDF本地检索
            results = engine.tfidf_search(query, top_k * 2)  # 使用TF-IDF检索
            all_results.extend(results)  # 添加到总结果列表

        # 4. 关键词补充检索：向量搜索可能遗漏包含精确数据的chunk（如图表数据）
        keyword_boost = self._keyword_supplement_search(original_query or query, target_docs)
        if keyword_boost:
            all_results.extend(keyword_boost)

        # 5. 按分数排序并取top_k
        all_results.sort(key=lambda r: r['score'], reverse=True)  # 按分数降序排列
        # 去重：同一个chunk（同页码+同内容前50字）只保留分数最高的
        seen = set()
        unique_results = []
        for r in all_results:
            dedup_key = f"{r.get('page_num', '')}_{r.get('text', '')[:50]}"
            if dedup_key not in seen:
                seen.add(dedup_key)
                unique_results.append(r)
        top_results = unique_results[:top_k]  # 取前top_k个结果

        # 强制包含关键词补充结果：确保包含精准关键词的chunk不被向量高分结果挤出
        keyword_supplement_set = {id(r) for r in (keyword_boost or [])}
        if keyword_boost:
            seen_pages_kw = {r.get('page_num') for r in top_results}
            for r in unique_results[top_k:]:
                if id(r) in keyword_supplement_set and r.get('page_num') not in seen_pages_kw:
                    top_results.append(r)

        # 强制包含图片/表格数据块：查询含图表类关键词时，确保相关图片块不被遗漏
        MEDIA_KEYWORDS = ['图表', '增长图', '结构图', '饼图', '条形图', '占比', '增长率', '应用结构', '市场规模']
        query_for_media = original_query or query
        if any(kw in query_for_media for kw in MEDIA_KEYWORDS):
            seen_pages = {r.get('page_num') for r in top_results}
            seen_types = set()
            for r in top_results:
                st = r.get('source_type', 'text')
                if st in ('image', 'table'):
                    seen_types.add(st)
            for r in unique_results[top_k:]:
                if r.get('source_type', 'text') in ('image', 'table') and r.get('page_num') not in seen_pages:
                    top_results.append(r)  # 补充图片/表格块到结果中

        if not top_results:  # 无结果时
            return "", []  # 返回空结果

        return self._format_context(top_results), top_results  # 返回格式化上下文和结果列表

    def _keyword_supplement_search(self, query: str, target_docs: List[str]) -> List[Dict]:
        """
        关键词补充检索：使用加权关键词字典，直接在CSV中搜索包含这些短语的chunk
        解决向量搜索遗漏包含精确数据的chunk的问题（如图表中的行业增长率数据）
        """
        # 加权关键词字典：精准关键词权重3.0，宽泛关键词权重1.0
        KEYWORD_WEIGHTS = {
            # 精准关键词（权重3.0）— 直接指向答案
            '募集资金': 3.0, '补充流动资金': 3.0, '流动资金': 3.0,
            '上游': 3.0, '下游': 3.0, '上游企业': 3.0, '下游行业': 3.0,
            '电子信息行业': 3.0, '电子信息': 2.0,
            # 中等关键词（权重2.0）— 与答案相关
            'IC市场': 2.0, '应用结构': 2.0, '增长率': 2.0,
            '市场规模': 2.0, '市场应用': 2.0,
            # 宽泛关键词（权重1.0）— 需要多个匹配
            '行业': 1.0, '领域': 1.0, '企业': 1.0,
            '条形图': 1.0, '图表': 1.0, '数据': 1.0,
            '比例': 1.0, '占比': 1.0,
            '负增长': 1.0, '最高': 1.0, '最快': 1.0, '最低': 1.0,
        }
        # 从查询中提取匹配的关键词并计算加权分数
        weighted_score = 0
        matched_keywords = []
        for kw, weight in KEYWORD_WEIGHTS.items():
            if kw in query:
                weighted_score += weight
                matched_keywords.append(kw)
        # 加权阈值1.5：精准关键词直接触发（如"募集资金"=3.0），宽泛词需多个组合
        if weighted_score < 1.5:
            return []

        supplement_results = []
        seen_texts = set()

        for doc_name in target_docs:
            engine = self.doc_engines.get(doc_name)
            if not engine:
                continue
            for chunk in engine.chunks:
                text = chunk.get('content', '')
                text_prefix = text[:50]
                if text_prefix in seen_texts:
                    continue
                # 计算chunk与查询关键词的加权匹配分数
                chunk_score = 0
                for kw in matched_keywords:
                    if kw in text:
                        chunk_score += KEYWORD_WEIGHTS[kw]
                if chunk_score >= 1.5:  # chunk至少匹配到一个精准关键词或多个宽泛词
                    seen_texts.add(text_prefix)
                    # 图片/表格块含图表具体数据，给予更高加分（向量检索容易漏掉）
                    is_media = chunk.get('source_type', 'text') in ('image', 'table')
                    boost_score = 0.50 if is_media else (0.15 + 0.05 * min(chunk_score, 5))
                    supplement_results.append({
                        'text': text,
                        'page_num': chunk.get('page_num', 0),
                        'score': round(boost_score, 4),
                        'source_type': chunk.get('source_type', 'text'),
                        'doc_name': doc_name,
                        'source': 'keyword_supplement',
                    })

        if supplement_results:
            logger.info(f"[关键词补充] 查询 '{query[:30]}...' -> 匹配关键词{matched_keywords} -> 补充 {len(supplement_results)} 条结果")
        return supplement_results

    @staticmethod
    def _format_context(results: List[Dict]) -> str:
        """将检索结果格式化为上下文字符串"""
        parts = []  # 初始化片段列表
        for r in results:  # 遍历每个检索结果
            doc_info = f"[{r.get('doc_name', '')}]" if r.get('doc_name') else ""  # 文档来源标记
            parts.append(f"{doc_info}[来源：第{r['page_num']}页]\n{r['text']}")  # 添加来源页码和文本
        return "\n\n".join(parts)  # 用双换行连接所有片段

    def build_prompt(self, question: str, context: str, lang: str = 'zh') -> str:
        """构建用于LLM生成的Prompt，支持中英文"""
        if not context:
            if lang == 'en':
                return f"Please answer the following question (no relevant document content found): {question}"
            return f"请回答问题（未检索到相关文档内容）：{question}"

        if lang == 'en':
            return (
                f"You are an intelligent Q&A assistant based on PDF documents. "
                f"Please accurately answer the user's question based on the following document content retrieved from the Prospectus. "
                f"If the document content is insufficient to answer the question, please honestly say you don't know. Do not fabricate. "
                f"IMPORTANT: You must respond in English, not Chinese.\n\n"
                f"Retrieved document content:\n{context}\n\n"
                f"User question: {question}\n\n"
                f"Please provide an accurate and concise answer based on the above document content. "
                f"If the question involves data, please directly cite the data from the original text."
            )

        return (
            f"你是一个基于PDF文档的智能问答助手。请根据以下从《招股意向书》中检索到的文档内容，"
            f"准确回答用户的问题。如果文档内容不足以回答问题，请如实说不知道，不要编造。\n\n"
            f"重要提示：检索内容包含三种类型：\n"
            f"1. [text] 文本段落——直接的文字内容\n"
            f"2. [table] 表格数据——结构化的数据表格\n"
            f"3. [图片] 图片描述——对PDF中图表/图片的文字描述，包含图表中的具体数据（如增长率、百分比等），请充分利用这些数据回答问题\n\n"
            f"检索到的文档内容：\n{context}\n\n"
            f"用户问题：{question}\n\n"
            f"请基于以上文档内容给出准确、简洁的回答。如果涉及数据，请直接引用原文中的数据。"
        )

    def query(self, user_query: str, top_k=25) -> Dict:
        """完整问答流程"""
        u_result = self.query_understand.understand(user_query)  # 对用户问题进行语义理解
        final_q = u_result["expanded_query"]  # 获取扩展后的查询语句
        context_str, results = self.get_context(final_q, top_k, original_query=user_query)  # 检索相关上下文（路由用原始查询）

        if not context_str:  # 未检索到上下文时
            return {  # 返回无结果响应
                "answer": "抱歉，在文档中没有找到与您问题相关的内容。",  # 默认回复
                "context": [],  # 上下文为空
                "query_info": u_result,  # 查询理解信息
            }

        prompt = self.build_prompt(user_query, context_str)  # 构建Prompt
        answer = self.llm_client.generate(prompt, context_str, user_query)  # 调用LLM生成回答

        contexts = [f"[{r.get('doc_name', '')} 第{r['page_num']}页] {r['text'][:200]}" for r in results]  # 格式化上下文摘要

        return {  # 返回完整问答结果
            "answer": answer,  # LLM生成的回答
            "context": contexts,  # 上下文摘要
            "results": results,  # 检索结果详情
            "query_info": u_result,  # 查询理解信息
        }
