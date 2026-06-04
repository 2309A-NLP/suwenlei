# -*- coding: utf-8 -*-
"""
RAG引擎核心模块 — 支持多文档路由检索

架构说明：
- 每个PDF文档维护独立的：CSV分块、TF-IDF模型、Milvus集合
- 查询路由：分析问题内容 → 判断在哪个文档中检索 → 返回结果
"""
import os
import re
import csv
import logging
from typing import List, Dict, Tuple, Optional

from llm_client import LLMClient
from query_understanding import QueryUnderstanding

logger = logging.getLogger(__name__)

# 中文停用词表：过滤虚词，提升TF-IDF对实义词的权重
CHINESE_STOP_WORDS = set([
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
])


class DocumentEngine:
    """单文档检索引擎：维护一个PDF的分块数据和bge-m3向量模型"""

    def __init__(self, doc_name: str, csv_path: str):
        self.doc_name = doc_name
        self.csv_path = csv_path
        self.chunks = []
        self.tfidf = None       # TF-IDF向量化器（降级备用）
        self.tfidf_mat = None   # 文档TF-IDF矩阵（降级备用）
        self._load_csv()
        self._init_embedder()

    def _load_csv(self):
        if not os.path.exists(self.csv_path):
            logger.warning(f"[DocEngine] CSV不存在: {self.csv_path}")
            return
        try:
            with open(self.csv_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.chunks.append({
                        "content": row.get('text', ''),
                        "page_num": int(row.get('page_num', 0)),
                        "chunk_idx": int(row.get('id', row.get('chunk_index', 0))),
                        "source_type": row.get('source_type', 'text'),
                    })
            logger.info(f"[DocEngine] {self.doc_name}: 加载{len(self.chunks)}个分块")
        except Exception as e:
            logger.error(f"[DocEngine] CSV加载失败: {e}")

    def _init_embedder(self):
        """初始化bge-m3向量模型（主检索）+ TF-IDF（降级备用）"""
        if not self.chunks:
            return
        texts = [c["content"] for c in self.chunks]
        # 训练TF-IDF备用检索（Milvus不可用时降级使用）
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            import jieba

            def jieba_tokenizer(text):
                words = list(jieba.cut(text))
                return [w for w in words if len(w) > 1 and w not in CHINESE_STOP_WORDS]

            self.tfidf = TfidfVectorizer(max_features=5000, tokenizer=jieba_tokenizer)
            self.tfidf_mat = self.tfidf.fit_transform(texts)
            logger.info(f"[DocEngine] {self.doc_name}: TF-IDF备用索引({len(self.tfidf.vocabulary_)}词)")
        except Exception as e:
            logger.warning(f"[DocEngine] {self.doc_name} TF-IDF备用索引训练失败: {e}")
        # 加载bge-m3模型（主向量化）
        try:
            from embedder import get_embedder
            embedder = get_embedder()
            sample = embedder.embed_query("test")
            logger.info(f"[DocEngine] {self.doc_name}: bge-m3已就绪, 维度={len(sample)}")
        except Exception as e:
            logger.error(f"[DocEngine] {self.doc_name} bge-m3加载失败: {e}")

    def generate_embeddings(self, texts: List[str]):
        """使用bge-m3将文本转为1024维向量"""
        try:
            from embedder import get_embedder
            return get_embedder().embed_texts(texts)
        except Exception as e:
            logger.error(f"[DocEngine] bge-m3编码失败: {e}")
            return []

    def tfidf_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """TF-IDF余弦检索 + 关键词命中加权重排序（提升精确匹配分数）"""
        query_keywords = [w for w in re.findall(r'[\u4e00-\u9fff]{2,}', query)
                          if w not in CHINESE_STOP_WORDS]
        if not self.chunks or self.tfidf is None:
            return []

        from sklearn.metrics.pairwise import cosine_similarity
        q_vec = self.tfidf.transform([query])
        scores = cosine_similarity(q_vec, self.tfidf_mat).flatten()

        # 扩大候选集（5倍），用于二次重排序
        candidate_count = min(top_k * 5, len(scores))
        candidate_indices = scores.argsort()[::-1][:candidate_count]

        scored_candidates = []
        for idx in candidate_indices:
            if scores[idx] <= 0:
                continue
            c = self.chunks[idx]
            text = c["content"]
            base_score = float(scores[idx])
            boost = 0.0

            # 关键词命中加权：精确出现+0.08，高频出现额外+0.15/次
            for kw in query_keywords:
                if kw in text:
                    boost += 0.08
                count = text.count(kw)
                if count >= 3:
                    boost += 0.15 * min(count, 5)

            # 关键词拼接后完整出现：强力加权+0.3
            query_clean = ''.join(query_keywords)
            if len(query_clean) > 2 and query_clean in text:
                boost += 0.3

            scored_candidates.append((idx, base_score + boost))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, final_score in scored_candidates[:top_k]:
            c = self.chunks[idx]
            results.append({
                "text": c["content"],
                "page_num": c["page_num"],
                "score": round(final_score, 4),
                "source_type": c.get("source_type", "text"),
                "doc_name": self.doc_name,
            })
        return results


class RAGEngine:
    """RAG引擎 — 多文档路由检索：路由→检索→补全→去重→Prompt构建"""

    def __init__(self, pdf_paths: Dict[str, str] = None, backend="auto"):
        self.pdf_paths = pdf_paths or {}
        self.backend = backend
        self.doc_engines = {}       # doc_name → DocumentEngine 映射
        self.milvus_store = self._init_milvus()  # Milvus可用则用向量检索
        self.llm_client = LLMClient()
        self.query_understand = QueryUnderstanding()
        if pdf_paths:
            self.init_rag()

    @staticmethod
    def _init_milvus():
        """Milvus初始化：连接失败时返回None，后续自动降级为TF-IDF"""
        try:
            from milvus_store import MilvusStore
            return MilvusStore()
        except Exception as e:
            logger.warning(f"MilvusStore 初始化失败: {e}")
            return None

    @staticmethod
    def _csv_path_for(pdf_path: str) -> str:
        """根据PDF路径推导分块CSV路径（约定命名规则）"""
        base = os.path.splitext(pdf_path)[0]
        if base.endswith('_chunks'):
            return f"{base}_v2.csv"
        return f"{base}_chunks_v2.csv"

    def init_rag(self):
        """为每个PDF文档创建独立的检索引擎实例"""
        logger.info("正在初始化RAG引擎（多文档模式）...")
        for doc_name, pdf_path in self.pdf_paths.items():
            csv_path = self._csv_path_for(pdf_path)
            # 兜底：若CSV不在PDF同目录，尝试从uploads目录匹配
            if not os.path.exists(csv_path):
                uploads_dir = os.path.join(os.path.dirname(pdf_path), 'uploads')
                if os.path.isdir(uploads_dir):
                    for f in os.listdir(uploads_dir):
                        if f.endswith('_chunks_v2.csv') and doc_name.replace('招股说明书', '') in f:
                            csv_path = os.path.join(uploads_dir, f)
                            break
            if os.path.exists(csv_path):
                engine = DocumentEngine(doc_name, csv_path)
                if engine.chunks:
                    self.doc_engines[doc_name] = engine
                    logger.info(f"文档引擎初始化完成: {doc_name} ({len(engine.chunks)}块)")
            else:
                logger.warning(f"文档 {doc_name} 的CSV不存在: {csv_path}")
        logger.info(f"RAG引擎初始化完成: {len(self.doc_engines)}个文档")

    def _route_query(self, query: str) -> List[str]:
        """查询路由策略：强匹配（公司名）→ 弱匹配（行业特征）→ 全文档搜索"""
        # 强关键词：精确匹配公司名/人名，命中即锁定单一文档
        STRONG_KEYWORDS = {
            '招股说明书1': ['兴图', '兴图新科', '武汉兴图', '程家明', 'C4ISR', '视频指挥'],
            '招股说明书2': ['力源', '力源信息', '武汉力源', '赵马克', 'Mark Zhao', '融冰投资'],
        }
        # 弱关键词：行业/业务特征词，用于模糊路由（多个文档可能同时命中）
        WEAK_KEYWORDS = {
            '招股说明书1': ['军用', '国防', '军队', '军事', '军方', '技术规范',
                          '法定代表人', '注册资本', '上游', '下游', '行业',
                          '补充流动资金', '募集资金', '国家科技进步'],
            '招股说明书2': ['仓储物流', '电子商务平台', '扩充产品', '关联方',
                          '关联交易', '控制关系', '博润', '听音投资'],
        }

        # 第一层：强匹配优先（精确锁定文档，避免跨文档噪声）
        strong_matches = []
        for doc_name, keywords in STRONG_KEYWORDS.items():
            if doc_name not in self.doc_engines:
                continue
            for kw in keywords:
                if kw in query:
                    strong_matches.append(doc_name)
                    break
        if strong_matches:
            logger.info(f"[路由] '{query[:30]}...' → 强匹配 {strong_matches}")
            return strong_matches

        # 第二层：弱匹配（根据业务特征词缩小检索范围）
        weak_matches = []
        for doc_name, keywords in WEAK_KEYWORDS.items():
            if doc_name not in self.doc_engines:
                continue
            for kw in keywords:
                if kw in query:
                    weak_matches.append(doc_name)
                    break
        if weak_matches:
            logger.info(f"[路由] '{query[:30]}...' → 弱匹配 {weak_matches}")
            return weak_matches

        # 第三层：无匹配时搜全部文档（保底策略）
        all_docs = list(self.doc_engines.keys())
        logger.info(f"[路由] '{query[:30]}...' → 全部 {all_docs}")
        return all_docs

    def get_context(self, query: str, top_k=5, original_query: str = None) -> Tuple[str, List[Dict]]:
        """检索流程：路由选择→向量/TF-IDF检索→关键词补全→去重排序"""
        route_query = original_query if original_query else query
        target_docs = self._route_query(route_query)  # 路由决策
        all_results = []

        for doc_name in target_docs:
            engine = self.doc_engines.get(doc_name)
            if not engine:
                continue
            # 检索策略：Milvus向量检索优先，失败降级TF-IDF关键词检索
            if self.milvus_store and self.milvus_store.is_available():
                try:
                    embeddings = engine.generate_embeddings([query])
                    if embeddings:
                        hits = self.milvus_store.search(embeddings[0], top_k * 3, doc_name=doc_name)
                        if hits:
                            for h in hits:
                                all_results.append({
                                    "text": h.entity.get("content", ""),
                                    "page_num": h.entity.get("page_num", 0),
                                    "score": round(h.score, 4),
                                    "source_type": h.entity.get("source_type", "text"),
                                    "doc_name": doc_name,
                                })
                            continue  # Milvus命中则跳过TF-IDF
                except Exception as e:
                    logger.warning(f"[{doc_name}] Milvus检索失败，降级TF-IDF: {e}")

            # 降级路径：TF-IDF+余弦相似度+关键词加权重排序
            results = engine.tfidf_search(query, top_k * 2)
            all_results.extend(results)

        # 关键词补全检索：直接在CSV中搜索含精确数据的chunk（补充向量检索遗漏）
        keyword_boost = self._keyword_supplement_search(original_query or query, target_docs)
        if keyword_boost:
            all_results.extend(keyword_boost)

        # 按分数降序排序
        all_results.sort(key=lambda r: r['score'], reverse=True)
        # 去重：基于页码+文本前缀
        seen = set()
        unique_results = []
        for r in all_results:
            dedup_key = f"{r.get('page_num', '')}_{r.get('text', '')[:50]}"
            if dedup_key not in seen:
                seen.add(dedup_key)
                unique_results.append(r)
        top_results = unique_results[:top_k]

        if not top_results:
            return "", []
        return self._format_context(top_results), top_results

    def _keyword_supplement_search(self, query: str, target_docs: List[str]) -> List[Dict]:
        """关键词补全检索：固定关键词+查询实体词双通道扫描CSV"""
        # 固定数据类关键词：匹配含数值/表格的chunk
        DATA_KEYWORDS = [
            'IC市场', '应用结构', '增长率', '条形图', '图表',
            '负增长', '最高', '最快', '最低',
            '收入', '销售额', '利润', '营收', '占比', '比重',
            '募集资金', '发行股数', '注册资本', '流动资金',
            '军用', '国防', '军队', '军事',
            '法定代表人', '实际控制人', '控股股东',
            '行业', '领域', '数据', '比例',
            '组织结构图', '股权结构图', '组织架构', '股权结构',
            '公司组织', '部门结构', '治理结构',
        ]
        # 第一通道：固定关键词匹配
        matched_keywords = [kw for kw in DATA_KEYWORDS if kw in query]

        # 第二通道：用jieba分词提取查询中的实体词，补充搜索
        try:
            import jieba
            query_entities = [w for w in jieba.cut(query)
                              if len(w) >= 2 and w not in CHINESE_STOP_WORDS
                              and not re.match(r'^[\d\.\-]+$', w)]
        except ImportError:
            # jieba不可用时降级为regex（取3+字避免公司名子串）
            query_entities = [w for w in re.findall(r'[\u4e00-\u9fff]{3,}', query)
                              if w not in CHINESE_STOP_WORDS]
        # 过滤掉公司名子串（"力源"、"信息技术"等在每页都出现，无筛选价值）
        COMPANY_NAMES = ['武汉力源信息技术股份有限公司', '武汉兴图新科电子股份有限公司',
                         '力源信息', '兴图新科', '力源有限']
        query_entities = [e for e in query_entities
                          if not any(e in cn or cn.startswith(e) for cn in COMPANY_NAMES)]
        # 合并两通道关键词（去重）
        all_keywords = list(dict.fromkeys(matched_keywords + query_entities))
        if not all_keywords:
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
                # 统计关键词命中数：固定关键词权重×2 + 实体词权重×1
                fixed_hits = sum(1 for kw in matched_keywords if kw in text)
                entity_hits = sum(1 for kw in query_entities if kw in text)
                keyword_match_count = fixed_hits * 2 + entity_hits
                # 至少命中2个实体词（或1个固定关键词）才纳入
                if keyword_match_count >= 2 or fixed_hits >= 1:
                    seen_texts.add(text_prefix)
                    boost_score = 0.6 + 0.05 * keyword_match_count
                    supplement_results.append({
                        'text': text,
                        'page_num': chunk.get('page_num', 0),
                        'score': round(boost_score, 4),
                        'source_type': chunk.get('source_type', 'text'),
                        'doc_name': doc_name,
                        'source': 'keyword_supplement',
                    })

        if supplement_results:
            logger.info(f"[关键词补充] '{query[:30]}...' → {len(supplement_results)} 条")
        return supplement_results

    @staticmethod
    def _format_context(results: List[Dict]) -> str:
        """将检索结果格式化为Prompt可用的上下文文本"""
        parts = []
        for r in results:
            doc_info = f"[{r.get('doc_name', '')}]" if r.get('doc_name') else ""
            parts.append(f"{doc_info}[来源：第{r['page_num']}页]\n{r['text']}")
        return "\n\n".join(parts)

    def build_prompt(self, question: str, context: str, history: list = None, lang: str = 'zh') -> str:
        """构建LLM Prompt：注入对话历史+检索上下文，支持多轮追问指代消解，支持中英文"""
        if not context:
            if lang == 'en':
                return f"Please answer the following question (no relevant document content found): {question}"
            return f"请回答问题（未检索到相关文档内容）：{question}"

        # 英文模式Prompt
        if lang == 'en':
            history_block = ""
            if history and len(history) > 0:
                recent = history[-6:]
                lines = []
                for msg in recent:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    lines.append(f"{role}: {msg['content'][:150]}")
                history_block = "Recent conversation history:\n" + "\n".join(lines) + "\n\n"
            return (
                f"You are an intelligent Q&A assistant based on PDF documents. "
                f"Please accurately answer the user's question based on the following document content retrieved from the Prospectus. "
                f"If the document content is insufficient to answer the question, please honestly say you don't know. Do not fabricate.\n\n"
                f"IMPORTANT: You must respond in English, not Chinese.\n\n"
                f"{history_block}"
                f"Retrieved document content:\n{context}\n\n"
                f"User question: {question}\n\n"
                f"Please provide an accurate and concise answer based on the above document content. "
                f"If the question involves data, please directly cite the data from the original text."
            )

        # 拼接最近3轮对话历史（6条消息），帮助LLM理解"这个公司"等指代
        history_block = ""
        if history and len(history) > 0:
            recent = history[-6:]
            lines = []
            for msg in recent:
                role = "用户" if msg["role"] == "user" else "助手"
                lines.append(f"{role}：{msg['content'][:150]}")
            history_block = "最近的对话历史：\n" + "\n".join(lines) + "\n\n"

        return (
            f"你是一个基于PDF文档的智能问答助手。请根据以下从《招股意向书》中检索到的文档内容，"
            f"准确回答用户的问题。如果文档内容不足以回答问题，请如实说不知道，不要编造。\n\n"
            f"{history_block}"
            f"重要提示：\n"
            f"1. 如果用户的问题是追问（如'这个公司'、'那XX呢'、'它的XX'），请结合上方的对话历史理解用户的完整意图，不要忽略省略的部分。\n"
            f"2. 检索内容包含三种类型：\n"
            f"   [text] 文本段落——直接的文字内容\n"
            f"   [table] 表格数据——结构化的数据表格\n"
            f"   [图片] 图片描述——对PDF中图表/图片的文字描述，包含图表中的具体数据，请充分利用\n\n"
            f"检索到的文档内容：\n{context}\n\n"
            f"用户问题：{question}\n\n"
            f"请基于以上文档内容和对话历史给出准确、简洁的回答。如果涉及数据，请直接引用原文中的数据。"
        )
