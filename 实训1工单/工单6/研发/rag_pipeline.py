# -*- coding: utf-8 -*-
"""
RAG引擎核心模块 — 混合检索：向量检索 + BM25全文检索 + RRF融合 + bge-reranker重排
工单编号: 人工智能NLP-RAG-混合检索任务

架构：
1. 向量检索：bge-m3生成1024维向量 → Milvus余弦相似度召回
2. 全文检索：jieba分词 → BM25倒排索引关键词召回
3. 混合融合：RRF(Reciprocal Rank Fusion)合并两路结果
4. 重排序：bge-reranker-v2-m3交叉编码器精细打分
5. 可选LLM重排：DeepSeek API对候选文档重新评分
"""
import os
import re
import csv
import logging
from typing import List, Dict, Tuple, Optional

from llm import LLMClient, QueryUnderstanding

logger = logging.getLogger(__name__)

# 中文停用词表（BM25和关键词补全共用）
CHINESE_STOP_WORDS = set([
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
    '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
    '没有', '看', '好', '自己', '这', '他', '她', '它', '们', '那', '些',
    '与', '及', '或', '对', '被', '把', '从', '向', '以', '为', '由',
    '于', '而', '但', '且', '之', '其', '所', '者', '了', '过', '将',
    '让', '使', '能', '可', '得', '已', '还', '又', '再', '才', '则',
    '等', '如', '若', '虽', '因', '故', '并', '非', '即', '既', '各',
    '每', '某', '该', '本', '哪', '何', '么', '吗', '呢', '吧', '啊',
    '哦', '嗯', '呀', '哈', '嘛', '样', '种', '些', '点', '多',
    '少', '个', '只', '第', '年', '月', '日', '元', '万', '亿',
    '涉及', '包括', '通过', '进行', '实现', '提供', '取得', '分别',
    '相关', '上述', '其中', '以及', '报告', '期内', '来自',
])


def _rrf_fusion(rankings: List[List[Dict]], k: int = 60) -> List[Dict]:
    """RRF(Reciprocal Rank Fusion)融合多路检索结果

    公式: RRF_score(d) = Σ 1/(k + rank_i(d))
    k=60是论文推荐值，平衡头部和尾部文档的权重

    参数:
        rankings: 多路检索结果列表，每个元素是按分数降序排列的结果列表
        k: RRF常数，默认60
    返回:
        融合后的去重结果列表，按RRF分数降序
    """
    doc_scores = {}  # dedup_key → (rrf_score, doc)
    doc_count = {}   # dedup_key → 命中路数（用于boost）

    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            # 去重键：文本前80字符
            dedup_key = doc.get('text', '')[:80]
            if not dedup_key:
                continue
            rrf_score = 1.0 / (k + rank + 1)  # rank从0开始
            if dedup_key in doc_scores:
                doc_scores[dedup_key] = (doc_scores[dedup_key][0] + rrf_score, doc)
                doc_count[dedup_key] = doc_count.get(dedup_key, 1) + 1
            else:
                doc_scores[dedup_key] = (rrf_score, doc)
                doc_count[dedup_key] = 1

    # 多路命中的文档获得bonus加权（在两路都出现的文档更可靠）
    results = []
    for dedup_key, (score, doc) in doc_scores.items():
        count = doc_count.get(dedup_key, 1)
        # 每多命中一路，RRF分数×1.3
        final_score = score * (1.3 ** (count - 1))
        doc['rrf_score'] = round(final_score, 6)
        doc['hit_count'] = count
        results.append(doc)

    results.sort(key=lambda x: x['rrf_score'], reverse=True)
    return results


class DocumentEngine:
    """单文档检索引擎：维护CSV分块数据和BM25索引"""

    def __init__(self, doc_name: str, csv_path: str):
        self.doc_name = doc_name
        self.csv_path = csv_path
        self.chunks = []           # CSV分块数据
        self.bm25_index = None     # BM25索引实例
        self._load_csv()
        self._build_bm25()

    def _load_csv(self):
        """从CSV加载分块数据"""
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
                        "doc_name": self.doc_name,
                    })
            logger.info(f"[DocEngine] {self.doc_name}: 加载{len(self.chunks)}个分块")
        except Exception as e:
            logger.error(f"[DocEngine] CSV加载失败: {e}")

    def _build_bm25(self):
        """构建BM25索引：jieba分词+倒排索引"""
        if not self.chunks:
            return
        try:
            from search import BM25Index
            # 构建BM25索引：每个chunk的text字段作为检索目标
            bm25_chunks = []
            for c in self.chunks:
                bm25_chunks.append({
                    'text': c['content'],
                    'page_num': c['page_num'],
                    'source_type': c['source_type'],
                    'doc_name': self.doc_name,
                })
            self.bm25_index = BM25Index()
            self.bm25_index.build(bm25_chunks)
        except ImportError:
            logger.warning("[DocEngine] bm25_search模块不可用，BM25检索禁用")
        except Exception as e:
            logger.error(f"[DocEngine] BM25索引构建失败: {e}")

    def bm25_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """BM25全文检索"""
        if self.bm25_index is None:
            return []
        return self.bm25_index.search(query, top_k=top_k)


class RAGEngine:
    """RAG引擎 — 混合检索：路由→向量检索+BM25→RRF融合→重排→Prompt构建"""

    def __init__(self, pdf_paths: Dict[str, str] = None, backend="auto"):
        self.pdf_paths = pdf_paths or {}
        self.backend = backend
        self.doc_engines = {}       # doc_name → DocumentEngine 映射
        self.milvus_store = self._init_milvus()
        self.llm_client = LLMClient()
        self.query_understand = QueryUnderstanding()

        # 混合检索配置
        self.search_mode = 'hybrid'  # vector/bm25/hybrid
        self.vector_weight = 0.6     # 向量检索权重（hybrid模式下）
        self.bm25_weight = 0.4       # BM25检索权重
        self.reranker_top_k = 10     # 重排器候选数（减少GPU推理量）
        self.final_top_k = 5         # 最终返回数

        # 延迟加载的嵌入和重排模型
        self._embedder = None
        self._reranker = None

        if pdf_paths:
            self.init_rag()

    def _get_embedder(self):
        """懒加载bge-m3嵌入模型"""
        if self._embedder is None:
            try:
                from models import EmbeddingClient
                self._embedder = EmbeddingClient()
            except Exception as e:
                logger.error(f"EmbeddingClient加载失败: {e}")
        return self._embedder

    def _get_reranker(self):
        """懒加载bge-reranker-v2-m3重排模型"""
        if self._reranker is None:
            try:
                from models import RerankerClient
                self._reranker = RerankerClient()
            except Exception as e:
                logger.error(f"RerankerClient加载失败: {e}")
        return self._reranker

    @staticmethod
    def _init_milvus():
        """Milvus初始化"""
        try:
            from search import MilvusStore
            return MilvusStore()
        except Exception as e:
            logger.warning(f"MilvusStore 初始化失败: {e}")
            return None

    @staticmethod
    def _csv_path_for(pdf_path: str) -> str:
        """根据PDF路径推导CSV路径"""
        base = os.path.splitext(pdf_path)[0]
        if base.endswith('_chunks'):
            return f"{base}_v2.csv"
        return f"{base}_chunks_v2.csv"

    def init_rag(self):
        """为每个PDF创建独立的检索引擎"""
        logger.info("正在初始化RAG引擎（混合检索模式）...")
        for doc_name, pdf_path in self.pdf_paths.items():
            csv_path = self._csv_path_for(pdf_path)
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

    def set_search_mode(self, mode: str, vector_weight: float = None, bm25_weight: float = None):
        """配置检索模式和权重"""
        if mode in ('vector', 'bm25', 'hybrid'):
            self.search_mode = mode
        if vector_weight is not None:
            self.vector_weight = max(0.0, min(1.0, vector_weight))
        if bm25_weight is not None:
            self.bm25_weight = max(0.0, min(1.0, bm25_weight))
        # 归一化权重
        total = self.vector_weight + self.bm25_weight
        if total > 0:
            self.vector_weight /= total
            self.bm25_weight /= total
        logger.info(f"[检索配置] 模式={self.search_mode}, 向量权重={self.vector_weight:.2f}, BM25权重={self.bm25_weight:.2f}")

    def _route_query(self, query: str) -> List[str]:
        """查询路由：强匹配（公司名）→ 弱匹配（行业特征）→ 全文档"""
        STRONG_KEYWORDS = {
            '招股说明书1': ['兴图', '兴图新科', '武汉兴图', '程家明', 'C4ISR', '视频指挥'],
            '招股说明书2': ['力源', '力源信息', '武汉力源', '赵马克', 'Mark Zhao', '融冰投资'],
        }
        WEAK_KEYWORDS = {
            '招股说明书1': ['军用', '国防', '军队', '军事', '军方', '技术规范',
                          '法定代表人', '注册资本', '上游', '下游', '行业',
                          '补充流动资金', '募集资金', '国家科技进步'],
            '招股说明书2': ['仓储物流', '电子商务平台', '扩充产品', '关联方',
                          '关联交易', '控制关系', '博润', '听音投资'],
        }

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

        all_docs = list(self.doc_engines.keys())
        logger.info(f"[路由] '{query[:30]}...' → 全部 {all_docs}")
        return all_docs

    def _vector_search(self, query: str, target_docs: List[str], top_k: int) -> List[Dict]:
        """向量检索：bge-m3向量化 → Milvus余弦相似度搜索"""
        import time
        embedder = self._get_embedder()
        if embedder is None or not self.milvus_store or not self.milvus_store.is_available():
            return []

        t_enc = time.time()
        query_emb = embedder.encode_query(query)
        logger.info(f"[向量编码] {time.time()-t_enc:.2f}s (dim={len(query_emb) if query_emb is not None else 0})")
        if query_emb is None:
            return []

        t_search = time.time()
        all_results = []
        for doc_name in target_docs:
            try:
                hits = self.milvus_store.search(query_emb, top_k, doc_name=doc_name)
                for h in hits:
                    all_results.append({
                        "text": h.entity.get("content", ""),
                        "page_num": h.entity.get("page_num", 0),
                        "score": round(h.score, 4),
                        "source_type": h.entity.get("source_type", "text"),
                        "doc_name": doc_name,
                        "vector_score": round(h.score, 4),
                    })
            except Exception as e:
                logger.warning(f"[向量检索] {doc_name} 失败: {e}")

        logger.info(f"[Milvus搜索] {time.time()-t_search:.2f}s, 命中{len(all_results)}条")
        all_results.sort(key=lambda r: r['score'], reverse=True)
        return all_results[:top_k]

    def _bm25_search_all(self, query: str, target_docs: List[str], top_k: int) -> List[Dict]:
        """BM25全文检索：对所有目标文档执行BM25搜索"""
        all_results = []
        for doc_name in target_docs:
            engine = self.doc_engines.get(doc_name)
            if not engine:
                continue
            results = engine.bm25_search(query, top_k=top_k)
            all_results.extend(results)

        all_results.sort(key=lambda r: r.get('bm25_score', r.get('score', 0)), reverse=True)
        return all_results[:top_k]

    def _keyword_supplement_search(self, query: str, target_docs: List[str]) -> List[Dict]:
        """关键词补全检索：固定关键词+查询实体词扫描CSV"""
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
        matched_keywords = [kw for kw in DATA_KEYWORDS if kw in query]

        try:
            import jieba
            query_entities = [w for w in jieba.cut(query)
                              if len(w) >= 2 and w not in CHINESE_STOP_WORDS
                              and not re.match(r'^[\d\.\-]+$', w)]
        except ImportError:
            query_entities = [w for w in re.findall(r'[\u4e00-\u9fff]{3,}', query)
                              if w not in CHINESE_STOP_WORDS]

        COMPANY_NAMES = ['武汉力源信息技术股份有限公司', '武汉兴图新科电子股份有限公司',
                         '力源信息', '兴图新科', '力源有限']
        query_entities = [e for e in query_entities
                          if not any(e in cn or cn.startswith(e) for cn in COMPANY_NAMES)]

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
                fixed_hits = sum(1 for kw in matched_keywords if kw in text)
                entity_hits = sum(1 for kw in query_entities if kw in text)
                keyword_match_count = fixed_hits * 2 + entity_hits
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

    def get_context(self, query: str, top_k=5, original_query: str = None) -> Tuple[str, List[Dict]]:
        """混合检索主流程：路由→向量+BM25→RRF融合→重排→关键词补全"""
        import time
        t0 = time.time()
        route_query = original_query if original_query else query
        target_docs = self._route_query(route_query)

        # 根据检索模式执行
        t1 = time.time()
        if self.search_mode == 'vector':
            results = self._vector_search(query, target_docs, top_k=self.reranker_top_k)
        elif self.search_mode == 'bm25':
            results = self._bm25_search_all(query, target_docs, top_k=self.reranker_top_k)
        else:
            vector_results = self._vector_search(query, target_docs, top_k=self.reranker_top_k)
            bm25_results = self._bm25_search_all(query, target_docs, top_k=self.reranker_top_k)
            if vector_results or bm25_results:
                results = _rrf_fusion([vector_results, bm25_results])
            else:
                results = []
        t2 = time.time()

        # bge-reranker重排序（仅对检索结果，不含关键词补全）
        rerank_used = False
        t_r0 = time.time()
        if results:
            reranker = self._get_reranker()
            t_r1 = time.time()
            logger.info(f"[reranker] 获取模型={t_r1-t_r0:.2f}s 候选数={len(results)}")
            if reranker and reranker.is_available():
                t_r2 = time.time()
                results = reranker.rerank(query, results, top_k=top_k)
                t_r3 = time.time()
                logger.info(f"[reranker] predict={t_r3-t_r2:.2f}s")
                rerank_used = True
            else:
                results.sort(key=lambda r: r.get('rrf_score', r.get('score', 0)), reverse=True)
                results = results[:top_k]
        t3 = time.time()

        # 关键词补全（rerank之后，不送reranker，避免194条chunk拖慢53秒）
        keyword_boost = self._keyword_supplement_search(original_query or query, target_docs)
        if keyword_boost:
            results.extend(keyword_boost)
        t4 = time.time()
        logger.info(f"[计时] 检索={t2-t1:.2f}s 重排={t3-t2:.2f}s 关键词补全={t4-t3:.2f}s 总计={t4-t0:.2f}s (reranker={'Y' if rerank_used else 'N'})")

        if not results:
            return "", []

        # 去重
        seen = set()
        unique_results = []
        for r in results:
            dedup_key = f"{r.get('page_num', '')}_{r.get('text', '')[:50]}"
            if dedup_key not in seen:
                seen.add(dedup_key)
                unique_results.append(r)

        top_results = unique_results[:top_k]
        return self._format_context(top_results), top_results

    @staticmethod
    def _format_context(results: List[Dict]) -> str:
        """将检索结果格式化为Prompt可用的上下文"""
        parts = []
        for r in results:
            doc_info = f"[{r.get('doc_name', '')}]" if r.get('doc_name') else ""
            parts.append(f"{doc_info}[来源：第{r['page_num']}页]\n{r['text']}")
        return "\n\n".join(parts)

    def build_prompt(self, question: str, context: str, history: list = None, lang: str = 'zh') -> str:
        """构建LLM Prompt：注入对话历史+检索上下文，支持中英文输出"""
        if not context:
            lang_hint = '请用英文简洁回答。' if lang == 'en' else '请用中文简洁回答。'
            return f"请回答问题（未检索到相关文档内容）：{question}\n{lang_hint}"

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
            f"准确回答用户的问题。如果文档内容不足以回答问题，请如实说不知道，不要编造。\n"
            f"最终输出语言要求：{'英文' if lang == 'en' else '中文'}。\n\n"
            f"{history_block}"
            f"重要提示：\n"
            f"1. 如果用户的问题是追问（如'这个公司'、'那XX呢'、'它的XX'），请结合上方的对话历史理解用户的完整意图。\n"
            f"2. 检索内容包含三种类型：\n"
            f"   [text] 文本段落\n"
            f"   [table] 表格数据\n"
            f"   [图片] 图片描述——对PDF中图表/图片的文字描述\n\n"
            f"检索到的文档内容：\n{context}\n\n"
            f"用户问题：{question}\n\n"
            f"请基于以上文档内容和对话历史给出准确、简洁的回答。如果涉及数据，请直接引用原文中的数据。如果最终要求英文，请把关键信息翻译成自然流畅的英文。"
        )
