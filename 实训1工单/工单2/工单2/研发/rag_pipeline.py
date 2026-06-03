# -*- coding: utf-8 -*-
"""
RAG引擎核心模块
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化

架构说明：
- pdf_parser/ 目录：独立的PDF预处理模块（PDF解析 → 分块 → CSV导出）
- RAGEngine：仅从CSV加载预处理的chunks数据，不再调用PDF解析
- 使用方式：先运行 pdf_parser 生成CSV，再启动 app 提供问答服务
- 本地TF-IDF检索为主，Milvus为可选后端
- 优化：增加中文停用词过滤 + 检索重排序（关键词密度 + 精确匹配）
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
# 优化：添加停用词过滤，减少无意义词对 TF-IDF 检索的干扰
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
    '相关', '上述', '其中', '以及',    '报告', '期内', '来自',
    # 英文停用词
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'or', 'and', 'not', 'this', 'that', 'it',
    'how', 'what', 'which', 'when', 'where', 'who', 'why', 'do', 'does', 'did', 'has', 'have', 'had', 'can', 'could', 'will', 'would', 'should', 'may', 'might',
    'if', 'then', 'than', 'but', 'so', 'no', 'yes', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'too', 'very',
])  # 优化：增加中文停用词表，提升 TF-IDF 检索精度


class RAGEngine:
    def __init__(self, pdf_path: str = None, backend="auto"):
        self.pdf_path = pdf_path  # PDF文件路径
        self.backend = backend  # RAG后端类型，auto→TF-IDF本地（Milvus不可用时）
        self.chunks = []  # 文档分块列表，每块包含{content, page_num, chunk_idx, source_type}
        self.tfidf = None  # TF-IDF向量化器
        self.svd = None  # SVD降维器
        self.tfidf_mat = None  # 全量TF-IDF矩阵，用于本地检索
        self.milvus_store = self._init_milvus()  # 初始化Milvus向量数据库（可选）
        self.llm_client = LLMClient()  # 创建LLM客户端实例
        self.query_understand = QueryUnderstanding()  # 创建查询理解实例
        if pdf_path:  # 如果提供了PDF路径
            self.init_rag()  # 立即初始化RAG引擎

    # ---- Milvus 延迟初始化 ----
    @staticmethod
    def _init_milvus():
        try:
            from milvus_store import MilvusStore  # 导入Milvus存储模块
            return MilvusStore()  # 创建Milvus存储实例
        except Exception as e:  # 导入或初始化失败时
            logger.warning(f"MilvusStore 初始化失败: {e}")  # 记录警告日志
            return None  # 返回None，降级到本地检索

    # ---- 初始化 ----
    @staticmethod
    def _csv_path_for(pdf_path: str) -> str:
        """根据PDF路径推导CSV文件名"""
        base = os.path.splitext(pdf_path)[0]  # 去除PDF扩展名
        # 如果已有 _chunks_v2 结尾，直接加后缀
        if base.endswith('_chunks'):  # 路径已以_chunks结尾
            return f"{base}_v2.csv"  # 直接追加_v2.csv
        return f"{base}_chunks_v2.csv"  # 返回标准CSV文件名

    def init_rag(self):
        """初始化RAG引擎（仅从CSV加载分块，不解析PDF）"""
        logger.info("正在初始化RAG引擎...")  # 打印初始化日志
        self.chunks = self.load_chunks_from_csv()  # 从CSV加载分块数据（不解析PDF）
        if self.chunks:  # 分块不为空时
            max_page = max(c.get('page_num', 0) for c in self.chunks)  # 获取最大页码
            logger.info(f"CSV加载完成: {len(self.chunks)} 块, 最大页码 {max_page}")  # 打印加载统计
        else:  # 分块为空时
            logger.warning("未加载到分块数据，请先运行 PDF 预处理")  # 记录警告
        self.init_embedder()  # 初始化嵌入模型

    def load_pdf(self, pdf_path: str):
        """兼容旧接口：设置PDF路径并加载CSV分块（不再解析PDF）"""
        self.pdf_path = pdf_path  # 设置PDF路径
        self.init_rag()  # 初始化RAG引擎（仅从CSV加载）

    def load_chunks_from_csv(self) -> List[Dict]:
        """仅从CSV加载分块数据（不解析PDF），CSV不存在时返回空列表"""
        return self.parse_pdf_to_chunks()

    def parse_pdf_to_chunks(self) -> List[Dict]:
        """仅从CSV加载分块数据（不再解析PDF），CSV不存在时返回空列表"""
        if not self.pdf_path:  # PDF路径为空时
            return []  # 返回空列表

        # 1. 尝试从已有的CSV加载
        csv_path = self._csv_path_for(self.pdf_path)  # 推导CSV文件路径
        if os.path.exists(csv_path):  # CSV文件存在时
            try:
                chunks = self._load_chunks_from_csv(csv_path)  # 从CSV加载分块
                if chunks:  # 加载成功且不为空
                    logger.info(f"直接从CSV加载: {csv_path} ({len(chunks)} 块)")  # 打印加载日志
                    return chunks  # 返回加载的分块
            except Exception as e:  # CSV加载失败时
                logger.warning(f"CSV加载失败: {e}，重新解析PDF")  # 记录警告并回退解析

        # 2. CSV不存在或加载失败，提示用户先运行 pdf_parser 预处理
        logger.error(
            "CSV分块文件不存在: {}。请先运行 pdf_parser 预处理PDF：\n"
            "    python -m pdf_parser.pdf_processor 招股说明书1.pdf".format(csv_path)
        )
        return []

    def _load_chunks_from_csv(self, csv_path: str) -> List[Dict]:
        """从已存在的CSV文件加载分块数据"""
        result = []  # 初始化结果列表
        with open(csv_path, 'r', encoding='utf-8-sig') as f:  # 以UTF-8带BOM格式打开CSV
            reader = csv.DictReader(f)  # 创建字典读取器
            for row in reader:  # 遍历每一行
                result.append({  # 添加分块字典
                    "content": row.get('text', ''),  # 文本内容
                    "page_num": int(row.get('page_num', 0)),  # 页码，默认0
                    "chunk_idx": int(row.get('id', row.get('chunk_index', 0))),  # 分块索引
                    "source_type": row.get('source_type', 'text'),  # 来源类型，默认text
                })
        return result  # 返回分块列表

    def _fallback_parse(self) -> List[Dict]:
        try:
            from PyPDF2 import PdfReader  # 导入PyPDF2读取器
            reader = PdfReader(self.pdf_path)  # 创建PDF读取器
            result = []  # 初始化结果列表
            idx = 0  # 分块索引计数器
            for pi, page in enumerate(reader.pages):  # 遍历每一页
                text = page.extract_text().strip()  # 提取并清理文本
                if not text:  # 空文本跳过
                    continue
                result.append({  # 添加分块字典
                    "content": text,  # 页面文本内容
                    "page_num": pi + 1,  # 页码（从1开始）
                    "chunk_idx": idx,  # 分块索引
                    "source_type": "text",  # 来源类型为文本
                })
                idx += 1  # 递增索引
            return result  # 返回解析结果
        except Exception as e:  # 回退解析失败
            logger.error(f"回退解析也失败: {e}")  # 记录错误
            return []  # 返回空列表

    # ---- 嵌入模型 ----
    def init_embedder(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # 导入TF-IDF向量化器
            from sklearn.decomposition import TruncatedSVD  # 导入SVD降维器
            import jieba  # 导入中文分词库
            texts = [c["content"] for c in self.chunks]  # 提取所有分块的文本内容
            if not texts:  # 文本列表为空时
                return  # 直接返回
            # 优化：使用 jieba 分词处理中文，过滤停用词
            def jieba_tokenizer(text):  # 定义jieba分词函数
                words = list(jieba.cut(text))  # jieba中文分词
                en_words = [w.lower() for w in re.findall(r'[a-zA-Z]+', text)]  # 提取英文单词
                return [w for w in words + en_words if len(w) > 1 and w not in CHINESE_STOP_WORDS]  # 合并去停用词
            self.tfidf = TfidfVectorizer(max_features=5000, tokenizer=jieba_tokenizer)  # 创建TF-IDF向量化器，最多5000特征
            self.tfidf_mat = self.tfidf.fit_transform(texts)  # 拟合文档并转换得到TF-IDF矩阵
            self.svd = TruncatedSVD(n_components=128, random_state=42)  # 创建SVD降维器，降至128维
            self.svd.fit(self.tfidf_mat)  # 拟合SVD模型
            logger.info(f"[Embedder] TF-IDF(jieba,{len(self.tfidf.vocabulary_)}词) -> SVD(128维)")  # 打印嵌入模型信息
        except Exception as e:  # 训练失败时
            logger.error(f"向量化模型训练失败: {e}")  # 记录错误

    def generate_embeddings(self, texts: List[str]):
        if self.tfidf is None:  # TF-IDF未初始化时
            return []  # 返回空列表
        tfidf_mat = self.tfidf.transform(texts)  # 将文本转为TF-IDF向量
        return self.svd.transform(tfidf_mat).tolist()  # SVD降维后返回Python列表

    # ---- 检索 ----
    def get_context(self, query: str, top_k=5) -> Tuple[str, List[Dict]]:
        """
        检索最相关的 top_k 个文档块
        优先 Milvus，降级到 TF-IDF 本地余弦相似度
        返回: (context_str, results_list)
        results_list: [{text, page_num, score}, ...]
        """
        # 1. 尝试 Milvus
        if self.milvus_store and self.milvus_store.is_available():  # Milvus可用时
            try:
                q_emb = self.generate_embeddings([query])[0]  # 生成查询向量
                hits = self.milvus_store.search(q_emb, top_k)  # 在Milvus中检索
                if hits:  # 检索到结果时
                    results = []  # 初始化结果列表
                    for h in hits:  # 遍历每个命中结果
                        results.append({  # 添加结果字典
                            "text": h.entity.get("content", ""),  # 文本内容
                            "page_num": h.entity.get("page_num", 0),  # 页码
                            "score": h.score,  # 相似度评分
                        })
                    return self._format_context(results), results  # 返回格式化上下文和结果列表
            except Exception as e:  # Milvus检索失败时
                logger.warning(f"Milvus检索失败，降级到TF-IDF: {e}")  # 记录警告并降级

        # 2. TF-IDF本地检索（余弦相似度）
        return self._tfidf_search(query, top_k)  # 使用本地TF-IDF检索

    def _tfidf_search(self, query: str, top_k=5) -> Tuple[str, List[Dict]]:
        """使用 TF-IDF + 余弦相似度进行本地检索"""
        # 优化：增加关键词提取用于重排序
        query_keywords = [w for w in re.findall(r'[\u4e00-\u9fff]{2,}', query)
                          if w not in CHINESE_STOP_WORDS]  # 优化：从查询中提取有效关键词用于重排序
        if not self.chunks or self.tfidf is None:  # 分块或TF-IDF未就绪时
            return "", []  # 返回空结果

        from sklearn.metrics.pairwise import cosine_similarity  # 导入余弦相似度计算函数
        q_vec = self.tfidf.transform([query])  # 将查询转为TF-IDF向量
        scores = cosine_similarity(q_vec, self.tfidf_mat).flatten()  # 计算与所有文档的余弦相似度并展平

        # 优化：前N个候选（取更多候选用于重排序）
        candidate_count = min(top_k * 5, len(scores))  # 优化：候选数量扩大5倍用于重排序
        candidate_indices = scores.argsort()[::-1][:candidate_count]  # 优化：先按TF-IDF初筛候选

        # 优化：对候选结果进行重排序（关键字密度 + 精确匹配加分）
        scored_candidates = []  # 优化：重排序候选列表
        for idx in candidate_indices:  # 优化：遍历每个候选
            if scores[idx] <= 0:  # 评分为0时跳过
                continue
            c = self.chunks[idx]  # 获取对应的分块
            text = c["content"]  # 分块文本
            base_score = float(scores[idx])  # TF-IDF基础分
            boost = 0.0  # 优化：重排序加分

            # 优化：文本分块优先（表格分块降低权重）
            if c.get("source_type") == "table":
                boost -= 0.1  # 表格分块减分
            else:
                boost += 0.05  # 文本分块加分

            # 优化：精确短语匹配加分（查询关键词连续出现在文本中）
            for kw in query_keywords:  # 优化：遍历每个关键词
                if kw in text:  # 优化：关键词在文本中出现
                    boost += 0.05  # 优化：每个命中关键词加0.05
                # 优化：关键词在文本中高频出现额外加分
                count = text.count(kw)  # 优化：统计关键词出现次数
                if count >= 3:  # 优化：出现3次以上
                    boost += 0.03 * min(count, 5)  # 优化：高频词额外加分

            # 优化：查询完整出现在文本中（边界匹配）
            query_clean = ''.join(query_keywords)  # 优化：合并关键词
            if len(query_clean) > 2 and query_clean in text:  # 优化：完整匹配
                boost += 0.2  # 优化：完整匹配大幅加分

            scored_candidates.append((idx, base_score + boost))  # 优化：记录总分

        # 优化：按重排序后的总分排序
        scored_candidates.sort(key=lambda x: x[1], reverse=True)  # 优化：按总分降序排列

        # 获取 top_k 结果
        results = []  # 初始化结果列表
        for idx, final_score in scored_candidates[:top_k]:  # 优化：遍历重排序后的结果
            c = self.chunks[idx]  # 获取对应的分块
            results.append({  # 添加结果字典
                "text": c["content"],  # 文本内容
                "page_num": c["page_num"],  # 页码
                "score": round(final_score, 4),  # 优化：使用重排序后的综合评分
            })

        return self._format_context(results), results  # 返回格式化上下文和结果列表

    @staticmethod
    def _format_context(results: List[Dict]) -> str:
        """将检索结果格式化为上下文字符串"""
        parts = []  # 初始化片段列表
        for r in results:  # 遍历每个检索结果
            parts.append(f"[来源：第{r['page_num']}页]\n{r['text']}")  # 添加来源页码和文本
        return "\n\n".join(parts)  # 用双换行连接所有片段

    # ---- Prompt构建 ----
    def build_prompt(self, question: str, context: str, lang: str = 'zh') -> str:
        """构建用于LLM生成的Prompt（始终用中文，翻译流程在外层处理）"""
        if not context:  # 无上下文时
            return f"请回答问题（未检索到相关文档内容）：{question}"  # 返回简单Prompt
        return (  # 返回完整Prompt（中文）
            f"你是一个基于PDF文档的智能问答助手。请根据以下从《招股意向书》中检索到的文档内容，"  # 系统角色定义
            f"准确回答用户的问题。如果文档内容不足以回答问题，请如实说不知道，不要编造。\n\n"  # 准确度要求
            f"检索到的文档内容：\n{context}\n\n"  # 检索上下文
            f"用户问题：{question}\n\n"  # 用户问题
            f"请基于以上文档内容给出准确、简洁的回答。如果涉及数据，请直接引用原文中的数据。"  # 回答要求
        )

    # ---- 主问答 ----
    def query(self, user_query: str, top_k=5, lang='zh') -> Dict:
        """完整问答流程"""
        u_result = self.query_understand.understand(user_query)  # 对用户问题进行语义理解
        final_q = u_result["expanded_query"]  # 获取扩展后的查询语句
        context_str, results = self.get_context(final_q, top_k)  # 检索相关上下文

        if not context_str:  # 未检索到上下文时
            answer_msg = "Sorry, no relevant content was found in the documents." if lang == 'en' else "抱歉，在文档中没有找到与您问题相关的内容。"
            return {  # 返回无结果响应
                "answer": answer_msg,  # 多语言回复
                "context": [],  # 上下文为空
                "query_info": u_result,  # 查询理解信息
            }

        prompt = self.build_prompt(user_query, context_str, lang=lang)  # 构建Prompt
        answer = self.llm_client.generate(prompt, context_str, user_query)  # 调用LLM生成回答

        contexts = [f"[第{r['page_num']}页] {r['text'][:200]}" for r in results]  # 格式化上下文摘要列表

        return {  # 返回完整问答结果
            "answer": answer,  # LLM生成的回答
            "context": contexts,  # 上下文摘要
            "results": results,  # 检索结果详情
            "query_info": u_result,  # 查询理解信息
        }
