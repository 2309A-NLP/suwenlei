# -*- coding: utf-8 -*-
"""
Q&A评测脚本 - RAG vs 纯LLM对比评估
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

评估方法：
1. RAG检索回答：PDF解析 → 分块 → 检索 → LLM/提取生成
2. 纯LLM回答：仅依赖LLM知识（无检索）
3. 对比指标：准确率、响应时间、检索命中率
"""

import os  # 操作系统接口模块，用于文件和路径操作
import sys  # 系统模块，用于修改Python解释器路径
import json  # JSON处理模块，用于读写JSON格式的评估结果
import time  # 时间模块，用于计算响应耗时
import logging  # 日志模块，用于输出调试和运行信息

sys.path.insert(0, os.path.dirname(__file__))  # 将当前文件所在目录加入系统路径，确保本地模块可导入

from rag_pipeline import RAGEngine  # 导入RAG引擎，提供PDF检索和上下文获取功能
from llm_client import LLMClient, GROUND_TRUTH  # 导入LLM客户端和标准答案字典
from query_understanding import QueryUnderstanding  # 导入查询理解模块，用于意图识别和查询扩展

logging.basicConfig(level=logging.INFO,  # 设置日志级别为INFO，输出信息级别及以上的日志
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # 日志格式：时间-名称-级别-消息
logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

# ==================== 评测问题列表（来自任务工单V1.1） ====================

EVAL_QUESTIONS = [  # 定义10个评估问题列表，每个包含ID和问题文本
    {"id": 260, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"},  # 问题260：军用领域收入
    {"id": 95, "question": "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？"},  # 问题95：参与制定的技术标准
    {"id": 33, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？"},  # 问题33：军用收入占比
    {"id": 34, "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？"},  # 问题34：上游企业
    {"id": 957, "question": "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？"},  # 问题957：重要供应商领域
    {"id": 793, "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？"},  # 问题793：下游行业
    {"id": 795, "question": "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？"},  # 问题795：获奖工程
    {"id": 543, "question": "武汉兴图新科电子股份有限公司注册资本是多少？"},  # 问题543：注册资本金额
    {"id": 531, "question": "武汉兴图新科电子股份有限公司法定代表人是谁？"},  # 问题531：法定代表人姓名
    {"id": 207, "question": "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？"}  # 问题207：募集资金用途
]


class RAGEvaluator:  # RAG系统评估器类，负责执行RAG与纯LLM的对比评估
    """RAG系统评估器"""

    def __init__(self, pdf_path):  # 构造函数：初始化评估器
        self.pdf_path = pdf_path  # 保存PDF文件路径
        self.rag_engine = None  # 初始化RAG引擎为None，延迟加载
        self.llm_client = None  # 初始化LLM客户端为None，延迟加载

    def init_engine(self):  # 初始化引擎：加载PDF并创建RAG引擎和LLM客户端
        self.rag_engine = RAGEngine(backend="auto")  # 创建RAG引擎实例，自动选择后端（TF-IDF/Milvus）
        self.rag_engine.load_pdf(self.pdf_path)  # 加载PDF文件，解析并分块索引
        self.llm_client = LLMClient()  # 创建LLM客户端实例
        self.llm_client.set_rag_engine(self.rag_engine)  # 将RAG引擎注入LLM客户端，供生成时使用
        return self  # 返回self，支持链式调用

    def rag_answer(self, question_id, question):  # RAG模式：执行查询理解→检索→生成的完整流程
        """RAG模式：Query Understanding → 检索 → 生成"""
        start = time.time()  # 记录开始时间，用于计算耗时

        # Query Understanding
        qu = QueryUnderstanding.understand(question)  # 执行查询理解：意图识别、消歧、扩展
        context, results = self.rag_engine.get_context(qu['expanded_query'], top_k=5)  # 使用扩展后的查询检索相关段落，取top-5
        prompt = self.rag_engine.build_prompt(question, context)  # 基于原问题和检索结果构建提示词
        answer = self.llm_client.generate(prompt, context, question)  # 调用LLM生成最终的答案文本

        elapsed = time.time() - start  # 计算RAG回答的总耗时
        gt = GROUND_TRUTH.get(question_id, "")  # 从标准答案字典中获取当前问题的标准答案

        # 准确率判断：精确匹配、包含关系、关键词覆盖
        accuracy = self._calc_accuracy(answer, gt)  # 计算生成答案与标准答案的准确率

        return {  # 返回RAG回答的完整结果字典
            'id': question_id,  # 问题ID
            'question': question,  # 原始问题文本
            'answer': answer,  # 生成的答案文本
            'retrieved_chunks': [  # 检索到的相关段落列表
                {'page': r['page_num'], 'score': r['score'], 'snippet': r['text'][:200]}  # 提取页码、相关度分数和片段前200字
                for r in results  # 遍历所有检索结果
            ],  # 构建检索摘要列表
            'time_seconds': round(elapsed, 2),  # 耗时，保留两位小数
            'matched_ground_truth': accuracy,  # 与标准答案的匹配准确率
            'intent': qu['intent'],  # 查询理解识别的意图类型
            'is_complex': qu['is_complex']  # 是否为复杂问题（含多个子问题）
        }

    def pure_llm_answer(self, question_id, question):  # 纯LLM模式：仅依赖LLM知识，不进行检索
        """纯LLM模式（无检索）"""
        start = time.time()  # 记录开始时间

        # 只给少量上下文，模拟无检索效果
        context = "武汉兴图新科电子股份有限公司是一家在科创板上市的公司，主营业务为视频指挥控制类产品。"  # 仅给一个公司简介的固定上下文
        prompt = self.rag_engine.build_prompt(question, context)  # 基于问题和少量上下文构建提示词
        answer = self.llm_client.generate(prompt, context, question)  # 调用LLM生成答案

        # 如果答案包含"未找到"或来自提取模式，则标记为无法回答
        gt = GROUND_TRUTH.get(question_id, "")  # 获取标准答案
        accuracy = self._calc_accuracy(answer, gt)  # 计算准确率

        elapsed = time.time() - start  # 计算纯LLM回答的总耗时

        return {  # 返回纯LLM回答的结果字典
            'id': question_id,  # 问题ID
            'question': question,  # 原始问题文本
            'answer': answer,  # 生成的答案文本
            'time_seconds': round(elapsed, 2),  # 耗时，保留两位小数
            'matched_ground_truth': accuracy  # 与标准答案的匹配准确率
        }

    @staticmethod  # 声明为静态方法
    def _calc_accuracy(answer, ground_truth):  # 计算答案准确率，基于数字匹配和关键词匹配的加权组合
        """
        计算答案准确率（0~1）
        基于：精确匹配、核心数字/实体包含
        """
        if not answer or not ground_truth:  # 如果答案或标准答案为空
            return 0.0  # 准确率为0
        if answer == ground_truth:  # 如果答案与标准答案完全一致
            return 1.0  # 准确率1.0（精确匹配）

        # 提取数字和关键实体
        import re  # 导入正则模块用于提取数字和关键词
        gt_nums = set(re.findall(r'[\d,]\.?\d*', ground_truth))  # 提取标准答案中的数字（含逗号分隔）
        ans_nums = set(re.findall(r'[\d,]\.?\d*', answer))  # 提取生成答案中的数字（含逗号分隔）
        gt_keywords = set(re.findall(r'[\u4e00-\u9fff]{3,}', ground_truth))  # 提取标准答案中的中文关键词（3字及以上）
        ans_keywords = set(re.findall(r'[\u4e00-\u9fff]{3,}', answer))  # 提取生成答案中的中文关键词（3字及以上）

        # 数字匹配率
        num_match = len(gt_nums & ans_nums) / max(len(gt_nums), 1)  # 计算数字交集大小与标准答案数字数的比值
        # 关键词匹配率
        kw_match = len(gt_keywords & ans_keywords) / max(len(gt_keywords), 1)  # 计算关键词交集大小与标准答案关键词数的比值

        return round((num_match * 0.5 + kw_match * 0.5), 4)  # 数字和关键词各占50%权重，加权求和后保留4位小数

    def evaluate_all(self):  # 执行全部10个问题的评估，对比RAG和纯LLM效果
        """执行全部评测"""
        if not self.rag_engine:  # 如果RAG引擎尚未初始化
            self.init_engine()  # 自动初始化引擎

        results = {  # 初始化评测结果容器
            'rag_results': [],  # RAG模式的结果列表
            'pure_llm_results': [],  # 纯LLM模式的结果列表
            'ground_truth': GROUND_TRUTH,  # 标准答案字典
            'summary': {}  # 汇总统计信息（稍后填充）
        }

        print("=" * 70)  # 打印分隔线
        print("RAG系统评测报告")  # 打印评测报告标题
        print("=" * 70)  # 打印分隔线
        print("\n评测模式：TF-IDF/Milvus向量检索 + LLM/提取生成")  # 打印评测模式说明
        print("评测问题数：{}\n".format(len(EVAL_QUESTIONS)))  # 打印问题总数

        rag_total_time = 0  # 累计RAG总耗时
        pure_total_time = 0  # 累计纯LLM总耗时
        rag_total_accuracy = 0.0  # 累计RAG准确率总和
        pure_total_accuracy = 0.0  # 累计纯LLM准确率总和
        rag_hit_count = 0  # RAG检索命中（有检索结果）的问题数量

        for i, q in enumerate(EVAL_QUESTIONS, 1):  # 遍历所有评估问题，i从1开始计数
            qid = q['id']  # 获取问题ID
            question = q['question']  # 获取问题文本

            print("\n{}".format("=" * 70))  # 打印问题分隔线
            print("问题 {}/10 (ID: {})".format(i, qid))  # 打印问题序号和ID
            print("问题：{}".format(question))  # 打印问题内容
            print("{}".format("=" * 70))  # 打印分隔线

            rag_r = self.rag_answer(qid, question)  # 执行RAG模式回答
            rag_total_time += rag_r['time_seconds']  # 累加RAG耗时
            rag_total_accuracy += rag_r['matched_ground_truth']  # 累加RAG准确率

            print("\n[ RAG检索回答 ]")  # 打印RAG回答标题
            print("   {}".format(rag_r['answer'][:200]))  # 打印RAG答案（截取前200字符）
            print("   耗时：{}s".format(rag_r['time_seconds']))  # 打印RAG耗时
            print("   准确率：{:.2%}".format(rag_r['matched_ground_truth']))  # 打印RAG准确率（百分比格式）
            print("   意图：{}".format(rag_r['intent']))  # 打印识别的意图类型

            if rag_r['retrieved_chunks']:  # 如果检索到相关段落
                rag_hit_count += 1  # 命中计数加1
                print("   参考来源：")  # 打印参考来源标题
                for c in rag_r['retrieved_chunks'][:3]:  # 最多显示前3个检索结果
                    print("      - 第{}页 (相关度：{:.4f})".format(c['page'], c['score']))  # 打印页码和相关性分数

            pure_r = self.pure_llm_answer(qid, question)  # 执行纯LLM模式回答
            pure_total_time += pure_r['time_seconds']  # 累加纯LLM耗时
            pure_total_accuracy += pure_r['matched_ground_truth']  # 累加纯LLM准确率

            print("\n[ 纯LLM回答 ]")  # 打印纯LLM回答标题
            print("   {}".format(pure_r['answer'][:200]))  # 打印纯LLM答案（截取前200字符）
            print("   耗时：{}s".format(pure_r['time_seconds']))  # 打印纯LLM耗时
            print("   准确率：{:.2%}".format(pure_r['matched_ground_truth']))  # 打印纯LLM准确率（百分比格式）

            gt = GROUND_TRUTH.get(qid, "（未提供标准答案）")  # 获取当前问题的标准答案
            print("\n[ 标准答案（来自招股说明书）]")  # 打印标准答案标题
            print("   {}".format(gt))  # 打印标准答案内容

            results['rag_results'].append(rag_r)  # 将RAG结果加入列表
            results['pure_llm_results'].append(pure_r)  # 将纯LLM结果加入列表

        n = len(EVAL_QUESTIONS)  # 问题总数（10）
        summary = {  # 构建评估汇总统计信息
            'total_questions': n,  # 总问题数
            'rag_avg_time': round(rag_total_time / n, 2),  # RAG平均耗时
            'pure_llm_avg_time': round(pure_total_time / n, 2),  # 纯LLM平均耗时
            'rag_total_time': round(rag_total_time, 2),  # RAG总耗时
            'pure_llm_total_time': round(pure_total_time, 2),  # 纯LLM总耗时
            'rag_avg_accuracy': round(rag_total_accuracy / n, 4),  # RAG平均准确率
            'pure_llm_avg_accuracy': round(pure_total_accuracy / n, 4),  # 纯LLM平均准确率
            'rag_retrieval_hit_rate': round(rag_hit_count / n, 4),  # RAG检索命中率（有检索结果的占比）
            'accuracy_improvement': round(  # RAG相比纯LLM的准确率提升百分比
                (rag_total_accuracy - pure_total_accuracy) / max(pure_total_accuracy, 0.01) * 100, 2  # 计算提升比例并转为百分比
            ),  # 准确率提升幅度
            'evaluation_components': [  # 评估系统组件清单
                'PDF解析：PyMuPDF',  # PDF解析使用的工具
                '文本分块：自适应段落分块（800字符/块，200字符重叠）',  # 文本分块策略
                '向量检索：TF-IDF + jieba分词 + 余弦相似度',  # 向量检索方法
                'Milvus向量库（可选，自动降级TF-IDF）',  # 向量库支持（可选降级）
                'Query Understanding：意图识别 + 消歧 + 查询扩展',  # 查询理解组件
                '答案生成：DeepSeek API优先 + 本地提取回退',  # 答案生成策略
            ]  # 系统组件清单列表
        }

        results['summary'] = summary  # 将汇总信息存入结果字典

        print("\n\n{}".format("=" * 70))  # 打印汇总分隔线
        print("评测汇总")  # 打印汇总标题
        print("{}".format("=" * 70))  # 打印分隔线
        print("RAG检索平均耗时：{:.2f}s".format(summary['rag_avg_time']))  # 打印RAG平均耗时
        print("纯LLM平均耗时：{:.2f}s".format(summary['pure_llm_avg_time']))  # 打印纯LLM平均耗时
        print("RAG检索平均准确率：{:.2%}".format(summary['rag_avg_accuracy']))  # 打印RAG平均准确率
        print("纯LLM平均准确率：{:.2%}".format(summary['pure_llm_avg_accuracy']))  # 打印纯LLM平均准确率
        print("RAG检索命中率：{:.2%}".format(summary['rag_retrieval_hit_rate']))  # 打印检索命中率
        print("准确率提升：{:.2f}%".format(summary['accuracy_improvement']))  # 打印准确率提升百分比
        print("\n评测组件包含：")  # 打印组件清单标题
        for comp in summary['evaluation_components']:  # 遍历组件清单
            print("  - {}".format(comp))  # 逐项打印组件信息

        return results  # 返回完整的评测结果字典


def main():  # 主函数：执行RAG系统评估流程
    pdf_path = os.path.join(os.path.dirname(__file__), '招股说明书1.pdf')  # 构建PDF文件路径（与脚本同目录）

    if not os.path.exists(pdf_path):  # 检查PDF文件是否存在
        print("错误：找不到PDF文件：{}".format(pdf_path))  # 打印错误信息
        return  # 提前退出

    evaluator = RAGEvaluator(pdf_path)  # 创建评估器实例，传入PDF路径
    results = evaluator.evaluate_all()  # 执行全部评估流程

    output_path = os.path.join(os.path.dirname(__file__), 'qa_evaluation_results.json')  # 构建结果输出路径
    with open(output_path, 'w', encoding='utf-8') as f:  # 以UTF-8编码打开输出文件
        json.dump(results, f, ensure_ascii=False, indent=2)  # 将评测结果序列化为JSON并写入文件（保留中文、缩进格式化）
    print("\n\n评测结果已保存到：{}".format(output_path))  # 提示结果保存路径


if __name__ == '__main__':  # 判断是否以主程序方式运行
    main()  # 调用主函数执行评估
