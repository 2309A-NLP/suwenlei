# -*- coding: utf-8 -*-
# main.py
"""
RAG 多角色医疗咨询系统 —— 主测试入口
功能：
1. 初始化 RAG 机器人
2. 测试短期记忆（上下文对话）
3. 测试多角色知识检索（高血压、中医、患者教育、法律）
4. 测试回答生成（命中/未命中知识库）
5. 测试多轮对话能力
6. 测试 Milvus 聊天记录存储
7. 全流程自动化测试验证系统可用性
"""

import logging
import sys
import time
# RAG 多角色医生核心类
from rag_project.rag_core import RAGMultiRoleDoctor
# 项目配置：BM25 开关
from rag_project.config import BM25_AVAILABLE

# 日志基础配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_main")


def main():
    logger.info("========== 开始测试 RAGMultiRoleDoctor ==========")

    # ===================== 1. 初始化 RAG 机器人 =====================
    try:
        bot = RAGMultiRoleDoctor()
        logger.info("✅ 机器人初始化成功")
        logger.info(f"可用知识库集合: {list(bot.collections.keys())}")
        logger.info(f"BM25 可用: {BM25_AVAILABLE}")
        # 如果 BM25 启用且索引加载成功，打印索引信息
        if BM25_AVAILABLE and bot.bm25_handler:
            logger.info(f"已加载 BM25 索引的知识库: {list(bot.bm25_handler.bm25_indexes.keys())}")
    except Exception as e:
        logger.error(f"❌ 初始化失败: {e}")
        sys.exit(1)  # 初始化失败直接退出程序

    # 使用唯一用户ID（时间戳），避免历史测试数据互相干扰
    test_user = f"test_user_{int(time.time())}"
    logger.info(f"测试用户ID: {test_user}")

    # ===================== 2. 短期记忆测试 =====================
    logger.info("\n========== 测试短期记忆 ==========")
    # 写入两条测试对话
    bot.update_short_term_memory(test_user, "测试问题1", "测试回答1")
    bot.update_short_term_memory(test_user, "测试问题2", "测试回答2")
    # 读取历史
    history = bot.get_short_term_memory(test_user)
    logger.info(f"获取历史记录: {history}")
    # 断言验证：必须是 2 条
    assert len(history) == 2, f"期望2条，实际{len(history)}条"
    logger.info("✅ 短期记忆写入和读取正常")

    # ===================== 3. 知识检索测试 =====================
    logger.info("\n========== 测试知识检索 ==========")
    # 标准高血压测试问题
    test_question = "高血压患者每日食盐摄入量应控制在多少克？"
    # 测试所有角色
    roles = ["hypertension", "tcm", "patient_edu", "lawyer"]
    for role in roles:
        # 检索对应角色知识库
        knowledge = bot.retriever.retrieve(test_question, role)
        logger.info(f"角色 {role:15} 检索知识长度: {len(knowledge)} 字符")
        if len(knowledge.strip()) >= 10:
            logger.info(f"知识片段预览: {knowledge[:200]}...")
        else:
            logger.warning(f"角色 {role} 未检索到有效知识")
    logger.info("✅ 知识检索测试完成")

    # ===================== 4. 生成回答测试（命中知识库） =====================
    logger.info("\n========== 测试生成回答（命中知识库） ==========")
    try:
        answer = bot.generate_answer(test_user, test_question, "hypertension")
        logger.info(f"用户: {test_question}")
        logger.info(f"机器人: {answer}")
        logger.info("✅ 知识库命中回答生成成功")
    except Exception as e:
        logger.error(f"❌ 生成回答失败: {e}")

    # ===================== 5. 生成回答测试（未命中知识库） =====================
    logger.info("\n========== 测试生成回答（未命中知识库） ==========")
    # 无关问题，不会命中医疗知识库
    general_question = "今天天气怎么样？"
    try:
        answer_general = bot.generate_answer(test_user, general_question, "hypertension")
        logger.info(f"用户: {general_question}")
        logger.info(f"机器人: {answer_general}")
        logger.info("✅ 未命中知识库的回答生成成功")
    except Exception as e:
        logger.error(f"❌ 通用回答生成失败: {e}")

    # ===================== 6. 多轮对话上下文测试 =====================
    logger.info("\n========== 测试多轮对话上下文 ==========")
    # 追问问题，依赖上文“食盐摄入”上下文
    follow_up = "那我应该选择哪种降压药？"
    try:
        answer_follow = bot.generate_answer(test_user, follow_up, "hypertension")
        logger.info(f"用户: {follow_up}")
        logger.info(f"机器人: {answer_follow}")
        # 获取完整对话历史
        full_history = bot.get_short_term_memory(test_user)
        logger.info(f"当前总对话轮数: {len(full_history)}")
        logger.info("✅ 多轮对话测试完成")
    except Exception as e:
        logger.error(f"多轮对话测试失败: {e}")

    # ===================== 7. 法律角色专项测试（可选） =====================
    # 如果系统包含民法典知识库，则执行法律检索测试
    if "civil_code_kb" in bot.collections.values():
        logger.info("\n========== 测试法律角色检索 ==========")
        law_question = "民法典中关于侵权责任的规定有哪些？"
        knowledge_law = bot.retriever.retrieve(law_question, "lawyer")
        if len(knowledge_law.strip()) >= 10:
            logger.info(f"法律知识检索成功，长度: {len(knowledge_law)}")
            answer_law = bot.generate_answer(test_user, law_question, "lawyer")
            logger.info(f"法律顾问回答: {answer_law[:300]}...")
        else:
            logger.warning("法律知识库可能为空，跳过回答生成")

    # ===================== 8. Milvus 聊天记录统计 =====================
    logger.info("\n========== 聊天记录统计 ==========")
    try:
        # 获取聊天记录表总数据量
        stats = bot.milvus_client.get_collection_stats("chat_records")
        row_count = stats.get("row_count", 0)
        logger.info(f"聊天记录总数: {row_count}")
    except Exception as e:
        logger.error(f"获取聊天记录总数失败: {e}")

    try:
        # 模糊查询：所有 test_user_ 开头的测试记录
        filter_expr = 'user_id like "test_user_%"'
        results = bot.milvus_client.query(
            collection_name="chat_records",
            filter=filter_expr,
            output_fields=["user_id", "question", "answer"]
        )
        logger.info(f"测试用户记录: {results}")
    except Exception as e:
        logger.error(f"查询测试用户记录失败: {e}")

    # 关闭机器人资源
    bot.close()
    logger.info("\n========== 所有测试执行完毕 ==========")
if __name__ == "__main__":
    main()