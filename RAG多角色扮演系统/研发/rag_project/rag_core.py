# -*- coding: utf-8 -*-
# rag_core.py RAGMultiRoleDoctor 主类

"""
RAG 多角色对话机器人核心模块
功能：实现基于混合检索（向量+BM25+重排序）和短期记忆的多角色问答系统
整体流程：
1. 用户提问 → 2. 检索知识 → 3. 拼接Prompt → 4. 大模型生成回答 → 5. 保存对话历史
支持：高血压、中医、患者教育、法律顾问等多角色切换
"""

import redis
import json
import logging
import time
from datetime import datetime
from typing import Dict
from pymilvus import MilvusClient
from openai import OpenAI

# 项目配置导入
from rag_project.config import (
    MILVUS_HOST, MILVUS_PORT, ROLE_TO_COLLECTION, DEFAULT_COLLECTION,
    REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_EXPIRE,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    BM25_AVAILABLE
)
from rag_project.logger_config import logger
from rag_project.utils import is_empty_knowledge
from rag_project.bm25_handler import BM25Handler
from rag_project.retriever import HybridRetriever
from role_config import ROLE_CONFIG, DEFAULT_ROLE
from rag_project.chat_history_milvus import ChatHistoryMilvus

# 全局的 Redis key 前缀
_CHAT_PREFIX = "hypertension_chat:"
_MAX_HISTORY = 20  # 最大保留 20 轮对话


class RAGMultiRoleDoctor:
    """多角色 RAG 对话机器人核心类"""

    def __init__(self):
        """初始化所有核心组件"""
        start_time = time.time()
        logger.info("开始初始化 RAGMultiRoleDoctor...")

        # ===== 1. Redis（短期对话记忆） =====
        self.redis_client = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True
        )
        logger.info(f"Redis 连接成功: {REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")

        # ===== 2. Milvus 客户端（向量数据库） =====
        self.milvus_client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")
        logger.info(f"MilvusClient 连接成功: {MILVUS_HOST}:{MILVUS_PORT}")

        # ===== 3. BM25 关键词检索 =====
        unique_colls = set(ROLE_TO_COLLECTION.values())
        self.bm25_handler = BM25Handler(unique_colls) if BM25_AVAILABLE else None
        logger.info("BM25 处理器初始化" if BM25_AVAILABLE else "BM25 不可用，跳过")

        # ===== 4. 混合检索器 =====
        self.retriever = HybridRetriever(self.milvus_client, self.bm25_handler)
        logger.info("混合检索器初始化完成")

        # ===== 5. 聊天记录持久化 =====
        self.chat_history_store = ChatHistoryMilvus(self.milvus_client)

        # ===== 6. LLM 客户端（加 15s 超时，避免请求卡死） =====
        self.llm_client = OpenAI(
            api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL, timeout=15.0
        )
        logger.info(f"LLM 客户端初始化，模型: {DEEPSEEK_MODEL}, 端点: {DEEPSEEK_BASE_URL}, 超时: 15s")

        # 角色配置
        self.role_config = ROLE_CONFIG
        self.collections = ROLE_TO_COLLECTION

        # ===== 7. Prompt 模板 =====
        self.prompt_template = (
            "你是一名{role_description}，请以专业、严谨、通俗的口吻回答用户问题。\n\n"
            "【硬性规则】\n"
            '1. 严格依据下方【检索知识片段】作答，**禁止编造、禁止脑补、禁止知识库外延伸**；\n'
            '2. 知识库无对应内容，必须直白回复：\u201c未找到相关内容\u201d，不得强行解释；\n'
            '3. 法律角色必须明确引用法条名称+条款，医疗角色禁止给出诊断、处方类建议；\n'
            '4. 回答逻辑清晰、分层易懂，字数严格控制在 200~400 字；\n'
            '5. 禁止夸大疗效、绝对化用语、虚假医学/法律断言。\n\n'
            '【检索到的知识片段】\n{retrieved_knowledge}\n\n'
            '【最近5轮对话上下文】\n{conversation_history}\n\n'
            '【用户当前问题】\n{user_question}\n\n'
            '【输出格式要求】\n'
            '1. 不用冗余开场白，直接正文回答；\n'
            '2. 语言通俗易懂，适合普通用户阅读；\n'
            '3. 专业术语可简单解释，不要堆砌专业词汇；\n'
            '4. 结尾不额外拓展、不追加无关建议。\n'
        )
        self.general_prompt_template = (
            "你是一名{role_description}，请专业、客观、谨慎回答用户问题。\n\n"
            "【重要提示】\n"
            "本次未检索到专属知识库内容，仅依靠模型通用知识作答。\n"
            "医疗、法律内容仅作科普参考，不可替代专业医师、律师诊断建议。\n\n"
            "【硬性规则】\n"
            "1. 不编造法条、不编造医学指南、不虚构数据；\n"
            "2. 风险类内容必须加免责提示；\n"
            "3. 语言通俗简洁，字数控制在 200~400 字。\n\n"
            "【最近5轮对话上下文】\n{conversation_history}\n\n"
            "【用户当前问题】\n{user_question}\n"
        )

        logger.info(f"RAGMultiRoleDoctor 初始化完成，耗时: {time.time() - start_time:.3f}s")

    # ==================== 知识检索（对外接口） ====================
    def retrieve_knowledge(self, query: str, role: str) -> str:
        """对外暴露的知识检索接口（适配 RAGAS 自动化评测）"""
        try:
            return self.retriever.retrieve(query, role)
        except Exception as e:
            logger.error(f"检索知识失败: {e}")
            return ""

    # ==================== 短期记忆管理（Redis） ====================
    def _load_history(self, user_id: str) -> list:
        """从 Redis 加载完整历史列表"""
        key = _CHAT_PREFIX + user_id
        data = self.redis_client.get(key)
        return json.loads(data) if data else []

    def _save_history(self, user_id: str, history: list):
        """将历史列表写回 Redis（带过期时间）"""
        key = _CHAT_PREFIX + user_id
        self.redis_client.setex(key, REDIS_EXPIRE, json.dumps(history))

    def get_short_term_memory(self, user_id: str) -> list:
        """获取用户最近最多 5 轮对话"""
        history = self._load_history(user_id)
        return history[-5:]

    def update_short_term_memory(self, user_id: str, question: str, answer: str):
        """更新 Redis 短期记忆（传入已有 history 避免二次读 Redis）"""
        history = self._load_history(user_id)
        history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_question": question,
            "bot_answer": answer
        })
        if len(history) > _MAX_HISTORY:
            history = history[-_MAX_HISTORY:]
        self._save_history(user_id, history)

    # ==================== 生成回答（主入口） ====================
    def generate_answer(self, user_id: str, user_question: str, role: str = "hypertension") -> str:
        """最终回答生成主函数（完整 RAG 流程）"""
        total_start = time.time()
        logger.info(f"生成回答，用户: {user_id}, 角色: {role}, 问题: {user_question[:50]}...")
        try:
            # 1. 加载历史 + 构建字符串（只读一次 Redis）
            history = self._load_history(user_id)
            history_str = "\n".join(
                f"用户：{h['user_question']}\n助手：{h['bot_answer']}" for h in history[-5:]
            ) or "无历史对话"

            # 2. 角色描述
            role_info = self.role_config.get(role, self.role_config.get(DEFAULT_ROLE))
            role_description = role_info.get("prompt", "专业助手")

            # 3. 混合检索
            knowledge = self.retriever.retrieve(user_question, role)

            # 4. 拼接 Prompt
            if is_empty_knowledge(knowledge):
                prompt = self.general_prompt_template.format(
                    role_description=role_description,
                    conversation_history=history_str,
                    user_question=user_question
                )
            else:
                prompt = self.prompt_template.format(
                    role_description=role_description,
                    retrieved_knowledge=knowledge,
                    conversation_history=history_str,
                    user_question=user_question
                )

            # 5. 调用 LLM
            response = self.llm_client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800
            )
            answer = response.choices[0].message.content.strip()
            logger.info(f"LLM 回答生成成功，长度: {len(answer)}")

            # 6. 更新短期记忆（传入已加载的 history，不再读 Redis）
            history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user_question": user_question,
                "bot_answer": answer
            })
            if len(history) > _MAX_HISTORY:
                history = history[-_MAX_HISTORY:]
            self._save_history(user_id, history)

            # 7. 持久化聊天记录（失败不影响主流程）
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.chat_history_store.store_message(user_id, role, user_question, answer, timestamp)
            except Exception as e:
                logger.error(f"聊天记录持久化失败: {e}", exc_info=True)

            logger.info(f"本次问答总耗时: {time.time() - total_start:.3f}s")
            return answer

        except Exception as e:
            logger.error(f"生成回答失败: {e}", exc_info=True)
            return "抱歉，我暂时无法回答您的问题，请稍后再试。"

    def close(self):
        """资源释放"""
        logger.info("关闭 RAGMultiRoleDoctor 资源...")
        self.redis_client.close()
        if hasattr(self, 'chat_history_store'):
            self.chat_history_store.close()
        logger.info("资源释放完毕")
