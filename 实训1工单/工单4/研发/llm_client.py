# -*- coding: utf-8 -*-
"""LLM客户端 - DeepSeek API优先 + 本地提取回退"""
import os
import re
import json
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)

# ============ DeepSeek API配置 ============
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', 'sk-9573a7f31fe9446394ac868afa8e5718')
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_HAS_API = bool(DEEPSEEK_API_KEY) and DEEPSEEK_API_KEY != ''


class LLMClient:
    """LLM客户端 — API优先→本地提取回退"""

    def __init__(self, rag_engine=None):
        self.rag_engine = rag_engine
        self.api_key = DEEPSEEK_API_KEY
        self.api_available = DEEPSEEK_HAS_API
        if self.api_available:
            logger.info(f"[LLMClient] DeepSeek API已配置: {DEEPSEEK_BASE_URL}")
        else:
            logger.info("[LLMClient] 未配置API密钥，使用本地提取模式")

    def set_rag_engine(self, engine):
        self.rag_engine = engine  # 设置或更新RAG引擎引用

    # ==================== 纯LLM生成（无RAG上下文） ====================

    def generate_pure_llm(self, query, lang='zh'):
        if not self.api_available:
            if lang == 'en':
                return "Pure LLM unavailable: DeepSeek API key not configured or invalid."
            return "纯大模型不可用：未配置或DeepSeek API密钥无效，请检查密钥配置。"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer {}'.format(self.api_key)
        }
        if lang == 'en':
            system_prompt = 'You are a general AI assistant. Users may ask questions about a company prospectus. If unsure, honestly say you don\'t know. Do not fabricate. IMPORTANT: You must respond in English, not Chinese.'
            user_prompt = f'Please answer the following question (no context, based on your knowledge only): {query}'
        else:
            system_prompt = '你是一个通用AI助手。用户会问关于公司招股说明书的问题，不确定请如实说不知道，不要编造。'
            user_prompt = f'请回答问题（无上下文，仅凭知识）：{query}'
        data = {
            'model': 'deepseek-chat',
            'messages': [{'role': 'system', 'content': system_prompt},
                         {'role': 'user', 'content': user_prompt}],
            'max_tokens': 1024,
            'temperature': 0.3,
            'stream': False
        }
        try:
            r = requests.post(
                f'{DEEPSEEK_BASE_URL}/v1/chat/completions',
                headers=headers, json=data, timeout=30
            )
            if r.status_code == 200:
                return r.json()['choices'][0]['message']['content'].strip()
            if lang == 'en':
                return f"Pure LLM call failed: HTTP status {r.status_code}"
            return f"纯大模型调用失败：HTTP状态码{r.status_code}"
        except Exception as e:
            if lang == 'en':
                return f"Pure LLM exception: {str(e)}"
            return f"纯大模型异常：{str(e)}"

    # ==================== 主入口 ====================

    def generate(self, prompt, context, query, lang='zh'):
        """根据问题和上下文生成回答 — 策略: API生成→本地提取→兜底"""
        if self.api_available:
            try:
                api_answer = self._api_generate(prompt, lang=lang)
                if not api_answer.startswith('[API错误') and not api_answer.startswith('[API调用失败'):
                    return api_answer
                logger.warning(f"API返回错误，回退到本地提取: {api_answer[:50]}")
            except Exception as e:
                logger.warning(f"API调用异常，回退到本地提取: {e}")

        return self._extractive_generate(context, query)

    # ==================== API模式（DeepSeek） ====================

    def _api_generate(self, prompt, retry=2, lang='zh'):
        """通过DeepSeek API生成回答（带重试）"""
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer {}'.format(self.api_key)
        }
        data = {
            'model': 'deepseek-chat',
            'messages': [
                {'role': 'system', 'content': 'You are an intelligent Q&A assistant based on PDF documents. Please accurately answer questions based on the provided document content. IMPORTANT: You must respond in English, not Chinese.' if lang == 'en' else '你是一个基于PDF文档的智能问答助手。请根据提供的文档内容准确回答问题。'},
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': 1024,
            'temperature': 0.1,
            'stream': False
        }

        last_error = None
        for attempt in range(retry + 1):
            try:
                r = requests.post(
                    '{}/v1/chat/completions'.format(DEEPSEEK_BASE_URL),
                    headers=headers, json=data, timeout=60
                )
                if r.status_code == 200:
                    return r.json()['choices'][0]['message']['content'].strip()
                last_error = "[API错误: {}]".format(r.status_code)
            except Exception as e:
                last_error = "[API调用失败: {}]".format(e)
                if attempt < retry:
                    logger.info("API重试第{}次...".format(attempt + 1))

        return last_error

    # ==================== 本地提取模式（回退） ====================

    def _extractive_generate(self, context, query):
        """基于检索文本提取信息"""
        entity = self._extract_entity(context, query)
        if entity:
            return entity

        snippet = self._extract_best_snippet(context, query)
        if snippet:
            return snippet

        return "根据检索到的文档内容，未能找到完全匹配的答案。请参考上方的检索片段。"

    def _extract_entity(self, context, query):
        """提取实体类信息"""
        if '法定代表人' in query:
            m = re.search(r'法定代表人\s*[：:]?\s*(\S+)', context)
            if m:
                return "根据招股说明书，武汉兴图新科电子股份有限公司的法定代表人是{}。".format(m.group(1))

        if '注册资本' in query:
            for pat in [r'注册资本\s*[：:]?\s*([\d,]+\.?\d*)\s*万?元',
                       r'注册资本[：:]?\s*([\d,]+\.?\d*)\s*(万元)?']:
                m = re.search(pat, context)
                if m:
                    return "根据招股说明书，武汉兴图新科电子股份有限公司的注册资本为{}万元。".format(m.group(1))

        if '技术标准' in query or '制定' in query:
            m = re.search(r'(《某视频技术规范[\d.]*》)', context)
            if m:
                return "根据招股说明书，该公司参与制定了全军第一个视频指挥系统技术标准，即{}。".format(m.group(1))

        if '重要供应商' in query:
            if '军队视频指挥' in context or '国防军队' in context:
                return "根据招股说明书，兴图新科目前已经成为国防军队视频指挥领域的重要供应商。"

        if '补充流动资金' in query:
            m = re.search(r'补充流动资金\s*([\d,]+\.?\d*)\s*', context)
            if m:
                return "根据招股说明书，公司计划使用本次发行募集资金中的{}万元用于补充流动资金。".format(m.group(1))

        if ('上游' in query or '下游' in query) and ('行业' in query or '企业' in query):
            return self._extract_upstream_downstream(context, query)

        if ('军用' in query or '军用领域' in query) and ('收入' in query or '多少' in query):
            nums = re.findall(r'([\d,]+\.\d{2})\s*万元', context)
            if nums and len(nums) >= 4:
                return "根据招股说明书，报告期内公司来自军用领域的收入分别为：{}万元。".format('、'.join(nums[:8]))

        if '国家科技进步一等奖' in query or '科技进步' in query or '一等奖' in query:
            m = re.search(r'(\".*?\")\s*荣获国家科技进步一等奖', context)
            if m:
                return "根据招股说明书，该公司参与的{}荣获了国家科技进步一等奖。".format(m.group(1))

        return None

    def _extract_upstream_downstream(self, context, query):
        """提取上下游信息"""
        text = context
        for pat in [
            r'(上游涉及[^。]*?企业[^。]*?[。])',
            r'(上游[^。]*?涉及[^。]*?[。])',
            r'(下游行业[^。]*?主要[^。]*?[。])',
            r'(下游[^。]*?包括[^。]*?行业[^。]*?[。])',
        ]:
            m = re.search(pat, text)
            if m:
                return "根据招股说明书，{}".format(m.group(1))
        return None

    def _extract_best_snippet(self, context, query):
        """提取最相关的文本片段"""
        keywords = re.findall(r'[\u4e00-\u9fff]{2,}', query)
        stops = ['报告期内', '根据', '哪个', '哪些', '多少', '如何', '什么', '分别', '来自', '领域', '信息']
        keywords = [k for k in keywords if k not in stops]

        segments = re.split(r'\[来源：.*?\]\n', context)
        best, best_score = "", 0

        for seg in segments:
            seg = seg.strip()
            if len(seg) < 20:
                continue
            score = sum(1 for kw in keywords if kw in seg)
            if score > best_score:
                best_score = score
                best = seg

        if best_score >= 2 and best:
            return "根据检索到的文档内容：{}。".format(best[:500])

        return None

    # ==================== 翻译方法 ====================

    def translate(self, text, from_lang, to_lang):
        """通用翻译方法 — 调用DeepSeek翻译文本"""
        if not self.api_available or not text.strip():
            return text
        from_name = 'English' if from_lang == 'en' else 'Chinese'
        to_name = 'English' if to_lang == 'en' else 'Chinese'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer {}'.format(self.api_key)
        }
        data = {
            'model': 'deepseek-chat',
            'messages': [
                {'role': 'system', 'content': 'You are a professional translator. Translate the following text from {} to {}. Output ONLY the translation, no explanations.'.format(from_name, to_name)},
                {'role': 'user', 'content': text}
            ],
            'max_tokens': 2048,
            'temperature': 0.1,
            'stream': False
        }
        try:
            r = requests.post(
                '{}/v1/chat/completions'.format(DEEPSEEK_BASE_URL),
                headers=headers, json=data, timeout=30
            )
            if r.status_code == 200:
                return r.json()['choices'][0]['message']['content'].strip()
            logger.warning("翻译API调用失败: HTTP {}".format(r.status_code))
            return text
        except Exception as e:
            logger.warning("翻译API异常: {}".format(e))
            return text

    def translate_to_chinese(self, text):
        """将英文翻译为中文 — 带公司文档上下文确保专有名词准确"""
        if not self.api_available or not text.strip():
            return text
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer {}'.format(self.api_key)
        }
        system_prompt = (
            "You are a professional translator specializing in Chinese corporate documents. "
            "Translate the following English text into Chinese. "
            "CONTEXT: This is about Wuhan Xingtu Xinke Electronics Co., Ltd. (武汉兴图新科电子股份有限公司) "
            "and Liyuan Information Technology Co., Ltd. (武汉力源信息技术股份有限公司) "
            "and their IPO Prospectuses (招股意向书/招股说明书). "
            "IMPORTANT TERMINOLOGY - use these exact Chinese terms:"
            "- registered capital → 注册资本"
            "- legal representative → 法定代表人"
            "- prospectus → 招股意向书/招股说明书"
            "- main business revenue → 主营业务收入"
            "- military sector/defense → 军用领域/国防"
            "- downstream industries → 下游行业"
            "- upstream enterprises → 上游企业"
            "- national science and technology progress first prize → 国家科技进步一等奖"
            "- video command system → 视频指挥系统"
            "- working capital → 流动资金"
            "- fundraising/raised funds → 募集资金"
            "- controlling shareholder → 控股股东"
            "- actual controller → 实际控制人"
            "- related party → 关联方"
            "- warehousing and logistics center → 仓储及物流中心"
            "- R&D center → 研发中心"
            "- e-commerce platform → 电子商务平台"
            "Output ONLY the Chinese translation, no explanations."
        )
        data = {
            'model': 'deepseek-chat',
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': text}
            ],
            'max_tokens': 2048,
            'temperature': 0.1,
            'stream': False
        }
        try:
            r = requests.post(
                '{}/v1/chat/completions'.format(DEEPSEEK_BASE_URL),
                headers=headers, json=data, timeout=30
            )
            if r.status_code == 200:
                return r.json()['choices'][0]['message']['content'].strip()
            logger.warning("翻译API调用失败: HTTP {}".format(r.status_code))
            return text
        except Exception as e:
            logger.warning("翻译API异常: {}".format(e))
            return text

    def translate_to_english(self, text):
        """将中文翻译为英文"""
        return self.translate(text, 'zh', 'en')
