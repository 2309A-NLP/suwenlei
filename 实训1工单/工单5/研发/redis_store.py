# -*- coding: utf-8 -*-
"""
多轮对话Redis存储模块 — 对话历史持久化

核心设计：
- 每个conversation_id对应一个List结构，存储消息序列
- key格式: rag:conv:{conversation_id}
- Redis不可用时自动降级为内存字典（开发/单机场景）
"""
import json
import uuid
import logging
from datetime import timedelta

import redis

logger = logging.getLogger(__name__)


class ConversationStore:
    """对话历史存储：Redis优先，自动降级内存"""

    def __init__(self, host='127.0.0.1', port=6379, db=0, password=None, ttl_hours=24):
        self.ttl = timedelta(hours=ttl_hours)  # 对话过期时间：默认24小时
        self.prefix = 'rag:conv:'  # Redis key前缀，隔离业务数据
        try:
            self.rclient = redis.Redis(
                host=host, port=port, db=db, password=password,
                decode_responses=True, socket_connect_timeout=5,
            )
            self.rclient.ping()  # 验证连接可用性
            self._available = True
            logger.info("Redis连接成功: {}:{}/{}".format(host, port, db))
        except Exception as e:
            # 降级策略：Redis不可用时使用内存字典（进程重启丢失）
            self._available = False
            self.rclient = None
            self._fallback = {}
            logger.warning("Redis不可用，降级内存存储: {}".format(e))

    @property
    def available(self):
        return self._available

    def _key(self, conversation_id):
        return self.prefix + conversation_id

    def generate_id(self):
        return str(uuid.uuid4())

    def get_history(self, conversation_id):
        if not conversation_id:
            return []
        if self._available:
            try:
                raw = self.rclient.get(self._key(conversation_id))
                return json.loads(raw) if raw else []
            except Exception as e:
                logger.warning("Redis读取失败: {}".format(e))
                return []
        else:
            return self._fallback.get(conversation_id, [])

    def save_history(self, conversation_id, history, ttl=None):
        if not conversation_id:
            return
        if ttl is None:
            ttl = self.ttl
        if self._available:
            try:
                data = json.dumps(history, ensure_ascii=False)
                self.rclient.setex(self._key(conversation_id), ttl, data)
            except Exception as e:
                logger.warning("Redis写入失败: {}".format(e))
        else:
            self._fallback[conversation_id] = history

    def append_message(self, conversation_id, role, content):
        """追加单条消息：读取→追加→回写（保证原子性，非高频场景可接受）"""
        history = self.get_history(conversation_id)
        history.append({'role': role, 'content': content})
        self.save_history(conversation_id, history)

    def delete_conversation(self, conversation_id):
        if self._available:
            try:
                self.rclient.delete(self._key(conversation_id))
            except Exception as e:
                logger.warning("Redis删除失败: {}".format(e))
        else:
            self._fallback.pop(conversation_id, None)

    def clear_all(self):
        if self._available:
            try:
                keys = self.rclient.keys(self.prefix + '*')
                if keys:
                    self.rclient.delete(*keys)
                    logger.info("Redis清空{}个key".format(len(keys)))
            except Exception as e:
                logger.warning("Redis清空失败: {}".format(e))
        else:
            self._fallback.clear()
