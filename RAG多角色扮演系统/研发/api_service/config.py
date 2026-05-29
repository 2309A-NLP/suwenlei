# -*- coding: utf-8 -*-
# config.py
"""
配置模块：所有环境变量和常量
"""
import os

# MySQL 连接配置
MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", "123456"),
    "database": os.getenv("MYSQL_DB", "2309a"),
}

# Redis 会话存储相关配置
REDIS_CHAT_PREFIX = "hypertension_chat"   # Redis key 前缀
SESSION_EXPIRE_SECONDS = 86400            # 会话过期时间（秒），24小时

# API 服务监听地址与端口
API_HOST = os.getenv("API_HOST", "10.223.11.86")
API_PORT = int(os.getenv("API_PORT", 8001))

# CORS 允许的来源（逗号分隔）
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*").split(",")