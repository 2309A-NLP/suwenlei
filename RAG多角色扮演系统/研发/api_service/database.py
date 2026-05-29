# -*- coding: utf-8 -*-
# database.py
"""
MySQL 数据库操作模块：初始化表、用户验证、用户注册
"""
import logging
import time
import mysql.connector
from mysql.connector import Error
from fastapi import HTTPException, status
from typing import Optional

from config import MYSQL_CONFIG

logger = logging.getLogger(__name__)


def init_db():
    """初始化数据库：如果 users 表不存在则创建"""
    start_time = time.time()
    logger.info("开始初始化数据库...")
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE if NOT EXISTS users (
                user_id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) NOT NULL UNIQUE,
                password VARCHAR(128) NOT NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        conn.commit()
        elapsed = time.time() - start_time
        logger.info(f"数据库表 users 已初始化，耗时: {elapsed:.4f} 秒")
    except Error as e:
        elapsed = time.time() - start_time
        logger.error(f"数据库初始化失败: {e}, 耗时: {elapsed:.4f} 秒")
        raise
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()
        elapsed = time.time() - start_time
        logger.info(f"数据库初始化连接已关闭，总耗时: {elapsed:.4f} 秒")


def get_db_connection():
    """创建一个新的 MySQL 数据库连接"""
    start_time = time.time()
    logger.debug("获取 MySQL 数据库连接")
    conn = mysql.connector.connect(**MYSQL_CONFIG)
    elapsed = time.time() - start_time
    logger.debug(f"获取 MySQL 数据库连接完成，耗时: {elapsed:.4f} 秒")
    return conn


def verify_user(username: str, password: str) -> Optional[str]:
    """
    校验用户名密码是否匹配。
    返回: 验证成功返回 user_id 字符串，否则返回 None
    异常: 数据库错误时抛出 HTTPException(503)
    """
    start_time = time.time()
    logger.info(f"验证用户登录: {username}")
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        # 移除密码加密，直接比对明文密码
        cursor.execute(
            "SELECT user_id FROM users WHERE username = %s AND password = %s",
            (username, password)
        )
        user = cursor.fetchone()
        if user:
            elapsed = time.time() - start_time
            logger.info(f"用户验证成功: {username}, 耗时: {elapsed:.4f} 秒")
            return str(user["user_id"])
        else:
            elapsed = time.time() - start_time
            logger.warning(f"用户验证失败: {username}, 耗时: {elapsed:.4f} 秒")
            return None
    except Error as e:
        elapsed = time.time() - start_time
        logger.error(f"MySQL验证失败: {e}, 耗时: {elapsed:.4f} 秒")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="数据库服务暂不可用")
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()


def register_user(username: str, password: str) -> bool:
    """
    注册新用户
    返回: True 注册成功，False 用户名已存在
    异常: 数据库错误时抛出 HTTPException(503)
    """
    start_time = time.time()
    logger.info(f"注册新用户: {username}")
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # 检查用户名是否已存在
        cursor.execute("SELECT user_id FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            elapsed = time.time() - start_time
            logger.warning(f"用户名已存在: {username}, 耗时: {elapsed:.4f} 秒")
            return False
        cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
        conn.commit()
        elapsed = time.time() - start_time
        logger.info(f"新用户注册成功: {username}, 耗时: {elapsed:.4f} 秒")
        return True
    except Error as e:
        elapsed = time.time() - start_time
        logger.error(f"注册失败: {e}, 耗时: {elapsed:.4f} 秒")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="注册失败，请稍后重试")
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()