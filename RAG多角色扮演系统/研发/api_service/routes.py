# -*- coding: utf-8 -*-
# routes.py
"""
路由模块：定义所有 API 端点
"""

import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from rag_project.config import ROLE_TO_COLLECTION
from fastapi import HTTPException, status, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from config import REDIS_CHAT_PREFIX, SESSION_EXPIRE_SECONDS
from models import ChatResponse, ChatData, UserRegisterRequest, UserLoginRequest, ChatRequest
from database import register_user, verify_user

logger = logging.getLogger(__name__)

def register_routes(app):
    """注册所有路由到 FastAPI app"""
    logger.info("开始注册 API 路由...")
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    templates = Jinja2Templates(directory=PROJECT_ROOT / "templates")

    @app.get("/", response_class=HTMLResponse)
    async def homepage(request: Request):
        logger.info("访问根路径 /")
        return templates.TemplateResponse("index.html", {"request": request})

    @app.post("/api/register", response_model=ChatResponse)
    async def user_register(request: UserRegisterRequest):
        """用户注册接口"""
        logger.info(f"收到注册请求，用户名: {request.username}")
        success = register_user(request.username, request.password)
        if not success:
            logger.warning(f"注册失败，用户名已存在: {request.username}")
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="用户名已存在")
        logger.info(f"用户注册成功: {request.username}")
        return ChatResponse(message="注册成功", data=ChatData(user_id=""))

    @app.post("/api/login", response_model=ChatResponse)
    async def user_login(request: UserLoginRequest, req: Request):
        """用户登录接口"""
        logger.info(f"收到登录请求，用户名: {request.username}")
        user_id = verify_user(request.username, request.password)
        if not user_id:
            logger.warning(f"登录失败，用户名或密码错误: {request.username}")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")
        session_key = f"{REDIS_CHAT_PREFIX}:{user_id}"
        doctor_bot = req.app.state.doctor_bot
        doctor_bot.redis_client.setex(session_key, SESSION_EXPIRE_SECONDS, json.dumps([]))
        logger.info(f"用户登录成功: {request.username} (ID: {user_id})，会话已创建")
        return ChatResponse(message="登录成功", data=ChatData(user_id=str(user_id)))

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat_interaction(request: ChatRequest, req: Request):
        """核心聊天接口"""
        logger.info(f"收到聊天请求，用户ID: {request.user_id}, 角色: {request.role}, 问题: {request.user_question[:50]}...")
        session_key = f"{REDIS_CHAT_PREFIX}:{request.user_id}"
        doctor_bot = req.app.state.doctor_bot
        if not doctor_bot.redis_client.exists(session_key):
            logger.warning(f"会话不存在或已过期: {request.user_id}")
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="未登录或会话过期")

        try:
            # 同步的 RAG 操作丢到线程池跑，不阻塞事件循环
            answer = await asyncio.to_thread(
                doctor_bot.generate_answer,
                request.user_id, request.user_question, request.role
            )
            logger.info(f"生成回答成功，用户ID: {request.user_id}")
        except asyncio.TimeoutError:
            logger.error(f"生成回答超时，用户ID: {request.user_id}")
            answer = "抱歉，请求超时，请稍后再试。"
        except Exception as e:
            logger.error(f"生成回答失败: {e}")
            answer = "抱歉，暂时无法回答您的问题。"

        print(f"用户 {request.user_id} 的问题: {request.user_question}")
        print(f"机器人回答: {answer}")

        return ChatResponse(
            data=ChatData(
                user_id=request.user_id,
                user_question=request.user_question,
                bot_answer=answer,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        )

    @app.get("/api/health")
    async def health_check(req: Request):
        """健康检查接口（使用 MilvusClient 替代 Collection）"""
        logger.info("执行健康检查...")
        doctor_bot = req.app.state.doctor_bot
        milvus_ok = False
        redis_ok = False
        bm25_ready = False

        # 检查 Milvus 连接：尝试获取任意集合的统计信息
        try:
            if doctor_bot and doctor_bot.milvus_client:
                # 使用第一个已知的知识库集合进行探测
                first_coll = next(iter(ROLE_TO_COLLECTION.values()))
                stats = doctor_bot.milvus_client.get_collection_stats(first_coll)
                if stats and stats.get("row_count", -1) >= 0:
                    milvus_ok = True
                    logger.info("Milvus 健康检查通过")
        except Exception as e:
            logger.warning(f"Milvus health check failed: {e}")

        # 检查 Redis
        try:
            redis_ok = doctor_bot.redis_client.ping() if doctor_bot else False
            logger.info(f"Redis 健康检查: {'通过' if redis_ok else '失败'}")
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")

        # 检查 BM25
        try:
            bm25_ready = all(
                coll_name in doctor_bot.bm25_handler.bm25_indexes
                for coll_name in ROLE_TO_COLLECTION.values()
            ) if doctor_bot and doctor_bot.bm25_handler else False
            logger.info(f"BM25 健康检查: {'就绪' if bm25_ready else '未就绪'}")
        except Exception:
            pass

        overall = "healthy" if (milvus_ok and redis_ok and bm25_ready) else "degraded"
        logger.info(f"健康检查总体状态: {overall}")
        return {
            "status": overall,
            "services": {
                "milvus": "connected" if milvus_ok else "disconnected",
                "redis": "connected" if redis_ok else "disconnected",
                "bm25": "ready" if bm25_ready else "unavailable",
                "llm": "ready"
            }
        }