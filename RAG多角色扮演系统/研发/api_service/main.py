# -*- coding: utf-8 -*-
# main.py
"""
应用入口：创建 FastAPI app，管理生命周期，注册路由
"""
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import ALLOW_ORIGINS, API_HOST, API_PORT
from database import init_db
from rag_project.rag_core import RAGMultiRoleDoctor
from routes import register_routes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理应用启动和关闭时的资源"""
    logger.info("初始化数据库...")
    init_db()
    logger.info("初始化 RAG 多角色机器人...")
    app.state.doctor_bot = RAGMultiRoleDoctor()
    logger.info("初始化完成")
    yield
    # 关闭时释放资源
    logger.info("释放资源...")
    if hasattr(app.state, "doctor_bot") and app.state.doctor_bot:
        app.state.doctor_bot.close()

def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用"""
    logger.info("创建 FastAPI 应用...")
    app = FastAPI(title="RAG多角色会话系统", version="2.1.0", lifespan=lifespan)

    # 添加 CORS 中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOW_ORIGINS,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    register_routes(app)
    logger.info("FastAPI 应用创建完成，路由已注册")
    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=False, workers=1)