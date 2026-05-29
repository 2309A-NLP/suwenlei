# -*- coding: utf-8 -*-
# models.py
"""
Pydantic 数据模型（用于请求/响应校验）
"""
from pydantic import BaseModel, Field
from typing import Optional

class ChatData(BaseModel):
    """单次聊天的数据结构，用于 API 响应中的 data 字段"""
    user_id: str = ""
    user_question: str = ""
    bot_answer: str = ""
    timestamp: str = ""

class ChatResponse(BaseModel):
    """统一的 API 响应格式"""
    code: int = 200
    message: str = "success"
    data: ChatData = Field(default_factory=ChatData)

class UserLoginRequest(BaseModel):
    """登录请求体模型"""
    username: str = Field(..., min_length=1, max_length=50)
    password: str = Field(..., min_length=1, max_length=128)

class UserRegisterRequest(BaseModel):
    """注册请求体模型"""
    username: str = Field(..., min_length=1, max_length=50)
    password: str = Field(..., min_length=6, max_length=128)

class ChatRequest(BaseModel):
    """聊天请求体模型"""
    user_id: str = Field(..., min_length=1, max_length=50)
    user_question: str = Field(..., min_length=1, max_length=500)
    role: str = Field("hypertension", max_length=50)