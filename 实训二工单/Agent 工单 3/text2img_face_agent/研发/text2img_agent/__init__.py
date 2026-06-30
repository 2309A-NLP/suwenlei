# -*- coding: utf-8 -*-
"""
工单编号: 人工智能NLP-Agent数字人项目-文生图智能体任务

文生图智能体: 给定一张面部图像, 用通义(DashScope) 图像编辑后端生成
"左转 / 端正 / 右转"三张视图并扩图; 通义大模型作为 Agent 大脑编排工具。
"""
from .agent import FaceViewAgent, AgentResult
from .config import AgentConfig

__all__ = ["FaceViewAgent", "AgentResult", "AgentConfig"]
__version__ = "2.0.0"
