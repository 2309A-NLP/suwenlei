# -*- coding: utf-8 -*-
# 工单编号: 人工智能NLP-Agent数字人项目-文生图智能体任务
"""全局配置: 通义 DashScope 后端 + Agent 参数 + 图像编辑指令模板。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class APIConfig:
    """通义 DashScope 接入配置。Key 取自环境变量 DASHSCOPE_API_KEY。"""
    api_key_env: str = "DASHSCOPE_API_KEY"
    llm_model: str = "qwen-plus"            # Agent 大脑 (function calling)
    image_edit_model: str = "qwen-image-edit"  # 转头/扩图图像编辑后端
    max_retries: int = 2


@dataclass
class InferenceConfig:
    """三视图与扩图参数。"""
    yaw_left: int = -30                     # 偏转角(度), 工单要求 ±30° 以内
    yaw_front: int = 0
    yaw_right: int = 30
    outpaint_ratio: float = 0.35            # 扩图四周扩展比例(相对原图边长)
    view_size: int = 1024                   # 三视图统一输出边长(模型返回尺寸不定, 归一到此值)


@dataclass
class RuntimeConfig:
    output_dir: str = "outputs"
    download_timeout: int = 90


# 三视图编辑指令: 各朝向指令 + 公共身份约束 IDENTITY_RULE + 禁止美颜化妆 NO_BEAUTY_RULE。
NO_BEAUTY_RULE: str = (
    "严禁任何美颜、化妆或修饰，必须保持原图本人未经修饰的真实素颜样貌："
    "不要美颜、不要磨皮、不要美白、不要提亮肤色、不要瘦脸、不要削尖下巴、"
    "不要放大眼睛、不要改变脸型与五官比例；"
    "不要化妆，不要添加口红、眼影、粉底、腮红、修容、高光、眉笔或任何彩妆；"
    "不要去除或淡化痣、痘、疤痕、皱纹、毛孔、眼袋、黑眼圈、油光等任何皮肤细节与瑕疵。"
    "完整保留原图真实的肤色、皮肤质感和全部面部细节，看起来与原图是同一个人、同样的素颜状态，"
    "呈现未经任何修图与化妆的真实写实效果。"
)
IDENTITY_RULE: str = (
    "严格保持这个人的身份不变：五官形状、眼睛颜色、肤色、发型、胡须和衣着都与原图完全一致，"
    "不要改变人物长相，不要换脸，不要把人物变好看。证件照构图，纯净的浅灰色背景，"
    "面部清晰锐利、无扭曲变形模糊，高清写实照片。" + NO_BEAUTY_RULE
)

POSE_INSTRUCTIONS: Dict[str, str] = {
    "left": (
        "把这个人的头向左转动约30度，呈现自然的四分之三左侧脸，略微露出左脸颊和左耳廓，"
        "鼻尖朝向画面左前方，视线看向左前方。是轻微转头的四分之三侧脸，不要正脸，也不要转成全侧脸(90度纯侧面)。"
    ) + IDENTITY_RULE,
    "front": "让面部朝向正前方，标准正脸证件照，视线直视镜头。" + IDENTITY_RULE,
    "right": (
        "把这个人的头向右转动约30度，呈现自然的四分之三右侧脸，略微露出右脸颊和右耳廓，"
        "鼻尖朝向画面右前方，视线看向右前方。是轻微转头的四分之三侧脸，不要正脸，也不要转成全侧脸(90度纯侧面)。"
    ) + IDENTITY_RULE,
}

OUTPAINT_INSTRUCTION: str = (
    "这是一张中心为人物、四周为纯色边框的图。请在完全保持中心人物（面部、五官、朝向、衣着）"
    "和已有像素不变的前提下，用自然的内容填满四周边框：向外延展肩部、衣着和更多纯浅灰色影室背景，"
    "与中心区域无缝衔接，不出现拼接痕迹或突兀感，颜色、光影和纹理与原图保持一致，"
    "整体证件照构图，高清写实。不要对中心人物做任何美颜、磨皮、美白、化妆或修饰，"
    "保留原图本人真实的素颜样貌、皮肤质感与瑕疵。"
)

VIEW_ORDER: List[str] = ["left", "front", "right"]
VIEW_LABEL_CN: Dict[str, str] = {"left": "面部左转", "front": "面部端正", "right": "面部右转"}


@dataclass
class AgentConfig:
    api: APIConfig = field(default_factory=APIConfig)
    infer: InferenceConfig = field(default_factory=InferenceConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def api_key(self) -> str:
        key = os.environ.get(self.api.api_key_env, "").strip()
        if not key:
            raise RuntimeError(
                f"未找到通义 API Key, 请设置环境变量 {self.api.api_key_env}。"
            )
        return key
