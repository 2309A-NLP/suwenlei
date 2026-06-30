# -*- coding: utf-8 -*-
# 工单编号: 人工智能NLP-Agent数字人项目-文生图智能体任务
"""Agent 工具集: align_face / generate_view / outpaint / make_contact_sheet。

TOOL_SPECS 是供通义 function-calling 选择的 JSON Schema; ToolRegistry 路由到实现。
"""
from __future__ import annotations

import os
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont

from . import face_utils
from .config import (
    AgentConfig, POSE_INSTRUCTIONS, OUTPAINT_INSTRUCTION,
    VIEW_ORDER, VIEW_LABEL_CN,
)
from .tongyi_backend import TongyiImageBackend


def _load_cjk_font(size: int):
    """加载中文字体用于拼图标注; 找不到退化为默认字体。"""
    for p in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
              r"C:\Windows\Fonts\simsun.ttc"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:  # noqa: BLE001
                continue
    return ImageFont.load_default()


def _tool(name: str, desc: str, props: Dict, required: List[str]) -> Dict:
    """构造一个通义/OpenAI 兼容的 function-calling 工具声明。"""
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required},
    }}


_STR = {"type": "string"}
_VIEW = {"type": "string", "enum": ["left", "front", "right"]}

TOOL_SPECS: List[Dict] = [
    _tool("align_face",
          "检测输入图像中的人脸并对齐裁剪为标准证件照构图，返回对齐后图片路径。流程第一步。",
          {"image_path": _STR}, ["image_path"]),
    _tool("generate_view",
          "基于对齐后的人脸，生成指定朝向的视图，保持同一个人不变。view 取 left(左转30°)/front(端正)/right(右转30°)。",
          {"image_path": _STR, "view": _VIEW}, ["image_path", "view"]),
    _tool("outpaint",
          "对一张人像视图进行扩图(向四周自然延展背景与肩部)，保持人物与背景一致、无拼接痕迹。",
          {"image_path": _STR, "view": _VIEW}, ["image_path", "view"]),
    _tool("make_contact_sheet",
          "把原图与三张最终视图拼成最终效果对比图。当左/端正/右三视图都已扩图完成后调用，结束任务。",
          {"original_path": _STR, "left_path": _STR, "front_path": _STR, "right_path": _STR},
          ["original_path", "left_path", "front_path", "right_path"]),
]


class ToolRegistry:
    """工具实现 + 路由。"""

    def __init__(self, cfg: AgentConfig, out_dir: str):
        self.cfg = cfg
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.backend = TongyiImageBackend(cfg)

    def align_face(self, image_path: str) -> Dict:
        img = face_utils.load_image(image_path)
        if not face_utils.has_face(img):
            return {"ok": False, "error": "未检测到人脸，请提供清晰的正面面部照片。"}
        path = os.path.join(self.out_dir, "00_aligned_face.png")
        face_utils.center_crop_face(img, out_size=768).save(path)
        return {"ok": True, "aligned_path": path}

    def generate_view(self, image_path: str, view: str) -> Dict:
        if view not in POSE_INSTRUCTIONS:
            return {"ok": False, "error": f"未知 view: {view}"}
        save_to = os.path.join(self.out_dir, f"10_view_{view}.png")
        path = self.backend.edit_image(image_path, POSE_INSTRUCTIONS[view], save_to)
        s = self.cfg.infer.view_size                       # 三视图统一到 view_size×view_size
        self.backend.fit_size(path, (s, s))
        return {"ok": True, "view": view, "view_path": path}

    def outpaint(self, image_path: str, view: str) -> Dict:
        save_to = os.path.join(self.out_dir, f"20_outpaint_{view}.png")
        path = self.backend.expand_image(
            image_path, OUTPAINT_INSTRUCTION, self.cfg.infer.outpaint_ratio, save_to,
        )
        return {"ok": True, "view": view, "outpaint_path": path}

    def make_contact_sheet(self, original_path: str, left_path: str, front_path: str, right_path: str) -> Dict:
        try:
            from .postprocess import flatten_backgrounds
            flatten_backgrounds([left_path, front_path, right_path])  # 抠主体换纯色, 三视图背景完全一致
        except Exception:  # noqa: BLE001
            pass
        views = {"left": left_path, "front": front_path, "right": right_path}
        path = os.path.join(self.out_dir, "30_final_contact_sheet.png")
        self._contact_sheet(original_path, views).save(path)
        return {"ok": True, "contact_sheet": path}

    def _contact_sheet(self, original_path: str, views: Dict[str, str]) -> Image.Image:
        cell, pad = 360, 18
        top_h = cell + 44
        W = 3 * cell + 4 * pad
        H = top_h + cell + 44 + pad
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        font = _load_cjk_font(26)

        def paste(path, x, y, label):
            im = Image.open(path).convert("RGB")
            im.thumbnail((cell, cell), Image.LANCZOS)
            canvas.paste(im, (x + (cell - im.width) // 2, y + (cell - im.height) // 2))
            tw = draw.textlength(label, font=font)
            draw.text((x + (cell - tw) // 2, y + cell + 10), label, fill=(30, 30, 30), font=font)

        paste(original_path, (W - cell) // 2, pad, "原图")
        y2 = top_h + pad
        for c, name in enumerate(VIEW_ORDER):
            paste(views[name], pad + c * (cell + pad), y2, VIEW_LABEL_CN[name])
        return canvas

    def dispatch(self, name: str, args: Dict) -> Dict:
        fn = getattr(self, name, None)
        if fn is None:
            return {"ok": False, "error": f"未知工具: {name}"}
        try:
            return fn(**args)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
