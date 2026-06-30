# -*- coding: utf-8 -*-
# 工单编号: 人工智能NLP-Agent数字人项目-文生图智能体任务
"""人脸预处理: OpenCV Haar 检测 + 以人脸为中心裁成证件照构图。"""
from __future__ import annotations

import io
import os
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

_HAAR = "haarcascade_frontalface_default.xml"


def detect_main_face(img: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """检测面积最大的正脸, 返回 (x, y, w, h); 未检出返回 None。"""
    arr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, _HAAR))
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)


def has_face(img: Image.Image) -> bool:
    return detect_main_face(img) is not None


_PROFILE = "haarcascade_profileface.xml"


def has_face_any(img: Image.Image) -> bool:
    """正脸 / 侧脸(含翻转)任一检出即为真。

    扩图校验用: 转头后的四分之三侧脸或全侧脸正脸级联检不出, 用此避免误杀成灰框。
    """
    if detect_main_face(img) is not None:
        return True
    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    prof = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, _PROFILE))
    for g in (gray, cv2.flip(gray, 1)):    # profileface 只识别朝一侧, 翻转再查另一侧
        if len(prof.detectMultiScale(g, scaleFactor=1.1, minNeighbors=4, minSize=(48, 48))) > 0:
            return True
    return False


def center_crop_face(
    img: Image.Image,
    out_size: int = 512,
    face_ratio: float = 0.62,
    bg_color: Tuple[int, int, int] = (228, 229, 231),
) -> Image.Image:
    """以人脸为中心裁出正方形缩放到 out_size; face_ratio 为人脸高度占画幅比例。

    未检出人脸时退化为等比缩放填充的正方形。
    """
    img = img.convert("RGB")
    box = detect_main_face(img)
    if box is None:
        return _square_pad_resize(img, out_size, bg_color)

    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    side = h / face_ratio
    cy -= 0.08 * side                      # 头顶多留空间
    left, top = cx - side / 2.0, cy - side / 2.0

    canvas = Image.new("RGB", (int(round(side)), int(round(side))), bg_color)
    canvas.paste(img, (int(round(-left)), int(round(-top))))
    return canvas.resize((out_size, out_size), Image.LANCZOS)


def _square_pad_resize(img: Image.Image, out_size: int, bg_color: Tuple[int, int, int]) -> Image.Image:
    w, h = img.size
    scale = out_size / max(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (out_size, out_size), bg_color)
    canvas.paste(resized, ((out_size - nw) // 2, (out_size - nh) // 2))
    return canvas


def load_image(path: str) -> Image.Image:
    """读取为 RGB; 用二进制读规避 Windows 中文路径问题。"""
    with open(path, "rb") as f:
        return Image.open(io.BytesIO(f.read())).convert("RGB")
