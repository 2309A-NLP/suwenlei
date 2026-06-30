# -*- coding: utf-8 -*-
# 工单编号: 人工智能NLP-Agent数字人项目-文生图智能体任务
"""后处理: 把多张图的背景灰度归一到同一目标值, 仅平移背景区、保护主体。"""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image


def _edge_bg_color(arr: np.ndarray, p: int = 14) -> np.ndarray:
    corners = np.concatenate([
        arr[:p, :p].reshape(-1, 3), arr[:p, -p:].reshape(-1, 3),
        arr[-p:, :p].reshape(-1, 3), arr[-p:, -p:].reshape(-1, 3),
    ])
    return corners.mean(axis=0)


def _bg_mask(arr: np.ndarray, bg: np.ndarray, tol: float = 26.0) -> np.ndarray:
    """背景掩码: 与边缘背景色接近的像素 (0~1, 已羽化)。"""
    dist = np.linalg.norm(arr.astype(np.float32) - bg[None, None, :], axis=2)
    m = np.clip(1.0 - dist / tol, 0.0, 1.0).astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), sigmaX=3)
    return m


def unify_backgrounds(
    image_paths: List[str],
    target_gray: Tuple[int, int, int] = (229, 230, 232),
    save: bool = True,
) -> List[str]:
    """
    把多张图的背景归一到 target_gray。原地保存(save=True)并返回路径列表。
    仅平移背景区像素, 主体通过掩码保护。
    """
    target = np.array(target_gray, dtype=np.float32)
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        arr = np.array(img).astype(np.float32)
        bg = _edge_bg_color(arr)
        shift = target - bg                      # 让背景平移到目标灰度
        mask = _bg_mask(arr, bg)[:, :, None]     # 只动背景
        out = arr + shift[None, None, :] * mask
        out = np.clip(out, 0, 255).astype(np.uint8)
        if save:
            Image.fromarray(out).save(path)
    return image_paths


def _subject_mask_grabcut(arr: np.ndarray, margin: float = 0.12, iters: int = 5) -> np.ndarray:
    """GrabCut 抠主体, 返回前景概率掩码 (H,W) float 0~1, 已羽化。

    主体居中, 用画面中央矩形作为初始前景框; 失败时返回全零(交由上层回退)。
    """
    h, w = arr.shape[:2]
    bgr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
    rect = (int(w * margin), int(h * margin), int(w * (1 - 2 * margin)), int(h * (1 - 2 * margin)))
    gc = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, gc, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    fg = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
    if fg.mean() < 0.05 or fg.mean() > 0.97:     # 抠失败(几乎全空或全满)
        return np.zeros((h, w), np.float32)
    fg = cv2.GaussianBlur(fg, (0, 0), sigmaX=2.0)
    return np.clip(fg, 0.0, 1.0)


def flatten_backgrounds(
    image_paths: List[str],
    target_gray: Tuple[int, int, int] = (229, 230, 232),
    save: bool = True,
) -> List[str]:
    """抠出主体, 把背景整体替换为纯色 target_gray; 三视图背景因此完全一致。

    对渐变/暗角背景比 unify_backgrounds 更彻底; GrabCut 失败的图自动回退色彩平移。
    """
    target = np.array(target_gray, dtype=np.float32)
    fallback: List[str] = []
    for path in image_paths:
        arr = np.array(Image.open(path).convert("RGB")).astype(np.float32)
        m = _subject_mask_grabcut(arr)[:, :, None]
        if m.max() <= 0.0:                       # 抠失败 → 收集起来走旧逻辑
            fallback.append(path)
            continue
        flat = np.broadcast_to(target, arr.shape)
        out = (arr * m + flat * (1.0 - m)).clip(0, 255).astype(np.uint8)
        if save:
            Image.fromarray(out).save(path)
    if fallback:
        unify_backgrounds(fallback, target_gray=target_gray, save=save)
    return image_paths
