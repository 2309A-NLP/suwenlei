# -*- coding: utf-8 -*-
# 工单编号: 人工智能NLP-Agent数字人项目-文生图智能体任务
"""通义(DashScope) 图像编辑后端: 转头(edit_image) 与扩图(expand_image)。"""
from __future__ import annotations

import os
import time
import urllib.request

import numpy as np
from PIL import Image
from dashscope import MultiModalConversation

from .config import AgentConfig
from . import face_utils


def _to_file_uri(path: str) -> str:
    return "file://" + os.path.abspath(path).replace("\\", "/")


def _edge_bg_color(img: Image.Image, p: int = 8) -> tuple:
    """取四角 p×p 区域均值作为背景色。"""
    a = np.array(img.convert("RGB"))
    corners = np.concatenate([
        a[:p, :p].reshape(-1, 3), a[:p, -p:].reshape(-1, 3),
        a[-p:, :p].reshape(-1, 3), a[-p:, -p:].reshape(-1, 3),
    ])
    return tuple(int(v) for v in corners.mean(axis=0).round())


class TongyiImageBackend:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.api_key = cfg.api_key()

    def edit_image(self, image_path: str, instruction: str, save_to: str) -> str:
        """按 instruction 编辑 image_path, 结果存到 save_to 并返回路径。"""
        messages = [{"role": "user", "content": [
            {"image": _to_file_uri(image_path)},
            {"text": instruction},
        ]}]
        last_err = None
        for attempt in range(self.cfg.api.max_retries + 1):
            try:
                rsp = MultiModalConversation.call(
                    model=self.cfg.api.image_edit_model,
                    messages=messages,
                    api_key=self.api_key,
                )
                if rsp.status_code == 200:
                    url = self._extract_image_url(rsp)
                    if url:
                        return self._download(url, save_to)
                    last_err = "响应中未找到图像 URL"
                else:
                    last_err = f"status={rsp.status_code} code={rsp.code} msg={rsp.message}"
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"通义图像编辑失败({self.cfg.api.image_edit_model}): {last_err}")

    def expand_image(self, image_path: str, instruction: str, ratio: float, save_to: str) -> str:
        """扩图: 先把图居中贴到放大画布, 再让模型只补四周(主体不被擦除)。

        生成后校验人脸仍在, 失败则重试; 最终仍失败回退本地灰底画布(不致空白)。
        """
        src = Image.open(image_path).convert("RGB")
        w, h = src.size
        m = int(round(min(w, h) * ratio))
        canvas = Image.new("RGB", (w + 2 * m, h + 2 * m), _edge_bg_color(src))
        canvas.paste(src, (m, m))
        pre_path = save_to.replace(".png", "_pre.png")
        canvas.save(pre_path)

        last_path = None
        for attempt in range(self.cfg.api.max_retries + 1):
            try:
                out = self.edit_image(pre_path, instruction, save_to)
                self.fit_size(out, canvas.size)        # 归一到画布尺寸, 保证各视图扩图等大
                if face_utils.has_face_any(Image.open(out)):
                    self._safe_remove(pre_path)
                    return out
                last_path = out
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1.0 * (attempt + 1))
        canvas.save(save_to)                   # 兜底: 主体保真, 四周纯色
        self._safe_remove(pre_path)
        return last_path or save_to

    @staticmethod
    def fit_size(path: str, size: tuple) -> str:
        """把图像归一到指定 (w, h); 模型返回尺寸不定时用于统一各图大小。"""
        im = Image.open(path)
        if im.size != tuple(size):
            im.convert("RGB").resize(tuple(size), Image.LANCZOS).save(path)
        return path

    @staticmethod
    def _safe_remove(path: str) -> None:
        try:
            os.remove(path)
        except OSError:
            pass

    @staticmethod
    def _extract_image_url(rsp) -> "str | None":
        try:
            for item in rsp.output["choices"][0]["message"]["content"]:
                if isinstance(item, dict) and item.get("image"):
                    return item["image"]
        except Exception:  # noqa: BLE001
            return None
        return None

    def _download(self, url: str, save_to: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(save_to)), exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=self.cfg.runtime.download_timeout) as r:
            data = r.read()
        with open(save_to, "wb") as f:
            f.write(data)
        return save_to
