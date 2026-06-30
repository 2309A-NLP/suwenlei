# -*- coding: utf-8 -*-
# 工单编号: 人工智能NLP-Agent数字人项目-文生图智能体任务
"""验收指标自动评估: 产物齐全/人脸保持/朝向差异/清晰度/扩图效果, 写入 docs/测试结果.json。"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

OUT = "outputs"
DOCS = "docs"
VIEWS = ["left", "front", "right"]
CN = {"left": "左转", "front": "端正", "right": "右转"}


def _cascade(name: str) -> cv2.CascadeClassifier:
    return cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, name))


def detect_face_any(img: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """正脸 + 左右侧脸级联综合检测, 返回最大人脸框 (兼容三视图的侧脸)。"""
    arr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    boxes = []
    for casc in ("haarcascade_frontalface_default.xml", "haarcascade_profileface.xml"):
        c = _cascade(casc)
        for f in c.detectMultiScale(gray, 1.1, 4, minSize=(60, 60)):
            boxes.append(tuple(int(v) for v in f))
    # 侧脸级联默认只识别一个方向, 翻转再检一次覆盖另一侧
    cprof = _cascade("haarcascade_profileface.xml")
    for f in cprof.detectMultiScale(cv2.flip(gray, 1), 1.1, 4, minSize=(60, 60)):
        x, y, w, h = (int(v) for v in f)
        boxes.append((gray.shape[1] - x - w, y, w, h))
    if not boxes:
        return None
    return max(boxes, key=lambda b: b[2] * b[3])


def _lap_var(img: Image.Image) -> float:
    g = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _face_metrics(img: Image.Image) -> Dict:
    """返回人脸: 是否检出, 中心水平位置(0~1), 面积占比。"""
    box = detect_face_any(img)
    if box is None:
        return {"detected": False, "cx": None, "area_ratio": None}
    x, y, w, h = box
    W, H = img.size
    return {
        "detected": True,
        "cx": round((x + w / 2) / W, 3),
        "area_ratio": round((w * h) / (W * H), 3),
    }


def _edge_bg_color(img: Image.Image):
    a = np.array(img.convert("RGB"))
    p = 12
    corners = np.concatenate([
        a[:p, :p].reshape(-1, 3), a[:p, -p:].reshape(-1, 3),
        a[-p:, :p].reshape(-1, 3), a[-p:, -p:].reshape(-1, 3),
    ])
    return corners.mean(axis=0)


def evaluate() -> Dict:
    report: Dict = {"功能层面": {}, "面部特征保持": {}, "角度差异性": {},
                    "图像清晰度": {}, "扩图效果": {}}

    aligned_p = os.path.join(OUT, "00_aligned_face.png")

    # 功能层面: 产物齐全性
    needed = [aligned_p, os.path.join(OUT, "30_final_contact_sheet.png")]
    needed += [os.path.join(OUT, f"10_view_{v}.png") for v in VIEWS]
    needed += [os.path.join(OUT, f"20_outpaint_{v}.png") for v in VIEWS]
    exists = {os.path.basename(p): os.path.exists(p) for p in needed}
    report["功能层面"] = {"产物齐全": all(exists.values()), "明细": exists}

    fmetrics: Dict[str, Dict] = {}
    for v in VIEWS:
        op = os.path.join(OUT, f"20_outpaint_{v}.png")   # 以最终交付的扩图为准
        if not os.path.exists(op):
            continue
        im = Image.open(op)
        fm = _face_metrics(im)
        fmetrics[v] = fm
        report["面部特征保持"][CN[v]] = fm["detected"]
        report["图像清晰度"][CN[v]] = round(_lap_var(im), 1)

    # 角度差异性: 三视图人脸水平中心彼此不同 -> 朝向有变化
    cxs = {v: fmetrics[v]["cx"] for v in VIEWS if v in fmetrics and fmetrics[v]["cx"] is not None}
    report["角度差异性"]["人脸水平中心"] = {CN[v]: cxs[v] for v in cxs}
    if len(cxs) >= 2:
        spread = round(max(cxs.values()) - min(cxs.values()), 3)
        report["角度差异性"]["三视图朝向有差异"] = spread > 0.04
        report["角度差异性"]["水平位置跨度"] = spread

    # 扩图效果: 主体占比下降(背景变多) + 背景色一致
    for v in VIEWS:
        vp, op = os.path.join(OUT, f"10_view_{v}.png"), os.path.join(OUT, f"20_outpaint_{v}.png")
        if not (os.path.exists(vp) and os.path.exists(op)):
            continue
        view, out = Image.open(vp), Image.open(op)
        fv, fo = _face_metrics(view), _face_metrics(out)
        color_diff = float(np.abs(_edge_bg_color(view) - _edge_bg_color(out)).mean())
        shrink = None
        if fv["area_ratio"] and fo["area_ratio"]:
            shrink = fo["area_ratio"] < fv["area_ratio"]   # 扩图后主体占比应更小
        report["扩图效果"][CN[v]] = {
            "主体占比_视图->扩图": [fv["area_ratio"], fo["area_ratio"]],
            "背景扩展成功(主体占比下降)": shrink,
            "扩图清晰度(lapvar)": round(_lap_var(out), 1),
            "背景色差(越小越一致)": round(color_diff, 1),
            "背景色一致": color_diff < 20,
        }

    return report


def main() -> None:
    rep = evaluate()
    os.makedirs(DOCS, exist_ok=True)
    with open(os.path.join(DOCS, "测试结果.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
