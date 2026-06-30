# -*- coding: utf-8 -*-
"""
工单编号: 人工智能NLP-Agent数字人项目-文生图智能体任务
冒烟测试: 校验"非网络"部分逻辑 (人脸对齐 / 配置 / 工具声明 / 拼图),
无需通义 API 即可运行。

    python tests/test_smoke.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text2img_agent import face_utils
from text2img_agent.config import AgentConfig, POSE_INSTRUCTIONS, VIEW_ORDER
from text2img_agent.tools import TOOL_SPECS, ToolRegistry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(ROOT, "inputs", "original_face.png")


def test_config_three_views():
    assert VIEW_ORDER == ["left", "front", "right"]
    assert set(POSE_INSTRUCTIONS) == {"left", "front", "right"}
    cfg = AgentConfig()
    assert cfg.infer.yaw_left < 0 < cfg.infer.yaw_right
    assert abs(cfg.infer.yaw_left) <= 30 and cfg.infer.yaw_right <= 30  # 工单: ±30°以内


def test_tool_specs_wellformed():
    names = {t["function"]["name"] for t in TOOL_SPECS}
    assert names == {"align_face", "generate_view", "outpaint", "make_contact_sheet"}
    for t in TOOL_SPECS:
        fn = t["function"]
        assert t["type"] == "function" and fn["description"]
        assert fn["parameters"]["type"] == "object"


def test_face_detection_and_align():
    img = face_utils.load_image(INPUT)
    assert face_utils.has_face(img), "应在工单原图中检测到人脸"
    sq = face_utils.center_crop_face(img, out_size=768)
    assert sq.size == (768, 768)


def test_align_tool_no_network(tmp_out="outputs/_test_tmp"):
    """align_face 工具是纯本地, 可在无 API key 下单独验证(临时清空 key)。"""
    import os as _os
    saved = _os.environ.pop("DASHSCOPE_API_KEY", None)
    try:
        # 不实例化 ToolRegistry(会校验 key), 直接测对齐逻辑等价路径
        img = face_utils.load_image(INPUT)
        aligned = face_utils.center_crop_face(img, 768)
        _os.makedirs(tmp_out, exist_ok=True)
        p = _os.path.join(tmp_out, "aligned.png")
        aligned.save(p)
        assert _os.path.exists(p)
    finally:
        if saved is not None:
            _os.environ["DASHSCOPE_API_KEY"] = saved


def _run_all():
    import os as _os
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("全部冒烟测试通过")


if __name__ == "__main__":
    _run_all()
