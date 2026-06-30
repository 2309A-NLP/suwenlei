# -*- coding: utf-8 -*-
"""
工单编号: 人工智能NLP-Agent数字人项目-文生图智能体任务
一键演示: 用工单原图跑完整 Agent 流程 (LLM 自主编排三视图 + 扩图 + 拼图)。

运行前需设置: 环境变量 DASHSCOPE_API_KEY
    python run_demo.py            # LLM 自主编排
    python run_demo.py pipeline   # 确定性流水线兜底
"""
from __future__ import annotations

import os
import sys

from text2img_agent import FaceViewAgent

INPUT = os.path.join("inputs", "original_face.png")


def main() -> None:
    mode_pipeline = len(sys.argv) > 1 and sys.argv[1] == "pipeline"
    agent = FaceViewAgent()
    res = (agent.run_pipeline if mode_pipeline else agent.run)(INPUT, out_dir="outputs")
    print("=" * 56)
    print("演示完成, 用时 %.1fs" % res.elapsed_sec)
    print("最终效果对比图:", res.contact_sheet)
    print("工具调用轨迹:")
    for s in res.steps:
        print("  -", s)
    print("=" * 56)


if __name__ == "__main__":
    main()
