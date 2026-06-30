# -*- coding: utf-8 -*-
# 工单编号: 人工智能NLP-Agent数字人项目-文生图智能体任务
"""命令行 / Gradio 入口 (通义后端, 需设置环境变量 DASHSCOPE_API_KEY)。

  python app.py --image inputs/original_face.png            # LLM 自主编排
  python app.py --image inputs/original_face.png --pipeline # 确定性流水线
  python app.py --ui                                        # Web UI (需 gradio)
"""
from __future__ import annotations

import argparse
import os
import sys

from text2img_agent import FaceViewAgent, AgentConfig


def _print_result(res) -> None:
    print("[Agent] 完成, 用时 %.1fs" % res.elapsed_sec)
    print("  对齐人脸 :", res.aligned_face)
    for k in ("left", "front", "right"):
        if k in res.views:
            print(f"  三视图[{k}] :", res.views[k])
    for k in ("left", "front", "right"):
        if k in res.outpainted:
            print(f"  扩图  [{k}] :", res.outpainted[k])
    print("  最终拼图 :", res.contact_sheet)
    print("  工具轨迹 :")
    for s in res.steps:
        print("    -", s)
    if res.summary:
        print("  总结     :", res.summary)


def run_cli(args) -> None:
    agent = FaceViewAgent(AgentConfig())
    print(f"[Agent] 处理: {args.image}  (模式: {'确定性流水线' if args.pipeline else 'LLM 自主编排'})")
    if args.pipeline:
        res = agent.run_pipeline(args.image, out_dir=args.out)
    else:
        res = agent.run(args.image, out_dir=args.out)
    _print_result(res)


def launch_ui() -> None:
    import gradio as gr

    agent = FaceViewAgent(AgentConfig())

    def _infer(image_path, use_pipeline):
        res = (agent.run_pipeline if use_pipeline else agent.run)(image_path, out_dir="outputs")
        gallery = [res.outpainted[k] for k in ("left", "front", "right") if k in res.outpainted]
        return res.contact_sheet, gallery, "\n".join(res.steps) + "\n\n" + res.summary

    with gr.Blocks(title="文生图智能体-面部三视图") as demo:
        gr.Markdown("# 文生图智能体: 面部左转/端正/右转 + 扩图\n"
                    "工单: 人工智能NLP-Agent数字人项目-文生图智能体任务 · 后端: 通义 qwen-image-edit")
        with gr.Row():
            inp = gr.Image(type="filepath", label="原始面部图像")
            with gr.Column():
                pipe_cb = gr.Checkbox(value=False, label="确定性流水线(不经LLM决策)")
                btn = gr.Button("生成", variant="primary")
        sheet = gr.Image(label="最终效果对比图")
        gallery = gr.Gallery(label="三视图(扩图后)", columns=3)
        trace = gr.Textbox(label="Agent 工具调用轨迹", lines=8)
        btn.click(_infer, [inp, pipe_cb], [sheet, gallery, trace])
    demo.launch()


def main() -> None:
    ap = argparse.ArgumentParser(description="文生图智能体: 面部三视图生成+扩图 (通义后端)")
    ap.add_argument("--image", help="输入面部图像路径")
    ap.add_argument("--out", default="outputs", help="输出目录")
    ap.add_argument("--pipeline", action="store_true", help="用确定性流水线(不经LLM决策)")
    ap.add_argument("--ui", action="store_true", help="启动 Gradio Web UI")
    args = ap.parse_args()

    if args.ui:
        launch_ui()
        return
    if not args.image:
        ap.error("请提供 --image 或使用 --ui")
    if not os.path.exists(args.image):
        sys.exit(f"找不到输入图像: {args.image}")
    run_cli(args)


if __name__ == "__main__":
    main()
