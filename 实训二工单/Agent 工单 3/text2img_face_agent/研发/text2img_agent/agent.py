# -*- coding: utf-8 -*-
# 工单编号: 人工智能NLP-Agent数字人项目-文生图智能体任务
"""文生图智能体: 通义大模型 function-calling 自主编排工具完成三视图+扩图。

run(): LLM 自主编排; run_pipeline(): 固定顺序的确定性兜底。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from dashscope import Generation

from .config import AgentConfig, VIEW_ORDER
from .tools import ToolRegistry, TOOL_SPECS


SYSTEM_PROMPT = (
    "你是一个文生图智能体，目标是：给定一张面部图像，生成同一个人的三张证件照视图"
    "（面部左转、端正、右转），并对每张视图扩图，最后拼成一张效果对比图。\n"
    "你必须通过调用提供的工具来完成，不要凭空编造结果。推荐流程：\n"
    "1) 先调用 align_face 对齐人脸；\n"
    "2) 分别用 generate_view 生成 left、front、right 三个视图；\n"
    "3) 对这三个视图分别调用 outpaint 扩图；\n"
    "4) 最后调用 make_contact_sheet 生成最终对比图并结束。\n"
    "每步都要使用上一步返回的真实文件路径。全部完成后用一句话中文总结产物。"
)


@dataclass
class AgentResult:
    aligned_face: Optional[str] = None
    views: Dict[str, str] = field(default_factory=dict)
    outpainted: Dict[str, str] = field(default_factory=dict)
    contact_sheet: Optional[str] = None
    elapsed_sec: float = 0.0
    steps: List[str] = field(default_factory=list)   # 工具调用轨迹
    summary: str = ""


class FaceViewAgent:
    def __init__(self, cfg: Optional[AgentConfig] = None):
        self.cfg = cfg or AgentConfig()
        self.api_key = self.cfg.api_key()

    def run(self, image_path: str, out_dir: Optional[str] = None, max_turns: int = 16) -> AgentResult:
        """LLM 自主编排模式。"""
        t0 = time.time()
        out_dir = out_dir or self.cfg.runtime.output_dir
        registry = ToolRegistry(self.cfg, out_dir)
        result = AgentResult()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"请处理这张面部图像并完成任务，输入图片路径：{image_path}"},
        ]

        for _ in range(max_turns):
            rsp = Generation.call(
                model=self.cfg.api.llm_model,
                messages=messages,
                tools=TOOL_SPECS,
                result_format="message",
                api_key=self.api_key,
            )
            if rsp.status_code != 200:
                raise RuntimeError(f"LLM 调用失败: {rsp.code} {rsp.message}")

            msg = rsp.output.choices[0].message
            messages.append(self._normalize_assistant_msg(msg))
            tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)

            if not tool_calls:
                result.summary = msg.get("content", "") if isinstance(msg, dict) else (msg.content or "")
                break

            for tc in tool_calls:
                name, args = self._parse_tool_call(tc)
                tool_result = registry.dispatch(name, args)
                self._record(result, name, args, tool_result)
                result.steps.append(f"{name}({json.dumps(args, ensure_ascii=False)}) -> ok={tool_result.get('ok')}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": (tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "")) or name,
                    "name": name,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                })

        result.elapsed_sec = round(time.time() - t0, 1)
        return result

    def run_pipeline(self, image_path: str, out_dir: Optional[str] = None) -> AgentResult:
        """确定性模式: 按固定顺序调用工具, 稳定可复现。"""
        t0 = time.time()
        out_dir = out_dir or self.cfg.runtime.output_dir
        reg = ToolRegistry(self.cfg, out_dir)
        result = AgentResult()

        a = reg.align_face(image_path)
        if not a.get("ok"):
            raise ValueError(a.get("error"))
        result.aligned_face = a["aligned_path"]
        result.steps.append("align_face -> ok")

        for v in VIEW_ORDER:
            gv = reg.generate_view(result.aligned_face, v)
            result.views[v] = gv["view_path"]
            result.steps.append(f"generate_view({v}) -> ok")
            op = reg.outpaint(gv["view_path"], v)
            result.outpainted[v] = op["outpaint_path"]
            result.steps.append(f"outpaint({v}) -> ok")

        cs = reg.make_contact_sheet(
            result.aligned_face, result.outpainted["left"],
            result.outpainted["front"], result.outpainted["right"],
        )
        result.contact_sheet = cs["contact_sheet"]
        result.steps.append("make_contact_sheet -> ok")
        result.summary = "已生成左转/端正/右转三视图及其扩图，并拼出最终效果对比图。"
        result.elapsed_sec = round(time.time() - t0, 1)
        return result

    @staticmethod
    def _parse_tool_call(tc):
        if isinstance(tc, dict):
            fn = tc.get("function", {})
            name = fn.get("name")
            raw = fn.get("arguments") or "{}"
        else:
            fn = tc.function
            name = fn.name
            raw = fn.arguments or "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except json.JSONDecodeError:
            args = {}
        return name, args

    @staticmethod
    def _normalize_assistant_msg(msg) -> Dict:
        """把 SDK 的 message 对象转成可回填 messages 的 dict。"""
        if isinstance(msg, dict):
            d = {"role": "assistant", "content": msg.get("content") or ""}
            if msg.get("tool_calls"):
                d["tool_calls"] = msg["tool_calls"]
            return d
        d = {"role": "assistant", "content": getattr(msg, "content", "") or ""}
        if getattr(msg, "tool_calls", None):
            d["tool_calls"] = msg.tool_calls
        return d

    @staticmethod
    def _record(result: AgentResult, name: str, args: Dict, tool_result: Dict):
        if not tool_result.get("ok"):
            return
        if name == "align_face":
            result.aligned_face = tool_result.get("aligned_path")
        elif name == "generate_view":
            result.views[tool_result.get("view")] = tool_result.get("view_path")
        elif name == "outpaint":
            result.outpainted[tool_result.get("view")] = tool_result.get("outpaint_path")
        elif name == "make_contact_sheet":
            result.contact_sheet = tool_result.get("contact_sheet")
