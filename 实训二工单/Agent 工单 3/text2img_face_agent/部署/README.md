# 文生图智能体（面部三视图 + 扩图）

> 工单编号：**人工智能NLP-Agent数字人项目-文生图智能体任务**
> Agent 数字人项目 · 八维文化与产业研究院

输入一张**面部图像**，自动生成同一个人的**左转 / 端正 / 右转**三张证件照视图，
并对每张视图**扩图**，最后拼成一张效果对比图。

## 架构：LLM 工具编排 Agent

- **大脑**：通义 `qwen-plus`，通过 function-calling 自主编排工具；
- **图像后端**：通义 `qwen-image-edit`（指令式图像编辑，保持同一个人改变朝向）；
- **工具**：`align_face`（人脸对齐）/ `generate_view`（转头）/ `outpaint`（扩图）/ `make_contact_sheet`（拼图）。

```
人脸 → align_face → generate_view×3(左/正/右) → outpaint×3 → make_contact_sheet → 对比图
```

## 快速开始

```bash
conda activate nlp_1
pip install -r requirements.txt
set DASHSCOPE_API_KEY=你的通义key          # PowerShell: $env:DASHSCOPE_API_KEY="..."

python run_demo.py            # LLM 自主编排（Agent 模式）
python run_demo.py pipeline   # 确定性流水线（稳定兜底）
python app.py --ui            # Web UI（需 pip install gradio）
python evaluate.py            # 验收评估
python tests/test_smoke.py    # 冒烟测试（无需 API）
```

## 产物（`outputs/`）

| 文件 | 说明 |
|---|---|
| `00_aligned_face.png` | 对齐后的输入脸 |
| `10_view_{left,front,right}.png` | 三视图（转头） |
| `20_outpaint_{left,front,right}.png` | 三视图扩图 |
| `30_final_contact_sheet.png` | **最终效果对比图** |

## 文档

- `docs/实现步骤与说明.md` —— 实现步骤、方案取舍、过程问题记录
- `docs/测试报告.md` —— 测试用例与验收对照
- `docs/测试结果.json` —— `evaluate.py` 输出的量化指标
- `comfyui/README.md` —— ComfyUI / 本地 SD 备选实现

## 说明

- 两种运行模式产出相同文件：`run`（LLM 自主编排）演示智能体能力，`run_pipeline`（确定性）用于稳定复现。
- 图像生成走通义云端，**无需本地 GPU/大模型**。
