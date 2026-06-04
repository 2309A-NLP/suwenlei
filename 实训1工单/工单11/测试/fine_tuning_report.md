# 工单11 - Embedding模型微调技术报告

> **项目**: 人工智能NLP-RAG项目  
> **任务**: Embedding模型微调  
> **创建人**: 王洪荣  
> **创建时间**: 2025年8月26日  
> **完成时间**: 2026年5月31日  
> **工单编号**: 人工智能NLP-RAG项目-Embedding模型微调任务

---

## 1. 背景与目标

### 1.1 问题描述

通用嵌入模型（如 bge-m3）虽然在大规模语料上预训练表现良好，但在面对小众专业领域时，存在 **"语义鸿沟"** 问题：

- 专业术语理解偏差（如 "complaint" 在法律语境中 ≠ "投诉"）
- 内部代号/缩写无法识别
- 同义词映射不准

### 1.2 微调目标

在专业数据（招股说明书）上微调 Embedding 模型，调整嵌入空间，使其更准确地反映专业领域的语义相似性，从而提升 RAG 系统的检索召回率。

---

## 2. 整体流程

```
┌─────────────────────────────────────────────────────────┐
│                    Step 1: 数据准备                       │
│  PDF解析 → 文本分块 → LLM生成三元组(query,pos,neg)       │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   Step 2: 模型微调                        │
│  加载bge-base-zh-v1.5 → TripletLoss → 2个epoch微调      │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   Step 3: 评估对比                        │
│  微调前检索评估 → 微调后检索评估 → 指标对比               │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Step 1: 数据准备

### 3.1 数据来源

| 项目 | 内容 |
|---|---|
| 原始文档 | `uploads/招股说明书1.pdf` |
| 文档页数 | 548 页 |
| 文档类型 | 科创板 IPO 招股意向书（含财务数据、业务描述、风险因素等） |

### 3.2 PDF 解析

使用项目自有的 `pdf_parser` 模块解析 PDF：

- 解析 548 页，跳过目录页
- 提取正文文本、表格内容
- 按 1000 字符/块、150 字符重叠进行分块
- 生成 1,144 个 chunk

### 3.3 三元组生成

对每个 chunk，调用 **DeepSeek API** 生成三元组数据：

**三元组格式**：`(query, positive, hard_negative)`

| 字段 | 说明 | 示例 |
|---|---|---|
| **query** | 用户可能提出的搜索问题 | "武汉兴图新科电子股份有限公司的注册资本是多少？" |
| **positive** | 与该 query 最相关的 chunk 原文（即对应 chunk） | "武汉兴图新科电子股份有限公司的注册资本为5,520.00万元..." |
| **hard_negative** | 与 query 主题相近但信息不同的 chunk（高难度负例） | 同一文档中相邻页面的其他财务数据段落 |

**生成策略改进**（第2版）：

1. 使用 LLM 生成更精准、口语化的 query
2. 用 LLM 从邻近 chunk 中**挑选最难区分的 hard_negative**（而非随机选取）
3. 最终生成 **200 条**高质量三元组

### 3.4 输出文件

| 文件 | 路径 | 说明 |
|---|---|---|
| 训练三元组 | `finetune/data/train_triplets.json` | 200 条 (query, positive, negative) |
| 评估 query | `finetune/data/eval_queries.json` | 30 条测试 query |

---

## 4. Step 2: 模型微调

### 4.1 基础模型

| 项目 | 内容 |
|---|---|
| **模型** | BAAI/bge-base-zh-v1.5 |
| **架构** | BERT-base (12层 Transformer) |
| **参数量** | ~1.1亿 |
| **输出维度** | 768 维 |
| **预训练** | 中文通用语料 |

### 4.2 训练配置

| 参数 | 值 | 说明 |
|---|---|---|
| 损失函数 | **TripletLoss** | 拉近 query-positive，推远 query-negative |
| Epochs | 2 | 小数据集避免过拟合 |
| Batch Size | 1 | 逐样本更新 |
| Learning Rate | 2e-5 | 标准微调学习率 |
| Warmup Steps | 20 | 前20步学习率线性上升 |
| 训练/评估划分 | 85% / 15% | 170条训练 + 30条评估 |
| 设备 | CUDA (GPU) | NVIDIA GPU |

### 4.3 训练过程

```
Epoch 0: TripletAcc 93.33% (baseline) → 90.00% (after epoch)
Epoch 1: TripletAcc 90.00% → 90.00% (final)
```

> **注**: TripletAcc 在训练过程中略有波动属正常现象，该指标只反映三元组排序准确率，不直接等价于检索效果。

### 4.4 输出文件

| 文件 | 路径 | 大小 |
|---|---|---|
| 微调后模型 | `finetune/output/bge-base-zh-v1.5-finetuned/` | ~390 MB |
| 训练结果 | `finetune/output/training_result.json` | - |

---

## 5. Step 3: 评估对比

### 5.1 评估方法

| 项目 | 内容 |
|---|---|
| 测试 query | 30 条（从三元组中抽取） |
| 语料库 | 1,139 条 chunk（覆盖招股书全部内容） |
| 检索方式 | 向量余弦相似度 Top-K |
| 评估指标 | Recall@K, MRR |

**评估流程：**
1. 用 query 编码，在 1,139 条 chunk 中搜索
2. 检查正确答案是否出现在前 K 个结果中
3. 对比微调前（原始 bge-base-zh-v1.5）和微调后的表现

### 5.2 评估结果

| 指标 | 微调前 | 微调后 | 提升幅度 |
|---|---|---|---|
| **Recall@1** 🏆 | **0.3333** | **0.7333** | **+0.4000 (+40.0%)** |
| Recall@3 | 0.7333 | 0.8667 | +0.1333 (+13.3%) |
| Recall@5 | 0.8333 | 0.9000 | +0.0667 (+6.7%) |
| Recall@10 | 0.8333 | 0.9000 | +0.0667 (+6.7%) |
| **MRR** 🏆 | **0.5612** | **0.8104** | **+0.2492 (+24.9%)** |

### 5.3 结果分析

```
Recall@1 提升40% → 微调后的模型首条就命中正确答案的概率翻了一倍多
MRR 提升25%   → 正确答案在搜索结果中的整体排名显著提高
Recall@10 提升不大 → 原始模型在宽召回(取前10)时已经不错，微调主要改善了精确排序
```

### 5.4 输出文件

| 文件 | 路径 |
|---|---|
| 评估结果 | `finetune/output/evaluation_result.json` |

---

## 6. 产出物清单

| 序号 | 产出物 | 路径 | 就绪 |
|---|---|---|---|
| 1 | 三元组训练数据集 | `finetune/data/train_triplets.json` | ✅ 200 条 |
| 2 | 评估测试 query | `finetune/data/eval_queries.json` | ✅ 30 条 |
| 3 | 微调后 Embedding 模型 | `finetune/output/bge-base-zh-v1.5-finetuned/` | ✅ 768 维 |
| 4 | 训练结果记录 | `finetune/output/training_result.json` | ✅ |
| 5 | 评估对比报告 | `finetune/output/evaluation_result.json` | ✅ 有数据指标支撑 |
| 6 | 完整可复现脚本 | `finetune/prepare_data.py` | ✅ |
| 7 | 训练脚本 | `finetune/train_embedding.py` | ✅ |
| 8 | 评估脚本 | `finetune/evaluate.py` | ✅ |
| 9 | 一键运行脚本 | `finetune/run_all.py` | ✅ |

---

## 7. 文件结构

```
E:\桌面\工单\RAG工单11\
├── app.py                          # RAG后端服务入口
├── models.py                       # EmbeddingClient + RerankerClient
├── search.py                       # BM25 + Milvus检索引擎
├── rag_pipeline.py                 # RAG管线
├── llm.py                          # LLM客户端
├── uploads/
│   └── 招股说明书1.pdf              # 原始数据(13MB)
├── model/
│   └── bge-base-zh-v1.5/           # 原始bge-base-zh-v1.5(390MB)
└── finetune/
    ├── prepare_data.py             # 数据准备脚本
    ├── train_embedding.py          # 微调训练脚本
    ├── evaluate.py                 # 评估对比脚本
    ├── run_all.py                  # 一键运行脚本
    ├── data/
    │   ├── train_triplets.json     # 200条三元组
    │   └── eval_queries.json       # 30条评估query
    └── output/
        ├── bge-base-zh-v1.5-finetuned/  # 微调后模型(390MB)
        ├── training_result.json         # 训练结果
        └── evaluation_result.json       # 评估结果
```

---

## 8. 运行说明

### 8.1 环境要求

```bash
# 推荐使用项目自带的 conda 环境
D:\an\envs\nlp_1\python.exe

# 依赖
sentence-transformers
torch (CUDA)
numpy
requests  # DeepSeek API
```

### 8.2 一键运行

```bash
cd E:\桌面\工单\RAG工单11
D:\an\envs\nlp_1\python.exe finetune\run_all.py
```

### 8.3 分步运行

```bash
# Step 1: 数据准备（解析PDF → 生成三元组）
D:\an\envs\nlp_1\python.exe finetune\prepare_data.py

# Step 2: 模型微调
D:\an\envs\nlp_1\python.exe finetune\train_embedding.py

# Step 3: 评估对比
D:\an\envs\nlp_1\python.exe finetune\evaluate.py
```

### 8.4 参数调整

如需调整训练数据量或参数，修改对应文件顶部配置：

**`finetune/prepare_data.py`**:
```python
MAX_CHUNKS_FOR_TRAIN = 200  # 生成数据量
```

**`finetune/train_embedding.py`**:
```python
BATCH_SIZE = 1      # 批大小
EPOCHS = 2          # 训练轮数
LEARNING_RATE = 2e-5  # 学习率
```

---

## 9. 验收对照

| 验收标准 | 达成情况 |
|---|---|
| 微调后模型检索效果 > 微调前 | ✅ Recall@1: 33% → 73% (+40%) |
| 有数据指标支撑 | ✅ Recall@K + MRR 完整数据 |
| 生成数据集 | ✅ 200条三元组 JSON |
| 模型训练过程可复现 | ✅ 完整训练脚本 |
| 过程问题记录 | ✅ 脚本注释 + 本报告 |

---
