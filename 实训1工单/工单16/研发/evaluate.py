"""
评估脚本：Qwen2-VL-2B微调前后效果对比
指标：BLEU、ROUGE、工业术语准确性、图纸推理正确性
"""
import json
import os
import re
import random
from collections import Counter

# === 配置 ===
WORK_DIR = "/mnt/e/桌面/工单/RAG工单16"
DATA_PATH = os.path.join(WORK_DIR, "selected_questions.jsonl")
IMAGES_DIR = os.path.join(WORK_DIR, "page_images")
OUTPUT_DIR = os.path.join(WORK_DIR, "eval_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 工业术语表（用于术语准确性评估）
INDUSTRY_TERMS = [
    "淬火", "回火", "正火", "退火", "渗碳", "渗氮", "调质",
    "公差配合", "过盈配合", "间隙配合", "过渡配合",
    "轴承", "齿轮", "凸轮", "连杆", "曲轴", "飞轮",
    "焊接", "铆接", "螺纹", "法兰", "密封",
    "液压", "气动", "电磁", "传感器", "执行器",
    "除尘器", "电极", "沉积", "气流", "过滤",
    "吸附", "脱附", "催化", "反应器", "换热器",
    "冲压", "锻造", "铸造", "注塑", "挤压",
    "表面粗糙度", "形位公差", "硬度", "强度", "韧性"
]


def load_data(path):
    """加载评估数据"""
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def extract_answer(text):
    """从模型输出中提取答案选项"""
    # 匹配 "答案是X" 或 "X" 或 "选X"
    patterns = [
        r'答案是\s*([A-D])',
        r'选\s*([A-D])',
        r'^([A-D])$',
        r'\b([A-D])\.',
    ]
    for p in patterns:
        m = re.search(p, text.strip())
        if m:
            return m.group(1)
    return text.strip()


def calc_bleu(reference, hypothesis):
    """简化版BLEU计算"""
    ref_tokens = list(reference)
    hyp_tokens = list(hypothesis)
    if not hyp_tokens:
        return 0.0
    # 1-gram精度
    ref_counter = Counter(ref_tokens)
    hyp_counter = Counter(hyp_tokens)
    overlap = sum((ref_counter & hyp_counter).values())
    precision = overlap / len(hyp_tokens) if hyp_tokens else 0
    # brevity penalty
    bp = min(1.0, len(hyp_tokens) / max(len(ref_tokens), 1))
    return bp * precision


def calc_rouge_l(reference, hypothesis):
    """ROUGE-L（最长公共子序列）"""
    ref = list(reference)
    hyp = list(hypothesis)
    if not ref or not hyp:
        return 0.0
    # LCS
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs_len = dp[m][n]
    recall = lcs_len / m
    precision = lcs_len / n
    if recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def check_term_accuracy(question, predicted, reference, options):
    """检查工业术语准确性"""
    # 从选项中提取术语
    all_option_text = " ".join(options)
    found_terms = [t for t in INDUSTRY_TERMS if t in all_option_text or t in question]
    if not found_terms:
        return None  # 无工业术语，跳过
    # 判断答案是否正确
    return predicted == reference


def check_diagram_reasoning(group):
    """检查图纸推理题（group 2和3）"""
    return group in [2, 3]


def evaluate_model(predictions, items):
    """综合评估"""
    results = {
        "total": len(items),
        "correct": 0,
        "accuracy": 0.0,
        "bleu_avg": 0.0,
        "rouge_l_avg": 0.0,
        "term_total": 0,
        "term_correct": 0,
        "term_accuracy": 0.0,
        "diagram_total": 0,
        "diagram_correct": 0,
        "diagram_accuracy": 0.0,
        "group_stats": {1: {"total": 0, "correct": 0}, 2: {"total": 0, "correct": 0}, 3: {"total": 0, "correct": 0}},
        "errors": []
    }

    bleu_scores = []
    rouge_scores = []

    for pred, item in zip(predictions, items):
        predicted = extract_answer(pred)
        reference = item['answer']
        group = item['group']
        is_correct = predicted == reference

        if is_correct:
            results["correct"] += 1

        # BLEU/ROUGE
        bleu_scores.append(calc_bleu(reference, predicted))
        rouge_scores.append(calc_rouge_l(reference, predicted))

        # 分组统计
        results["group_stats"][group]["total"] += 1
        if is_correct:
            results["group_stats"][group]["correct"] += 1

        # 工业术语准确性
        term_result = check_term_accuracy(item['question'], predicted, reference, item['options'])
        if term_result is not None:
            results["term_total"] += 1
            if term_result:
                results["term_correct"] += 1

        # 图纸推理
        if check_diagram_reasoning(group):
            results["diagram_total"] += 1
            if is_correct:
                results["diagram_correct"] += 1

        # 记录错误
        if not is_correct:
            results["errors"].append({
                "question": item['question'][:100],
                "predicted": predicted,
                "reference": reference,
                "group": group,
                "doc": item['document']
            })

    results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0
    results["bleu_avg"] = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    results["rouge_l_avg"] = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
    results["term_accuracy"] = results["term_correct"] / results["term_total"] if results["term_total"] > 0 else 0
    results["diagram_accuracy"] = results["diagram_correct"] / results["diagram_total"] if results["diagram_total"] > 0 else 0

    for g in results["group_stats"]:
        gs = results["group_stats"][g]
        gs["accuracy"] = gs["correct"] / gs["total"] if gs["total"] > 0 else 0

    return results


def run_inference(model_path, items, is_api=False, api_key=None):
    """运行推理（本地或API）"""
    predictions = []
    if is_api:
        import dashscope
        from dashscope import MultiModalConversation
        dashscope.api_key = api_key
        for idx, item in enumerate(items):
            img_name = f"{item['document'].replace('.pdf','')}_p{{page_num}}.png"
            page_match = re.search(r'第(\d+)页', item['question'])
            page_num = int(page_match.group(1)) if page_match else 1
            img_path = os.path.join(IMAGES_DIR, f"{item['document'].replace('.pdf','')}_p{page_num}.png")
            if not os.path.exists(img_path):
                img_path = os.path.join(IMAGES_DIR, f"{item['document'].replace('.pdf','')}_p1.png")

            options_text = "\n".join(item['options'])
            prompt = f"{item['question']}\n{options_text}\n请只回答选项字母。"

            try:
                resp = MultiModalConversation.call(
                    model='qwen-vl-chat',
                    messages=[{
                        "role": "user",
                        "content": [
                            {"image": img_path},
                            {"text": prompt}
                        ]
                    }]
                )
                answer = resp.output.choices[0].message.content[0]['text']
                predictions.append(answer)
            except Exception as e:
                predictions.append(f"ERROR: {e}")

            if (idx + 1) % 100 == 0:
                print(f"  推理进度: {idx+1}/{len(items)}")
    else:
        # 本地推理（需要GPU环境）
        from transformers import AutoProcessor, AutoModelForCausalLM
        import torch
        from PIL import Image

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map="auto"
        )

        for idx, item in enumerate(items):
            page_match = re.search(r'第(\d+)页', item['question'])
            page_num = int(page_match.group(1)) if page_match else 1
            img_path = os.path.join(IMAGES_DIR, f"{item['document'].replace('.pdf','')}_p{page_num}.png")
            if not os.path.exists(img_path):
                img_path = os.path.join(IMAGES_DIR, f"{item['document'].replace('.pdf','')}_p1.png")

            options_text = "\n".join(item['options'])
            prompt = f"{item['question']}\n{options_text}\n请只回答选项字母。"

            try:
                image = Image.open(img_path).convert("RGB")
                msgs = [{"role": "user", "content": [{"image": img_path}, {"text": prompt}]}]
                text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=32)
                answer = processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                predictions.append(answer)
            except Exception as e:
                predictions.append(f"ERROR: {e}")

            if (idx + 1) % 100 == 0:
                print(f"  推理进度: {idx+1}/{len(items)}")

    return predictions


def main():
    # 加载数据
    items = load_data(DATA_PATH)
    # 抽取200条评估样本（按组均衡）
    random.seed(42)
    eval_samples = []
    for g in [1, 2, 3]:
        group_items = [i for i in items if i['group'] == g]
        eval_samples.extend(random.sample(group_items, min(67, len(group_items))))
    random.shuffle(eval_samples)
    print(f"评估样本数: {len(eval_samples)}")

    # === 基线评估（用DashScope API调用Qwen-VL-Chat） ===
    print("\n=== 基线评估（Qwen-VL-Chat API） ===")
    import sys
    sys.path.insert(0, WORK_DIR)
    api_key = os.environ.get('DASHSCOPE_API_KEY', '')
    baseline_preds = run_inference(None, eval_samples, is_api=True, api_key=api_key)
    baseline_results = evaluate_model(baseline_preds, eval_samples)

    # 保存基线结果
    with open(os.path.join(OUTPUT_DIR, "baseline_results.json"), 'w', encoding='utf-8') as f:
        json.dump(baseline_results, f, ensure_ascii=False, indent=2)

    print(f"基线准确率: {baseline_results['accuracy']:.2%}")
    print(f"基线术语准确率: {baseline_results['term_accuracy']:.2%}")
    print(f"基线图纸推理准确率: {baseline_results['diagram_accuracy']:.2%}")

    # === 微调后评估（需要微调完成后手动运行） ===
    print("\n=== 微调后评估 ===")
    ft_model_path = os.path.join(WORK_DIR, "merged_model")
    if os.path.exists(ft_model_path):
        ft_preds = run_inference(ft_model_path, eval_samples, is_api=False)
        ft_results = evaluate_model(ft_preds, eval_samples)
        with open(os.path.join(OUTPUT_DIR, "finetuned_results.json"), 'w', encoding='utf-8') as f:
            json.dump(ft_results, f, ensure_ascii=False, indent=2)
        print(f"微调后准确率: {ft_results['accuracy']:.2%}")
    else:
        print("微调模型尚未就绪，跳过。请微调完成后重新运行。")
        ft_results = None

    # === 生成对比报告 ===
    generate_report(baseline_results, ft_results)


def generate_report(baseline, finetuned=None):
    """生成评估报告"""
    report = []
    report.append("# Qwen2-VL-2B 微调前后评估报告\n")
    report.append(f"评估样本数: {baseline['total']}\n")

    report.append("## 1. 整体指标对比\n")
    report.append("| 指标 | 基线模型 | 微调模型 | 提升 |")
    report.append("|------|---------|---------|------|")

    metrics = [
        ("整体准确率", "accuracy"),
        ("BLEU均值", "bleu_avg"),
        ("ROUGE-L均值", "rouge_l_avg"),
        ("工业术语准确率", "term_accuracy"),
        ("图纸推理准确率", "diagram_accuracy"),
    ]

    for name, key in metrics:
        base_val = baseline[key]
        if finetuned:
            ft_val = finetuned[key]
            diff = ft_val - base_val
            sign = "+" if diff > 0 else ""
            report.append(f"| {name} | {base_val:.2%} | {ft_val:.2%} | {sign}{diff:.2%} |")
        else:
            report.append(f"| {name} | {base_val:.2%} | 待评估 | - |")

    report.append("\n## 2. 分组准确率对比\n")
    report.append("| 分组 | 描述 | 基线模型 | 微调模型 | 提升 |")
    report.append("|------|------|---------|---------|------|")
    group_names = {1: "文本理解", 2: "图纸推理", 3: "复杂图纸推理"}
    for g in [1, 2, 3]:
        base_acc = baseline["group_stats"][g]["accuracy"]
        if finetuned:
            ft_acc = finetuned["group_stats"][g]["accuracy"]
            diff = ft_acc - base_acc
            sign = "+" if diff > 0 else ""
            report.append(f"| Group {g} | {group_names[g]} | {base_acc:.2%} | {ft_acc:.2%} | {sign}{diff:.2%} |")
        else:
            report.append(f"| Group {g} | {group_names[g]} | {base_acc:.2%} | 待评估 | - |")

    report.append("\n## 3. 失败案例分析\n")
    errors = baseline.get("errors", [])[:10]
    for i, err in enumerate(errors, 1):
        report.append(f"### 案例{i}")
        report.append(f"- 问题: {err['question']}")
        report.append(f"- 预测: {err['predicted']} | 正确: {err['reference']}")
        report.append(f"- 分组: Group {err['group']} | 文档: {err['doc']}")
        report.append("")

    report.append("\n## 4. 改进建议\n")
    report.append("1. **数据增强**: 增加图纸推理题的训练样本比例")
    report.append("2. **超参数调优**: 尝试更高LoRA rank（16/32）提升模型容量")
    report.append("3. **数据清洗**: 剔除答案有歧义的样本")
    report.append("4. **图像预处理**: 尝试更高分辨率的页面截图")
    report.append("5. **多轮训练**: 增加epoch数观察是否欠拟合")

    report_text = "\n".join(report)
    report_path = os.path.join(OUTPUT_DIR, "evaluation_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n报告已保存: {report_path}")


if __name__ == "__main__":
    main()
