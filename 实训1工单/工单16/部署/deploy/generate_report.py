"""
生成评估报告：微调前后对比 + 失败案例分析 + 改进建议
"""
import json, os

WORK_DIR = "/root/RAG工单16"
OUTPUT_DIR = os.path.join(WORK_DIR, "eval_results")

def load_results(name):
    path = os.path.join(OUTPUT_DIR, name)
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    return None

def main():
    baseline = load_results("baseline_results.json")
    finetuned = load_results("finetuned_results.json")

    if not baseline:
        print("缺少基线评估结果，请先运行 eval_baseline.py")
        return

    report = []
    report.append("# Qwen2-VL-2B-Instruct 微调前后评估报告\n")
    report.append(f"评估样本数: {baseline['total']}\n")

    # 1. 整体指标
    report.append("## 1. 整体指标对比\n")
    report.append("| 指标 | 基线模型 | 微调模型 | 提升 |")
    report.append("|------|---------|---------|------|")
    metrics = [("整体准确率","accuracy"), ("BLEU均值","bleu_avg"), ("ROUGE-L均值","rouge_l_avg"),
               ("工业术语准确率","term_accuracy"), ("图纸推理准确率","diagram_accuracy")]
    for name, key in metrics:
        bv = baseline[key]
        if finetuned:
            fv = finetuned[key]; diff = fv - bv; sign = "+" if diff > 0 else ""
            report.append(f"| {name} | {bv:.2%} | {fv:.2%} | {sign}{diff:.2%} |")
        else:
            report.append(f"| {name} | {bv:.2%} | 待评估 | - |")

    # 2. 分组准确率
    report.append("\n## 2. 分组准确率对比\n")
    report.append("| 分组 | 描述 | 基线模型 | 微调模型 | 提升 |")
    report.append("|------|------|---------|---------|------|")
    gn = {1: "文本理解", 2: "图纸推理", 3: "复杂图纸推理"}
    for g in [1,2,3]:
        ba = baseline["group_stats"][str(g)]["accuracy"]
        if finetuned:
            fa = finetuned["group_stats"][str(g)]["accuracy"]; diff = fa-ba; sign = "+" if diff>0 else ""
            report.append(f"| Group {g} | {gn[g]} | {ba:.2%} | {fa:.2%} | {sign}{diff:.2%} |")
        else:
            report.append(f"| Group {g} | {gn[g]} | {ba:.2%} | 待评估 | - |")

    # 3. 失败案例
    report.append("\n## 3. 失败案例分析\n")
    errors = (finetuned or baseline).get("errors", [])[:10]
    if not errors:
        report.append("无失败案例。\n")
    for i, err in enumerate(errors, 1):
        report.append(f"### 案例{i}")
        report.append(f"- 问题: {err['question']}")
        report.append(f"- 预测: {err['predicted']} | 正确: {err['reference']}")
        report.append(f"- 分组: Group {err['group']} | 文档: {err['doc']}")
        report.append("")

    # 4. 改进建议
    report.append("\n## 4. 改进建议\n")
    report.append("1. **数据增强**: 增加图纸推理题的训练样本比例，当前Group 2/3准确率偏低")
    report.append("2. **超参数调优**: 尝试更高LoRA rank（16/32）提升模型容量")
    report.append("3. **数据清洗**: 剔除答案有歧义的样本，人工校验训练数据质量")
    report.append("4. **图像预处理**: 尝试更高分辨率的页面截图，保留更多细节信息")
    report.append("5. **多轮训练**: 增加epoch数（3-5轮）观察是否欠拟合")
    report.append("6. **学习率调整**: 尝试更小的学习率（1e-4）或warmup比例调大")
    report.append("7. **领域术语增强**: 在训练数据中增加更多包含专业术语的问答对")

    # 5. 训练配置
    report.append("\n## 5. 训练配置\n")
    report.append("| 参数 | 值 |")
    report.append("|------|-----|")
    report.append("| 模型 | Qwen2-VL-2B-Instruct |")
    report.append("| 微调方法 | LoRA |")
    report.append("| LoRA rank | 8 |")
    report.append("| LoRA alpha | 16 |")
    report.append("| 量化 | 4bit (bitsandbytes) |")
    report.append("| 训练轮数 | 2 |")
    report.append("| 学习率 | 2e-4 |")
    report.append("| Batch size | 4 x 4 = 16 |")
    report.append("| 图片最大像素 | 262144 (512x512) |")

    report_text = "\n".join(report)
    report_path = os.path.join(OUTPUT_DIR, "evaluation_report.md")
    with open(report_path, 'w') as f: f.write(report_text)
    print(f"报告已保存: {report_path}")

if __name__ == "__main__": main()
