"""
基线评估脚本：用本地Qwen2-VL-2B对200条样本做推理评估
"""
import json
import os
import re
import random
from collections import Counter
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# === 配置 ===
WORK_DIR = "/mnt/e/桌面/工单/RAG工单16"
DATA_PATH = os.path.join(WORK_DIR, "selected_questions.jsonl")
IMAGES_DIR = os.path.join(WORK_DIR, "page_images")
OUTPUT_DIR = os.path.join(WORK_DIR, "eval_results")
MODEL_PATH = "/mnt/e/models/Qwen/Qwen2-VL-2B-Instruct"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 工业术语表
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


def extract_answer(text):
    """从模型输出中提取答案选项"""
    text = text.strip()
    patterns = [
        r'答案是\s*([A-D])',
        r'选\s*([A-D])',
        r'^([A-D])$',
        r'\b([A-D])\.',
        r'([A-D])\s*是正确的',
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1)
    # 直接取第一个A-D字母
    for ch in text:
        if ch in 'ABCD':
            return ch
    return text


def calc_bleu(reference, hypothesis):
    ref_tokens = list(reference)
    hyp_tokens = list(hypothesis)
    if not hyp_tokens:
        return 0.0
    ref_counter = Counter(ref_tokens)
    hyp_counter = Counter(hyp_tokens)
    overlap = sum((ref_counter & hyp_counter).values())
    precision = overlap / len(hyp_tokens)
    bp = min(1.0, len(hyp_tokens) / max(len(ref_tokens), 1))
    return bp * precision


def calc_rouge_l(reference, hypothesis):
    ref, hyp = list(reference), list(hypothesis)
    if not ref or not hyp:
        return 0.0
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
    return 2 * recall * precision / (recall + precision) if recall + precision > 0 else 0


def evaluate(predictions, items):
    results = {
        "total": len(items),
        "correct": 0,
        "accuracy": 0.0,
        "bleu_avg": 0.0,
        "rouge_l_avg": 0.0,
        "term_total": 0, "term_correct": 0, "term_accuracy": 0.0,
        "diagram_total": 0, "diagram_correct": 0, "diagram_accuracy": 0.0,
        "group_stats": {1: {"total": 0, "correct": 0},
                        2: {"total": 0, "correct": 0},
                        3: {"total": 0, "correct": 0}},
        "errors": []
    }
    bleu_scores, rouge_scores = [], []

    for pred, item in zip(predictions, items):
        predicted = extract_answer(pred)
        reference = item['answer']
        group = item['group']
        is_correct = predicted == reference
        if is_correct:
            results["correct"] += 1
        bleu_scores.append(calc_bleu(reference, predicted))
        rouge_scores.append(calc_rouge_l(reference, predicted))
        results["group_stats"][group]["total"] += 1
        if is_correct:
            results["group_stats"][group]["correct"] += 1

        all_text = " ".join(item['options']) + " " + item['question']
        found_terms = [t for t in INDUSTRY_TERMS if t in all_text]
        if found_terms:
            results["term_total"] += 1
            if is_correct:
                results["term_correct"] += 1

        if group in [2, 3]:
            results["diagram_total"] += 1
            if is_correct:
                results["diagram_correct"] += 1

        if not is_correct:
            results["errors"].append({
                "question": item['question'][:100],
                "predicted": predicted,
                "reference": reference,
                "group": group,
                "doc": item['document']
            })

    results["accuracy"] = results["correct"] / results["total"]
    results["bleu_avg"] = sum(bleu_scores) / len(bleu_scores)
    results["rouge_l_avg"] = sum(rouge_scores) / len(rouge_scores)
    results["term_accuracy"] = results["term_correct"] / results["term_total"] if results["term_total"] > 0 else 0
    results["diagram_accuracy"] = results["diagram_correct"] / results["diagram_total"] if results["diagram_total"] > 0 else 0
    for g in results["group_stats"]:
        gs = results["group_stats"][g]
        gs["accuracy"] = gs["correct"] / gs["total"] if gs["total"] > 0 else 0
    return results


def main():
    # 加载数据
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        items = [json.loads(line) for line in f]

    # 按组均衡抽200条
    random.seed(42)
    eval_samples = []
    for g in [1, 2, 3]:
        group_items = [i for i in items if i['group'] == g]
        eval_samples.extend(random.sample(group_items, min(67, len(group_items))))
    random.shuffle(eval_samples)
    print(f"评估样本数: {len(eval_samples)}")

    # 加载模型
    print("加载模型...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
        quantization_config=bnb_config
    )
    print("模型加载完成")

    # 推理
    predictions = []
    for idx, item in enumerate(eval_samples):
        page_match = re.search(r'第(\d+)页', item['question'])
        page_num = int(page_match.group(1)) if page_match else 1
        img_path = os.path.join(IMAGES_DIR, f"{item['document'].replace('.pdf','')}_p{page_num}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(IMAGES_DIR, f"{item['document'].replace('.pdf','')}_p1.png")

        options_text = "\n".join(item['options'])
        prompt = f"{item['question']}\n{options_text}\n请只回答选项字母。"

        try:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": f"file://{img_path}"},
                {"type": "text", "text": prompt}
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                               padding=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=32)
            answer = processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            predictions.append(answer)
        except Exception as e:
            predictions.append(f"ERROR: {e}")

        if (idx + 1) % 20 == 0:
            print(f"  推理进度: {idx+1}/{len(eval_samples)}")

    # 评估
    results = evaluate(predictions, eval_samples)

    # 保存结果
    output_path = os.path.join(OUTPUT_DIR, "baseline_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n=== 基线评估结果 ===")
    print(f"准确率: {results['accuracy']:.2%}")
    print(f"术语准确率: {results['term_accuracy']:.2%}")
    print(f"图纸推理准确率: {results['diagram_accuracy']:.2%}")
    print(f"结果已保存: {output_path}")


if __name__ == "__main__":
    main()
