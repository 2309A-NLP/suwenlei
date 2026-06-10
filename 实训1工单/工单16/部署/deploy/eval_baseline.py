"""
autoDL基线评估脚本
"""
import json, os, re, random
from collections import Counter
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

WORK_DIR = "/root/RAG工单16"
DATA_PATH = os.path.join(WORK_DIR, "selected_questions.jsonl")
IMAGES_DIR = os.path.join(WORK_DIR, "page_images")
OUTPUT_DIR = os.path.join(WORK_DIR, "eval_results")
MODEL_PATH = "/root/autodl-tmp/models/Qwen/Qwen2-VL-2B-Instruct"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    text = text.strip()
    for p in [r'答案是\s*([A-D])', r'选\s*([A-D])', r'^([A-D])$', r'\b([A-D])\.', r'([A-D])\s*是正确的']:
        m = re.search(p, text)
        if m: return m.group(1)
    for ch in text:
        if ch in 'ABCD': return ch
    return text

def calc_bleu(ref, hyp):
    rt, ht = list(ref), list(hyp)
    if not ht: return 0.0
    overlap = sum((Counter(rt) & Counter(ht)).values())
    p = overlap / len(ht)
    bp = min(1.0, len(ht) / max(len(rt), 1))
    return bp * p

def calc_rouge_l(ref, hyp):
    r, h = list(ref), list(hyp)
    if not r or not h: return 0.0
    m, n = len(r), len(h)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1]+1 if r[i-1]==h[j-1] else max(dp[i-1][j], dp[i][j-1])
    l = dp[m][n]
    rc, pc = l/m, l/n
    return 2*rc*pc/(rc+pc) if rc+pc>0 else 0

def evaluate(predictions, items):
    results = {"total": len(items), "correct": 0, "accuracy": 0.0, "bleu_avg": 0.0, "rouge_l_avg": 0.0,
               "term_total": 0, "term_correct": 0, "term_accuracy": 0.0,
               "diagram_total": 0, "diagram_correct": 0, "diagram_accuracy": 0.0,
               "group_stats": {1: {"total":0,"correct":0}, 2: {"total":0,"correct":0}, 3: {"total":0,"correct":0}},
               "errors": []}
    bleus, rouges = [], []
    for pred, item in zip(predictions, items):
        p, r, g = extract_answer(pred), item['answer'], item['group']
        ok = p == r
        if ok: results["correct"] += 1
        bleus.append(calc_bleu(r, p)); rouges.append(calc_rouge_l(r, p))
        results["group_stats"][g]["total"] += 1
        if ok: results["group_stats"][g]["correct"] += 1
        txt = " ".join(item['options']) + " " + item['question']
        if any(t in txt for t in INDUSTRY_TERMS):
            results["term_total"] += 1
            if ok: results["term_correct"] += 1
        if g in [2,3]:
            results["diagram_total"] += 1
            if ok: results["diagram_correct"] += 1
        if not ok:
            results["errors"].append({"question": item['question'][:100], "predicted": p, "reference": r, "group": g, "doc": item['document']})
    results["accuracy"] = results["correct"] / results["total"]
    results["bleu_avg"] = sum(bleus)/len(bleus)
    results["rouge_l_avg"] = sum(rouges)/len(rouges)
    results["term_accuracy"] = results["term_correct"]/results["term_total"] if results["term_total"]>0 else 0
    results["diagram_accuracy"] = results["diagram_correct"]/results["diagram_total"] if results["diagram_total"]>0 else 0
    for g in results["group_stats"]:
        gs = results["group_stats"][g]
        gs["accuracy"] = gs["correct"]/gs["total"] if gs["total"]>0 else 0
    return results

def main():
    with open(DATA_PATH) as f: items = [json.loads(l) for l in f]
    random.seed(42)
    samples = []
    for g in [1,2,3]:
        gi = [i for i in items if i['group']==g]
        samples.extend(random.sample(gi, min(67, len(gi))))
    random.shuffle(samples)
    print(f"评估样本: {len(samples)}")

    print("加载模型...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")
    print("模型加载完成")

    preds = []
    for idx, item in enumerate(samples):
        pm = re.search(r'第(\d+)页', item['question'])
        pn = int(pm.group(1)) if pm else 1
        img = os.path.join(IMAGES_DIR, f"{item['document'].replace('.pdf','')}_p{pn}.png")
        if not os.path.exists(img): img = os.path.join(IMAGES_DIR, f"{item['document'].replace('.pdf','')}_p1.png")
        opts = "\n".join(item['options'])
        prompt = f"{item['question']}\n{opts}\n请只回答选项字母。"
        try:
            msgs = [{"role":"user","content":[{"type":"image","image":f"file://{img}"},{"type":"text","text":prompt}]}]
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            iv, vv = process_vision_info(msgs)
            inputs = processor(text=[text], images=iv, videos=vv, padding=True, return_tensors="pt").to(model.device)
            with torch.no_grad(): out = model.generate(**inputs, max_new_tokens=32)
            preds.append(processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
        except Exception as e:
            preds.append(f"ERROR: {e}")
        if (idx+1) % 20 == 0: print(f"  {idx+1}/{len(samples)}")

    results = evaluate(preds, samples)
    with open(os.path.join(OUTPUT_DIR, "baseline_results.json"), 'w') as f: json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n=== 基线评估 ===")
    print(f"准确率: {results['accuracy']:.2%}")
    print(f"术语准确率: {results['term_accuracy']:.2%}")
    print(f"图纸推理准确率: {results['diagram_accuracy']:.2%}")

if __name__ == "__main__": main()
