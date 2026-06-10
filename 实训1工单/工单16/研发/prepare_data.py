"""
数据准备脚本：从IMDR PDF提取页面截图，生成Qwen-VL-Chat微调格式JSONL
"""
import json
import os
import re
import zipfile
import fitz  # PyMuPDF

# === 配置 ===
ZIP_PATH = "/mnt/e/桌面/资料/14-17附件/original_problems.zip"
WORK_DIR = "/mnt/e/桌面/工单/RAG工单16"
DOCS_DIR = os.path.join(WORK_DIR, "images")
SELECTED_PATH = os.path.join(WORK_DIR, "selected_questions.jsonl")
OUTPUT_PATH = os.path.join(WORK_DIR, "finetune_data.jsonl")

# === 0. 读取选中数据，确定需要的文档 ===
print("=== 步骤0: 确定需要的文档 ===")
with open(SELECTED_PATH, 'r', encoding='utf-8') as f:
    qa_items = [json.loads(line) for line in f]
needed_docs = set(item['document'] for item in qa_items)
print(f"选中 {len(qa_items)} 条QA，涉及 {len(needed_docs)} 个文档")

# === 1. 只解压需要的PDF文档 ===
print("\n=== 步骤1: 解压需要的PDF文档 ===")
os.makedirs(DOCS_DIR, exist_ok=True)

z = zipfile.ZipFile(ZIP_PATH)
extracted = 0
for pf in z.namelist():
    fname = os.path.basename(pf)
    if fname in needed_docs:
        dest = os.path.join(DOCS_DIR, fname)
        if not os.path.exists(dest):
            with z.open(pf) as src, open(dest, 'wb') as dst:
                dst.write(src.read())
            extracted += 1
print(f"新解压 {extracted} 个文件，共 {len(needed_docs)} 个")

# === 2. 提取页面截图并生成微调数据 ===
print("\n=== 步骤3: 提取页面截图 & 生成微调数据 ===")
page_pattern = re.compile(r'第(\d+)页')
IMG_DIR = os.path.join(WORK_DIR, "page_images")
os.makedirs(IMG_DIR, exist_ok=True)

finetune_data = []
errors = []

for idx, item in enumerate(qa_items):
    doc_name = item['document']
    question = item['question']
    options = item['options']
    answer = item['answer']
    group = item['group']

    # 解析页码
    page_match = page_pattern.search(question)
    if page_match:
        page_num = int(page_match.group(1)) - 1  # 0-indexed
    else:
        page_num = 0  # 默认第一页

    # 提取页面截图
    pdf_path = os.path.join(DOCS_DIR, doc_name)
    if not os.path.exists(pdf_path):
        errors.append(f"PDF不存在: {doc_name}")
        continue

    try:
        doc = fitz.open(pdf_path)
        # 确保页码有效
        if page_num >= len(doc):
            page_num = 0
        if page_num >= len(doc):
            errors.append(f"PDF无有效页面: {doc_name}")
            doc.close()
            continue

        page = doc[page_num]
        # 渲染为图片（2倍分辨率）
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_name = f"{doc_name.replace('.pdf','')}_p{page_num+1}.png"
        img_path = os.path.join(IMG_DIR, img_name)
        pix.save(img_path)
        doc.close()

        # 构建选项文本
        options_text = "\n".join(options)

        # 构建问答对（选择题格式）
        user_msg = f"<img>{img_path}</img>\n{question}\n{options_text}"
        assistant_msg = f"答案是{answer}"

        finetune_item = {
            "id": f"{doc_name}_{page_num+1}_{idx}",
            "conversations": [
                {"from": "user", "value": user_msg},
                {"from": "assistant", "value": assistant_msg}
            ]
        }
        finetune_data.append(finetune_item)

    except Exception as e:
        errors.append(f"处理失败 {doc_name}: {str(e)}")

    if (idx + 1) % 500 == 0:
        print(f"  已处理 {idx+1}/{len(qa_items)}")

# === 3. 保存微调数据 ===
print(f"\n=== 步骤3: 保存微调数据 ===")
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    for item in finetune_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"成功生成: {len(finetune_data)} 条")
print(f"失败: {len(errors)} 条")
if errors:
    print("错误详情（前10条）:")
    for e in errors[:10]:
        print(f"  - {e}")

# 统计分组
groups = {}
for item in finetune_data:
    g = item['id'].split('_')[0]  # 这里不太准确，用原始数据统计
print(f"\n输出文件: {OUTPUT_PATH}")
print("=== 完成 ===")
