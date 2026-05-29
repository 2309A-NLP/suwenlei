# -*- coding: utf-8 -*-
# pdf_preprocessing_advanced.py  深度版 · 国家基层高血压防治管理指南2025
import pdfplumber
import pandas as pd
import re
import logging
import os
from pathlib import Path
# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# -------------------------- 配置区 --------------------------
CONFIG = {
    "max_chunk_len": 600,        # 最大块字符数
    "overlap_len": 60,           # 重叠长度
    "min_chunk_len": 100,        # 最小有效块长度
    "section_titles": [          # 指南标准章节（用于精准匹配）
        "基层高血压管理基本要求",
        "基层高血压诊疗管理流程",
        "诊疗关键点",
        "高血压诊断与评估",
        "血压测量",
        "高血压诊断标准",
        "初诊评估",
        "高血压治疗",
        "治疗原则",
        "降压目标",
        "生活方式干预",
        "降压药物治疗",
        "综合干预管理",
        "高血压与中医药",
        "转诊",
        "高血压长期随访管理",
        "筛查与预防",
        "健康教育"
    ]
}

# ---------------------------------------------------------------------

def clean_text(text: str) -> str:
    """深度文本清洗：去多余空格、换行、页码、杂志信息、乱码"""
    if not text:
        return ""
    # 去除杂志页眉页脚（中国循环杂志 2025年9月...）
    text = re.sub(r"中国循环杂志.*?\d{4}\s*年\d*\s*月.*?\d+", "", text)
    text = re.sub(r"Chinese Circulation Journal.*", "", text)
    text = re.sub(r"指南与共识", "", text)
    text = re.sub(r"摘要|Abstract|关键词|Key words", "", text)
    text = re.sub(r"通讯作者.*", "", text)
    text = re.sub(r"中图分类号.*|文献标识码.*|文章编号.*|DOI:.*", "", text)
    # 统一空白
    text = re.sub(r"\s+", " ", text).strip()
    # 去除孤立数字页码
    text = re.sub(r"^\s*\d+\s*$", "", text)
    return text

def extract_title_from_text(text: str) -> str:
    """智能识别章节标题（优先匹配指南标准章节）"""
    for title in CONFIG["section_titles"]:
        if title in text:
            return title
    # 匹配 1、1.1、1.1.1 层级标题
    level1 = re.search(r"^(\d+\s+[^。\s]{4,20})", text)
    level2 = re.search(r"^(\d+\.\d+\s+[^。\s]{4,20})", text)
    level3 = re.search(r"^(\d+\.\d+\.\d+\s+[^。\s]{4,20})", text)
    if level3:
        return level3.group(1).strip()
    if level2:
        return level2.group(1).strip()
    if level1:
        return level1.group(1).strip()
    return "未分类章节"

def extract_pdf_content(pdf_path: str) -> list:
    """
    深度提取PDF：文本 + 表格合并
    返回：每页结构化数据（page_num, raw_text, tables_text, clean_text, section_title）
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        logging.info(f"PDF总页数：{len(pdf.pages)}")
        for idx, page in enumerate(pdf.pages):
            page_num = idx + 1
            # 提取纯文本
            raw_text = page.extract_text() or ""
            # 提取表格并转为文本
            tables = page.extract_tables()
            table_text = ""
            for table in tables:
                for row in table:
                    row_clean = [str(c).strip() if c else "" for c in row]
                    table_text += " | ".join(row_clean) + "\n"
            # 合并清洗
            full_text = raw_text + "\n" + table_text
            clean = clean_text(full_text)
            if len(clean) < 20:
                continue  # 跳过空白页
            # 识别章节
            section = extract_title_from_text(clean)
            pages.append({
                "page_num": page_num, #页码
                "raw_text": raw_text, #原文
                "table_text": table_text.strip(), #表格
                "clean_text": clean, #清洗文本
                "section_title": section #章节名
            })
    logging.info(f"有效文本页：{len(pages)} 页")
    return pages

def split_by_semantic(pages: list) -> list:
    """
    语义切块：
    1. 优先在句号、换行、标题处断开
    2. 保留章节与页码
    3. 过滤过短无效块
    """
    chunks = []
    chunk_id = 1
    max_len = CONFIG["max_chunk_len"] # 每块最大长度
    overlap = CONFIG["overlap_len"]
    min_len = CONFIG["min_chunk_len"]
    #遍历每页，取出文本、章节、页码
    for page in pages:
        text = page["clean_text"] # 当前页的干净文本
        section = page["section_title"] # 章节标题
        page_num = page["page_num"] # 页码
        start = 0
        total = len(text)
        # 循环切片：只要没切到末尾就继续
        while start < total:
            end = min(start + max_len, total)
            chunk = text[start:end]

            # 智能断句：在句号/换行处切割
            if end < total:
                split_pos = max(
                    chunk.rfind("。"),
                    chunk.rfind(".\n"),
                    chunk.rfind("；"),
                    chunk.rfind("，")
                )
                if split_pos > 50:
                    end = start + split_pos + 1
                    chunk = text[start:end]

            # 过滤太短
            if len(chunk) >= min_len:
                chunks.append({
                    "chunk_id": f"chunk_{chunk_id:05d}", #块 ID
                    "guide_name": "国家基层高血压防治管理指南2025版", #指南名
                    "section_title": section, #章节
                    "page_num": page_num, #页码
                    "chunk_length": len(chunk), #长度
                    "chunk_text": chunk.strip() #文本内容
                })
                chunk_id += 1

            # 滑动窗口
            start = end - overlap if (end - overlap) > start else end
            if start >= total:
                break

    logging.info(f"最终生成有效语义块：{len(chunks)} 个")
    return chunks

def save_to_beautiful_csv(chunks: list, output_path: str):
    """输出美观、可直接入库的结构化CSV"""
    df = pd.DataFrame(chunks)
    # 调整列顺序（更直观）
    df = df[[
        "chunk_id", "guide_name", "section_title",
        "page_num", "chunk_length", "chunk_text"
    ]]
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logging.info(f"✅ 深度处理完成，已保存：{output_path}")
    return df

def main():
    """主流程：PDF → 深度清洗 → 语义切块 → 规范CSV"""
    # 【请修改为你本地路径】
    BASE_DIR = Path(__file__).resolve().parent.parent / "dataset"
    PDF_PATH    = str(BASE_DIR / "国家基层高血压防治管理手册2025版.pdf")
    OUTPUT_CSV  = str(BASE_DIR / "guideline_chunks_advanced.csv")

    logging.info("🚀 开始深度预处理：国家基层高血压防治管理指南2025")
    pages = extract_pdf_content(PDF_PATH)
    chunks = split_by_semantic(pages)
    save_to_beautiful_csv(chunks, OUTPUT_CSV)

if __name__ == "__main__":
    main()