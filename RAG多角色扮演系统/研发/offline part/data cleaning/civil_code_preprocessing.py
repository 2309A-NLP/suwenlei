# -*- coding: utf-8 -*-
# civil_code_preprocessing.py
# 将《中华人民共和国民法典》docx内容清洗为结构化CSV
import re
import pandas as pd
import logging
from pathlib import Path
import docx  # 顶部导入

def parse_civil_code_txt(docx_path):
    # 读取 docx 文件所有文本
    doc = docx.Document(docx_path)
    lines = []
    for paragraph in doc.paragraphs:
        line = paragraph.text.strip()
        if line:
            lines.append(line)

    text = '\n'.join(lines)

    # 分割成行
    lines = text.split('\n')

    chunks = []
    chunk_id = 1

    current_book = ""  # 当前编（第一编、第二编...）
    current_chapter = ""  # 当前章（第一章、第二章...）
    current_section = ""  # 当前节（第一节、第二节...）- 可选
    #三个变量记录当前位置，属于哪一编、哪一章、哪一节
    i = 0
    total_lines = len(lines)

    while i < total_lines:
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # 识别编：第一编　总则
        book_match = re.match(r'^第[一二三四五六七八九十]+编\s+(.+)$', line)
        if book_match:
            current_book = book_match.group(1).strip()
            current_chapter = ""
            current_section = ""
            i += 1
            continue

        # 识别章：第一章　基本规定
        chapter_match = re.match(r'^第[一二三四五六七八九十]+章\s+(.+)$', line)
        if chapter_match:
            current_chapter = chapter_match.group(1).strip()
            i += 1
            continue

        # 识别节：第一节　民事权利能力和民事行为能力
        section_match = re.match(r'^第[一二三四五六七八九十]+节\s+(.+)$', line)
        if section_match:
            current_section = section_match.group(1).strip()
            i += 1
            continue

        # 识别具体条款：第一条　为了保护民事主体的合法权益...
        article_match = re.match(r'^第([一二三四五六七八九十百千万\d]+)条\s+(.+)$', line)
        if article_match:
            article_num = article_match.group(1)
            article_content = article_match.group(2)

            # 收集可能多行的条款内容
            full_content = article_content
            j = i + 1
            while j < total_lines and not re.match(r'^第[一二三四五六七八九十百千万\d]+条', lines[j].strip()) \
                    and not re.match(r'^第[一二三四五六七八九十]+编', lines[j].strip()) \
                    and not re.match(r'^第[一二三四五六七八九十]+章', lines[j].strip()) \
                    and not re.match(r'^第[一二三四五六七八九十]+节', lines[j].strip()):
                next_line = lines[j].strip()
                if next_line and not re.match(r'^\([一二三四五六七八九十]+\)', next_line):
                    full_content += ' ' + next_line
                j += 1

            # 创建文本块
            section_title = f"{current_book} - {current_chapter}"
            if current_section:
                section_title += f" - {current_section}"

            # 按条款存储，每条条款单独作为一个chunk（法律条款不宜切割）
            chunks.append({
                'chunk_id': f"civil_{article_num:0>6d}" if article_num.isdigit() else f"civil_{article_num}", #唯一 ID
                'chunk_text': f"【民法典第{article_num}条】{full_content}", #条款文本（带标题）
                'section_title': section_title,#编章节
                'article_num': article_num, #第 X 条
                'book': current_book, #编
                'chapter': current_chapter, #章
                'section': current_section #节
            })
            chunk_id += 1
            i = j
            continue

        i += 1

    logging.info(f"共解析出 {len(chunks)} 个法律条款")
    return chunks


def split_long_articles(chunks, max_chars=800):
    """
    对于特别长的条款进行分割（民法典中部分条款较长）
    """
    final_chunks = []
    for chunk in chunks:
        text = chunk['chunk_text']
        if len(text) <= max_chars:
            final_chunks.append(chunk)
            continue

        # 长条款分割
        parts = []
        start = 0
        part_idx = 1
        while start < len(text):
            end = min(start + max_chars, len(text))
            # 如果还没到文本末尾（不是最后一段），就尝试找句号
            if end < len(text):
                last_period = text.rfind('。', start, end)
                #如果找到了句号，且不是在起始位置
                if last_period > start:
                    #把结束位置调整到句号后面，实现按句子断开
                    end = last_period + 1

            part_text = text[start:end].strip()
            new_chunk = chunk.copy()
            new_chunk['chunk_id'] = f"{chunk['chunk_id']}_p{part_idx}"
            new_chunk['chunk_text'] = part_text
            final_chunks.append(new_chunk)

            start = end
            part_idx += 1

    return final_chunks


def preprocess_civil_code_to_csv(docx_path, output_csv):
    """主函数：民法典文档 → CSV 文本块"""
    logging.info(f"开始解析民法典：{docx_path}")
    chunks = parse_civil_code_txt(docx_path)

    logging.info(f"长条款分割前：{len(chunks)} 条")
    chunks = split_long_articles(chunks, max_chars=800)
    logging.info(f"长条款分割后：{len(chunks)} 个文本块")

    df = pd.DataFrame(chunks)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    logging.info(f"预处理完成，输出至 {output_csv}")

    # 打印统计信息
    print("\n=== 民法典知识库统计 ===")
    print(f"总文本块数：{len(df)}")
    print(f"涉及编数：{df['book'].nunique()}")
    print(f"涉及章数：{df['chapter'].nunique()}")
    print("\n各编文本块分布：")
    print(df['book'].value_counts())

    return df


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent / "dataset"
    DOCX_PATH = str(BASE_DIR / "中华人民共和国民法典_20200528.docx")
    OUTPUT_CSV = str(BASE_DIR / "civil_code_chunks.csv")
    preprocess_civil_code_to_csv(DOCX_PATH, OUTPUT_CSV)