# -*- coding: utf-8 -*-
"""
PDF处理器 — 独立的PDF文档处理模块

功能：
1. 解析PDF文件，去除水印/页眉/页脚
2. 提取表格内容并转为结构化文本
3. 提取图片并通过Qwen-VL多模态API生成图片语义描述
4. 文本清理：去除换行符、特殊符号、多余空格
5. 文本分块（自适应段落分块，带重叠）
6. 输出CSV文件

依赖：pymupdf（必须）
可选：requests（DeepSeek多模态API图片描述）、rapidocr_onnxruntime（OCR）

用法：
    from pdf_parser.pdf_processor import PDFProcessor
    csv_path = PDFProcessor.process_pdf("招股说明书1.pdf")
"""
import os
import re
import csv
import json
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import pymupdf  # PDF解析核心库

logger = logging.getLogger(__name__)

# 分块参数：1000字符/块，150字符重叠，平衡语义完整性和检索粒度
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
ENABLE_IMAGE_EXTRACT = True

DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', 'sk-850d2fe1c44744cabc20cc754d3921ce')
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_VISION_MODEL = "deepseek-chat"
MIN_IMAGE_SIZE = 150    # 过滤水印/图标等小图片

QWEN_VL_API_KEY = os.environ.get('QWEN_VL_API_KEY', 'sk-f720ccff5b5d42f69984de2f0300a9e4')  # 阿里云DashScope密钥
QWEN_VL_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_VL_MODEL = "qwen-vl-plus"  # 阿里云多模态模型，用于图片语义描述


# ==================== 数据结构 ====================

@dataclass
class ExtractedTable:
    """PDF中提取的表格：转为管道符分隔的文本"""
    page_num: int
    rows: List[List[str]]
    row_count: int
    col_count: int

    def to_text(self) -> str:
        lines = []
        for row in self.rows:
            cells = [cell.replace('\n', ' ').strip() if cell else "" for cell in row]
            lines.append(" | ".join(cells))
        return "\n".join(lines)


@dataclass
class PageContent:
    """单页处理结果：正文+表格+图片"""
    page_num: int
    main_text: str
    tables: List[ExtractedTable] = field(default_factory=list)
    images: List[Dict] = field(default_factory=list)


@dataclass
class Chunk:
    """文本分块：带来源类型标记，用于区分text/table/image"""
    id: int
    page_num: int
    text: str
    char_count: int
    source_type: str = "text"


# ==================== PDF处理器 ====================

class PDFProcessor:
    """PDF文档处理器：解析→去噪→提取表格/图片→分块→CSV"""

    def __init__(self, pdf_path: str, extract_images: bool = ENABLE_IMAGE_EXTRACT):
        self.pdf_path = pdf_path
        self.extract_images_enabled = extract_images
        self.doc = None
        self._page_width = 0
        self._page_height = 0

    def open(self):
        self.doc = pymupdf.open(self.pdf_path)
        if len(self.doc) > 0:
            rect = self.doc[0].rect
            self._page_width = rect.width
            self._page_height = rect.height
        logger.info(f"[PDFProcessor] 已打开PDF: {os.path.basename(self.pdf_path)}, "
                     f"{len(self.doc)}页, 图片提取={self.extract_images_enabled}")
        return self

    def close(self):
        if self.doc:
            self.doc.close()
            self.doc = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *args):
        self.close()

    # ==================== 文本清理 ====================

    @staticmethod
    def clean_text(text: str) -> str:
        """清理文本：统一空白符，过滤非常规Unicode字符"""
        if not text:
            return ""
        text = text.replace('\n', ' ').replace('\r', '').replace('\t', ' ')
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。、；：！？""''（）《》【】\[\]{}(),.;:!?\'\"-]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    # ==================== 图片OCR识别 ====================

    _ocr_engine = None

    @classmethod
    def _get_ocr_engine(cls):
        """惰性初始化RapidOCR：失败时设为False避免重复尝试"""
        if cls._ocr_engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
                cls._ocr_engine = RapidOCR()
                logger.info("[PDFProcessor] RapidOCR初始化成功")
            except ImportError:
                logger.warning("[PDFProcessor] RapidOCR未安装，图片文字识别不可用")
                cls._ocr_engine = False
            except Exception as e:
                logger.warning(f"[PDFProcessor] RapidOCR初始化失败: {e}")
                cls._ocr_engine = False
        return None if cls._ocr_engine is False else cls._ocr_engine

    def _describe_with_qwen_vl(self, image_bytes: bytes, page_num: int, img_idx: int) -> str:
        """通过Qwen-VL多模态API生成图片语义描述（图表/组织架构图等）"""
        import base64
        import requests
        import time

        img_base64 = base64.b64encode(image_bytes).decode('utf-8')
        headers = {
            "Authorization": f"Bearer {QWEN_VL_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": QWEN_VL_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                    {"type": "text", "text": "请详细描述这张图片的内容，包括图表类型、数据趋势、关键数值、文字标注等。如果是组织架构图或股权结构图，请列出所有公司名称、持股比例和层级关系。如果是流程图，描述各步骤。如果是财务图表，列出主要数据点。用中文回答，控制在500字以内。"}
                ]
            }],
            "max_tokens": 600
        }

        for attempt in range(3):
            try:
                response = requests.post(
                    f"{QWEN_VL_BASE_URL}/chat/completions",
                    headers=headers, json=payload, timeout=30
                )
                if response.status_code == 200:
                    result = response.json()
                    description = self.clean_text(result['choices'][0]['message']['content'])
                    logger.info(f"[PDFProcessor] 第{page_num}页图片{img_idx}: Qwen-VL成功({len(description)}字)")
                    return f"[多模态识别] {description[:500]}"
                elif response.status_code == 429:
                    wait_time = (attempt + 1) * 5
                    logger.warning(f"[PDFProcessor] Qwen-VL限流，等待{wait_time}秒")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"[PDFProcessor] Qwen-VL错误: {response.status_code}")
                    return ""
            except requests.exceptions.Timeout:
                if attempt < 2:
                    time.sleep(2)
                    continue
                return ""
            except Exception as e:
                if attempt < 2:
                    time.sleep((attempt + 1) * 3)
                    continue
                logger.warning(f"[PDFProcessor] Qwen-VL失败: {e}")
                return ""
        return ""

    def describe_image(self, image_bytes: bytes, page_num: int, img_idx: int) -> str:
        """图片描述三级降级：OCR文字识别 → Qwen-VL多模态 → 文件大小推断"""
        try:
            # 第一级：OCR识别图片中的文字
            ocr = self._get_ocr_engine()
            if ocr is not None and image_bytes:
                import io
                import numpy as np
                from PIL import Image
                img = Image.open(io.BytesIO(image_bytes))
                result, _ = ocr(np.array(img))
                if result:
                    texts = [item[1] for item in result if item[1].strip()]
                    if texts:
                        ocr_text = self.clean_text(" ".join(texts))
                        if len(ocr_text) > 10:
                            return f"[OCR识别] {ocr_text[:200]}"

            # 第二级：Qwen-VL多模态API（处理OCR无法识别的图表）
            logger.info(f"[PDFProcessor] 第{page_num}页图片{img_idx}: OCR无结果，尝试Qwen-VL")
            vl_description = self._describe_with_qwen_vl(image_bytes, page_num, img_idx)
            if vl_description:
                import time
                time.sleep(0.5)  # 限流保护
                return vl_description

            # 第三级：根据文件大小粗推断图片类型
            size_kb = len(image_bytes) / 1024 if image_bytes else 0
            img_type = "图表" if size_kb > 50 else ("示意图" if size_kb > 10 else "图片")
            return f"[{img_type}] 大小{size_kb:.1f}KB"

        except Exception as e:
            logger.warning(f"[PDFProcessor] 图片识别失败: {e}")
            return ""

    # ==================== 水印去除 ====================

    def remove_watermarks(self, page) -> None:
        """水印检测：基于透明度、字号、关键词三重判断"""
        try:
            blocks = page.get_text("dict")
            if not blocks or "blocks" not in blocks:
                return

            watermark_keywords = [
                "内部资料", "机密", " confidential", "内部文件",
                "仅供参考", "不得外传", "严禁复制", "受版权保护",
                "draft", "样本", "示例", "内部使用"
            ]

            for block in blocks.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        font_size = span.get("size", 0)
                        opacity = span.get("opacity", 1.0)

                        is_watermark = False
                        # 低透明度+大字号 → 水印
                        if opacity < 0.8 and font_size > 20:
                            is_watermark = True
                        # 包含水印关键词+较大字号 → 水印
                        if any(kw in text.lower() for kw in watermark_keywords) and font_size > 15:
                            is_watermark = True
                        # 高重复字符+大字号 → 水印（如"机密机密机密"）
                        if len(set(text)) <= 2 and len(text) > 5 and font_size > 20:
                            is_watermark = True

                        if is_watermark:
                            bbox = span.get("bbox")
                            if bbox:
                                page.add_redact_annot(bbox, fill=(1, 1, 1), text="")
            page.apply_redactions()
        except Exception as e:
            logger.warning(f"[PDFProcessor] 水印去除失败: {e}")

    # ==================== 页眉页脚去除 ====================

    def clean_page_text(self, page) -> str:
        """提取文本并过滤页眉页脚：基于Y坐标阈值+模式匹配"""
        blocks = page.get_text("blocks")
        if not blocks:
            return ""

        header_y_threshold = 60                # 页眉Y坐标阈值（顶部60px内）
        footer_y_threshold = self._page_height - 60  # 页脚Y坐标阈值（底部60px内）
        header_text_pattern = re.compile(r'武汉兴图新科电子股份有限公司\s*.*\s*招股意向书')

        kept_blocks = []
        for b in blocks:
            y0, y1 = b[1], b[3]
            text = b[4].strip()
            if not text:
                continue
            # 过滤日期格式页眉
            if re.fullmatch(r'\d+-\d+-\d+\s*', text):
                continue
            # 过滤公司名+招股书页眉
            if y1 < header_y_threshold:
                if header_text_pattern.search(text):
                    continue
                if text.strip() == '武汉兴图新科电子股份有限公司':
                    continue
            # 过滤页码
            if y0 > footer_y_threshold and text.isdigit():
                continue
            kept_blocks.append(b[4])

        result = '\n'.join(kept_blocks).strip()
        # 清理残留的页眉页脚行
        result = re.sub(r'^\d+-\d+-\d+\s*$', '', result, flags=re.MULTILINE)
        result = re.sub(r'^武汉兴图新科电子股份有限公司\s+招股意向书\s*$', '', result, flags=re.MULTILINE)
        result = re.sub(r'\n{3,}', '\n\n', result)
        return result.strip()

    # ==================== 表格提取 ====================

    def extract_tables(self, page, page_num: int) -> List[ExtractedTable]:
        """提取PDF页面中的表格：利用pymupdf的find_tables能力"""
        tables = []
        try:
            found = page.find_tables()
            for t in found.tables:
                rows = t.extract()
                if rows and len(rows) > 0:
                    tables.append(ExtractedTable(
                        page_num=page_num, rows=rows,
                        row_count=t.row_count, col_count=t.col_count
                    ))
        except Exception as e:
            logger.warning(f"[PDFProcessor] 第{page_num}页表格提取失败: {e}")
        return tables

    # ==================== 图片提取 ====================

    def extract_images(self, page, page_num: int) -> List[Dict]:
        """提取页面图片：过滤小图标，提取文字描述，调用多模态API生成语义描述。
        对矢量绘图（提取为黑色/空白图片）自动兜底：渲染整页为图片后裁剪"""
        image_info = []
        try:
            img_list = page.get_images()
            for img_idx, img_ref in enumerate(img_list):
                xref = img_ref[0]
                base_img = self.doc.extract_image(xref)
                if not base_img:
                    continue
                w = base_img.get("width", 0)
                h = base_img.get("height", 0)
                # 过滤小图标/水印（小于150px）
                if w < MIN_IMAGE_SIZE or h < MIN_IMAGE_SIZE:
                    continue

                nearby = self._get_nearby_text(page, img_idx)  # 图片周围的标题/说明文字
                image_bytes = base_img.get("image", b"")
                size_kb = len(image_bytes) / 1024 if image_bytes else 0
                # 过滤极小图片（<3KB，通常是装饰元素）
                if size_kb < 3:
                    continue

                description = self.describe_image(image_bytes, page_num, img_idx) if image_bytes else ""

                # 兜底：当图片描述为"纯黑色"/"空白"/"黑色矩形"时，用页面渲染兜底
                # 这类图片通常是PDF矢量绘图（如组织结构图），pymupdf无法正确提取
                if description and any(kw in description for kw in ['黑色', '空白', '矩形框', '无内容']):
                    rendered = self._render_page_region(page, page_num, img_idx)
                    if rendered:
                        desc2 = self.describe_image(rendered, page_num, img_idx)
                        if desc2 and desc2 != description:
                            description = desc2
                            image_bytes = rendered
                            size_kb = len(rendered) / 1024

                image_info.append({
                    "page_num": page_num, "img_idx": img_idx,
                    "width": w, "height": h,
                    "size_kb": round(size_kb, 1),
                    "nearby_text": nearby,
                    "description": description,
                })
        except Exception as e:
            logger.warning(f"[PDFProcessor] 第{page_num}页图片提取失败: {e}")
        return image_info

    def _render_page_region(self, page, page_num: int, img_idx: int) -> bytes:
        """矢量绘图兜底：将页面指定区域渲染为高清PNG，解决矢量绘图提取为空白的问题"""
        try:
            import io
            img_info = page.get_image_info()
            if img_idx >= len(img_info):
                return b""
            bbox = img_info[img_idx].get("bbox")
            if not bbox:
                return b""
            x0, y0, x1, y1 = bbox
            # 扩大裁剪区域20%，捕获标题和边框
            margin_x = (x1 - x0) * 0.1
            margin_y = (y1 - y0) * 0.1
            clip = pymupdf.Rect(
                max(0, x0 - margin_x), max(0, y0 - margin_y),
                min(page.rect.width, x1 + margin_x), min(page.rect.height, y1 + margin_y)
            )
            # 2倍渲染分辨率，确保文字清晰可读
            mat = pymupdf.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat, clip=clip)
            img_bytes = pix.tobytes("png")
            logger.info(f"[PDFProcessor] 第{page_num}页图片{img_idx}: 页面渲染兜底({len(img_bytes)//1024}KB)")
            return img_bytes
        except Exception as e:
            logger.warning(f"[PDFProcessor] 页面渲染兜底失败: {e}")
            return b""

    def _get_nearby_text(self, page, img_idx: int) -> str:
        """获取图片周围文本（上方标题+下方说明），作为图片的上下文补充"""
        try:
            blocks = page.get_text("blocks")
            img_info = page.get_image_info()
            if img_idx < len(img_info):
                bbox = img_info[img_idx].get("bbox")
                if bbox:
                    x0, y0, x1, y1 = bbox
                    above, below = [], []
                    for b in blocks:
                        block_y0, block_y1 = b[1], b[3]
                        block_text = b[4].strip()
                        if not block_text:
                            continue
                        # 上方400px内的文本（标题，范围扩大以捕获远距离标题）
                        if block_y1 < y0 and block_y1 > y0 - 400:
                            above.append((y0 - block_y1, block_text))
                        # 下方400px内的文本（图注，范围扩大以捕获远距离说明）
                        if block_y0 > y1 and block_y0 < y1 + 400:
                            below.append((block_y0 - y1, block_text))
                    parts = []
                    if above:
                        above.sort()
                        parts.append(above[0][1][:100])
                    if below:
                        below.sort()
                        parts.append(below[0][1][:200])
                    return " ".join(parts)
        except Exception:
            pass
        return ""

    # ==================== 文本分块 ====================

    def chunk_text(self, text: str, page_num: int,
                   start_id: int = 0, source_type: str = "text") -> List[Chunk]:
        """自适应段落分块：优先按段落切分，超长段落再按句子切分"""
        chunks = []
        chunk_id = start_id
        if not text or not text.strip():
            return chunks

        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        current = ""
        for para in paragraphs:
            # 当前块未满，合并相邻段落
            if current and len(current) + len(para) + 1 <= CHUNK_SIZE:
                current += "\n\n" + para
            else:
                if current:
                    chunks.append(Chunk(chunk_id, page_num, current, len(current), source_type))
                    chunk_id += 1
                # 超长段落：按句子边界拆分
                if len(para) > CHUNK_SIZE:
                    sub = self._split_long(para, page_num, chunk_id, source_type)
                    chunks.extend(sub)
                    chunk_id += len(sub)
                    current = ""
                else:
                    current = para

        if current:
            chunks.append(Chunk(chunk_id, page_num, current, len(current), source_type))
        return chunks

    def _split_long(self, text: str, page_num: int,
                    start_id: int, source_type: str) -> List[Chunk]:
        """超长文本拆分：优先在句子边界处切分，保留上下文重叠"""
        chunks = []
        pos = 0
        cid = start_id
        total = len(text)
        while pos < total:
            end = min(pos + CHUNK_SIZE, total)
            if end < total:
                # 在候选区间内寻找最近的句子分隔符（。；；\n）
                seg = text[max(pos, end - 100):end + 50]
                for sep in ['。', '；', '\n', '.', ';']:
                    idx = seg.rfind(sep)
                    if idx > 20:
                        end = max(pos, end - 100) + idx + 1
                        break
            ct = text[pos:end].strip()
            if ct:
                chunks.append(Chunk(cid, page_num, ct, len(ct), source_type))
                cid += 1
            # 重叠窗口：确保跨块信息不丢失
            pos = max(end - CHUNK_OVERLAP, pos + CHUNK_SIZE // 2)
            if pos >= total:
                break
        return chunks

    # ==================== 目录页检测 ====================

    @staticmethod
    def _is_toc_page(text: str) -> bool:
        """目录页检测：多行含省略号（...）的页面判定为目录"""
        if not text:
            return False
        dot_lines = sum(1 for line in text.split('\n') if '...' in line and len(line) > 20)
        return dot_lines >= 3

    # ==================== 主处理流程 ====================

    def process_all(self) -> Tuple[List[Chunk], List[PageContent]]:
        """完整处理PDF：逐页解析→去水印→提取正文/表格/图片→分块"""
        if not self.doc:
            self.open()

        all_chunks = []
        pages_detail = []
        global_chunk_id = 0
        total_pages = len(self.doc)
        skipped_toc_pages = 0
        image_descriptions_count = 0

        for page_num in range(1, total_pages + 1):
            page = self.doc[page_num - 1]
            # 水印去除（在提取文本之前执行）
            self.remove_watermarks(page)

            raw_text = page.get_text()
            # 目录页跳过（含大量省略号的索引页）
            if self._is_toc_page(raw_text):
                skipped_toc_pages += 1
                if skipped_toc_pages == 1:
                    logger.info(f"[PDFProcessor] 第{page_num}页: 检测到目录页，跳过")
                continue

            # 清理页眉页脚后提取正文
            main_text = self.clean_page_text(page)
            if main_text:
                main_text = self.clean_text(main_text)

            tables = self.extract_tables(page, page_num)

            images = []
            if self.extract_images_enabled:
                images = self.extract_images(page, page_num)
                for img in images:
                    if img.get("description"):
                        image_descriptions_count += 1

            pages_detail.append(PageContent(
                page_num=page_num, main_text=main_text,
                tables=tables, images=images,
            ))

            # 正文分块
            if main_text:
                tc = self.chunk_text(main_text, page_num, global_chunk_id, "text")
                for c in tc:
                    c.id = global_chunk_id
                    global_chunk_id += 1
                all_chunks.extend(tc)

            # 表格分块：转为管道符分隔的文本，标记source_type=table
            for table in tables:
                table_text = table.to_text()
                if table_text.strip():
                    table_text = self.clean_text(table_text)
                    tc = self.chunk_text(
                        f"[表格] 第{table.page_num}页 {table_text}",
                        page_num, global_chunk_id, "table"
                    )
                    for c in tc:
                        c.id = global_chunk_id
                        global_chunk_id += 1
                    all_chunks.extend(tc)

            # 图片分块：优先用API描述，回退用周围文本
            for img in images:
                desc = img.get("description", "")
                nearby = img.get("nearby_text", "")
                if desc:
                    t = f"[图片] 第{page_num}页 ({img['width']}x{img['height']}) {desc}"
                    all_chunks.append(Chunk(global_chunk_id, page_num, t, len(t), "image"))
                    global_chunk_id += 1
                elif nearby:
                    t = f"[图片] 第{page_num}页 ({img['width']}x{img['height']}) {nearby}"
                    all_chunks.append(Chunk(global_chunk_id, page_num, t, len(t), "image"))
                    global_chunk_id += 1

            if page_num % 50 == 0 or page_num == total_pages:
                logger.info(f"[PDFProcessor] {page_num}/{total_pages}页: {len(all_chunks)}个分块")

        logger.info(
            f"[PDFProcessor] 完成: {total_pages}页, {len(all_chunks)}个分块, "
            f"{sum(1 for c in all_chunks if c.source_type=='table')}个表格, "
            f"{sum(1 for c in all_chunks if c.source_type=='image')}个图片"
        )
        if image_descriptions_count > 0:
            logger.info(f"[PDFProcessor] 图片描述: {image_descriptions_count}张")
        return all_chunks, pages_detail

    # ==================== CSV导出 ====================

    def export_to_csv(self, chunks: List[Chunk], output_path: str = None) -> str:
        """导出分块到CSV：作为TF-IDF检索的数据源"""
        if output_path is None:
            base = os.path.splitext(self.pdf_path)[0]
            output_path = f"{base}_chunks.csv"

        with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'page_num', 'text', 'char_count', 'chunk_index', 'source_type'])
            for chunk in chunks:
                clean_text = chunk.text.replace('\n', ' ').replace('\r', '').strip()
                writer.writerow([chunk.id, chunk.page_num, clean_text,
                                 len(clean_text), chunk.id, chunk.source_type])

        logger.info(f"[PDFProcessor] CSV已导出: {output_path} ({len(chunks)}行)")
        return output_path

    # ==================== 便捷入口 ====================

    @staticmethod
    def process_pdf(pdf_path: str, output_csv: str = None,
                    extract_images: bool = False) -> str:
        """一键处理：打开PDF→解析→分块→导出CSV"""
        with PDFProcessor(pdf_path, extract_images=extract_images) as proc:
            chunks, _ = proc.process_all()
            csv_path = proc.export_to_csv(chunks, output_csv)
        return csv_path


# ==================== 命令行入口 ====================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAG PDF文档处理器")
    parser.add_argument("pdf", nargs="?", help="PDF文件路径")
    parser.add_argument("-o", "--output", help="输出CSV文件路径")
    parser.add_argument("--chunk-size", type=int, default=1000, help="分块大小(字符)")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="分块重叠(字符)")
    parser.add_argument("--extract-images", action="store_true", default=True, help="提取图片(默认开启)")
    parser.add_argument("--no-images", action="store_true", help="禁用图片提取")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    pdf_path = args.pdf or os.path.join(os.path.dirname(__file__), "招股说明书1.pdf")
    if not os.path.exists(pdf_path):
        print(f"错误: 找不到PDF文件: {pdf_path}")
        return 1

    if args.chunk_size:
        global CHUNK_SIZE, CHUNK_OVERLAP
        CHUNK_SIZE = args.chunk_size
        CHUNK_OVERLAP = args.chunk_overlap

    print(f"开始处理PDF: {pdf_path}")
    print(f"分块: {CHUNK_SIZE}字符, 重叠{CHUNK_OVERLAP}字符")
    print(f"提取图片: {not args.no_images}")

    use_images = not args.no_images
    csv_path = PDFProcessor.process_pdf(pdf_path, args.output, use_images)
    print(f"\n处理完成! CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    exit(main())
