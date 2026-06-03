# -*- coding: utf-8 -*-
"""
PDF处理器 — 独立的PDF文档处理模块
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化

功能：
1. 解析PDF文件，去除水印/页眉/页脚
2. 提取表格内容并转为结构化文本
3. 提取图片及关联文本（可选，默认关闭）
4. 文本分块（自适应段落分块，带重叠）
5. 输出CSV文件（与原格式兼容）

依赖：pymupdf（必须）
可选依赖：tesseract-ocr（图片OCR）

用法：
    from pdf_parser.pdf_processor import PDFProcessor
    csv_path = PDFProcessor.process_pdf("招股说明书1.pdf")
    
    或命令行：
    python pdf_processor.py 招股说明书1.pdf -o output.csv
"""

import os  # 导入操作系统接口模块
import re  # 导入正则表达式模块
import csv  # 导入CSV文件处理模块
import logging  # 导入日志模块
from typing import List, Dict, Optional, Tuple  # 导入类型提示
from dataclasses import dataclass, field  # 导入数据类装饰器

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

# ==================== 配置 ====================

CHUNK_SIZE = 1000  # 每块目标字符数（优化：从800增至1000，财务法律文本需更大上下文）
CHUNK_OVERLAP = 150  # 块间重叠字符数（优化：从200减至150，减少冗余分块）
ENABLE_IMAGE_EXTRACT = False  # 是否提取图片（默认关闭，太慢）


# ==================== 数据结构 ====================

@dataclass
class ExtractedTable:
    """从PDF提取的表格"""
    page_num: int  # 表格所在页码
    rows: List[List[str]]  # 表格行数据
    row_count: int  # 行数
    col_count: int  # 列数

    def to_text(self) -> str:
        """将表格转为结构化文本"""
        lines = []  # 初始化行列表
        for row in self.rows:  # 遍历每一行
            cells = []  # 初始化单元格列表
            for cell in row:  # 遍历每个单元格
                if cell is None:  # 空单元格处理
                    cells.append("")  # 添加空字符串
                else:  # 非空单元格
                    cells.append(cell.replace('\n', ' ').strip())  # 替换换行并去除首尾空格
            lines.append(" | ".join(cells))  # 用分隔符合并单元格
        return "\n".join(lines)  # 用换行合并所有行


@dataclass
class PageContent:
    """单页处理结果"""
    page_num: int  # 页码
    main_text: str  # 正文文本
    tables: List[ExtractedTable] = field(default_factory=list)  # 本页的表格列表
    images: List[Dict] = field(default_factory=list)  # 本页的图片信息列表


@dataclass
class Chunk:
    """文本块"""
    id: int  # 分块唯一ID
    page_num: int  # 来源页码
    text: str  # 分块文本内容
    char_count: int  # 字符数
    source_type: str = "text"  # 来源类型：text / table / image


# ==================== PDF处理器 ====================

class PDFProcessor:
    """
    PDF文档处理器
    负责：解析 -> 去噪 -> 提取表格/图片 -> 分块 -> 导出CSV
    """

    def __init__(self, pdf_path: str, extract_images: bool = ENABLE_IMAGE_EXTRACT):
        self.pdf_path = pdf_path  # PDF文件路径
        self.extract_images = extract_images  # 是否提取图片
        self.doc = None  # pymupdf文档对象，初始为None
        self._page_width = 0  # 页面宽度（像素）
        self._page_height = 0  # 页面高度（像素）

    def open(self):
        import pymupdf  # 导入PDF处理库
        self.doc = pymupdf.open(self.pdf_path)  # 打开PDF文件
        if len(self.doc) > 0:  # PDF有页面时
            rect = self.doc[0].rect  # 获取第一页的页面矩形
            self._page_width = rect.width  # 记录页面宽度
            self._page_height = rect.height  # 记录页面高度
        logger.info(f"[PDFProcessor] 已打开PDF: {os.path.basename(self.pdf_path)}, "  # 打印打开日志
                     f"{len(self.doc)}页, 提取图片={self.extract_images}")  # 页数和图片提取设置
        return self  # 返回自身以支持链式调用

    def close(self):
        if self.doc:  # 文档已打开时
            self.doc.close()  # 关闭文档
            self.doc = None  # 清空文档引用

    def __enter__(self):
        return self.open()  # 上下文管理器进入时打开文档

    def __exit__(self, *args):
        self.close()  # 上下文管理器退出时关闭文档

    # ==================== 页眉页脚去除 ====================

    def clean_page_text(self, page) -> str:
        """
        从页面提取文本并去除页眉页脚
        使用pymupdf的blocks定位：页眉在顶部(y<60)，页脚在底部(y>页高-60)
        """
        blocks = page.get_text("blocks")  # 获取页面的文本块列表
        if not blocks:  # 无文本块时
            return ""  # 返回空字符串

        # 过滤条件
        header_y_threshold = 60  # 页眉Y坐标阈值
        footer_y_threshold = self._page_height - 60  # 页脚Y坐标阈值
        # 公司名称+招股意向书的组合页眉
        header_text_pattern = re.compile(  # 编译页眉匹配正则
            r'武汉兴图新科电子股份有限公司\s*.*\s*招股意向书'  # 匹配公司+招股书组合
        )

        kept_blocks = []  # 保留的文本块列表
        for b in blocks:  # 遍历每个文本块
            y0, y1 = b[1], b[3]  # 文本块的顶部和底部Y坐标
            text = b[4].strip()  # 提取并清理文本内容

            # 跳过空白块
            if not text:  # 文本为空时
                continue  # 跳过

            # 跳过纯页码块（1-1-0, 1-1-1等），无论位置
            if re.fullmatch(r'\d+-\d+-\d+\s*', text):  # 匹配页码格式
                continue  # 跳过

            # 跳过页眉：顶部且匹配公司行
            if y1 < header_y_threshold:  # 在页眉区域
                if header_text_pattern.search(text):  # 匹配公司名称页眉
                    continue  # 跳过
                if text.strip() == '武汉兴图新科电子股份有限公司':  # 单纯公司名
                    continue  # 跳过

            # 跳过页脚：底部页码
            if y0 > footer_y_threshold:  # 在页脚区域
                if text.isdigit():  # 内容为纯数字（页码）
                    continue  # 跳过

            kept_blocks.append(b[4])  # 保留此文本块

        result = '\n'.join(kept_blocks).strip()  # 用换符合并保留的文本块

        # 文本级后处理：去除残留的单独页码行
        result = re.sub(r'^\d+-\d+-\d+\s*$', '', result, flags=re.MULTILINE)  # 去除残留页码行
        # 去除单独的"武汉兴图新科电子股份有限公司  招股意向书"行
        result = re.sub(r'^武汉兴图新科电子股份有限公司\s+招股意向书\s*$', '', result, flags=re.MULTILINE)  # 去除公司名称行
        # 清理多余空行
        result = re.sub(r'\n{3,}', '\n\n', result)  # 将3个以上连续换行替换为2个

        return result.strip()  # 返回清理后的文本

    # ==================== 表格提取 ====================

    def extract_tables(self, page, page_num: int) -> List[ExtractedTable]:
        """提取当前页的所有表格"""
        tables = []  # 初始化表格列表
        try:
            found = page.find_tables()  # 查找页面中的表格
            for t in found.tables:  # 遍历每个找到的表格
                rows = t.extract()  # 提取表格行数据
                if rows and len(rows) > 0:  # 有数据时
                    tables.append(ExtractedTable(  # 添加提取的表格
                        page_num=page_num,  # 页码
                        rows=rows,  # 行数据
                        row_count=t.row_count,  # 行数
                        col_count=t.col_count  # 列数
                    ))
        except Exception as e:  # 提取失败时
            logger.warning(f"[PDFProcessor] 第{page_num}页表格提取失败: {e}")  # 记录警告
        return tables  # 返回表格列表

    # ==================== 图片提取 ====================

    def extract_images(self, page, page_num: int) -> List[Dict]:
        """提取当前页的图片信息（仅元数据）"""
        image_info = []  # 初始化图片信息列表
        try:
            img_list = page.get_images()  # 获取页面图片列表
            for img_idx, img_ref in enumerate(img_list):  # 遍历每张图片
                xref = img_ref[0]  # 图片的交叉引用编号
                base_img = self.doc.extract_image(xref)  # 提取图片数据
                if not base_img:  # 提取失败时
                    continue  # 跳过
                w = base_img.get("width", 0)  # 图片宽度
                h = base_img.get("height", 0)  # 图片高度
                if w < 30 or h < 30:  # 过小的图片（图标）跳过
                    continue  # 跳过
                # 获取图片周围的文本
                nearby = self._get_nearby_text(page, img_idx)  # 获取图片下方的说明文本
                image_info.append({  # 添加图片信息
                    "page_num": page_num,  # 页码
                    "img_idx": img_idx,  # 图片索引
                    "width": w,  # 宽度
                    "height": h,  # 高度
                    "size_kb": round(len(base_img["image"]) / 1024, 1),  # 大小（KB）
                    "nearby_text": nearby,  # 附近文本
                })
        except Exception as e:  # 提取失败时
            logger.warning(f"[PDFProcessor] 第{page_num}页图片提取失败: {e}")  # 记录警告
        return image_info  # 返回图片信息列表

    def _get_nearby_text(self, page, img_idx: int) -> str:
        """获取图片下方的说明文本"""
        try:
            blocks = page.get_text("blocks")  # 获取页面文本块
            img_info = page.get_image_info()  # 获取页面图片信息
            if img_idx < len(img_info):  # 图片索引有效时
                bbox = img_info[img_idx].get("bbox")  # 获取图片边界框
                if bbox:  # 有边界框时
                    y1 = bbox[3]  # 图片底部Y坐标
                    candidates = []  # 候选文本块列表
                    for b in blocks:  # 遍历所有文本块
                        if b[1] > y1 + 5 and b[1] < y1 + 150:  # 图片下方5-150像素内的文本
                            candidates.append((b[1], b[4]))  # 添加（Y坐标，文本）
                    if candidates:  # 有候选文本时
                        candidates.sort()  # 按Y坐标排序
                        return candidates[0][1][:200]  # 返回最近文本的前200字
        except Exception:  # 忽略所有异常
            pass  # 静默处理
        return ""  # 未找到附近文本时返回空字符串

    # ==================== 文本分块 ====================

    def chunk_text(self, text: str, page_num: int,
                   start_id: int = 0,
                   source_type: str = "text") -> List[Chunk]:
        """将文本分块，自适应段落边界，带重叠"""
        chunks = []  # 初始化分块列表
        chunk_id = start_id  # 分块起始ID
        if not text or not text.strip():  # 文本为空时
            return chunks  # 返回空列表

        # 按空行分段落
        paragraphs = re.split(r'\n\s*\n', text)  # 按空行分割段落
        paragraphs = [p.strip() for p in paragraphs if p.strip()]  # 清理并过滤空段落

        current = ""  # 当前累积的文本
        for para in paragraphs:  # 遍历每个段落
            if current and len(current) + len(para) + 1 <= CHUNK_SIZE:  # 合并后不超过分块大小
                current += "\n\n" + para  # 合并到当前块
            else:  # 需要新分块
                if current:  # 当前块非空
                    chunks.append(Chunk(chunk_id, page_num, current, len(current), source_type))  # 保存当前块
                    chunk_id += 1  # 递增ID
                if len(para) > CHUNK_SIZE:  # 段落超过分块大小时
                    sub = self._split_long(para, page_num, chunk_id, source_type)  # 分割长段落
                    chunks.extend(sub)  # 添加分割后的子块
                    chunk_id += len(sub)  # 更新ID计数器
                    current = ""  # 重置当前块
                else:  # 段落大小合适
                    current = para  # 开始新块

        if current:  # 处理最后一个块
            chunks.append(Chunk(chunk_id, page_num, current, len(current), source_type))  # 保存最后一块
        return chunks  # 返回分块列表

    def _split_long(self, text: str, page_num: int,
                    start_id: int, source_type: str) -> List[Chunk]:
        """分割超长段落（句子边界分割，带重叠）"""
        chunks = []  # 初始化分块列表
        pos = 0  # 当前读取位置
        cid = start_id  # 起始ID
        total = len(text)  # 文本总长度
        while pos < total:  # 未处理完所有文本时
            end = min(pos + CHUNK_SIZE, total)  # 计算结束位置
            if end < total:  # 不是最后一段时
                seg = text[max(pos, end - 100):end + 50]  # 在边界附近取一段文本用于查找分隔符
                for sep in ['。', '；', '\n', '.', ';']:  # 按优先级查找句子分隔符
                    idx = seg.rfind(sep)  # 从右向左查找分隔符
                    if idx > 20:  # 分隔符位置合适（避免分割过短）
                        end = max(pos, end - 100) + idx + 1  # 调整结束位置到分隔符后
                        break  # 找到分隔符后跳出
            ct = text[pos:end].strip()  # 提取当前分块文本
            if ct:  # 非空时
                chunks.append(Chunk(cid, page_num, ct, len(ct), source_type))  # 添加分块
                cid += 1  # 递增ID
            pos = max(end - CHUNK_OVERLAP, pos + CHUNK_SIZE // 2)  # 计算下一个起始位置（带重叠）
            if pos >= total:  # 已处理完时
                break  # 退出循环
        return chunks  # 返回分块列表

    # ==================== 目录页检测 ====================

    @staticmethod
    def _is_toc_page(text: str) -> bool:
        """
        检测是否为目录页（TOC页面）
        目录页特征：包含多行"......."点线连接标题和页码
        返回: True=是目录页，应跳过
        """
        if not text:  # 空文本
            return False  # 不是目录页
        # 统计包含连续3个以上"."且长度超过20字符的行数
        dot_lines = 0  # 含点线的行数计数器
        for line in text.split('\n'):  # 逐行检查
            if '...' in line and len(line) > 20:  # 含点线且行较长
                dot_lines += 1  # 计数加1
        # 目录页通常有大量点线行（>=3行），正文页没有
        return dot_lines >= 3  # 点线行数>=3视为目录页

    # ==================== 主处理流程 ====================

    def process_all(self) -> Tuple[List[Chunk], List[PageContent]]:
        """
        完整处理PDF：解析所有页面 -> 去噪 -> 提取表格/图片 -> 分块
        返回: (chunks列表, pages详情列表)
        """
        if not self.doc:  # 文档未打开时
            self.open()  # 打开文档

        all_chunks = []  # 所有分块列表
        pages_detail = []  # 页面详情列表
        global_chunk_id = 0  # 全局分块ID计数器
        total_pages = len(self.doc)  # 文档总页数
        skipped_toc_pages = 0  # 跳过的目录页计数

        for page_num in range(1, total_pages + 1):  # 遍历每一页（从1开始）
            page = self.doc[page_num - 1]  # 获取当前页面对象

            # 0. 获取原始文本用于目录页检测
            raw_text = page.get_text()  # 提取页面原始文本
            if self._is_toc_page(raw_text):  # 检测是否为目录页
                skipped_toc_pages += 1  # 目录页计数加1
                if skipped_toc_pages == 1:  # 首次遇到目录页时
                    logger.info(f"[PDFProcessor] 第{page_num}页: 检测到目录页，跳过处理")  # 日志提示
                continue  # 跳过当前页，不处理

            # 1. 获取纯净正文（去除页眉页脚）
            main_text = self.clean_page_text(page)  # 清理页眉页脚后获取正文

            # 2. 提取表格
            tables = self.extract_tables(page, page_num)  # 提取本页所有表格

            # 3. 提取图片（可选）
            images = []  # 初始化图片列表
            if self.extract_images:  # 启用图片提取时
                images = self.extract_images(page, page_num)  # 提取图片信息

            # 保存页面详情
            pages_detail.append(PageContent(  # 添加页面详情
                page_num=page_num,  # 页码
                main_text=main_text,  # 正文文本
                tables=tables,  # 表格列表
                images=images,  # 图片列表
            ))

            # 4. 正文分块
            if main_text:  # 有正文时
                tc = self.chunk_text(main_text, page_num, global_chunk_id, "text")  # 对正文进行分块
                for c in tc:  # 遍历每个分块
                    c.id = global_chunk_id  # 设置全局ID
                    global_chunk_id += 1  # 递增全局ID
                all_chunks.extend(tc)  # 添加到全部分块列表

            # 5. 表格转文本后分块
            for table in tables:  # 遍历每个表格
                table_text = table.to_text()  # 将表格转为文本
                if table_text.strip():  # 表格文本非空时
                    tc = self.chunk_text(  # 对表格文本进行分块
                        f"[表格] 第{table.page_num}页\n{table_text}",  # 添加表格标记和页码
                        page_num, global_chunk_id, "table"  # 来源类型为table
                    )
                    for c in tc:  # 遍历每个分块
                        c.id = global_chunk_id  # 设置全局ID
                        global_chunk_id += 1  # 递增全局ID
                    all_chunks.extend(tc)  # 添加到全部分块列表

            # 6. 图片关联文本
            for img in images:  # 遍历每张图片
                if img.get("nearby_text"):  # 有附近文本时
                    t = f"[图片] 第{page_num}页 ({img['width']}x{img['height']}) {img['nearby_text']}"  # 构造图片文本描述
                    all_chunks.append(Chunk(global_chunk_id, page_num, t, len(t), "image"))  # 添加图片分块
                    global_chunk_id += 1  # 递增全局ID

            # 进度日志
            if page_num % 50 == 0 or page_num == total_pages:  # 每50页或最后一页时
                logger.info(  # 打印进度日志
                    f"[PDFProcessor] {page_num}/{total_pages}页: "  # 当前页/总页数
                    f"{len(all_chunks)}个分块"  # 当前分块总数
                )

        logger.info(  # 打印处理完成日志
            f"[PDFProcessor] 处理完成: {total_pages}页, {len(all_chunks)}个分块, "  # 总页数和分块数
            f"{sum(1 for c in all_chunks if c.source_type=='table')}个表格块"  # 表格块数量
        )
        return all_chunks, pages_detail  # 返回全部分块和页面详情

    # ==================== CSV导出 ====================

    def export_to_csv(self, chunks: List[Chunk], output_path: str = None) -> str:
        """
        导出CSV文件
        格式: id, page_num, text, char_count, chunk_index, source_type
        """
        if output_path is None:  # 未指定输出路径时
            base = os.path.splitext(self.pdf_path)[0]  # 获取PDF文件名（不含扩展名）
            output_path = f"{base}_chunks.csv"  # 生成默认CSV路径

        with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:  # 以UTF-8 BOM格式写入CSV
            writer = csv.writer(f)  # 创建CSV写入器
            writer.writerow(['id', 'page_num', 'text', 'char_count', 'chunk_index', 'source_type'])  # 写入表头
            for chunk in chunks:  # 遍历每个分块
                # 去除文本中的换行符，确保每行就是一条完整记录
                clean_text = chunk.text.replace('\n', ' ').replace('\r', '').strip()  # 将换行替换为空格
                writer.writerow([  # 写入一行记录
                    chunk.id, chunk.page_num, clean_text,  # ID、页码、清理后文本
                    len(clean_text), chunk.id, chunk.source_type,  # 字符数、索引、来源类型
                ])

        logger.info(f"[PDFProcessor] CSV已导出: {output_path} ({len(chunks)}行)")  # 打印导出成功日志
        fsize = os.path.getsize(output_path)  # 获取文件大小
        logger.info(f"[PDFProcessor] 文件大小: {fsize:,} 字节")  # 打印文件大小
        return output_path  # 返回CSV文件路径

    # ==================== 便捷入口 ====================

    @staticmethod
    def process_pdf(pdf_path: str, output_csv: str = None,
                    extract_images: bool = False) -> str:
        """一键处理：解析 -> 去噪 -> 提取 -> 分块 -> CSV导出"""
        with PDFProcessor(pdf_path, extract_images=extract_images) as proc:  # 创建处理器并作为上下文管理器
            chunks, _ = proc.process_all()  # 完整处理PDF
            csv_path = proc.export_to_csv(chunks, output_csv)  # 导出CSV
        return csv_path  # 返回CSV路径


# ==================== 命令行入口 ====================

def main():
    import argparse  # 导入命令行参数解析模块
    parser = argparse.ArgumentParser(description="RAG PDF文档处理器")  # 创建参数解析器
    parser.add_argument("pdf", nargs="?", help="PDF文件路径")  # PDF文件位置参数（可选）
    parser.add_argument("-o", "--output", help="输出CSV文件路径")  # 输出文件参数
    parser.add_argument("--chunk-size", type=int, default=1000, help="分块大小(字符)")  # 分块大小参数（优化值：1000）
    parser.add_argument("--chunk-overlap", type=int, default=150, help="分块重叠(字符)")  # 分块重叠参数（优化值：150）
    parser.add_argument("--extract-images", action="store_true", help="提取图片信息(较慢)")  # 提取图片开关
    args = parser.parse_args()  # 解析命令行参数

    logging.basicConfig(level=logging.INFO,  # 配置日志级别
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # 设置日志格式

    pdf_path = args.pdf or os.path.join(os.path.dirname(__file__), "招股说明书1.pdf")  # 没有参数时使用默认路径
    if not os.path.exists(pdf_path):  # 文件不存在时
        print(f"错误: 找不到PDF文件: {pdf_path}")  # 打印错误信息
        return 1  # 返回非零退出码

    if args.chunk_size:  # 用户指定了分块大小时
        global CHUNK_SIZE, CHUNK_OVERLAP  # 声明使用全局配置变量
        CHUNK_SIZE = args.chunk_size  # 更新分块大小
        CHUNK_OVERLAP = args.chunk_overlap  # 更新分块重叠大小

    print(f"开始处理PDF: {pdf_path}")  # 打印开始信息
    print(f"分块: {CHUNK_SIZE}字符, 重叠{CHUNK_OVERLAP}字符")  # 打印分块配置
    print(f"提取图片: {args.extract_images}")  # 打印图片提取设置

    csv_path = PDFProcessor.process_pdf(pdf_path, args.output, args.extract_images)  # 调用一键处理
    print(f"\n处理完成! CSV: {csv_path}")  # 打印完成信息
    return 0  # 返回成功退出码


if __name__ == "__main__":
    exit(main())  # 调用主函数并以退出码结束
