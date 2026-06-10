# RAGFlow架构分析与PDF解析技术总结

## 一、RAGFlow系统架构概述

RAGFlow是InfiniFlow开源的RAG引擎，核心组件包括：
- **API Server**：提供外部接口和平台功能
- **Task Executor**：负责文件解析、分块、向量化和索引
- **Redis Stream**：任务消息队列，连接API Server和Task Executor
- **Elasticsearch/Infinity**：向量存储和全文检索
- **MinIO**：文件对象存储
- **MySQL**：元数据存储

## 二、PDF解析技术（DeepDoc模块）

### 2.1 解析器架构

RAGFlow内置多种解析器，通过`ParserType`枚举区分：

| 解析器类型 | 说明 | 适用场景 |
|-----------|------|---------|
| `naive`/`general` | 通用解析器，自动识别PDF类型 | 默认PDF |
| `paper` | 论文解析，按章节分块 | 学术论文 |
| `book` | 书籍解析，按章节分块 | 书籍 |
| `table` | 表格解析 | 表格为主 |
| `one` | 整文档不切分 | 短文档 |
| `knowledge_graph`/`kg` | 知识图谱模式 | 图谱构建 |

### 2.2 PDF解析技术栈

DeepDoc的PDF解析流程：

```
PDF文件
  ├── pdfplumber 提取文本框和字符位置
  ├── pypdf 提取文本和元数据
  ├── LayoutRecognizer（布局检测）
  │     ├── ONNX Runtime（默认）
  │     └── AscendLayoutRecognizer（昇腾NPU）
  ├── OCR（光学字符识别）
  │     ├── TextDetector（文本检测）
  │     └── TextRecognizer（文本识别）
  └── TableStructureRecognizer（表格结构识别）
```

### 2.3 布局检测（LayoutRecognizer）

- **模型**：基于ONNX的布局检测模型
- **识别类别**：标题、正文、图片、表格、表格标题、图片标题、参考文献、页眉页脚等
- **流程**：
  1. 将PDF页面转为图片（zoomin=3倍放大）
  2. LayoutRecognizer检测布局区域
  3. 对每个区域进行OCR或直接提取文本

### 2.4 OCR模块

- **TextDetector**：检测文本区域边界框
- **TextRecognizer**：识别文本内容
- **支持模式**：CRNN、SRN、SAR、SVTR、ABINet、CAN等多种识别引擎
- **自动回退**：pdfplumber提取的文本如果乱码（font encoding garbled），自动回退到OCR

### 2.5 表格结构识别（TableStructureRecognizer）

- 识别表格的行列结构
- 支持HTML格式输出
- 自动检测表格方向（0°/90°/180°/270°旋转）
- 对旋转表格重新OCR

### 2.6 乱码检测与处理

RAGFlow有完善的乱码检测机制：
1. **字符频率检测**（`_is_garbled_char`）：检查非常用字符比例
2. **字体编码检测**（`_is_garbled_by_font_encoding`）：检查font subset前缀
3. **文本完整性检测**（`_is_garbled_text`）：检查乱码字符超过50%则判定为乱码页
4. **自动回退**：乱码页清除pdfplumber文本，改用OCR提取

## 三、任务处理流程（do_handle_task）

### 3.1 任务触发与消费

```
文件上传 → API Server → Redis Stream消息队列
                                    ↓
                    Task Executor消费（collect函数）
                                    ↓
                    do_handle_task(task) 处理
```

- **消息队列**：Redis Stream，Consumer Group模式
- **消费方式**：`REDIS_CONN.queue_consumer()`从队列获取任务
- **任务状态**：支持取消、超时、进度回调

### 3.2 do_handle_task主流程

```python
async def do_handle_task(task):
    1. 绑定嵌入模型（embedding model）
    2. 初始化知识库（init_kb）
    3. 根据task_type分支处理：
       - "memory" → 保存到记忆
       - "dataflow" → 数据流处理
       - "raptor" → RAPTOR树摘要
       - "graphrag" → 知识图谱增强
       - 默认 → build_chunks → 向量化 → 索引
```

### 3.3 build_chunks分块流程

```python
async def build_chunks(task, progress_callback):
    1. 从MinIO获取文件二进制数据
    2. 合并知识库级parser_config
    3. 调用chunker.chunk()进行分块：
       - 根据parser_id选择对应的解析器
       - 解析器执行布局检测+OCR+分块
    4. 生成chunk内容和元数据
    5. 可选：自动生成关键词（auto_keywords）
    6. 可选：自动生成问题（auto_questions）
    7. 可选：提取元数据（enable_metadata）
    8. 上传到MinIO并建立向量索引
```

### 3.4 paper_id与解析策略

当文件类型为PDF时，`parser_id`决定解析策略：

| parser_id | 解析器 | 分块策略 |
|-----------|--------|---------|
| `naive`/`general` | RAGFlowPdfParser | 自动布局检测+OCR，按chunk_token_num切分 |
| `paper` | PaperParser | 按论文结构（摘要、引言、方法、实验）分块 |
| `book` | BookParser | 按章节分块 |
| `one` | PlainParser | 整文档不切分 |
| `table` | TableParser | 表格专用解析 |
| `knowledge_graph`/`kg` | naive | 同通用解析，用于知识图谱构建 |

## 四、关键技术实现方案

### 4.1 布局检测+OCR混合解析

RAGFlow的核心创新在于**布局感知的混合解析**：
1. 先用LayoutRecognizer识别页面布局
2. 文本区域直接提取（pdfplumber）
3. 图片区域OCR识别
4. 表格区域用TableStructureRecognizer解析结构
5. 乱码区域自动回退到OCR

### 4.2 分块策略

- `chunk_token_num`：每块最大token数（默认128）
- `overlapped_percent`：重叠比例
- `delimiter`：分隔符（默认`\n!?。；！？`）
- 支持按布局边界智能分块

### 4.3 向量化

- 支持多种嵌入模型（通过TEI服务或内置）
- 默认使用Qwen3-Embedding-0.6B
- 批量编码，支持GPU加速

## 五、部署架构

### Docker部署组成

| 服务 | 镜像 | 端口 | 说明 |
|------|------|------|------|
| ragflow | infiniflow/ragflow:v0.25.6 | 80/9380 | 主服务 |
| elasticsearch | elasticsearch:8.11.3 | 1200 | 向量检索 |
| mysql | mysql:8.0.39 | 3306 | 元数据 |
| redis | valkey/valkey:8 | 6379 | 消息队列+缓存 |
| minio | pgsty/minio | 9000/9001 | 文件存储 |
| tei | text-embeddings-inference | 6380 | 嵌入模型服务 |

### GPU支持

docker-compose-gpu配置中通过`deploy.resources.reservations.devices`启用NVIDIA GPU。

## 六、针对IMDR数据集的优化建议

### 问题分析

IMDR数据集特点：
1. **图片型PDF** — 文字是图片，无法直接提取
2. **低分辨率** — OCR识别难度大
3. **图文混合** — 需要同时理解文本和技术图纸
4. **部件位置关系** — 需要空间推理

### 优化方向

1. **解析方法选择**：使用`naive`解析器（自动布局检测+OCR）
2. **分块策略调整**：
   - 增大`chunk_token_num`（如256-512），确保图文关联
   - 设置合理的`overlapped_percent`（如20%）
3. **OCR精度提升**：
   - 确保GPU可用，加速OCR推理
   - 调整zoomin参数（放大倍数）
4. **ReRank模型**：配置Rerank模型提升检索精度
5. **视觉理解**：对于图片问题，可能需要多模态模型（Vision LLM）辅助理解
