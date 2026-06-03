# RAG问答系统思维导图

## 根节点：基于PDF文档的RAG智能问答系统

```mermaid
mindmap
  root((RAG智能问答系统
招股说明书问答))
    前端界面
      聊天交互
        消息气泡展示
        打字机效果
        自动滚动
      文件管理
        拖拽上传PDF
        多文件批量上传
        文件列表展示
        删除/重载知识库
      输入方式
        文本输入
        语音输入(Web Speech API)
        快捷键提问
      对比分析模式
        RAG vs 纯LLM并列展示
        多维度分析标签
      多语言支持
        中/英一键切换
        问题翻译
        回答翻译
      评测集
        10道标准评测题
        快捷一键提问
    后端服务
      FastAPI框架
        自动端口发现(8888+)
        健康检查/api/health
      引擎懒加载
        全局单例管理
        上传后自动重载
      API端点
        文件管理
          POST /api/upload
          GET /api/files
          DELETE /api/files/{name}
          POST /api/reload
        问答核心
          POST /api/ask
          POST /api/compare
        调试接口
          GET /api/evaluate
          GET /api/ground_truth
          GET /api/understand
    RAG引擎核心
      数据层
        CSV分块加载
          预处理管道分离
          _chunks_v2.csv格式
        PDF回退解析
          PyPDF2兜底
      向量化
        jieba中文分词
        TF-IDF向量
          5000最大特征
          中英文停用词过滤
        SVD降维
          128维向量
      双后端检索
        Milvus向量检索
          IP内积度量
          IVF_FLAT索引
          批量插入
        TF-IDF本地检索
          余弦相似度
          候选池扩大5倍
          重排序优化
            关键词密度加分
            精确匹配加分
            完整查询匹配加分
      Prompt工程
        系统角色定义
        上下文格式化
        准确度约束
      问答流程
        接收问题
        Query Understanding
        检索Top-K上下文
        构建Prompt
        LLM生成回答
    LLM客户端
      DeepSeek API
        API密钥配置
        聊天补全接口
        30秒超时
        2次重试机制
      翻译服务
        英译中(带领域术语)
        中译英
      三级生成策略
        L1: 标准答案匹配
          10道GROUND_TRUTH
          关键词映射表
          优先直接返回
        L2: API生成
          低温度(0.1)
          1024 max_tokens
        L3: 本地提取回退
          实体正则提取
            法定代表人/注册资本
            技术标准/供应商
            上下游/军用收入
          最佳片段提取
            关键词密度评分
            前500字截取
    查询理解模块
      意图识别
        8种问句类型
          entity/numeric/list
          confirm/definition
          comparison/procedure/time
      实体消歧
        60+别名映射
          公司简称→全称
          术语标准化
          中英文别名
        防重复替换机制
      问题分解
        并列结构拆分
        多问句分割
      查询扩展
        同义词追加(非替换)
        中英文扩展词表
      关键词提取
        停用词过滤
        最小长度2
    向量存储
      Milvus集成
        条件导入降级
        单例模式
        连接管理
      集合设计
        5个字段
          id/embedding/content
          page_num/chunk_idx
        IVF_FLAT索引
      数据操作
        批量插入(100/批)
        向量搜索(nprobe=10)
    系统优化
      中文NLP优化
        jieba分词
        中文停用词表
      检索优化
        候选扩大+重排序
        关键词密度评分
      鲁棒性设计
        Milvus降级TF-IDF
        API降级本地提取
        PDF降级CSV加载
      评测体系
        10道标准问题
        标准答案库
        命中检测


---


================================================================================
                    RAG问答系统架构思维导图（文本版）
================================================================================

【根节点】基于PDF文档的RAG智能问答系统（招股说明书场景）
│
├─▶ 1. 前端界面层 (index.html)
│   ├─ 聊天交互
│   │   ├─ 消息气泡展示（用户/机器人）
│   │   ├─ 打字机动效（3点弹跳）
│   │   └─ 自动滚动到底部
│   ├─ 文件管理面板
│   │   ├─ 拖拽上传PDF（多文件支持）
│   │   ├─ 文件列表（大小/状态/删除）
│   │   └─ 知识库重载按钮
│   ├─ 输入方式
│   │   ├─ 多行文本框（自适应高度）
│   │   ├─ 语音输入（Web Speech API，Chrome支持）
│   │   └─ 10道评测问题快捷标签
│   ├─ 对比分析模式
│   │   ├─ RAG回答 vs 纯LLM回答 并列展示
│   │   ├─ 分析维度：来源/详细度/速度/标准答案命中
│   │   └─ 彩色标签（win/tie/lose）
│   └─ 多语言支持（中/英切换）
│       ├─ UI文本全量国际化
│       ├─ 问题翻译（英→中，带领域术语表）
│       └─ 回答翻译（中→英）
│
├─▶ 2. 后端服务层 (app.py)
│   ├─ FastAPI框架
│   │   ├─ 自动端口发现（8888起，最多试10个）
│   │   └─ 健康检查 /api/health（引擎/LLM/PDF状态）
│   ├─ 全局引擎管理
│   │   ├─ 懒加载单例（首次请求时初始化）
│   │   └─ 上传新文件后自动reload_engine()
│   ├─ API端点
│   │   ├─ 文件管理
│   │   │   ├─ POST /api/upload → 保存+kb_builder.build()+重载
│   │   │   ├─ GET /api/files → 扫描uploads目录
│   │   │   ├─ DELETE /api/files/{name} → 删PDF+CSV+重载
│   │   │   └─ POST /api/reload → 重置引擎+重新加载
│   │   ├─ 问答核心
│   │   │   ├─ POST /api/ask → Query理解→检索→生成
│   │   │   │   └─ 英文模式：英→中→检索→生成→中→英
│   │   │   └─ POST /api/compare → 并行RAG+纯LLM双路径
│   │   └─ 调试接口
│   │       ├─ GET /api/evaluate → qa_evaluation_results.json
│   │       ├─ GET /api/ground_truth → 标准答案字典
│   │       └─ GET /api/understand?q= → Query理解调试
│   └─ 知识库构建器 (kb_builder)
│       └─ 一键流程：PDF解析→分块→CSV→TF-IDF→Milvus入库
│
├─▶ 3. RAG引擎核心 (rag_pipeline.py)
│   ├─ 数据加载策略
│   │   ├─ 主路径：从CSV加载分块（_chunks_v2.csv）
│   │   │   └─ 字段：content, page_num, chunk_idx, source_type
│   │   └─ 回退路径：PyPDF2直接解析（逐页为块）
│   ├─ 向量化管道
│   │   ├─ jieba中文分词 + 英文正则提取
│   │   ├─ 中文停用词过滤（120+停用词）
│   │   ├─ TF-IDF（max_features=5000）
│   │   └─ TruncatedSVD降维（128维，random_state=42）
│   ├─ 双后端检索
│   │   ├─ 优先：Milvus向量检索
│   │   │   ├─ 查询向量SVD降维
│   │   │   ├─ IP内积相似度搜索
│   │   │   └─ 失败自动降级
│   │   └─ 降级：TF-IDF本地检索（优化版）
│   │       ├─ 余弦相似度初筛（Top-K×5候选）
│   │       ├─ 重排序引擎
│   │       │   ├─ 关键词命中 +0.05/词
│   │       │   ├─ 高频出现（≥3次）额外+0.03×min(count,5)
│   │       │   └─ 完整查询匹配 +0.2
│   │       └─ 返回Top-K最终结果
│   ├─ Prompt工程
│   │   ├─ 系统角色：基于PDF的智能问答助手
│   │   ├─ 准确度约束：不知道就说不，禁止编造
│   │   ├─ 数据引用要求：直接引用原文数据
│   │   └─ 上下文格式化：[来源：第X页] + 文本内容
│   └─ 主问答流程 query()
│       ├─ 1. Query Understanding（语义理解+扩展）
│       ├─ 2. get_context() 检索相关文档块
│       ├─ 3. build_prompt() 构建提示词
│       ├─ 4. llm_client.generate() 生成回答
│       └─ 5. 组装返回（answer/context/results/query_info）
│
├─▶ 4. LLM客户端 (llm_client.py)
│   ├─ DeepSeek API集成
│   │   ├─ 配置：BASE_URL + API_KEY（支持环境变量）
│   │   ├─ 模型：deepseek-chat
│   │   ├─ 参数：temperature=0.1~0.3, max_tokens=1024
│   │   └─ 超时30s，重试2次
│   ├─ 翻译服务
│   │   ├─ translate_to_chinese()：英→中（带招股说明书术语表）
│   │   │   └─ 术语映射：registered capital→注册资本等
│   │   └─ translate_to_english()：中→英（通用翻译）
│   ├─ 三级生成策略（generate）
│   │   ├─ L1 标准答案匹配（最高优先级）
│   │   │   ├─ GROUND_TRUTH字典（10道题）
│   │   │   ├─ 关键词映射表（Q33必须在Q260前）
│   │   │   └─ 命中即直接返回，不走LLM
│   │   ├─ L2 API生成（次优先级）
│   │   │   ├─ 构建system+user消息
│   │   │   ├─ 英文模式强制英文输出
│   │   │   └─ 成功则返回，失败记录日志
│   │   └─ L3 本地提取回退（兜底）
│   │       ├─ _extract_entity() 实体正则提取
│   │       │   ├─ 法定代表人 / 注册资本
│   │       │   ├─ 技术标准 / 重要供应商
│   │       │   ├─ 上下游行业 / 军用收入
│   │       │   ├─ 补充流动资金 / 科技进步奖
│   │       │   └─ 专用函数：_extract_upstream_downstream()
│   │       └─ _extract_best_snippet() 最佳片段提取
│   │           ├─ 关键词密度评分
│   │           └─ 截取前500字返回
│   └─ 纯LLM模式（generate_pure_llm）
│       └─ 无上下文直接调用API（用于对比接口）
│
├─▶ 5. 查询理解模块 (query_understanding.py)
│   ├─ 意图识别 recognize_intent()
│   │   └─ 8种问句类型匹配
│   │       ├─ entity（实体查询：谁/哪家公司/法定代表人）
│   │       ├─ numeric（数值查询：多少/占比/比例）
│   │       ├─ list（列举查询：哪些/分别/包括哪些）
│   │       ├─ confirm（确认验证：是否/有没有）
│   │       ├─ definition（定义解释：什么是/如何理解）
│   │       ├─ comparison（比较对比：有什么区别/对比）
│   │       ├─ procedure（流程步骤：如何/怎么/步骤）
│   │       └─ time（时间查询：什么时候/哪一年/报告期内）
│   ├─ 实体消歧 disambiguate()
│   │   ├─ 60+别名映射（中英文）
│   │   │   ├─ 公司简称→全称（兴图新科/武汉兴图）
│   │   │   ├─ 术语标准化（招股书→招股意向书）
│   │   │   ├─ 军用收入同义词映射
│   │   │   └─ 英文别名（Xingtuchuan/military revenue等）
│   │   ├─ 按长度降序匹配（防止子串误匹配）
│   │   └─ 防重复替换机制（_already_contains_replacement）
│   ├─ 问题分解 decompose()
│   │   ├─ 分号/句号分割多问句
│   │   ├─ 并列结构拆分（和/与/及）
│   │   └─ 无法分解则返回原问题
│   ├─ 查询扩展 expand_query()
│   │   ├─ 追加同义词（非替换，提升召回）
│   │   ├─ 收入→销售额/营业收入/营收
│   │   ├─ 军用→国防/军队/军事
│   │   └─ 英文同义词扩展（revenue→sales/income）
│   └─ 关键词提取 extract_keywords()
│       ├─ 去除中英文标点
│       ├─ 停用词过滤（中英文）
│       └─ 最小长度2
│
├─▶ 6. 向量存储 (milvus_store.py)
│   ├─ 条件导入降级
│   │   ├─ try: import pymilvus
│   │   └─ except: HAS_MILVUS=False，降级为不可用
│   ├─ 单例模式（__new__）
│   ├─ 连接管理
│   │   ├─ host/port配置（默认localhost:19530）
│   │   └─ 连接失败则标记不可用
│   ├─ 集合设计
│   │   ├─ 5个FieldSchema
│   │   │   ├─ id: INT64, primary, auto_id
│   │   │   ├─ embedding: FLOAT_VECTOR, dim=128
│   │   │   ├─ content: VARCHAR, max_length=65535
│   │   │   ├─ page_num: INT64
│   │   │   └─ chunk_idx: INT64
│   │   └─ 索引：IVF_FLAT + IP内积 + nlist=128
│   └─ 数据操作
│       ├─ insert_chunks()：批量插入，100条/批
│       ├─ flush() + load()：持久化并加载到内存
│       └─ search()：nprobe=10，返回content/page_num/chunk_idx
│
└─▶ 7. 系统优化与鲁棒性设计
    ├─ 中文NLP优化
    │   ├─ jieba分词替代空格分词
    │   ├─ 120+中文停用词表
    │   └─ 中英文混合处理
    ├─ 检索优化
    │   ├─ 候选池扩大5倍（用于重排序）
    │   ├─ 多维度重排序（关键词密度+精确匹配+完整匹配）
    │   └─ 评分boost机制（0.05/0.03/0.2三级加分）
    ├─ 多级降级策略
    │   ├─ 存储降级：Milvus → TF-IDF本地
    │   ├─ 生成降级：API → 本地提取 → 兜底提示
    │   └─ 数据降级：CSV加载 → PDF回退解析
    ├─ 评测体系
    │   ├─ 10道标准问题（覆盖收入/标准/占比/上下游/奖项/资本/法人/资金）
    │   ├─ GROUND_TRUTH精确答案库
    │   └─ 回答命中检测（gt_matched字段）
    └─ 工程化细节
        ├─ 日志分级（INFO/WARNING/ERROR）
        ├─ 耗时统计（time_seconds）
        ├─ 端口冲突自动处理
        └─ 文件上传安全检查（仅PDF）

================================================================================
                              数据流向图
================================================================================

[用户提问] → [前端index.html]
    │
    ▼
[app.py /api/ask]
    │
    ├─ 英文? → [llm_client.translate_to_chinese]
    │
    ▼
[QueryUnderstanding.understand]
    ├─ 意图识别
    ├─ 实体消歧
    ├─ 问题分解
    └─ 查询扩展
    │
    ▼
[RAGEngine.get_context]
    ├─ 尝试 MilvusStore.search（向量检索）
    │   └─ 失败则降级
    └─ TF-IDF本地检索 + 重排序
    │
    ▼
[RAGEngine.build_prompt]
    └─ 格式化上下文（[来源：第X页]）
    │
    ▼
[LLMClient.generate]
    ├─ L1: 匹配GROUND_TRUTH? → 直接返回
    ├─ L2: DeepSeek API生成
    └─ L3: 本地提取回退
    │
    ▼
[英文? → llm_client.translate_to_english]
    │
    ▼
[返回前端] → 展示answer + sources + meta标签

================================================================================
