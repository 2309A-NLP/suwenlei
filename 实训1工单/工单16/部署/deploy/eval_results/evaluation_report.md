# Qwen2-VL-2B 微调前后评估报告

评估样本数: 201

## 1. 整体指标对比

| 指标 | 基线模型 | 微调模型 | 提升 |
|------|---------|---------|------|
| 整体准确率 | 0.00% | 待评估 | - |
| BLEU均值 | 0.00% | 待评估 | - |
| ROUGE-L均值 | 0.00% | 待评估 | - |
| 工业术语准确率 | 0.00% | 待评估 | - |
| 图纸推理准确率 | 0.00% | 待评估 | - |

## 2. 分组准确率对比

| 分组 | 描述 | 基线模型 | 微调模型 | 提升 |
|------|------|---------|---------|------|
| Group 1 | 文本理解 | 0.00% | 待评估 | - |
| Group 2 | 图纸推理 | 0.00% | 待评估 | - |
| Group 3 | 复杂图纸推理 | 0.00% | 待评估 | - |

## 3. 失败案例分析

### 案例1
- 问题: 根据专利文本，现有技术中热能回收交换机采用分体式安装的主要缺点是什么？
- 预测: ERROR: 'NoneType' object has no attribute 'choices' | 正确: C
- 分组: Group 1 | 文档: CN202659443U.pdf

### 案例2
- 问题: 根据专利的声明信息，在聚丁二酸丁二醇酯制造方法中，以下关于缩聚反应器的描述哪个是正确的？
- 预测: ERROR: 'NoneType' object has no attribute 'choices' | 正确: B
- 分组: Group 1 | 文档: CN102597054A.pdf

### 案例3
- 问题: 在文件中第9页的电路图2中，如果变频器(10)停止工作，并且能耗制动回路(30)正常工作，那么接下来会发生什么？
- 预测: ERROR: 'NoneType' object has no attribute 'choices' | 正确: B
- 分组: Group 3 | 文档: CN108452960B.pdf

### 案例4
- 问题: 根据专利文本，以下哪个部件负责感应焊缝的具体位置？
- 预测: ERROR: 'NoneType' object has no attribute 'choices' | 正确: C
- 分组: Group 1 | 文档: CN216066104U.pdf

### 案例5
- 问题: 根据专利文本，以下关于保温砖本体的描述哪一项是正确的？
- 预测: ERROR: 'NoneType' object has no attribute 'choices' | 正确: C
- 分组: Group 1 | 文档: CN211597342U.pdf

### 案例6
- 问题: 在文件中第7页的示意图中，部件9相对于部件7的位置关系是什么？
- 预测: ERROR: 'NoneType' object has no attribute 'choices' | 正确: C
- 分组: Group 2 | 文档: CN213657597U.pdf

### 案例7
- 问题: 根据文件中第9页的图片和文本信息，如果第四连接板(39)发生移动，与其直接相连并同步移动的部件有哪些？
- 预测: ERROR: 'NoneType' object has no attribute 'choices' | 正确: B
- 分组: Group 3 | 文档: CN112696060A.pdf

### 案例8
- 问题: 根据文件中第5页的示意图，一次除尘器（10）之后连接的设备是什么？
- 预测: ERROR: 'NoneType' object has no attribute 'choices' | 正确: A
- 分组: Group 3 | 文档: CN202390368U.pdf

### 案例9
- 问题: 如果需要更换文件中第4页图示中的孔盖5，在拆卸所有连接部件后，孔盖5是通过哪个部件实现旋转开启的？
- 预测: ERROR: 'NoneType' object has no attribute 'choices' | 正确: D
- 分组: Group 3 | 文档: CN201712993U.pdf

### 案例10
- 问题: 在文件中第8页的示意图中，压缩空气如果从编号3的部件流入，经过编号6部件后，下一步会流入哪个部件？
- 预测: ERROR: 'NoneType' object has no attribute 'choices' | 正确: A
- 分组: Group 3 | 文档: CN2932258Y.pdf


## 4. 改进建议

1. **数据增强**: 增加图纸推理题的训练样本比例
2. **超参数调优**: 尝试更高LoRA rank（16/32）提升模型容量
3. **数据清洗**: 剔除答案有歧义的样本
4. **图像预处理**: 尝试更高分辨率的页面截图
5. **多轮训练**: 增加epoch数观察是否欠拟合