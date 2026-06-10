#!/bin/bash
# autoDL 一键部署运行脚本
# 用法: cd /root/RAG工单16 && bash run_all.sh

set -e
cd /root/RAG工单16

echo "=============================="
echo "  RAG工单16 VLM微调全流程"
echo "=============================="

# 1. 安装依赖
echo ""
echo "=== 步骤1: 安装依赖 ==="
pip install qwen-vl-utils peft -q
cd /root/LLaMA-Factory
pip install -e ".[torch,metrics,bitsandbytes]" -q 2>/dev/null
cd /root/RAG工单16

# 2. 注册数据集
echo ""
echo "=== 步骤2: 注册数据集 ==="
cp imdr_vlm.json /root/LLaMA-Factory/data/
python3 -c "
import json
path = '/root/LLaMA-Factory/data/dataset_info.json'
with open(path) as f: data = json.load(f)
data['imdr_vlm'] = {
    'file_name': 'imdr_vlm.json',
    'formatting': 'sharegpt',
    'columns': {'messages': 'messages', 'images': 'images'},
    'tags': {'role_tag': 'role', 'content_tag': 'content', 'user_tag': 'user', 'assistant_tag': 'assistant'}
}
with open(path, 'w') as f: json.dump(data, f, ensure_ascii=False, indent=2)
print('数据集注册完成')
"

# 3. 下载模型（如果没有）
echo ""
echo "=== 步骤3: 检查模型 ==="
if [ ! -d "/root/autodl-tmp/models/Qwen/Qwen2-VL-2B-Instruct" ]; then
    echo "下载Qwen2-VL-2B-Instruct..."
    python3 -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2-VL-2B-Instruct', cache_dir='/root/autodl-tmp/models')
print('模型下载完成')
"
else
    echo "模型已存在，跳过下载"
fi

# 4. 基线评估
echo ""
echo "=== 步骤4: 基线评估 ==="
python3 eval_baseline.py

# 5. LoRA微调
echo ""
echo "=== 步骤5: LoRA微调 ==="
cd /root/LLaMA-Factory
llamafactory-cli train /root/RAG工单16/qwen2vl_lora_sft.yaml
cd /root/RAG工单16

# 6. 微调后评估
echo ""
echo "=== 步骤6: 微调后评估 ==="
python3 eval_finetuned.py

# 7. 生成报告
echo ""
echo "=== 步骤7: 生成报告 ==="
python3 generate_report.py

echo ""
echo "=============================="
echo "  全部完成！"
echo "  报告: /root/RAG工单16/eval_results/evaluation_report.md"
echo "=============================="
