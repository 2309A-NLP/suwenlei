#!/bin/bash
# autoDL 一键部署脚本
# 用法: bash autodl_setup.sh

set -e

echo "=== 1. 安装依赖 ==="
pip install modelscope transformers accelerate bitsandbytes peft qwen-vl-utils pillow torch torchvision -q

echo "=== 2. 下载Qwen2-VL-2B-Instruct模型 ==="
python -c "
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2-VL-2B-Instruct', cache_dir='/root/autodl-tmp/models')
print(f'模型下载完成: {model_dir}')
"

echo "=== 3. 安装LLaMA-Factory ==="
cd /root
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git 2>/dev/null || true
cd LLaMA-Factory
pip install -e ".[torch,metrics,bitsandbytes]" -q

echo "=== 4. 注册数据集 ==="
cd /root/RAG工单16

# 复制数据到LLaMA-Factory
cp imdr_vlm.json /root/LLaMA-Factory/data/

# 更新dataset_info.json
python -c "
import json
path = '/root/LLaMA-Factory/data/dataset_info.json'
with open(path) as f:
    data = json.load(f)
data['imdr_vlm'] = {
    'file_name': 'imdr_vlm.json',
    'formatting': 'sharegpt',
    'columns': {'messages': 'messages', 'images': 'images'},
    'tags': {'role_tag': 'role', 'content_tag': 'content', 'user_tag': 'user', 'assistant_tag': 'assistant'}
}
with open(path, 'w') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print('数据集注册完成')
"

echo "=== 5. 运行LoRA微调 ==="
cd /root/LLaMA-Factory
llamafactory-cli train /root/RAG工单16/qwen2vl_lora_sft.yaml

echo "=== 6. 运行微调后评估 ==="
cd /root/RAG工单16
python eval_finetuned.py

echo "=== 7. 生成报告 ==="
python generate_report.py

echo "=== 全部完成 ==="
