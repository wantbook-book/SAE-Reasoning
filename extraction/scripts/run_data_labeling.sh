#!/bin/bash

# 运行数据标注模块
# 从OpenThoughts和LMSYS数据集创建reasoning和non-reasoning混合数据集

echo "🏷️ 开始运行数据标注模块..."

python 1_data_labeling.py \
    --output_path data/labeled_dataset \
    --total_samples 10000 \
    --reasoning_ratio 0.5 \
    --reasoning_dataset "open-thoughts/OpenThoughts-114k" \
    --non_reasoning_dataset "lmsys/lmsys-chat-1m" \
    --reasoning_text_column "text" \
    --non_reasoning_text_column "text" \
    --min_text_length 1 \
    --max_text_length 200000000000 \
    --tokenizer_path "/pubshare/LLM/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

echo "✅ 数据标注完成！数据集保存在: data/labeled_dataset"
echo "📊 查看统计信息: cat data/labeled_dataset/dataset_stats.json"
