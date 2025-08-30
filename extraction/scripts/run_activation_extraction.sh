#!/bin/bash

# 运行激活提取模块
# 从LLM指定层提取activations，对每个token用SAE编码，然后对每个句子的SAE特征做平均
export CUDA_VISIBLE_DEVICES=0
echo "🧠 开始运行激活提取模块..."

python 2_activation_extraction.py \
    --dataset_path data/labeled_dataset_2_test \
    --model_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --sae_path "andreuka18/deepseek-r1-distill-llama-8b-lmsys-openthoughts" \
    --hook_point "blocks.19.hook_resid_post" \
    --output_path data/sae_features \
    --sae_id "blocks.19.hook_resid_post" \
    --text_column "text" \
    --batch_size 2 \
    --max_length 32768 \
    --device "auto"

echo "✅ 激活提取完成！SAE特征保存在: data/sae_features"
echo "📊 查看配置信息: cat data/sae_features/extraction_config.json"
