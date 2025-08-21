#!/bin/bash
set -ex

save_dir="outputs/"
gpu_util=0.8
ntrain=0
export VLLM_USE_V1=0
export TOKENIZERS_PARALLELISM=false
# ================need to modify=======================
export CUDA_VISIBLE_DEVICES=0,1,2,3
temperature=0.6
top_p=0.95
max_new_tokens=32768
models=(
    "/angel/fwk/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)
chat_template_args="--apply_chat_template"
n_sampling=4
# ================need to modify=======================

for model in "${models[@]}"; do
    if [ ! -d "$model" ]; then
        echo "Model path $model does not exist"
        exit 1
    fi
    echo "Evaluating model: $model"
    python evaluate_gpqa.py \
        --ntrain "$ntrain" \
        --save_dir "$save_dir" \
        --model "$model" \
        --gpu_util "$gpu_util" \
        --temperature "$temperature" \
        --top_p "$top_p" \
        --max_new_tokens "$max_new_tokens" \
        --n_sampling "$n_sampling" \
        $chat_template_args
done
