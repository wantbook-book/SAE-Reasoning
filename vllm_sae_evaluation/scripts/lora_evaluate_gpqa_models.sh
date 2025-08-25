#!/bin/bash
set -ex

save_dir="outputs/"
gpu_util=0.9
ntrain=0
export VLLM_USE_V1=0
export TOKENIZERS_PARALLELISM=false
# ================need to modify=======================
export CUDA_VISIBLE_DEVICES=4,5,6,7
temperature=0.6
top_p=0.95
max_new_tokens=32768
models=(
    "/pubshare/LLM/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)
LORA_PATH_LIST=(
    /pubshare/fwk/code/sae/SAE-Reasoning2/sft/ckpts/pubshare/LLM/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/andreuka18/deepseek-r1-distill-llama-8b-lmsys-openthoughts/pretrained_blocks.19.hook_resid_post/rlpr/61104_1.0_5.0/checkpoint-500
    /pubshare/fwk/code/sae/SAE-Reasoning2/sft/ckpts/pubshare/LLM/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/andreuka18/deepseek-r1-distill-llama-8b-lmsys-openthoughts/pretrained_blocks.19.hook_resid_post/rlpr/61104_1.0_5.0/checkpoint-1500
    /pubshare/fwk/code/sae/SAE-Reasoning2/sft/ckpts/pubshare/LLM/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/andreuka18/deepseek-r1-distill-llama-8b-lmsys-openthoughts/pretrained_blocks.19.hook_resid_post/rlpr/61104_1.0_5.0/checkpoint-3000
    /pubshare/fwk/code/sae/SAE-Reasoning2/sft/ckpts/pubshare/LLM/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/andreuka18/deepseek-r1-distill-llama-8b-lmsys-openthoughts/pretrained_blocks.19.hook_resid_post/rlpr/61104_1.0_5.0/checkpoint-6000
    /pubshare/fwk/code/sae/SAE-Reasoning2/sft/ckpts/pubshare/LLM/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/andreuka18/deepseek-r1-distill-llama-8b-lmsys-openthoughts/pretrained_blocks.19.hook_resid_post/rlpr/61104_1.0_5.0/checkpoint-9000
    /pubshare/fwk/code/sae/SAE-Reasoning2/sft/ckpts/pubshare/LLM/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/andreuka18/deepseek-r1-distill-llama-8b-lmsys-openthoughts/pretrained_blocks.19.hook_resid_post/rlpr/61104_1.0_5.0/checkpoint-15000
)

chat_template_args="--apply_chat_template"
n_sampling=4
# ================need to modify=======================

for model in "${models[@]}"; do
    if [ ! -d "$model" ]; then
        echo "Model path $model does not exist"
        exit 1
    fi
    for lora_path in "${LORA_PATH_LIST[@]}";do
        python evaluate_gpqa.py \
            --ntrain "$ntrain" \
            --save_dir "$save_dir" \
            --model "$model" \
            --gpu_util "$gpu_util" \
            --temperature "$temperature" \
            --top_p "$top_p" \
            --max_new_tokens "$max_new_tokens" \
            --n_sampling "$n_sampling" \
            --lora_path ${lora_path} \
            $chat_template_args
    done
done
