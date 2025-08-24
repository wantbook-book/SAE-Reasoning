#!/bin/bash
set -ex

save_dir="outputs/"
gpu_util=0.9
ntrain=0
export VLLM_USE_V1=0
export TOKENIZERS_PARALLELISM=false
# ================need to modify=======================
export CUDA_VISIBLE_DEVICES=0
temperature=0.6
top_p=0.95
max_new_tokens=32768
models=(
    "/angel/fwk/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)
chat_template_args="--apply_chat_template"
n_sampling=4
sae_release="andreuka18/deepseek-r1-distill-llama-8b-lmsys-openthoughts"
sae_id="blocks.19.hook_resid_post"
intervention_configs=(
    # '{"feature_idx": 1160, "max_activation": 5.395, "strength": 1.0}'
    # '{"feature_idx": 3466, "max_activation": 1.63, "strength": 1.0}'
    # '{"feature_idx": 14907, "max_activation": 6.228, "strength": 1.0}'
    '{"15796": {"max_activation": 5.084, "strength": 1.0}, "61104": {"max_activation": 5.0, "strength": 1.0}}'
    '{"15796": {"max_activation": 5.084, "strength": 1.0}, "1160": {"max_activation": 5.395, "strength": 1.0}}'
    
)
intervention_type=intervention
# ================need to modify=======================

for model in "${models[@]}"; do
    if [ ! -d "$model" ]; then
        echo "Model path $model does not exist"
        exit 1
    fi
    echo "Evaluating model: $model"
    
    for intervention_config in "${intervention_configs[@]}"; do
        echo "Using intervention config: $intervention_config"
        
        # Create a unique save directory for each config
        # config_hash=$(echo "$intervention_config" | md5sum | cut -d' ' -f1 | cut -c1-8)
        # current_save_dir="${save_dir}/config_${config_hash}"
        # mkdir -p "$current_save_dir"
        
        python evaluate_gpqa.py \
            --ntrain "$ntrain" \
            --save_dir "$save_dir" \
            --model "$model" \
            --sae_release $sae_release \
            --sae_id $sae_id \
            --intervention_config "$intervention_config" \
            --gpu_util "$gpu_util" \
            --temperature "$temperature" \
            --top_p "$top_p" \
            --max_new_tokens "$max_new_tokens" \
            --n_sampling "$n_sampling" \
            --intervention_type $intervention_type \
            $chat_template_args
    done
done
