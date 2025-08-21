#!/bin/bash
set -ex

save_dir="eval_results/"
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
sae_release="andreuka18/deepseek-r1-distill-llama-8b-lmsys-openthoughts"
sae_id="blocks.19.hook_resid_post"
intervention_configs=(
    '{"feature_idx": 1160, "max_activation": 5.395, "strength": 1.0}'
    '{"feature_idx": 3466, "max_activation": 1.63, "strength": 1.0}'
    '{"feature_idx": 14907, "max_activation": 6.228, "strength": 1.0}'
    '{"feature_idx": 15796, "max_activation": 5.084, "strength": 1.0}'
    '{"feature_idx": 18202, "max_activation": 6.036, "strength": 1.0}'
    '{"feature_idx": 20201, "max_activation": 7.796, "strength": 1.0}'
    '{"feature_idx": 24648, "max_activation": 14.844, "strength": 1.0}'
    '{"feature_idx": 33362, "max_activation": 5.684, "strength": 1.0}'
    '{"feature_idx": 44815, "max_activation": 13.059, "strength": 1.0}'
    '{"feature_idx": 48684, "max_activation": 1.96, "strength": 1.0}'
    '{"feature_idx": 49883, "max_activation": 6.022, "strength": 1.0}'
    '{"feature_idx": 50219, "max_activation": 2.674, "strength": 1.0}'
    '{"feature_idx": 50876, "max_activation": 3.455, "strength": 1.0}'
    '{"feature_idx": 54529, "max_activation": 15.304, "strength": 1.0}'
    '{"feature_idx": 54788, "max_activation": 3.836, "strength": 1.0}'
    '{"feature_idx": 60248, "max_activation": 3.292, "strength": 1.0}'
    '{"feature_idx": 3942, "max_activation": 5.0, "strength": 1.0}'
    '{"feature_idx": 4395, "max_activation": 5.0, "strength": 1.0}'
    '{"feature_idx": 16441, "max_activation": 5.0, "strength": 1.0}'
    '{"feature_idx": 16778, "max_activation": 5.0, "strength": 1.0}'
    '{"feature_idx": 25953, "max_activation": 5.0, "strength": 1.0}'
    '{"feature_idx": 46691, "max_activation": 5.0, "strength": 1.0}'
    '{"feature_idx": 61104, "max_activation": 5.0, "strength": 1.0}'
)
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
        
        python vllm_sae_generate.py \
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
            $chat_template_args
    done
done
