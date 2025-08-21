#!/bin/bash
set -ex

# Start time
start_time=$(date +%s)
export VLLM_USE_V1=0
export TOKENIZERS_PARALLELISM=false
PROMPT_TYPE="pure"
NUM_TEST_SAMPLE=-1
SPLIT="test"
DATA_DIR="dataset/"
gpu_util=0.9

# ================need to modify=======================
# List of model paths
MODEL_PATH_LIST=(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)
export CUDA_VISIBLE_DEVICES=2
DATA_NAME="math_500,aime24"

N_SAMPLING=4
top_p=0.95
TEMPRATURE=0.6
MAX_TOKENS=32768
PROMPT_FILE=prompts/math_cot.txt
CHAT_TEMPLATE_ARG="--apply_chat_template"

sae_release="andreuka18/deepseek-r1-distill-llama-8b-lmsys-openthoughts"
sae_id="blocks.19.hook_resid_post"
intervention_configs=(
    '{"feature_idx": 24648, "max_activation": 14.844, "strength": 1.0}'
    '{"feature_idx": 33362, "max_activation": 5.684, "strength": 1.0}'
    '{"feature_idx": 44815, "max_activation": 13.059, "strength": 1.0}'
)
# ================need to modify=======================

for MODEL_NAME_OR_PATH in "${MODEL_PATH_LIST[@]}"; do
    OUTPUT_DIR="${MODEL_NAME_OR_PATH}/math_eval_sampling_${N_SAMPLING}"
    
    for intervention_config in "${intervention_configs[@]}"; do
        python3 -u generate_math.py \
            --model_name_or_path "${MODEL_NAME_OR_PATH}" \
            --data_name "${DATA_NAME}" \
            --output_dir "${OUTPUT_DIR}" \
            --split "${SPLIT}" \
            --prompt_type "${PROMPT_TYPE}" \
            --num_test_sample "${NUM_TEST_SAMPLE}" \
            --seed 0 \
            --temperature ${TEMPRATURE} \
            --n_sampling ${N_SAMPLING} \
            --top_p ${top_p} \
            --start 0 \
            --end -1 \
            --use_vllm \
            --save_outputs \
            --max_tokens_per_call ${MAX_TOKENS} \
            --overwrite \
            --data_dir "${DATA_DIR}" \
            --prompt_file ${PROMPT_FILE} \
            --sae_release $sae_release \
            --sae_id $sae_id \
            --intervention_config "$intervention_config" \
            --gpu_util ${gpu_util} \
            ${CHAT_TEMPLATE_ARG}
    done
done

# End time
end_time=$(date +%s)

# Calculate and print the execution time
execution_time=$((end_time - start_time))
echo "Total execution time: $execution_time seconds"
