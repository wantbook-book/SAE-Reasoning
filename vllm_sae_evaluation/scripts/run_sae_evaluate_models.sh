#!/bin/bash

# Example script for running GPQA evaluation with SAE integration
# This script demonstrates different ways to use SAE parameters
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=0
ntrain=0
temperature=0.6
top_p=0.95
max_new_tokens=32768
chat_template_args="--apply_chat_template"
model="/angel/fwk/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
sae_release="andreuka18/deepseek-r1-distill-llama-8b-lmsys-openthoughts"
sae_id="blocks.19.hook_resid_post"
intervention_config='{"feature_idx": 100, "max_activation": 5.0, "strength": 1.0}'

echo "Running GPQA evaluation with SAE integration examples..."

# # Example 1: Using local SAE path with single feature intervention
# echo "\n=== Example 1: Local SAE with single feature intervention ==="
# python vllm_sae_generate.py \
#     --model "meta-llama/Llama-2-7b-hf" \
#     --sae_path "/path/to/local/sae" \
#     --intervention_config '{"feature_idx": 100, "max_activation": 5.0, "strength": 1.0}' \
#     --hook_layer 8 \
#     --save_dir "results/sae_local_single" \
#     --ntrain 3 \
#     --temperature 0.6

# Example 2: Using HuggingFace SAE release with multi-feature intervention
echo "\n=== Example 2: HuggingFace SAE with multi-feature intervention ==="
python vllm_sae_generate.py \
    --model $model \
    --sae_release $sae_release \
    --sae_id $sae_id \
    --intervention_config "$intervention_config" \
    --save_dir "results/sae_hf_multi" \
    --ntrain $ntrain \
    --temperature $temperature \
    --top_p $top_p \
    --max_new_tokens $max_new_tokens \
    $chat_template_args

# # Example 3: Using intervention config file
# echo "\n=== Example 3: Using intervention config file ==="
# # First create a config file
# cat > intervention_config.json << EOF
# {
#     "50": {"max_activation": 4.0, "strength": 1.2},
#     "150": {"max_activation": 6.0, "strength": 0.9},
#     "250": {"max_activation": 3.5, "strength": 1.1}
# }
# EOF

# python vllm_sae_generate.py \
#     --model "meta-llama/Llama-2-7b-hf" \
#     --sae_path "/path/to/local/sae" \
#     --intervention_config "intervention_config.json" \
#     --hook_layer 10 \
#     --hook_point "hook_resid_post" \
#     --save_dir "results/sae_config_file" \
#     --ntrain 3 \
#     --temperature 0.6

# # Example 4: Baseline run without SAE (for comparison)
# echo "\n=== Example 4: Baseline without SAE ==="
# python vllm_sae_generate.py \
#     --model "meta-llama/Llama-2-7b-hf" \
#     --save_dir "results/baseline" \
#     --ntrain 3 \
#     --temperature 0.6

# echo "\nAll examples completed. Check the results/ directory for outputs."
# echo "Compare the accuracy between SAE-enabled and baseline runs."

# # Clean up
# rm -f intervention_config.json