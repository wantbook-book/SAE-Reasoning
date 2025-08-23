#!/bin/bash

export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # Set the GPUs you want to use
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")

echo ""
echo "Number of GPUs: ${GPU_COUNT}"
echo ""

PY_SCRIPT="/pubshare/fwk/code/sae/SAE-Reasoning2/sft/sae_guided_sft.py"
PY_CONFIG="/pubshare/fwk/code/sae/SAE-Reasoning2/configs/sae_guided_sft.yaml"
ACCELERATE_DS_CONFIG="/pubshare/fwk/code/sae/SAE-Reasoning2/configs/accelerate_ds_zero2.yaml"

ACCELERATE_LOG_LEVEL=info accelerate launch \
        --config_file "${ACCELERATE_DS_CONFIG}" \
        --main_process_port=29501 \
        --num_processes="${GPU_COUNT}" \
            "${PY_SCRIPT}" --config "${PY_CONFIG}"

echo "END TIME: $(date)"
echo "DONE"
