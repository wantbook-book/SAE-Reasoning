#!/bin/bash

# Start time
start_time=$(date +%s)

# ================need to modify=======================
OUTPUT_DIRS=(
    "/angel/fwk/code/SAE-Reasoning/vllm_sae_evaluation/outputs/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/math_eval_sampling_4"
)
# SUBDIRS=("math_500" "math_hard" "asdiv" "college_math" "tabmwp")
SUBDIRS=("math_500" "aime24")
# SUBDIRS=("aime24")
# FILE_NAME="test_pure_-1_seed0_t0.0_s0_e-1.jsonl"
# FILE_NAME="test_pure_-1_seed0_t0.6_s0_e-1.jsonl"
FILE_NAME="test_pure_-1_seed175569182_t0.6_pf_math_cot_s0_e-1.jsonl"
# ================need to modify=======================

EVAL_SCRIPT="evaluate_math_outputs.py"
for OUTPUT_DIR in "${OUTPUT_DIRS[@]}"; do
    # Check if the root directory exists
    if [ ! -d "$OUTPUT_DIR" ]; then
        echo "Skipped $OUTPUT_DIR: Directory does not exist."
        continue
    fi
    # Iterate through subdirectories
    for SUBDIR in "${SUBDIRS[@]}"; do
        # Check if the subdirectory exists
        SUBDIR="$OUTPUT_DIR/$SUBDIR"
        if [ ! -d "$SUBDIR" ]; then
            echo "Skipped $SUBDIR: Directory does not exist."
            continue
        fi

        INPUT_JSONL="$SUBDIR/$FILE_NAME"
        if [ -f "$INPUT_JSONL" ]; then
            OUTPUT_JSONL="${INPUT_JSONL%.jsonl}_output.jsonl"
            echo "Evaluating: $INPUT_JSONL"
            python "$EVAL_SCRIPT" --input_jsonl "$INPUT_JSONL" --output_jsonl "$OUTPUT_JSONL" --gold_is_latex
        else
            echo "Skipped $SUBDIR: test.jsonl not found."
        fi
    done
done

# End time
end_time=$(date +%s)

# Calculate and print the execution time
execution_time=$((end_time - start_time))
echo "Execution time: $execution_time seconds"