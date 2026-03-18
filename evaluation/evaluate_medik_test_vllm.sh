#!/bin/bash

# Batch evaluation script for vLLM-served models
#
# Prerequisites: Start the required vLLM server(s) first:
#   VLLM_MODEL=openai/gpt-oss-20b VLLM_GPU=0 VLLM_PORT=8001 bash setup_vllm.sh   # GPT-OSS 20B
#   VLLM_MODEL=openai/gpt-oss-120b VLLM_GPU=1 VLLM_PORT=8002 bash setup_vllm.sh  # GPT-OSS 120B
#
# Configurations (uncomment ONE block):
#
# --- Config A: GPT-OSS 20B only (single round) ---
# VAL_MODEL="gpt-oss-20b-vllm"
# ROUND2_MODEL=""
# OUTPUT_FILE="$BASE_DIR/evaluation/gptoss20b_vllm_results.db"
#
# --- Config B: GPT-OSS 120B only (single round, default) ---
# VAL_MODEL="gpt-oss-120b-vllm"
# ROUND2_MODEL=""
# OUTPUT_FILE="$BASE_DIR/evaluation/gptoss120b_vllm_results.db"
#
# --- Config C: GPT-OSS 20B (R1) + 120B (R2) two-round ---
# VAL_MODEL="gpt-oss-20b-vllm"
# ROUND2_MODEL="gpt-oss-120b-vllm"
# OUTPUT_FILE="$BASE_DIR/evaluation/gptoss_20b_r1_120b_r2_results.db"

# Set base directory
BASE_DIR="/home/grads/cqm5886/work/LLM_PMID_Checker"
INPUT_FILE="$BASE_DIR/data/test_data_biolink.tsv"
OUTPUT_FILE="${OUTPUT_FILE:-$BASE_DIR/evaluation/gptoss_120b_r1_evaluation_only_results.tsv}"
VAL_MODEL="${VAL_MODEL:-gpt-oss-120b-vllm}"
ROUND2_MODEL="${ROUND2_MODEL:}"
NODE_DICT="$BASE_DIR/data/kg2_data/kg2c-2.10.2-v1.0-nodes.jsonl.gz"
PREDICATE_FILE="$BASE_DIR/data/biolink_data/biolink_predicates.tsv"
MAX_CONCURRENT=30

echo "========================================"
echo "Running evaluation with vLLM model"
echo "========================================"
echo "Input:  $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo "Model:  $VAL_MODEL"
if [ -n "$ROUND2_MODEL" ]; then
    echo "Round 2 Model: $ROUND2_MODEL (enabled)"
else
    echo "Round 2 Model: disabled"
fi
if [ -n "$NODE_DICT" ]; then
    echo "Node Dict: $NODE_DICT"
else
    echo "Node Dict: disabled"
fi
if [ -n "$PREDICATE_FILE" ]; then
    echo "Predicate File: $PREDICATE_FILE"
else
    echo "Predicate File: disabled"
fi
echo "Max Concurrent: $MAX_CONCURRENT"
echo "========================================"
echo ""

cd "$BASE_DIR"

# Activate conda environment if available
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate llm_pmid_env 2>/dev/null || echo "Note: Could not activate llm_pmid_env, using current environment"
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE not found!"
    exit 1
fi

# Build command
cmd="python main.py --input \"$INPUT_FILE\" --output \"$OUTPUT_FILE\" --val_model \"$VAL_MODEL\" --max_concurrent $MAX_CONCURRENT"

# Add Round 2 model if specified
if [ -n "$ROUND2_MODEL" ]; then
    cmd="$cmd --round2_model \"$ROUND2_MODEL\""
fi

# Add node_dict if specified
if [ -n "$NODE_DICT" ]; then
    cmd="$cmd --node_dict \"$NODE_DICT\""
fi

# Add predicate file if specified
if [ -n "$PREDICATE_FILE" ]; then
    cmd="$cmd --predicate_file \"$PREDICATE_FILE\""
fi

# Run batch evaluation
echo "Starting batch evaluation (this will be much faster than sequential processing)..."
echo ""
eval $cmd

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Batch evaluation completed successfully!"
    echo "========================================"
    echo "Results saved to: $OUTPUT_FILE"

else
    echo ""
    echo "========================================"
    echo "Error: Batch evaluation failed!"
    echo "========================================"
    exit $exit_code
fi
