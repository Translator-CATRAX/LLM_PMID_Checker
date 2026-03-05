#!/bin/bash

# Batch evaluation script for vLLM-served models
# Processes all rows from test_data.tsv using vLLM backend
#
# Prerequisites: Start the required vLLM server(s) first:
#   bash setup_vllm.sh                                                      # Hermes 4 on GPU 3, port 8000
#   VLLM_MODEL=openai/gpt-oss-20b VLLM_GPU=0 VLLM_PORT=8001 bash setup_vllm.sh   # GPT-OSS 20B
#   VLLM_MODEL=openai/gpt-oss-120b VLLM_GPU=1 VLLM_PORT=8002 bash setup_vllm.sh  # GPT-OSS 120B
#
# Configurations (uncomment ONE block):
#
# --- Config A: Hermes 4 only (single round) ---
# VAL_MODEL="hermes4-vllm"
# ROUND2_MODEL=""
# OUTPUT_FILE="$BASE_DIR/evaluation/hermes4_vllm_evaluation_results.tsv"
#
# --- Config B: GPT-OSS 20B only (single round) ---
# VAL_MODEL="gpt-oss-20b-vllm"
# ROUND2_MODEL=""
# OUTPUT_FILE="$BASE_DIR/evaluation/gptoss20b_vllm_evaluation_results.tsv"
#
# --- Config C: GPT-OSS 20B (R1) + 120B (R2) two-round ---
# VAL_MODEL="gpt-oss-20b-vllm"
# ROUND2_MODEL="gpt-oss-120b-vllm"
# OUTPUT_FILE="$BASE_DIR/evaluation/gptoss_20b_r1_120b_r2_evaluation_results.tsv"

# Set base directory
BASE_DIR="/home/grads/cqm5886/work/LLM_PMID_Checker"
INPUT_FILE="$BASE_DIR/data/test_data.tsv"
OUTPUT_FILE="${OUTPUT_FILE:-$BASE_DIR/evaluation/gptoss_20b_r1_120b_r2_evaluation_results.tsv}"
VAL_MODEL="${VAL_MODEL:-gpt-oss-20b-vllm}"
ROUND2_MODEL="${ROUND2_MODEL:-gpt-oss-120b-vllm}"
NODE_DICT="$BASE_DIR/data/kg2_data/kg2c-2.10.2-v1.0-nodes.jsonl.gz"
MAX_CONCURRENT=24

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
cmd="python evaluation/evaluate_batch.py --input \"$INPUT_FILE\" --output \"$OUTPUT_FILE\" --val_model \"$VAL_MODEL\" --max_concurrent $MAX_CONCURRENT"

# Add Round 2 model if specified
if [ -n "$ROUND2_MODEL" ]; then
    cmd="$cmd --round2_model \"$ROUND2_MODEL\""
fi

# Add node_dict if specified
if [ -n "$NODE_DICT" ]; then
    cmd="$cmd --node_dict \"$NODE_DICT\""
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
    echo ""

    # Calculate metrics if the Python script exists
    METRICS_FILE="${OUTPUT_FILE%.tsv}_metrics.txt"
    if [ -f "$BASE_DIR/evaluation/calculate_metrics.py" ]; then
        echo "Calculating metrics..."
        python "$BASE_DIR/evaluation/calculate_metrics.py" "$OUTPUT_FILE" "$METRICS_FILE"
        echo ""
        echo "========================================"
        echo "Metrics Summary"
        echo "========================================"
        cat "$METRICS_FILE"
    fi

else
    echo ""
    echo "========================================"
    echo "Error: Batch evaluation failed!"
    echo "========================================"
    exit $exit_code
fi
