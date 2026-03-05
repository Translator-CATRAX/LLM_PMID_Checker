#!/bin/bash

# Batch evaluation script for OpenAI models
# Processes all rows from test_data.tsv using OpenAI backend (gpt-5-nano, etc.)
#
# Prerequisites: OPENAI_API_KEY must be set in .env or environment

# Set base directory
BASE_DIR="/home/grads/cqm5886/work/LLM_PMID_Checker"
INPUT_FILE="$BASE_DIR/evaluation/test_data.tsv"
OUTPUT_FILE="$BASE_DIR/evaluation/gpt_5_nano_evaluation_results.tsv"
VAL_MODEL="gpt-5-nano"
ROUND2_MODEL=""  # Optional: Set to enable Round 2 re-evaluation (e.g., "gpt-5-mini")
NODE_DICT="$BASE_DIR/data/kg2_data/kg2c-2.10.2-v1.0-nodes.jsonl.gz"  # Optional: Set to KG2 nodes file for entity context
MAX_CONCURRENT=24

echo "========================================"
echo "Running evaluation with OpenAI model"
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

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ] && ! grep -q "OPENAI_API_KEY" .env 2>/dev/null; then
    echo "Error: OPENAI_API_KEY not found in environment or .env file!"
    echo "Please set your OpenAI API key before running this script."
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
