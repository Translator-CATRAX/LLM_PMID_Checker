#!/bin/bash

# Test version of evaluation script - processes all rows from test_data.tsv
# This script evaluates triples using CURIEs and measures runtime for each query
# 

# Note: Uses Ollama server which should be running on GPU:3 via setup_ollama.sh
# GPU is configured in setup_ollama.sh with CUDA_VISIBLE_DEVICES=3

# Set base directory
BASE_DIR="/home/grads/cqm5886/work/llm_pmid_support"
INPUT_FILE="$BASE_DIR/evaluation/test_data.tsv"
OUTPUT_FILE="$BASE_DIR/evaluation/hermes4_70b_iq4_xs_evaluation_results_v2.tsv"
METRICS_FILE="$BASE_DIR/evaluation/hermes4_70b_iq4_xs_evaluation_metrics_v2.txt"
VAL_MODEL="hermes4:70b-iq4-xs"
CHECKER_MODEL=""  # Optional: Set to enable verification
MAX_CONCURRENT=5  # Number of concurrent evaluations (adjust based on your system)

echo "========================================"
echo "Running TEST evaluation with Ollama models"
echo "========================================"
echo "Input:  $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo "Model:  $VAL_MODEL"
if [ -n "$CHECKER_MODEL" ]; then
    echo "Checker Model: $CHECKER_MODEL (enabled)"
else
    echo "Checker Model: disabled"
fi
echo "Max Concurrent: $MAX_CONCURRENT"
echo "========================================"
echo ""

# Change to base directory
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

# Add checker model if specified
if [ -n "$CHECKER_MODEL" ]; then
    cmd="$cmd --checker_model \"$CHECKER_MODEL\""
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
