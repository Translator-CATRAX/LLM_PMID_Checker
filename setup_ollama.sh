#!/bin/bash

# Ollama setup script for LLM PMID Checker

set -e

# Configuration
PRIMARY_PORT=11434
PRIMARY_GPU=3
PROJECT_DIR=~/work/llm_pmid_support
OLLAMA_DIR=~/.ollama_pmid
LOG_DIR="$PROJECT_DIR/logs"

# Batch processing configuration (passed to ollama serve command)
OLLAMA_NUM_PARALLEL=4          # Max parallel requests per model (default: 1-4)
OLLAMA_MAX_LOADED_MODELS=1     # How many models can be loaded at once
OLLAMA_MAX_QUEUE=512           # Max requests in queue (default: 512)
OLLAMA_KEEP_ALIVE=5m           # Keep model loaded for 5 minutes to avoid reload costs

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi is not available. Make sure NVIDIA drivers are installed."
    echo "   Continuing with CPU-only setup..."
    USE_GPU=false
else
    USE_GPU=true
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "Found $GPU_COUNT GPU(s) available"
fi


# Create directories
mkdir -p "$OLLAMA_DIR/models"
mkdir -p "$LOG_DIR"

# Kill any existing Ollama processes
echo "Stopping any existing Ollama instances..."
pkill -f "ollama serve" || true
sleep 3

# Start Ollama instance
echo "Starting Ollama instance on GPU:$PRIMARY_GPU (Port: $PRIMARY_PORT)"

if [ "$USE_GPU" = true ]; then
    # GPU setup
    CUDA_VISIBLE_DEVICES=$PRIMARY_GPU \
    OLLAMA_HOST=0.0.0.0:$PRIMARY_PORT \
    OLLAMA_MODELS="$OLLAMA_DIR/models" \
    OLLAMA_NUM_PARALLEL=$OLLAMA_NUM_PARALLEL \
    OLLAMA_MAX_LOADED_MODELS=$OLLAMA_MAX_LOADED_MODELS \
    OLLAMA_MAX_QUEUE=$OLLAMA_MAX_QUEUE \
    OLLAMA_KEEP_ALIVE=$OLLAMA_KEEP_ALIVE \
    ollama serve > "$LOG_DIR/ollama_gpu${PRIMARY_GPU}_port${PRIMARY_PORT}.log" 2>&1 &
    echo "   Using GPU:$PRIMARY_GPU"
    echo "   Batch processing: OLLAMA_NUM_PARALLEL=$OLLAMA_NUM_PARALLEL"
else
    # CPU setup
    OLLAMA_HOST=0.0.0.0:$PRIMARY_PORT \
    OLLAMA_MODELS="$OLLAMA_DIR/models" \
    OLLAMA_NUM_PARALLEL=$OLLAMA_NUM_PARALLEL \
    OLLAMA_MAX_LOADED_MODELS=$OLLAMA_MAX_LOADED_MODELS \
    OLLAMA_MAX_QUEUE=$OLLAMA_MAX_QUEUE \
    OLLAMA_KEEP_ALIVE=$OLLAMA_KEEP_ALIVE \
    ollama serve > "$LOG_DIR/ollama_cpu_port${PRIMARY_PORT}.log" 2>&1 &
    echo "   Using CPU (no GPU available)"
    echo "   Batch processing: OLLAMA_NUM_PARALLEL=$OLLAMA_NUM_PARALLEL"
fi
sleep 5

# Check if the instance is running
if curl -s http://localhost:$PRIMARY_PORT/api/version > /dev/null; then
    echo "Ollama instance running on port $PRIMARY_PORT"
else
    echo "Failed to start Ollama instance on port $PRIMARY_PORT"
    echo "Check log: $LOG_DIR/ollama_gpu${PRIMARY_GPU}_port${PRIMARY_PORT}.log"
    exit 1
fi

# Pull Hermes 4 70B
# manually generate model manifest from huggingface
# run manual_install_hermes4.sh

# Pull GPT-OSS models
echo "Pulling GPT-OSS models..."
OLLAMA_HOST=localhost:$PRIMARY_PORT ollama pull gpt-oss:20b || echo "GPT-OSS 20B not available, continuing..."

# List available models
echo ""
echo "Available models:"
OLLAMA_HOST=localhost:$PRIMARY_PORT ollama list

echo ""
echo "Batch Processing Configuration:"
echo "  OLLAMA_NUM_PARALLEL: $OLLAMA_NUM_PARALLEL"
echo "  OLLAMA_MAX_LOADED_MODELS: $OLLAMA_MAX_LOADED_MODELS"
echo "  OLLAMA_MAX_QUEUE: $OLLAMA_MAX_QUEUE"
echo "  OLLAMA_KEEP_ALIVE: $OLLAMA_KEEP_ALIVE"
echo ""
echo "Note: These variables are set for the Ollama server process."
echo "They will enable concurrent request processing for better performance."
echo ""
echo "OLLAMA SETUP COMPLETE!"
echo "========================="