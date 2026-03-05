#!/bin/bash

# vLLM setup script for LLM PMID Checker
#
# Supports multiple models via environment variables:
#
#   # Hermes 4 70B (default)
#   bash setup_vllm.sh
#
#   # GPT-OSS 20B
#   VLLM_MODEL=openai/gpt-oss-20b VLLM_MODEL_NAME=gpt-oss-20b-vllm VLLM_GPU=0 VLLM_PORT=8001 bash setup_vllm.sh
#
#   # GPT-OSS 120B
#   VLLM_MODEL=openai/gpt-oss-120b VLLM_MODEL_NAME=gpt-oss-120b-vllm VLLM_GPU=1 VLLM_PORT=8002 bash setup_vllm.sh

set -e

# ============================================================
# Configuration
# ============================================================
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_GPU=${VLLM_GPU:-3}
MODEL_ID=${VLLM_MODEL:-"cyankiwi/Hermes-4-70B-AWQ-4bit"}

# Auto-derive served model name from HuggingFace repo if not set explicitly
if [ -n "$VLLM_MODEL_NAME" ]; then
    MODEL_NAME="$VLLM_MODEL_NAME"
elif [[ "$MODEL_ID" == *"gpt-oss-20b"* ]]; then
    MODEL_NAME="gpt-oss-20b-vllm"
elif [[ "$MODEL_ID" == *"gpt-oss-120b"* ]]; then
    MODEL_NAME="gpt-oss-120b-vllm"
elif [[ "$MODEL_ID" == *"Hermes-4"* ]] || [[ "$MODEL_ID" == *"hermes4"* ]]; then
    MODEL_NAME="hermes4-vllm"
else
    # Fallback: use last part of repo id, lowercased, with slashes replaced
    MODEL_NAME=$(echo "$MODEL_ID" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')"-vllm"
fi

PROJECT_DIR=~/work/llm_pmid_support
LOG_DIR="$PROJECT_DIR/logs"
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"

# vLLM server settings
MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-8192}
GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEM:-0.95}
MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-24}
DTYPE="auto"

# ============================================================
# Preflight checks
# ============================================================
echo "============================================================"
echo " vLLM Server Setup"
echo " Model: $MODEL_ID"
echo " Served as: $MODEL_NAME"
echo "============================================================"

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. NVIDIA GPU required for vLLM."
    exit 1
fi

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "Found $GPU_COUNT GPU(s)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
echo ""

# Check if vLLM is installed
if ! python -c "import vllm" 2>/dev/null; then
    echo "ERROR: vLLM is not installed. Please install it first:"
    echo "  pip install vllm"
    echo ""
    echo "For CUDA 12.x with A100 GPUs:"
    echo "  pip install vllm"
    exit 1
fi

VLLM_VERSION=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
echo "vLLM version: $VLLM_VERSION"
echo ""

# ============================================================
# Create directories
# ============================================================
mkdir -p "$LOG_DIR"

# ============================================================
# Kill any existing vLLM processes on the same port
# ============================================================
echo "Checking for existing vLLM processes on port $VLLM_PORT..."
EXISTING_PID=$(lsof -ti:$VLLM_PORT 2>/dev/null || true)
if [ -n "$EXISTING_PID" ]; then
    echo "  Killing existing process on port $VLLM_PORT (PID: $EXISTING_PID)"
    kill -9 $EXISTING_PID 2>/dev/null || true
    sleep 2
fi

# ============================================================
# Start vLLM server
# ============================================================
echo ""
echo "Starting vLLM server..."
echo "  Model:        $MODEL_ID"
echo "  Served as:    $MODEL_NAME"
echo "  GPU:          $VLLM_GPU"
echo "  Port:         $VLLM_PORT"
echo "  Max seq len:  $MAX_MODEL_LEN"
echo "  Max batch:    $MAX_NUM_SEQS"
echo "  GPU mem util: $GPU_MEMORY_UTILIZATION"
echo "  Dtype:        $DTYPE"
echo ""

VLLM_LOG="$LOG_DIR/vllm_gpu${VLLM_GPU}_port${VLLM_PORT}.log"

CUDA_VISIBLE_DEVICES=$VLLM_GPU \
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --served-model-name "$MODEL_NAME" \
    --port $VLLM_PORT \
    --dtype $DTYPE \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-num-seqs $MAX_NUM_SEQS \
    --enable-prefix-caching \
    --trust-remote-code \
    --disable-log-requests \
    > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "vLLM server started (PID: $VLLM_PID)"
echo "Log file: $VLLM_LOG"

# ============================================================
# Wait for server to be ready
# ============================================================
echo ""
echo "Waiting for vLLM server to load model and start serving..."
echo "(This may take a few minutes for model download on first run)"
echo ""

MAX_WAIT=600  # 10 minutes max wait
WAITED=0
INTERVAL=10

while [ $WAITED -lt $MAX_WAIT ]; do
    # Check if process is still alive
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo ""
        echo "ERROR: vLLM server process died unexpectedly."
        echo "Check log file: $VLLM_LOG"
        echo "Last 20 lines:"
        tail -20 "$VLLM_LOG"
        exit 1
    fi
    
    # Check if API is responding
    if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
        echo ""
        echo "vLLM server is ready!"
        break
    fi
    
    echo "  Waiting... ($WAITED seconds elapsed)"
    sleep $INTERVAL
    WAITED=$((WAITED + INTERVAL))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo ""
    echo "ERROR: vLLM server did not start within $MAX_WAIT seconds."
    echo "Check log file: $VLLM_LOG"
    tail -20 "$VLLM_LOG"
    exit 1
fi

# ============================================================
# Verify server
# ============================================================
echo ""
echo "Verifying server..."
echo ""

# List available models
echo "Available models:"
curl -s http://localhost:$VLLM_PORT/v1/models | python -m json.tool 2>/dev/null || \
    curl -s http://localhost:$VLLM_PORT/v1/models

echo ""
echo "============================================================"
echo " vLLM SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "Server:  http://localhost:$VLLM_PORT"
echo "API:     http://localhost:$VLLM_PORT/v1"
echo "Health:  http://localhost:$VLLM_PORT/health"
echo "Model:   $MODEL_NAME ($MODEL_ID)"
echo "PID:     $VLLM_PID"
echo ""
echo "To use with evaluate_batch.py:"
echo "  python evaluation/evaluate_batch.py \\"
echo "      --input evaluation/test_data.tsv \\"
echo "      --output evaluation/results_${MODEL_NAME}.tsv \\"
echo "      --val_model $MODEL_NAME \\"
echo "      --node_dict data/kg2_data/kg2c-2.10.2-v1.0-nodes.jsonl.gz \\"
echo "      --max_concurrent 24"
echo ""
echo "To stop the server:"
echo "  kill $VLLM_PID"
echo ""
echo "Environment variables to customize:"
echo "  VLLM_MODEL=<hf_repo>         # HuggingFace model repo"
echo "  VLLM_MODEL_NAME=<name>       # Served model name (auto-derived if not set)"
echo "  VLLM_PORT=8000               # API port"
echo "  VLLM_GPU=3                   # GPU device index"
echo "  VLLM_MAX_MODEL_LEN=4096      # Max sequence length"
echo "  VLLM_MAX_NUM_SEQS=24         # Max concurrent sequences"
echo "  VLLM_GPU_MEM=0.95            # GPU memory utilization (0-1)"
echo ""
echo "Examples:"
echo "  # Hermes 4 70B (default)"
echo "  bash setup_vllm.sh"
echo ""
echo "  # GPT-OSS 20B on GPU 0, port 8001"
echo "  VLLM_MODEL=openai/gpt-oss-20b VLLM_GPU=0 VLLM_PORT=8001 bash setup_vllm.sh"
echo ""
echo "  # GPT-OSS 120B on GPU 1, port 8002"
echo "  VLLM_MODEL=openai/gpt-oss-120b VLLM_GPU=1 VLLM_PORT=8002 bash setup_vllm.sh"
