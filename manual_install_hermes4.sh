TARGET_MODEL_NAME=${1:-hermes4:70b-q4-m}
GGUF_FILENAME=${2:-Hermes-4-70B-UD-Q4_K_M.gguf}

pip install huggingface_hub --quiet
cd ~/.ollama_pmid/models
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='unsloth/Hermes-4-70B-GGUF', filename='${GGUF_FILENAME}', local_dir='.')"

# create modelfile
cat > /tmp/hermes4-modelfile << EOF
FROM ~/.ollama_pmid/models/${GGUF_FILENAME}

TEMPLATE """<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

"""

SYSTEM """You are Hermes 4, a helpful AI assistant. Be concise and accurate in your responses."""

PARAMETER temperature 0.1
PARAMETER top_p 0.95
PARAMETER top_k 20
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|start_header_id|>"
EOF

# create model
OLLAMA_HOST=localhost:11434 ollama create ${TARGET_MODEL_NAME} -f /tmp/hermes4-modelfile