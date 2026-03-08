# LLM PMID Checker

A system for checking whether research triples are supported by PubMed abstracts using large language models.

## Overview

Given a TSV file of research triples (e.g., `SIX1 stimulates Cell Proliferation`) with associated PubMed IDs, this system:

1. **Extracts abstracts** from PMIDs via NCBI E-utilities
2. **Evaluates support** using vLLM-served models with concurrent batch processing
3. **Saves results** to a SQLite database or TSV file, preserving all input columns alongside evaluation outputs

## Quick Start

### 1. Install Dependencies

```bash
conda activate llm_pmid_env
pip install -r requirements.txt
```

### 2. Start vLLM Server(s)

Use the provided setup script to launch one or more vLLM servers:

```bash
# Hermes 4 70B
VLLM_MODEL=cyankiwi/Hermes-4-70B-AWQ-4bit VLLM_MODEL_NAME=hermes4-vllm VLLM_GPU=0 VLLM_PORT=8000 bash setup_vllm.sh

# GPT-OSS 20B
VLLM_MODEL=openai/gpt-oss-20b VLLM_MODEL_NAME=gpt-oss-20b-vllm VLLM_GPU=0 VLLM_PORT=8001 bash setup_vllm.sh

# GPT-OSS 120B
VLLM_MODEL=openai/gpt-oss-120b VLLM_MODEL_NAME=gpt-oss-120b-vllm VLLM_GPU=0 VLLM_PORT=8002 bash setup_vllm.sh
```

### 3. Pre-fetch PMID Abstracts (Recommended)

Abstracts fetched from NCBI are automatically cached in a local SQLite database (`data/pmid_cache.db`). For large datasets, pre-fetch all abstracts before running evaluation to avoid rate limits during batch processing:

```bash
python scripts/prefetch_pmid_abstracts.py --tsv-file data/test_data.tsv

# Adjust batch size and rate limiting
python scripts/prefetch_pmid_abstracts.py --tsv-file data/test_data.tsv --batch-size 200 --delay 1.0

# Force re-fetch (overwrite cached entries)
python scripts/prefetch_pmid_abstracts.py --tsv-file data/test_data.tsv --force
```

To diagnose cache issues:

```bash
# Find PMIDs that failed to cache (errors or missing abstracts)
python scripts/check_failed_pmids.py --tsv-file data/test_data.tsv

# Check overall cache status
python scripts/check_cache_status.py
```

### 4. Configure Environment

Create a `.env` file in the project root:

```bash
# NCBI E-utilities
NCBI_EMAIL=your.email@example.com
NCBI_API_KEY=your_ncbi_api_key_here

# Batch processing
MAX_CONCURRENT_REQUESTS=5

# vLLM Configuration
VLLM_BASE_URL=http://localhost:8000

# Per-model URLs (comma-separated model=url pairs)
VLLM_MODEL_URLS=hermes4-vllm=http://localhost:8000,gpt-oss-20b-vllm=http://localhost:8001,gpt-oss-120b-vllm=http://localhost:8002

# Available vLLM models (must match --served-model-name used when starting vLLM)
AVAILABLE_VLLM_MODELS=hermes4-vllm,gpt-oss-20b-vllm,gpt-oss-120b-vllm
```

### 5. Run Evaluation

## Usage

```
python main.py --input INPUT_TSV --output OUTPUT_FILE [options]
```

Output format is auto-detected from the file extension:
- `.db` / `.sqlite` / `.sqlite3` → SQLite database
- `.tsv` / `.txt` → Tab-separated values

| Flag | Description |
|---|---|
| `--input` | **(required)** Input TSV file |
| `--output` | **(required)** Output file (`.db` for SQLite, `.tsv` for TSV) |
| `--val_model` | Validation model (default: first in `AVAILABLE_VLLM_MODELS`) |
| `--round2_model` | Optional Round 2 model for re-evaluating yes/maybe results |
| `--table` | SQLite table name, only for `.db` output (default: `evaluations`) |
| `--node_dict` | KG2 nodes file for richer entity context |
| `--max_concurrent` | Max concurrent requests (default: `MAX_CONCURRENT_REQUESTS` from `.env`) |
| `--verbose` / `-v` | Enable DEBUG logging |

### Examples

```bash
# Basic evaluation (SQLite output)
python main.py --input data/test_data.tsv --output results.db --val_model gpt-oss-20b-vllm

# TSV output
python main.py --input data/test_data.tsv --output results.tsv --val_model gpt-oss-20b-vllm

# High concurrency
python main.py --input data/test_data.tsv --output results.tsv --val_model hermes4-vllm --max_concurrent 30

# Two-round evaluation (Round 1 with 20B, Round 2 with 120B)
python main.py --input data/test_data.tsv --output results.tsv \
    --val_model gpt-oss-20b-vllm --round2_model gpt-oss-120b-vllm

# With node_dict for entity context enrichment
python main.py --input data/test_data.tsv --output results.tsv \
    --val_model gpt-oss-20b-vllm --node_dict data/kg2_data/kg2c-2.10.2-v1.0-nodes.jsonl.gz

# Write to a custom table name (useful for multiple runs in the same DB)
python main.py --input data/test_data.tsv --output results.tsv \
    --val_model gpt-oss-20b-vllm --table run_20b_v1
```

## Input Format

The input TSV file **must** contain these columns:

| Column | Description |
|---|---|
| `subject_curie` | Subject entity CURIE (e.g., `CHEBI:70723`) |
| `predicate` | Relationship (e.g., `stimulates`, `inhibits`) |
| `object_curie` | Object entity CURIE (e.g., `PR:000004517`) |
| `PMID` | PubMed ID to check against |

Any additional columns in the TSV are carried through to the output unchanged.

### Example TSV

```
subject	predicate	object	PMID	subject_curie	object_curie	Supported
INCENP protein, human	stimulates	aurora kinase B	18767990	MESH:C083767	PR:000004517	true
quercetagetin	inhibits	aurora kinase B	25298094	CHEBI:8695	PR:000004517	true
```

## Output Format

Results are written to a **SQLite database** (`.db`) or a **TSV file** (`.tsv`), depending on the `--output` extension. Both formats contain all columns from the input TSV plus these evaluation columns:

| Column | Type | Description |
|---|---|---|
| `predicted` | bool | Whether the triple is supported (`support == "yes"`) |
| `support` | text | `yes`, `no`, or `maybe` |
| `subject_mentioned` | bool | Whether the subject appears in the abstract |
| `object_mentioned` | bool | Whether the object appears in the abstract |
| `supporting_sentences` | text | Exact sentences from the abstract (pipe-separated) |
| `reasoning` | text | LLM's reasoning for the judgment |
| `runtime_seconds` | real | Wall-clock time for this evaluation |

## Available Models

| Model | HuggingFace Repo |
|---|---|
| `hermes4-vllm` | [cyankiwi/Hermes-4-70B-AWQ-4bit](https://huggingface.co/cyankiwi/Hermes-4-70B-AWQ-4bit) |
| `gpt-oss-20b-vllm` | [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) |
| `gpt-oss-120b-vllm` | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) |

