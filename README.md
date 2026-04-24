# LLM PMID Checker

A system for checking whether research triples are supported by PubMed abstracts using large language models.

## Overview

Given a TSV file of research triples (e.g., `aspirin biolink:treats_or_applied_or_studied_to_treat headache`) with associated PubMed IDs, this system:

1. **Extracts abstracts** from PMIDs via NCBI E-utilities
2. **Evaluates support** using vLLM-served models with concurrent batch processing
3. **Saves results** to a SQLite database or TSV file, preserving all input columns alongside evaluation outputs

## Quick Start

### 1. Install Dependencies

```bash
conda activate llm_pmid_env
pip install -r requirements.txt
```

### 2. Prepare SemMedDB KGX Data

Download the SemMedDB KGX dataset from [Translator-CATRAX/SemMedDB-KGX](https://github.com/Translator-CATRAX/SemMedDB-KGX) into `data/semmedb_kgx/`:

```bash
cd data/semmedb_kgx
python download_semmeddb_uncapped.py
```

Then extract the per-PMID edges TSV from the normalized edges JSONL:

```bash
python scripts/extract_semmeddb_edges.py \
    -i data/semmedb_kgx/normalized_edges.jsonl \
    -o data/semmedb_kgx/semmeddb_edges_extracted.tsv
```

This expands each multi-PMID edge into one row per PMID, producing columns: `subject_curie`, `predicate`, `object_curie`, `PMID`, `SemMedDB_sentence`.

### 3. Node File & CURIE Names for Richer Entity Context (Recommended)

The SemMedDB KGX download (step 2) includes `normalized_nodes.jsonl`, which provides entity names, categories, descriptions, and equivalent identifiers for every CURIE. Combined with `curie_all_names.tsv` (produced below), they can provide richer entity context.

```bash
python scripts/extract_curie_names.py \
    --input data/semmedb_kgx/semmeddb_edges_extracted.tsv \
    --output data/semmedb_kgx/curie_all_names.tsv \
    --batch-size 500 --max-concurrent 10
```

This queries the [Node Normalization API](https://nodenormalization-sri.renci.org/docs) for all 66,024 unique CURIEs and collects every known name variant (primary label + labels from equivalent identifiers), case-insensitively deduplicated. Runtime: ~5–15 minutes.



### 4. Start vLLM Server(s)

Use the provided setup script to launch one or more vLLM servers:

```bash
# GPT-OSS 20B
VLLM_MODEL=openai/gpt-oss-20b VLLM_MODEL_NAME=gpt-oss-20b-vllm VLLM_GPU=0 VLLM_PORT=8001 bash setup_vllm.sh

# GPT-OSS 120B
VLLM_MODEL=openai/gpt-oss-120b VLLM_MODEL_NAME=gpt-oss-120b-vllm VLLM_GPU=1 VLLM_PORT=8002 bash setup_vllm.sh
```

### 5. Extract Biolink Predicate Definitions (Optional)

If your input uses Biolink predicates (e.g., `biolink:affects`, `biolink:treats_or_applied_or_studied_to_treat`), extract predicate definitions from the Biolink Model YAML to provide the LLM with formal predicate semantics:

```bash
python scripts/extract_biolink_predicates.py \
    --input data/biolink_data/biolink-model.yaml \
    --output data/biolink_data/biolink_predicates.tsv
```

The output TSV has two columns: `predicate` (e.g., `biolink:affects`) and `description`. Pass it to `main.py` via `--predicate_file`.

### 6. Pre-fetch PMID Abstracts (Recommended)

Abstracts fetched from NCBI are automatically cached in a local SQLite database (`data/pmid_cache.db`). For large datasets, pre-fetch all abstracts before running evaluation to avoid rate limits during batch processing:

```bash
python scripts/prefetch_pmid_abstracts.py \
    --tsv-file data/semmedb_kgx/semmeddb_edges_extracted.tsv \
    --batch-size 200 --delay 1.0

# Force re-fetch (overwrite cached entries)
python scripts/prefetch_pmid_abstracts.py \
    --tsv-file data/semmedb_kgx/semmeddb_edges_extracted.tsv --force
```

If the initial fetch has transient network failures, retry only the failed PMIDs:

```bash
python scripts/retry_failed_pmids.py --batch-size 200 --delay 1.0
```

To diagnose cache issues:

```bash
# Find PMIDs that failed to cache (errors or missing abstracts)
python scripts/check_failed_pmids.py \
    --tsv-file data/semmedb_kgx/semmeddb_edges_extracted.tsv

# Check overall cache status
python scripts/check_cache_status.py
```

### 7. Configure Environment

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
VLLM_MODEL_URLS=gpt-oss-20b-vllm=http://localhost:8000,gpt-oss-120b-vllm=http://localhost:8002

# Available vLLM models (must match --served-model-name used when starting vLLM)
AVAILABLE_VLLM_MODELS=gpt-oss-20b-vllm,gpt-oss-120b-vllm
```

### 8. Run Evaluation

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
| `--node_dict` | Nodes file (`.jsonl`, `.jsonl.gz`) for richer entity context |
| `--names_file` | `curie_all_names.tsv` to supplement `--node_dict` with richer equivalent names |
| `--predicate_file` | Biolink predicates TSV with predicate definitions (columns: `predicate`, `description`) |
| `--max_concurrent` | Max concurrent requests (default: `MAX_CONCURRENT_REQUESTS` from `.env`) |
| `--overwrite` | Discard existing output and start fresh (default: auto-resume) |
| `--verbose` / `-v` | Enable DEBUG logging |

### Stop & Resume

Results are written incrementally — every completed row is flushed to disk immediately. You can safely `Ctrl+C` at any time and re-run the exact same command to resume:

```bash
# First run (or resume after interruption) — same command each time
python main.py --input data/semmedb_kgx/edges_with_abstract.tsv --output results.tsv \
    --val_model gpt-oss-20b-vllm \
    --predicate_file data/biolink_data/biolink_predicates.tsv \
    --node_dict data/semmedb_kgx/normalized_nodes.jsonl \
    --names_file data/semmedb_kgx/curie_all_names.tsv

# To discard previous progress and start over
python main.py --input data/semmedb_kgx/edges_with_abstract.tsv --output results.tsv \
    --val_model gpt-oss-20b-vllm --overwrite \
    --predicate_file data/biolink_data/biolink_predicates.tsv \
    --node_dict data/semmedb_kgx/normalized_nodes.jsonl \
    --names_file data/semmedb_kgx/curie_all_names.tsv
```

On resume, the program reads the existing output file, determines which `(subject_curie, predicate, object_curie, PMID)` rows are already evaluated, and only processes the remaining rows.

### Examples

```bash
# Basic evaluation (SQLite output)
python main.py --input data/semmedb_kgx/semmeddb_edges_extracted.tsv --output results.db \
    --val_model gpt-oss-20b-vllm \
    --predicate_file data/biolink_data/biolink_predicates.tsv \
    --node_dict data/semmedb_kgx/normalized_nodes.jsonl \
    --names_file data/semmedb_kgx/curie_all_names.tsv

# TSV output
python main.py --input data/semmedb_kgx/semmeddb_edges_extracted.tsv --output results.tsv \
    --val_model gpt-oss-20b-vllm \
    --predicate_file data/biolink_data/biolink_predicates.tsv \
    --node_dict data/semmedb_kgx/normalized_nodes.jsonl \
    --names_file data/semmedb_kgx/curie_all_names.tsv

# High concurrency
python main.py --input data/semmedb_kgx/semmeddb_edges_extracted.tsv --output results.tsv \
    --val_model gpt-oss-20b-vllm --max_concurrent 30 \
    --predicate_file data/biolink_data/biolink_predicates.tsv \
    --node_dict data/semmedb_kgx/normalized_nodes.jsonl \
    --names_file data/semmedb_kgx/curie_all_names.tsv

# Two-round evaluation (Round 1 with 20B, Round 2 with 120B)
python main.py --input data/semmedb_kgx/semmeddb_edges_extracted.tsv --output results.tsv \
    --val_model gpt-oss-20b-vllm --round2_model gpt-oss-120b-vllm \
    --predicate_file data/biolink_data/biolink_predicates.tsv \
    --node_dict data/semmedb_kgx/normalized_nodes.jsonl \
    --names_file data/semmedb_kgx/curie_all_names.tsv

# Write to a custom table name (useful for multiple runs in the same DB)
python main.py --input data/semmedb_kgx/semmeddb_edges_extracted.tsv --output results.db \
    --val_model gpt-oss-20b-vllm --table run_20b_v1 \
    --predicate_file data/biolink_data/biolink_predicates.tsv \
    --node_dict data/semmedb_kgx/normalized_nodes.jsonl \
    --names_file data/semmedb_kgx/curie_all_names.tsv
```

## Input Format

The input TSV file **must** contain these columns:

| Column | Description |
|---|---|
| `subject_curie` | Subject entity CURIE (e.g., `CHEBI:70723`) |
| `predicate` | Relationship (e.g., `biolink:affects`, `biolink:treats_or_applied_or_studied_to_treat`) |
| `object_curie` | Object entity CURIE (e.g., `PR:000004517`) |
| `PMID` | PubMed ID to check against |

Any additional columns in the TSV are carried through to the output unchanged.

### Example TSV

```
subject_curie	predicate	object_curie	PMID	SemMedDB_sentence
NCBIGene:1	biolink:treats_or_applied_or_studied_to_treat	UMLS:C0007634	PMID:24096582	Our previous study revealed that...
NCBIGene:1	biolink:affects	UMLS:C0005935	PMID:29798367	Removal of tympanosclerosis lesions...
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
| `gpt-oss-20b-vllm` | [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) |
| `gpt-oss-120b-vllm` | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) |

