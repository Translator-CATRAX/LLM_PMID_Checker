# LLM PMID Checker

A system for checking whether research triples are supported by PubMed abstracts using large language models.

## Overview

Given a Parquet (or TSV) file of research triples (e.g., `aspirin biolink:treats_or_applied_or_studied_to_treat headache`) with associated PubMed IDs, this system:

1. **Extracts abstracts** from PMIDs via NCBI E-utilities
2. **Evaluates support** using vLLM-served models with concurrent batch processing
3. **Saves results** to a SQLite database (recommended) or TSV file, preserving all input columns alongside evaluation outputs

## Pre-computed Evaluation Results

Pre-computed evaluation results for the full SemMedDB KGX dataset are available as a [GitHub release](https://github.com/RTXteam/LLM_PMID_Checker/releases/tag/v1.0). If you only need the results, you can skip the setup steps below and download them directly.

### Download Results

To download the pre-computed results, please use the provided download script (requires only Python stdlib):

```bash
python scripts/download_release_data.py --output-dir results --tag v1.0
```

After running the command above, the following files will be downloaded:
```
results/
    results_no_abstract_with_names.parquet
    results_with_names.parquet                           <-- use this (5.9G, 26,719,183 rows with 13 cols)
    LLM_Pmid_Evaluation_SemMedDB_with_names_v1.0.tar.gz 
```

### Explanation of Results

The release contains two Parquet files:

| File | Size | Rows | Description |
|---|---|---|---|
| `results_with_names.parquet` | ~5.9 GB | 26,719,183 | Evaluation results for triples whose PMIDs had an available abstract |
| `results_no_abstract_with_names.parquet` | ~69 MB | 1,292,499 | Triples whose PMIDs had no abstract available (not evaluated) |

**`results_with_names.parquet`** contains the original input columns plus entity names and LLM evaluation outputs:

| Column | Type | Description |
|---|---|---|
| `subject_curie` | string | Subject entity CURIE (e.g., `NCBITaxon:562`) |
| `subject_name` | string | Subject entity name (e.g., `Escherichia coli`) |
| `predicate` | string | Biolink predicate (e.g., `biolink:has_part`) |
| `object_curie` | string | Object entity CURIE (e.g., `NCBIGene:100`) |
| `object_name` | string | Object entity name (e.g., `ADA`) |
| `PMID` | string | PubMed ID used for evaluation (e.g., `PMID:3047400`) |
| `SemMedDB_sentences` | string | Original SemMedDB sentence(s) for this edge (pipe-separated if multiple) |
| `predicted` | bool | Whether the triple is supported (`True` if `support == "yes"`) |
| `support` | string | LLM judgment: `yes`, `no`, or `maybe` |
| `subject_mentioned` | bool | Whether the subject entity is mentioned in the abstract |
| `object_mentioned` | bool | Whether the object entity is mentioned in the abstract |
| `supporting_sentences` | string | Exact sentences from the abstract that support the triple (pipe-separated if multiple) |
| `reasoning` | string | LLM's reasoning for the judgment |

Each row represents one unique `(subject_curie, predicate, object_curie, PMID)` combination. The LLM reads the PubMed abstract for the given PMID, checks whether the subject and object are mentioned, and judges whether the abstract supports the stated relationship.

### Dataset Statistics

**Coverage:**

| Metric | Count |
|---|---|
| Total unique (subject, predicate, object, PMID) combinations | 28,011,682 |
| Triples with abstract available (evaluated) | 26,719,183 (95.4%) |
| Triples without abstract (not evaluated) | 1,292,499 (4.6%) |
| Unique PMIDs (with abstract) | 12,000,111 |
| Unique PMIDs (without abstract) | 1,094,551 |
| Unique subject CURIEs | 54,920 |
| Unique object CURIEs | 47,290 |
| Unique Biolink predicates | 19 |

**Support distribution** (among 26,719,183 evaluated triples):

| Support | Count | Percentage |
|---|---|---|
| `yes` | 16,139,271 | 60.4% |
| `no` | 9,447,319 | 35.4% |
| `maybe` | 1,132,593 | 4.2% |

**Entity mention rates** (among 26,719,183 evaluated triples):

| Metric | Count | Percentage |
|---|---|---|
| Both subject and object mentioned | 20,659,114 | 77.3% |
| Subject only mentioned | 2,427,312 | 9.1% |
| Object only mentioned | 2,909,116 | 10.9% |
| Neither mentioned | 723,641 | 2.7% |

## How to Run

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

Then extract the per-PMID edges into a Parquet file from the normalized edges JSONL:

```bash
python scripts/extract_semmeddb_edges.py \
    -i data/semmedb_kgx/normalized_edges.jsonl \
    -o data/semmedb_kgx/semmeddb_edges_extracted.parquet
```

This groups edges by `(subject_curie, predicate, object_curie, PMID)`. When multiple edge records share the same key but have different supporting sentences, the sentences are concatenated with `" | "` in the `SemMedDB_sentences` column. Output columns: `subject_curie`, `predicate`, `object_curie`, `PMID`, `SemMedDB_sentences`.

### 3. Node File & CURIE Names for Richer Entity Context (Recommended)

The SemMedDB KGX download (step 2) includes `normalized_nodes.jsonl`, which provides entity names, categories, descriptions, and equivalent identifiers for every CURIE. Combined with `curie_all_names.tsv` (generated via the following command), they can provide richer entity context.

```bash
python scripts/extract_curie_names.py \
    --input data/semmedb_kgx/semmeddb_edges_extracted.parquet \
    --output data/semmedb_kgx/curie_all_names.tsv \
    --batch-size 500 --max-concurrent 10
```

This queries the [Node Normalization API](https://nodenormalization-sri.renci.org/docs) for all unique CURIEs and collects every known name variant (primary label + labels from 
equivalent identifiers), case-insensitively deduplicated.

### 4. Start vLLM Server(s)

Use the provided setup script to launch one or more vLLM servers:

```bash
# GPT-OSS 20B on GPU 0
VLLM_MODEL=openai/gpt-oss-20b VLLM_MODEL_NAME=gpt-oss-20b-vllm VLLM_GPU=0 VLLM_PORT=8000 bash setup_vllm.sh

# GPT-OSS 120B on GPU 1
VLLM_MODEL=openai/gpt-oss-120b VLLM_MODEL_NAME=gpt-oss-120b-vllm VLLM_GPU=1 VLLM_PORT=9000 bash setup_vllm.sh
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
    --tsv-file data/semmedb_kgx/semmeddb_edges_extracted.parquet \
    --batch-size 200 --delay 1.0

# Force re-fetch (overwrite cached entries)
python scripts/prefetch_pmid_abstracts.py \
    --tsv-file data/semmedb_kgx/semmeddb_edges_extracted.parquet --force
```

If the initial fetch has transient network failures, retry only the failed PMIDs:

```bash
python scripts/retry_failed_pmids.py --batch-size 200 --delay 1.0
```

To diagnose cache issues:

```bash
# Find PMIDs that failed to cache (errors or missing abstracts)
python scripts/check_failed_pmids.py \
    --tsv-file data/semmedb_kgx/semmeddb_edges_extracted.parquet

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

See [Usage](#usage) below for full command-line options and examples.

## Usage

```
python main.py --input INPUT_FILE --output OUTPUT_FILE [options]
```

Input format is auto-detected from the file extension:
- `.parquet` / `.pq` → Parquet (recommended, preserves text exactly)
- `.tsv` / `.txt` → Tab-separated values

Output format is auto-detected from the file extension:
- `.db` / `.sqlite` / `.sqlite3` → SQLite database (recommended for stop/resume)
- `.tsv` / `.txt` → Tab-separated values

When using SQLite output (`.db`), rows without a cached abstract are written to a `evaluations_no_abstract` table in the same database. When using TSV output, they are written to a separate `*_no_abstract.tsv` file.

| Flag | Description |
|---|---|
| `--input` | **(required)** Input file (`.parquet` or `.tsv`; must contain `subject_curie`, `predicate`, `object_curie`, `PMID`) |
| `--output` | **(required)** Output file (`.db` for SQLite recommended, `.tsv` for TSV) |
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

Results are written incrementally -- every completed row is flushed to disk immediately. You can safely `Ctrl+C` at any time and re-run the exact same command to resume:

```bash
# First run (or resume after interruption) -- same command each time
python main.py --input data/semmedb_kgx/semmeddb_edges_extracted.parquet --output results.db \
    --val_model gpt-oss-120b-vllm \
    --predicate_file data/biolink_data/biolink_predicates.tsv \
    --node_dict data/semmedb_kgx/normalized_nodes.jsonl \
    --names_file data/semmedb_kgx/curie_all_names.tsv

# To discard previous progress and start over
python main.py --input data/semmedb_kgx/semmeddb_edges_extracted.parquet --output results.db \
    --val_model gpt-oss-120b-vllm --overwrite \
    --predicate_file data/biolink_data/biolink_predicates.tsv \
    --node_dict data/semmedb_kgx/normalized_nodes.jsonl \
    --names_file data/semmedb_kgx/curie_all_names.tsv
```

On resume, the program reads the existing output, determines which `(subject_curie, predicate, object_curie, PMID)` rows are already evaluated, and only processes the remaining rows.

### Examples

```bash
# Standard evaluation with Parquet input and SQLite output (recommended)
python main.py --input data/semmedb_kgx/semmeddb_edges_extracted.parquet --output results.db \
    --val_model gpt-oss-120b-vllm --max_concurrent 24 \
    --predicate_file data/biolink_data/biolink_predicates.tsv \
    --node_dict data/semmedb_kgx/normalized_nodes.jsonl \
    --names_file data/semmedb_kgx/curie_all_names.tsv

# Two-round evaluation (Round 1 with 20B, Round 2 with 120B)
python main.py --input data/semmedb_kgx/semmeddb_edges_extracted.parquet --output results.db \
    --val_model gpt-oss-20b-vllm --round2_model gpt-oss-120b-vllm \
    --predicate_file data/biolink_data/biolink_predicates.tsv \
    --node_dict data/semmedb_kgx/normalized_nodes.jsonl \
    --names_file data/semmedb_kgx/curie_all_names.tsv

# Write to a custom table name (useful for multiple runs in the same DB)
python main.py --input data/semmedb_kgx/semmeddb_edges_extracted.parquet --output results.db \
    --val_model gpt-oss-20b-vllm --table run_20b_v1 \
    --predicate_file data/biolink_data/biolink_predicates.tsv \
    --node_dict data/semmedb_kgx/normalized_nodes.jsonl \
    --names_file data/semmedb_kgx/curie_all_names.tsv
```

## Input Format

The input file (Parquet or TSV) **must** contain these columns:

| Column | Description |
|---|---|
| `subject_curie` | Subject entity CURIE (e.g., `CHEBI:70723`) |
| `predicate` | Relationship (e.g., `biolink:affects`, `biolink:treats_or_applied_or_studied_to_treat`) |
| `object_curie` | Object entity CURIE (e.g., `PR:000004517`) |
| `PMID` | PubMed ID to check against |

Any additional columns are carried through to the output unchanged.

## Output Format

Results are written to a **SQLite database** (`.db`, recommended) or a **TSV file** (`.tsv`), depending on the `--output` extension. Both formats contain all columns from the input plus these evaluation columns:

| Column | Type | Description |
|---|---|---|
| `subject_name` | text | Subject entity name from node dictionary (inserted after `subject_curie`) |
| `predicted` | bool | Whether the triple is supported (`support == "yes"`) |
| `object_name` | text | Object entity name from node dictionary (inserted after `object_curie`) |
| `support` | text | `yes`, `no`, or `maybe` |
| `subject_mentioned` | bool | Whether the subject appears in the abstract |
| `object_mentioned` | bool | Whether the object appears in the abstract |
| `supporting_sentences` | text | Exact sentences from the abstract (pipe-separated with `" \| "`) |
| `reasoning` | text | LLM's reasoning for the judgment |

### Post-evaluation Utilities

**Convert SQLite to Parquet** (for final delivery or analytical queries):

```bash
python scripts/convert_db_to_parquet.py --db results.db --output-dir .
```

This produces `results.parquet` (from the `evaluations` table) and `results_no_abstract.parquet` (from the `evaluations_no_abstract` table). The `runtime_seconds` column is dropped by default; use `--drop-columns` with no arguments to keep all columns.

**Verify coverage** (ensure all input rows are accounted for):

```bash
python scripts/compare_coverage.py \
    --extracted data/semmedb_kgx/semmeddb_edges_extracted.parquet \
    --results-db results.db
```

This reports unique 4-key counts, duplicates, overlap between tables, coverage percentage, and lists any missing or extra keys.

## Available Models

| Model | HuggingFace Repo |
|---|---|
| `gpt-oss-20b-vllm` | [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) |
| `gpt-oss-120b-vllm` | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) |
