# LLM PMID Checker

A system for checking whether research triples are supported by PubMed abstracts using large language models.

## Overview

This system checks whether a given research triple (e.g., `['SIX1', 'affects', 'Cell Proliferation']`) is supported by a list of PubMed IDs (PMIDs). It:

1. **Normalizes entities** using a multi-source approach:
   - **HGNC** (Hugo Gene Nomenclature Committee) for gene/protein entities
   - **ARAX TRAPI API** to convert names to CURIEs
   - **Node Normalization API** to find equivalent identifiers
   - **UMLS** (Unified Medical Language System) for additional medical term synonyms
2. **Extracts abstracts** from PMIDs using NCBI E-utilities
3. **Checks support** using AI models:
   - **Local models** with **concurrent batch processing** via Ollama (Hermes 4, GPT-OSS)
   - **Cloud models** via OpenAI API (GPT-5 nano, GPT-5 mini)
4. **Reports results** with confidence scores and supporting sentences

## Quick Start

### 1. Install Dependencies

If you haven't pre-installed Ollama, you can download and install it from [here](https://ollama.com/download).

```bash
conda activate llm_pmid_env
pip install -r requirements.txt
```

### 2. Setup Ollama

Run the provided setup script:

```bash
pkill -f "ollama serve"
./setup_ollama.sh
```

This bash script will automatically launch Ollama and pull the necessary models. However, the hermes4 model is not available in the Ollama model hub yet, so you will need to manually install this model from Hugging Face. You can simply run the script `manual_install_hermes4.sh` to install the model.

```bash
./manual_install_hermes4.sh
```

### 3. Configure Environment

Create a `.env` file:

```bash
# Ollama Configuration  
OLLAMA_BASE_URL=http://localhost:11434

# Available Ollama models (comma-separated list)
# Add or remove models as needed based on your 'ollama list' output
# The first model in the list will be used as the default
AVAILABLE_OLLAMA_MODELS=hermes4:70b-iq4-xs,hermes4:70b-q4-s,hermes4:70b-q4-m,gpt-oss:20b

# NCBI E-utilities Configuration
NCBI_EMAIL=your.email@example.com
NCBI_API_KEY=your_ncbi_api_key_here

# UMLS Settings (Optional - enhances name resolution)
UMLS_API_KEY=your_umls_api_key_here
USE_UMLS=true

# ARAX and Node Normalization APIs
# ARAX_BASE_URL=https://arax.transltr.io/api/arax/v1.4
# NODE_NORM_BASE_URL=https://nodenormalization-sri.renci.org

# OpenAI Configuration
# API key for OpenAI commercial models (GPT-5 nano, GPT-5-mini, etc.)
OPENAI_API_KEY=your_openai_api_key_here

# Available OpenAI models (comma-separated list)
AVAILABLE_OPENAI_MODELS=gpt-5-nano,gpt-5-mini

# Enable OpenAI's built-in web_search tool (Optional)
# Default: false
OPENAI_ENABLE_WEB_SEARCH=false

# Batch Processing Configuration
# Maximum concurrent requests for batch processing (default: 5)
# Higher values = faster but more memory/CPU usage
MAX_CONCURRENT_REQUESTS=5
```

**Note**: UMLS integration is optional but recommended. It enhances entity name resolution by combining UMLS Terminology Services with Node Normalization. See [UMLS_INTEGRATION.md](UMLS_INTEGRATION.md) for details.

### 4. Run Checker

#### Web Interface (Recommended)

Launch the Streamlit web interface:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501` and use the interactive interface to:
- Enter research triples to check
- Input PMIDs manually or via file upload  
- Select AI models
- View results with charts and export options

#### Command Line Interface

**Basic Triples Using Entity Names:**
```bash
# Using local Ollama models
python main.py --val_model gpt-oss:20b --triple_name 'SIX1' 'affects' 'Cell Proliferation' --pmids 34513929 16488997
python main.py --val_model hermes4:70b-q4-m --triple_name 'SIX1' 'affects' 'Cell Proliferation' --pmids 34513929

# Using OpenAI cloud models
python main.py --val_model gpt-5-nano --triple_name 'SIX1' 'affects' 'Cell Proliferation' --pmids 34513929 16488997
python main.py --val_model gpt-5-mini --triple_name 'SIX1' 'affects' 'Cell Proliferation' --pmids 34513929
```

**Basic Triples Using CURIEs:**
```bash
# Using local Ollama models
python main.py --val_model gpt-oss:20b --triple_curie 'NCBIGene:6495' 'affects' 'UMLS:C0596290' --pmids 34513929 16488997
python main.py --val_model hermes4:70b-q4-s --triple_name 'SIX1' 'affects' 'Cell Proliferation' --pmids-file pmids.txt

# Using OpenAI cloud models
python main.py --val_model gpt-5-nano --triple_curie 'NCBIGene:6495' 'affects' 'UMLS:C0596290' --pmids 34513929 16488997
```

**Qualified Triples:**
```bash
# With all qualifiers
python main.py --val_model hermes4:70b-q4-m --triple_name 'SIX1' 'affects' 'Cell Proliferation' \
  --qualified_predicate 'causes' \
  --qualified_object_aspect 'activity' \
  --qualified_object_direction 'increased' \
  --pmids 34513929

# With only direction qualifier
python main.py --val_model gpt-oss:20b --triple_curie 'NCBIGene:6495' 'affects' 'UMLS:C0596290' \
  --qualified_predicate 'causes' \
  --qualified_object_direction 'upregulated' \
  --pmids 34513929

# With only aspect qualifier  
python main.py --val_model hermes4:70b-q4-m --triple_name 'SIX1' 'affects' 'Cell Proliferation' \
  --qualified_predicate 'causes' \
  --qualified_object_aspect 'activity_or_abundance' \
  --pmids 34513929
  
# With verification enabled
python main.py --val_model hermes4:70b-q4-m --checker_model gpt-oss:20b --triple_name 'SIX1' 'affects' 'Cell Proliferation' --pmids 34513929
python main.py --val_model gpt-5-nano --checker_model gpt-5-mini --triple_name 'SIX1' 'affects' 'Cell Proliferation' --pmids 34513929
```

Qualifiers include:
- **Qualified Predicate** (required): More specific relationship (e.g., `causes`)
- **Object Aspect** (optional): What aspect is affected (e.g., `activity`, `abundance`, `activity_or_abundance`, `localization`)
- **Object Direction** (optional): Direction of change (e.g., `increased`, `decreased`, `upregulated`, `downregulated`)

**Constraint**: If using qualifiers, `qualified_predicate` is required and at least one of `qualified_object_aspect` or `qualified_object_direction` must be provided.

## Model Selection and Verification

### Validation and Checker Models

The system supports two separate model selections:

1. **Validation Model** (`--val_model`): Model used to evaluate triples against abstracts
   - Local Ollama models: `hermes4:70b-iq4-xs`, `hermes4:70b-q4-s`, `hermes4:70b-q4-m`, `gpt-oss:20b`
   - OpenAI cloud models: `gpt-5-nano`, `gpt-5-mini`
   - This is the primary model that performs the evaluation

2. **Checker Model** (`--checker_model`): Model used to verify and correct the validation results
   - Optional: If not provided, verification is disabled
   - Can be the same as or different from the validation model
   - Can mix local and cloud models (e.g., validate with Hermes, verify with GPT-5)

**Example Combinations:**
```bash
# Use Hermes 4 (Q4_M) for validation, GPT-OSS for verification
python main.py --val_model hermes4:70b-q4-m --checker_model gpt-oss:20b ...

# Use GPT-5 models for both validation and verification
python main.py --val_model gpt-5-nano --checker_model gpt-5-mini ...

# Mix local and cloud: validate with OpenAI, verify with Ollama
python main.py --val_model gpt-5-nano --checker_model gpt-oss:20b ...

# Use Hermes 4 (IQ4_XS) without verification (omit --checker_model)
python main.py --val_model hermes4:70b-iq4-xs ...
```

## Available Models

### Local Ollama Models
- **Hermes 4 70B (IQ4_XS)**: Fastest Hermes variant with the lowest VRAM usage (~28 GB). Great for lighter GPUs.
- **Hermes 4 70B (Q4_S)**: Balanced quality/speed option requiring ~34 GB VRAM. Recommended default for most workflows.
- **Hermes 4 70B (Q4_M)**: Highest quality quantization requiring ~42 GB VRAM. Best accuracy when resources permit.
- **GPT-OSS 20B**: Open-source 20B parameter model (~12 GB VRAM). Good baseline and faster responses.

### OpenAI Cloud Models
- **GPT-5 Nano**: OpenAI's smallest GPT-5 model. Fast, cost-effective, suitable for high-volume tasks.
- **GPT-5 Mini**: OpenAI's mid-tier GPT-5 model. Balanced performance and cost for general use.

**Note**: OpenAI models require an API key and incur usage costs. See [OpenAI Pricing](https://openai.com/pricing) for details.
