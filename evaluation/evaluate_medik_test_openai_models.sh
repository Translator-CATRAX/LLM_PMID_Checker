#!/bin/bash

# Test evaluation script for OpenAI models - processes all rows from test_data.tsv
# This script evaluates triples using OpenAI models and tracks:
# - Accuracy metrics
# - Runtime per request
# - Token usage (input, output, cached tokens)


# Set base directory
BASE_DIR="/home/grads/cqm5886/work/llm_pmid_support"
INPUT_FILE="$BASE_DIR/datava/test_data.tsv"
OUTPUT_FILE="$BASE_DIR/evaluation/gpt_5_1_chat_latest_evaluation_results.tsv"
METRICS_FILE="$BASE_DIR/evaluation/gpt_5_1_chat_latest_evaluation_metrics.txt"
TIMING_FILE="$BASE_DIR/evaluation/gpt_5_1_chat_latest_timing_results.txt"
TOKENS_FILE="$BASE_DIR/evaluation/gpt_5_1_chat_latest_token_usage.txt"
VAL_MODEL="gpt-5.1-chat-latest"
CHECKER_MODEL="gpt-5.1-chat-latest"

echo "========================================"
echo "Running TEST evaluation with OpenAI models"
echo "Input: test_data.tsv"
echo "Model: $VAL_MODEL"
if [ -n "$CHECKER_MODEL" ]; then
    echo "Checker Model: $CHECKER_MODEL (enabled)"
else
    echo "Checker Model: disabled"
fi
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

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ] && ! grep -q "OPENAI_API_KEY" .env 2>/dev/null; then
    echo "Error: OPENAI_API_KEY not found in environment or .env file!"
    echo "Please set your OpenAI API key before running this script."
    exit 1
fi

# Create output file with header (with detailed columns)
echo -e "subject\tpredicate\tobject\tground_truth\tPMID\tis_supported_from_llm\tevidence_category\tsubject_mentioned\tobject_mentioned\tsupporting_sentence\treasoning\tsubject_curie\tobject_curie\truntime_seconds\tprompt_tokens\tcompletion_tokens\ttotal_tokens\tcached_tokens" > "$OUTPUT_FILE"

# Create timing file with header
echo -e "PMID\truntime_seconds" > "$TIMING_FILE"

# Create tokens file with header
echo -e "PMID\tprompt_tokens\tcompletion_tokens\ttotal_tokens\tcached_tokens\tprompt_tokens_uncached" > "$TOKENS_FILE"

# Counter for progress
counter=0
total=$(tail -n +2 "$INPUT_FILE" | wc -l)

echo "Processing $total test cases..."
echo ""

# Skip header line and process each row
tail -n +2 "$INPUT_FILE" | while IFS=$'\t' read -r subject predicate object pmid subject_curie object_curie supported; do
    # Skip empty lines
    if [ -z "$subject_curie" ] || [ -z "$predicate" ] || [ -z "$object_curie" ] || [ -z "$pmid" ]; then
        continue
    fi
    
    counter=$((counter + 1))
    
    # Map predicate to qualified parameters
    case "$predicate" in
        "stimulates")
            qualified_predicate="causes"
            qualified_object_aspect="activity_or_abundance"
            qualified_object_direction="increased"
            ;;
        "inhibits")
            qualified_predicate="causes"
            qualified_object_aspect="activity_or_abundance"
            qualified_object_direction="decreased"
            ;;
        "produces")
            qualified_predicate="causes"
            qualified_object_aspect="activity_or_abundance"
            qualified_object_direction="increased"
            ;;
        *)
            echo "Warning: Unknown predicate '$predicate' for subject '$subject', object '$object', PMID '$pmid'"
            qualified_predicate="causes"
            qualified_object_aspect="activity_or_abundance"
            qualified_object_direction="increased"
            ;;
    esac
    
    echo "[$counter/$total] Processing: $subject_curie | $predicate | $object_curie"
    echo "    PMID: $pmid | Ground truth: $supported"
    
    # Record start time
    start_time=$(date +%s.%N)
    
    # Build command with optional checker model
    cmd="python main.py --val_model \"$VAL_MODEL\" --triple_curie \"$subject_curie\" \"$predicate\" \"$object_curie\" --qualified_predicate \"$qualified_predicate\" --qualified_object_aspect \"$qualified_object_aspect\" --qualified_object_direction \"$qualified_object_direction\" --pmids \"$pmid\" --verbose"
    
    # Add checker model if specified
    if [ -n "$CHECKER_MODEL" ]; then
        cmd="$cmd --checker_model \"$CHECKER_MODEL\""
    fi
    
    # Run main.py and capture output
    # Save both stdout (reasoning) and stderr (logs) to separate files, then combine
    output=$(eval $cmd 2>&1 | tee /tmp/llm_output_$$.log)
    
    exit_code=$?
    
    # Record end time and calculate duration
    end_time=$(date +%s.%N)
    runtime=$(echo "$end_time - $start_time" | bc)
    
    # Extract token usage from logs (OpenAI client logs this info)
    prompt_tokens=$(echo "$output" | grep -oP "prompt_tokens=\K\d+" | tail -1)
    completion_tokens=$(echo "$output" | grep -oP "completion_tokens=\K\d+" | tail -1)
    total_tokens=$(echo "$output" | grep -oP "total_tokens=\K\d+" | tail -1)
    cached_tokens=$(echo "$output" | grep -oP "Prompt cache hit: \K\d+" | tail -1)
    
    # Set defaults if extraction failed
    prompt_tokens=${prompt_tokens:-0}
    completion_tokens=${completion_tokens:-0}
    total_tokens=${total_tokens:-0}
    cached_tokens=${cached_tokens:-0}
    
    # Calculate uncached prompt tokens
    prompt_tokens_uncached=$((prompt_tokens - cached_tokens))
    
    # Check if main.py ran successfully
    if [ $exit_code -ne 0 ]; then
        echo "    ERROR: main.py failed with exit code $exit_code"
        echo "    Runtime: ${runtime}s"
        # Write error row
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$subject" "$predicate" "$object" "$supported" "$pmid" \
            "Error" "Error" "No" "No" "" "" \
            "$subject_curie" "$object_curie" "$runtime" "0" "0" "0" "0" >> "$OUTPUT_FILE"
        printf '%s\t%s\n' "$pmid" "$runtime" >> "$TIMING_FILE"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$pmid" "0" "0" "0" "0" "0" >> "$TOKENS_FILE"
    else
        # Parse LLM output using Python parser (pipe via stdin to avoid shell escaping issues)
        parsed_json=$(echo "$output" | python "$BASE_DIR/evaluation/parse_llm_output.py" -)
        
        # Extract fields from JSON
        is_supported=$(echo "$parsed_json" | python -c "import sys, json; print(json.load(sys.stdin)['is_supported'])" 2>/dev/null || echo "Unknown")
        evidence_category=$(echo "$parsed_json" | python -c "import sys, json; print(json.load(sys.stdin)['evidence_category'])" 2>/dev/null || echo "Unknown")
        subject_mentioned=$(echo "$parsed_json" | python -c "import sys, json; print('Yes' if json.load(sys.stdin)['subject_mentioned'] else 'No')" 2>/dev/null || echo "No")
        object_mentioned=$(echo "$parsed_json" | python -c "import sys, json; print('Yes' if json.load(sys.stdin)['object_mentioned'] else 'No')" 2>/dev/null || echo "No")
        supporting_sentence=$(echo "$parsed_json" | python -c "import sys, json; print(json.load(sys.stdin)['supporting_sentence'])" 2>/dev/null || echo "")
        reasoning=$(echo "$parsed_json" | python -c "import sys, json; print(json.load(sys.stdin)['reasoning'])" 2>/dev/null || echo "")
        
        # Sanitize text fields: strip any remaining newlines and tabs
        supporting_sentence=$(echo "$supporting_sentence" | tr '\n\t' '  ')
        reasoning=$(echo "$reasoning" | tr '\n\t' '  ')
        
        # Convert Python bool to string
        if [ "$is_supported" == "True" ]; then
            predicted="True"
        elif [ "$is_supported" == "False" ]; then
            predicted="False"
        else
            predicted="Unknown"
        fi
        
        echo "    Prediction: $predicted"
        echo "    Evidence Category: $evidence_category"
        echo "    Subject: $subject_mentioned, Object: $object_mentioned"
        echo "    Runtime: ${runtime}s"
        echo "    Tokens: ${prompt_tokens} input (${cached_tokens} cached), ${completion_tokens} output, ${total_tokens} total"
        echo ""
        
        # Append detailed results to output file (use printf to avoid echo -e interpreting escape sequences)
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$subject" "$predicate" "$object" "$supported" "$pmid" \
            "$predicted" "$evidence_category" "$subject_mentioned" "$object_mentioned" \
            "$supporting_sentence" "$reasoning" \
            "$subject_curie" "$object_curie" "$runtime" \
            "$prompt_tokens" "$completion_tokens" "$total_tokens" "$cached_tokens" >> "$OUTPUT_FILE"
        printf '%s\t%s\n' "$pmid" "$runtime" >> "$TIMING_FILE"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$pmid" "$prompt_tokens" "$completion_tokens" "$total_tokens" "$cached_tokens" "$prompt_tokens_uncached" >> "$TOKENS_FILE"
    fi
    
    # Small delay to avoid rate limiting
    sleep 1
done

echo "========================================"
echo "Test evaluation complete!"
echo "========================================"
echo "Results saved to $OUTPUT_FILE"
echo ""
echo "Now calculating metrics..."
echo ""

# Calculate metrics using Python
python "$BASE_DIR/evaluation/calculate_metrics.py" "$OUTPUT_FILE" "$METRICS_FILE"

echo ""
echo "========================================"
echo "Accuracy Metrics"
echo "========================================"
cat "$METRICS_FILE"

echo ""
echo "========================================"
echo "Timing Statistics"
echo "========================================"
python "$BASE_DIR/evaluation/calculate_timing_stats.py" "$TIMING_FILE"

echo ""
echo "========================================"
echo "Token Usage Statistics"
echo "========================================"
python "$BASE_DIR/evaluation/calculate_token_stats.py" "$TOKENS_FILE"

echo ""
echo "========================================"
echo "Test complete!"
echo "========================================"

