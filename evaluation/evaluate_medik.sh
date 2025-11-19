#!/bin/bash

# Bash script to evaluate processed_mediK_results_v2.tsv using main.py
# This script processes each row and runs main.py with appropriate parameters
#
# Note: Uses Ollama server which should be running on GPU:3 via setup_ollama.sh
# GPU is configured in setup_ollama.sh with CUDA_VISIBLE_DEVICES=3

# Set base directory
BASE_DIR="/home/grads/cqm5886/work/llm_pmid_support"
INPUT_FILE="$BASE_DIR/data/andy_team_data/processed_mediK_results_v2.tsv"
OUTPUT_FILE="$BASE_DIR/data/andy_team_data/evaluation_results.tsv"
METRICS_FILE="$BASE_DIR/data/andy_team_data/evaluation_metrics.txt"
VAL_MODEL="hermes4:70b-q4-m"
CHECKER_MODEL="hermes4:70b-q4-m"

# Change to base directory
cd "$BASE_DIR"

# Activate conda environment if available
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate llm_pmid_env 2>/dev/null || echo "Note: Could not activate llm_pmid_env, using current environment"
fi

# Create output file with header (with detailed columns)
echo -e "subject\tpredicate\tobject\tground_truth\tPMID\tis_supported_from_llm\tevidence_category\tsubject_mentioned\tobject_mentioned\tsupporting_sentence\treasoning\tsubject_curie\tobject_curie" > "$OUTPUT_FILE"

# Skip header line and process each row  
# Main input file has 7 fields: subject, predicate, object, supported, pmid, subject_curie, object_curie
tail -n +2 "$INPUT_FILE" | while IFS=$'\t' read -r subject predicate object supported pmid subject_curie object_curie; do
    # Skip empty lines
    if [ -z "$subject" ] || [ -z "$predicate" ] || [ -z "$object" ] || [ -z "$pmid" ]; then
        continue
    fi
    
    # Remove [Chemical/Ingredient] and other type annotations from subject and object
    subject="${subject// \[Chemical\/Ingredient\]/}"
    object="${object// \[Chemical\/Ingredient\]/}"
    
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
    
    echo "Processing: $subject | $predicate | $object | PMID: $pmid | Ground truth: $supported"
    
    # Run main.py and capture output (with --verbose to get reasoning)
    output=$(python main.py \
        --val_model "$VAL_MODEL" \
        --checker_model "$CHECKER_MODEL" \
        --triple_name "$subject" "$predicate" "$object" \
        --qualified_predicate "$qualified_predicate" \
        --qualified_object_aspect "$qualified_object_aspect" \
        --qualified_object_direction "$qualified_object_direction" \
        --pmids "$pmid" \
        --verbose \
        2>&1)
    
    exit_code=$?
    
    # Check if main.py ran successfully
    if [ $exit_code -ne 0 ]; then
        echo "ERROR: main.py failed with exit code $exit_code"
        # Write error row
        echo -e "$subject\t$predicate\t$object\t$supported\t$pmid\tError\tError\tNo\tNo\t\t\t$subject_curie\t$object_curie" >> "$OUTPUT_FILE"
    else
        # Parse LLM output using Python parser
        parsed_json=$(python "$BASE_DIR/evaluation/parse_llm_output.py" "$output")
        
        # Extract fields from JSON (sanitization is done in parse_llm_output.py)
        is_supported=$(echo "$parsed_json" | python -c "import sys, json; print(json.load(sys.stdin)['is_supported'])" 2>/dev/null || echo "Unknown")
        evidence_category=$(echo "$parsed_json" | python -c "import sys, json; print(json.load(sys.stdin)['evidence_category'])" 2>/dev/null || echo "Unknown")
        subject_mentioned=$(echo "$parsed_json" | python -c "import sys, json; print('Yes' if json.load(sys.stdin)['subject_mentioned'] else 'No')" 2>/dev/null || echo "No")
        object_mentioned=$(echo "$parsed_json" | python -c "import sys, json; print('Yes' if json.load(sys.stdin)['object_mentioned'] else 'No')" 2>/dev/null || echo "No")
        supporting_sentence=$(echo "$parsed_json" | python -c "import sys, json; print(json.load(sys.stdin)['supporting_sentence'])" 2>/dev/null || echo "")
        reasoning=$(echo "$parsed_json" | python -c "import sys, json; print(json.load(sys.stdin)['reasoning'])" 2>/dev/null || echo "")
        
        # Convert Python bool to string
        if [ "$is_supported" == "True" ]; then
            predicted="True"
        elif [ "$is_supported" == "False" ]; then
            predicted="False"
        else
            predicted="Unknown"
        fi
        
        echo "Prediction: $predicted | Category: $evidence_category"
        
        # Append detailed results to output file
        echo -e "$subject\t$predicate\t$object\t$supported\t$pmid\t$predicted\t$evidence_category\t$subject_mentioned\t$object_mentioned\t$supporting_sentence\t$reasoning\t$subject_curie\t$object_curie" >> "$OUTPUT_FILE"
    fi
    
    echo "---"
    
    # Small delay to avoid overwhelming the system
    sleep 2
done

echo "Evaluation complete. Results saved to $OUTPUT_FILE"
echo "Now calculating metrics..."

# Calculate metrics using Python
python "$BASE_DIR/evaluation/calculate_metrics.py" "$OUTPUT_FILE" "$METRICS_FILE"

echo "Metrics saved to $METRICS_FILE"
cat "$METRICS_FILE"

