#!/usr/bin/env python3
"""
Batch evaluation script for processing TSV files with concurrent evaluation.
This script processes multiple rows concurrently for faster evaluation.
"""
import asyncio
import pandas as pd
import time
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.triple_evaluator import TripleEvaluatorSystem
from src.config import settings


async def evaluate_single_row(evaluator, row, row_idx, total_rows):
    """Evaluate a single row from the TSV file."""
    try:
        # Extract data from row
        subject = row['subject']
        predicate = row['predicate']
        object_ = row['object']
        pmid = str(row['PMID'])
        subject_curie = row['subject_curie']
        object_curie = row['object_curie']
        
        # Get ground truth (column should be renamed by this point)
        ground_truth = row.get('ground_truth', None)
        
        # Map predicate to qualified parameters
        predicate_mapping = {
            'stimulates': {
                'qualified_predicate': 'causes',
                'qualified_object_aspect': 'activity_or_abundance',
                'qualified_object_direction': 'increased'
            },
            'inhibits': {
                'qualified_predicate': 'causes',
                'qualified_object_aspect': 'activity_or_abundance',
                'qualified_object_direction': 'decreased'
            },
            'produces': {
                'qualified_predicate': 'causes',
                'qualified_object_aspect': 'activity_or_abundance',
                'qualified_object_direction': 'increased'
            }
        }
        
        qualifiers = predicate_mapping.get(predicate, {
            'qualified_predicate': 'causes',
            'qualified_object_aspect': 'activity_or_abundance',
            'qualified_object_direction': 'increased'
        })
        
        print(f"[{row_idx+1}/{total_rows}] Processing: {subject_curie} | {predicate} | {object_curie}")
        print(f"    PMID: {pmid} | Ground truth: {ground_truth}")
        
        # Record start time
        start_time = time.time()
        
        # Evaluate triple with PMID
        result = await evaluator.evaluate_triple_with_names(
            subject=subject,
            predicate=predicate,
            object_=object_,
            subject_names=[subject],  # Will be normalized by evaluator
            object_names=[object_],   # Will be normalized by evaluator
            pmids=[pmid],
            qualified_predicate=qualifiers.get('qualified_predicate'),
            qualified_object_aspect=qualifiers.get('qualified_object_aspect'),
            qualified_object_direction=qualifiers.get('qualified_object_direction')
        )
        
        # Calculate runtime
        runtime = time.time() - start_time
        
        # Extract evaluation result
        if result.evaluations:
            eval_result = result.evaluations[0]
            
            # Map is_supported based on evidence_category
            is_supported = eval_result.evidence_category == "direct_support"
            
            return {
                'subject': subject,
                'predicate': predicate,
                'object': object_,
                'ground_truth': ground_truth,
                'PMID': pmid,
                'is_supported_from_llm': is_supported,
                'evidence_category': eval_result.evidence_category,
                'subject_mentioned': 'Yes' if eval_result.subject_mentioned else 'No',
                'object_mentioned': 'Yes' if eval_result.object_mentioned else 'No',
                'supporting_sentence': eval_result.supporting_sentence or '',
                'reasoning': eval_result.reasoning or '',
                'subject_curie': subject_curie,
                'object_curie': object_curie,
                'runtime_seconds': runtime
            }
        else:
            return {
                'subject': subject,
                'predicate': predicate,
                'object': object_,
                'ground_truth': ground_truth,
                'PMID': pmid,
                'is_supported_from_llm': False,
                'evidence_category': 'Error',
                'subject_mentioned': 'No',
                'object_mentioned': 'No',
                'supporting_sentence': '',
                'reasoning': 'No evaluation result',
                'subject_curie': subject_curie,
                'object_curie': object_curie,
                'runtime_seconds': runtime
            }
            
    except Exception as e:
        print(f"    ERROR: {str(e)}")
        return {
            'subject': row.get('subject', ''),
            'predicate': row.get('predicate', ''),
            'object': row.get('object', ''),
            'ground_truth': row.get('ground_truth', ''),
            'PMID': str(row.get('PMID', '')),
            'is_supported_from_llm': False,
            'evidence_category': 'Error',
            'subject_mentioned': 'No',
            'object_mentioned': 'No',
            'supporting_sentence': '',
            'reasoning': f'Error: {str(e)}',
            'subject_curie': row.get('subject_curie', ''),
            'object_curie': row.get('object_curie', ''),
            'runtime_seconds': 0.0
        }


async def evaluate_batch(input_file, output_file, val_model, checker_model=None, max_concurrent=None):
    """Evaluate all rows in the TSV file with concurrent processing."""
    
    # Override max_concurrent if specified
    if max_concurrent:
        original_max = settings.max_concurrent_requests
        settings.max_concurrent_requests = max_concurrent
        print(f"Using max concurrent requests: {settings.max_concurrent_requests}")
    else:
        print(f"Using default max concurrent requests: {settings.max_concurrent_requests}")
    
    # Read input TSV
    print(f"Reading input file: {input_file}")
    df = pd.read_csv(input_file, sep='\t')
    
    # Rename 'Supported' to 'ground_truth' for consistency if it exists
    if 'Supported' in df.columns and 'ground_truth' not in df.columns:
        df = df.rename(columns={'Supported': 'ground_truth'})
        print("Note: Renamed 'Supported' column to 'ground_truth' for consistency")
    
    # Filter out rows with empty CURIEs or PMIDs
    df = df.dropna(subset=['subject_curie', 'object_curie', 'PMID', 'predicate'])
    
    total_rows = len(df)
    print(f"Total rows to process: {total_rows}")
    print(f"Model: {val_model}")
    if checker_model:
        print(f"Checker model: {checker_model}")
    else:
        print("Checker model: disabled")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = TripleEvaluatorSystem(
        llm_provider=val_model,
        checker_model=checker_model
    )
    
    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
    
    async def evaluate_with_semaphore(row, idx):
        """Evaluate with semaphore rate limiting."""
        async with semaphore:
            return await evaluate_single_row(evaluator, row, idx, total_rows)
    
    # Start evaluation
    overall_start_time = time.time()
    
    # Create tasks for all rows
    tasks = [
        evaluate_with_semaphore(row, idx)
        for idx, row in df.iterrows()
    ]
    
    # Execute all tasks concurrently
    print(f"\nStarting batch evaluation with {settings.max_concurrent_requests} concurrent workers...")
    print("=" * 60)
    results = await asyncio.gather(*tasks)
    
    # Calculate total runtime
    total_runtime = time.time() - overall_start_time
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(output_file, sep='\t', index=False)
    
    print("\n" + "=" * 60)
    print("Batch evaluation complete!")
    print("=" * 60)
    print(f"Total time: {total_runtime:.2f} seconds")
    print(f"Average time per row: {total_runtime / total_rows:.2f} seconds")
    print(f"Results saved to: {output_file}")
    
    # Calculate metrics
    if 'ground_truth' in results_df.columns and 'is_supported_from_llm' in results_df.columns:
        # Convert ground_truth to boolean
        results_df['ground_truth_bool'] = results_df['ground_truth'].astype(str).str.lower().isin(['true', '1', 'yes'])
        results_df['predicted_bool'] = results_df['is_supported_from_llm'].astype(bool)
        
        # Calculate accuracy
        correct = (results_df['ground_truth_bool'] == results_df['predicted_bool']).sum()
        accuracy = correct / total_rows * 100
        
        print(f"\nAccuracy: {correct}/{total_rows} ({accuracy:.1f}%)")
        
        # Calculate confusion matrix
        tp = ((results_df['ground_truth_bool'] == True) & (results_df['predicted_bool'] == True)).sum()
        tn = ((results_df['ground_truth_bool'] == False) & (results_df['predicted_bool'] == False)).sum()
        fp = ((results_df['ground_truth_bool'] == False) & (results_df['predicted_bool'] == True)).sum()
        fn = ((results_df['ground_truth_bool'] == True) & (results_df['predicted_bool'] == False)).sum()
        
        print(f"\nConfusion Matrix:")
        print(f"  True Positives:  {tp}")
        print(f"  True Negatives:  {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        
        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nMetrics:")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1 Score:  {f1:.3f}")
    
    # Restore original max_concurrent if it was changed
    if max_concurrent:
        settings.max_concurrent_requests = original_max
    
    return results_df


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Batch evaluation script for processing TSV files with concurrent evaluation"
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Input TSV file (e.g., test_50_rows.tsv)'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output TSV file for results'
    )
    parser.add_argument(
        '--val_model',
        default=settings.default_model,
        help=f'Validation model (default: {settings.default_model})'
    )
    parser.add_argument(
        '--checker_model',
        default=None,
        help='Optional checker model for verification'
    )
    parser.add_argument(
        '--max_concurrent',
        type=int,
        default=None,
        help=f'Maximum concurrent requests (default: {settings.max_concurrent_requests})'
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    asyncio.run(evaluate_batch(
        input_file=args.input,
        output_file=args.output,
        val_model=args.val_model,
        checker_model=args.checker_model,
        max_concurrent=args.max_concurrent
    ))


if __name__ == "__main__":
    main()






