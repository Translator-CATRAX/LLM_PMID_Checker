#!/usr/bin/env python3
"""
Batch evaluation script for processing TSV files with concurrent evaluation.
This script processes multiple rows concurrently for faster evaluation.
"""
import asyncio
import pandas as pd
import numpy as np
import time
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.triple_evaluator import TripleEvaluatorSystem
from src.node_dict_loader import NodeDictLoader
from src.config import settings

PREDICATE_QUALIFIERS = {
    'stimulates': {
        'qualified_predicate': 'causes',
        'qualified_object_aspect': 'activity_or_abundance',
        'qualified_object_direction': 'increased',
    },
    'inhibits': {
        'qualified_predicate': 'causes',
        'qualified_object_aspect': 'activity_or_abundance',
        'qualified_object_direction': 'decreased',
    },
    'produces': {
        'qualified_predicate': 'causes',
        'qualified_object_aspect': 'activity_or_abundance',
        'qualified_object_direction': 'increased',
    },
}

DEFAULT_QUALIFIERS = {
    'qualified_predicate': 'causes',
    'qualified_object_aspect': 'activity_or_abundance',
    'qualified_object_direction': 'increased',
}


async def evaluate_single_row(evaluator, row, row_idx, total_rows, node_dict=None):
    """Evaluate a single row from the TSV file."""
    try:
        subject = row['subject']
        predicate = row['predicate']
        object_ = row['object']
        pmid = str(row['PMID'])
        subject_curie = row['subject_curie']
        object_curie = row['object_curie']
        ground_truth = row.get('ground_truth', None)

        qualifiers = PREDICATE_QUALIFIERS.get(predicate, DEFAULT_QUALIFIERS)

        print(f"[{row_idx+1}/{total_rows}] Processing: {subject_curie} | {predicate} | {object_curie}")
        print(f"    PMID: {pmid} | Ground truth: {ground_truth}")

        subject_info = node_dict.get_node_info(subject_curie) if node_dict else None
        object_info = node_dict.get_node_info(object_curie) if node_dict else None

        start_time = time.time()

        result = await evaluator.evaluate_triple_with_names(
            subject=subject,
            predicate=predicate,
            object_=object_,
            subject_names=[subject],
            object_names=[object_],
            pmids=[pmid],
            subject_info=subject_info,
            object_info=object_info,
            qualified_predicate=qualifiers.get('qualified_predicate'),
            qualified_object_aspect=qualifiers.get('qualified_object_aspect'),
            qualified_object_direction=qualifiers.get('qualified_object_direction'),
        )

        runtime = time.time() - start_time

        if result.evaluations:
            ev = result.evaluations[0]
            sentences_str = " | ".join(ev.supporting_sentences) if ev.supporting_sentences else ""
            return {
                'subject': subject,
                'predicate': predicate,
                'object': object_,
                'ground_truth': ground_truth,
                'PMID': pmid,
                'predicted': ev.is_supported,
                'support': ev.support,
                'subject_mentioned': 'Yes' if ev.subject_mentioned else 'No',
                'object_mentioned': 'Yes' if ev.object_mentioned else 'No',
                'supporting_sentences': sentences_str,
                'reasoning': ev.reasoning or '',
                'subject_curie': subject_curie,
                'object_curie': object_curie,
                'runtime_seconds': runtime,
            }
        else:
            return {
                'subject': subject,
                'predicate': predicate,
                'object': object_,
                'ground_truth': ground_truth,
                'PMID': pmid,
                'predicted': False,
                'support': 'no',
                'subject_mentioned': 'No',
                'object_mentioned': 'No',
                'supporting_sentences': '',
                'reasoning': 'No evaluation result',
                'subject_curie': subject_curie,
                'object_curie': object_curie,
                'runtime_seconds': runtime,
            }

    except Exception as e:
        print(f"    ERROR: {str(e)}")
        return {
            'subject': row.get('subject', ''),
            'predicate': row.get('predicate', ''),
            'object': row.get('object', ''),
            'ground_truth': row.get('ground_truth', ''),
            'PMID': str(row.get('PMID', '')),
            'predicted': False,
            'support': 'no',
            'subject_mentioned': 'No',
            'object_mentioned': 'No',
            'supporting_sentences': '',
            'reasoning': f'Error: {str(e)}',
            'subject_curie': row.get('subject_curie', ''),
            'object_curie': row.get('object_curie', ''),
            'runtime_seconds': 0.0,
        }


def save_timing_log(results_df, output_file, total_runtime, total_rows, concurrency, model_name):
    """Save a comprehensive timing log alongside the results file."""
    output_path = Path(output_file)
    base_name = output_path.stem.replace("_evaluation_results", "").replace("_results", "")
    eval_dir = output_path.parent

    timing_tsv_path = eval_dir / f"{base_name}_timing_results.txt"
    if 'runtime_seconds' in results_df.columns:
        timing_df = results_df[['PMID', 'runtime_seconds']].copy()
        timing_df.to_csv(timing_tsv_path, sep='\t', index=False)
        print(f"Per-row timing saved to: {timing_tsv_path}")

    timing_summary_path = eval_dir / f"{base_name}_timing_summary.txt"

    runtimes = results_df['runtime_seconds'].values if 'runtime_seconds' in results_df.columns else np.array([])

    if len(runtimes) > 0:
        mean_request = np.mean(runtimes)
        std_request = np.std(runtimes, ddof=1) if len(runtimes) > 1 else 0.0
        median_request = np.median(runtimes)
        min_request = np.min(runtimes)
        max_request = np.max(runtimes)
        sum_request = np.sum(runtimes)
        p25_request = np.percentile(runtimes, 25)
        p75_request = np.percentile(runtimes, 75)
        p95_request = np.percentile(runtimes, 95)
    else:
        mean_request = std_request = median_request = 0.0
        min_request = max_request = sum_request = 0.0
        p25_request = p75_request = p95_request = 0.0

    wall_clock_per_row = total_runtime / total_rows if total_rows > 0 else 0.0
    effective_throughput = total_rows / total_runtime if total_runtime > 0 else 0.0

    summary_lines = [
        "=" * 60,
        "  Timing Summary",
        "=" * 60,
        "",
        f"Timestamp:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model:           {model_name}",
        f"Total rows:      {total_rows}",
        f"Concurrency:     {concurrency}",
        "",
        "--- Wall-Clock Performance ---",
        f"Total wall-clock time:       {total_runtime:>10.2f} s  ({total_runtime/60:.2f} min)",
        f"Avg wall-clock time per row:  {wall_clock_per_row:>10.2f} s",
        f"Effective throughput:          {effective_throughput:>10.2f} rows/s",
        "",
        "--- Per-Request (Individual) Timing ---",
        f"Sum of all request times:    {sum_request:>10.2f} s  ({sum_request/60:.2f} min)",
        f"Mean:                        {mean_request:>10.2f} s",
        f"Std Dev:                     {std_request:>10.2f} s",
        f"Median:                      {median_request:>10.2f} s",
        f"Min:                         {min_request:>10.2f} s",
        f"Max:                         {max_request:>10.2f} s",
        f"25th percentile:             {p25_request:>10.2f} s",
        f"75th percentile:             {p75_request:>10.2f} s",
        f"95th percentile:             {p95_request:>10.2f} s",
        "",
        "--- Estimated Effective Per-Row Throughput ---",
        f"Approx mean (request_mean / concurrency):  {mean_request / concurrency:.2f} s"
        if concurrency > 0 else "Approx mean: N/A (concurrency=0)",
        f"Approx std  (request_std  / concurrency):  {std_request  / concurrency:.2f} s"
        if concurrency > 0 else "Approx std:  N/A (concurrency=0)",
        "",
        "=" * 60,
    ]

    summary_text = "\n".join(summary_lines) + "\n"
    with open(timing_summary_path, 'w') as f:
        f.write(summary_text)

    timing_json_path = eval_dir / f"{base_name}_timing_summary.json"
    timing_data = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "total_rows": total_rows,
        "concurrency": concurrency,
        "wall_clock": {
            "total_seconds": round(total_runtime, 4),
            "avg_per_row_seconds": round(wall_clock_per_row, 4),
            "effective_throughput_rows_per_sec": round(effective_throughput, 4),
        },
        "per_request": {
            "sum_seconds": round(float(sum_request), 4),
            "mean_seconds": round(float(mean_request), 4),
            "std_seconds": round(float(std_request), 4),
            "median_seconds": round(float(median_request), 4),
            "min_seconds": round(float(min_request), 4),
            "max_seconds": round(float(max_request), 4),
            "p25_seconds": round(float(p25_request), 4),
            "p75_seconds": round(float(p75_request), 4),
            "p95_seconds": round(float(p95_request), 4),
        },
        "effective_throughput_estimate": {
            "approx_mean_per_row_seconds": round(mean_request / concurrency, 4) if concurrency > 0 else None,
            "approx_std_per_row_seconds": round(std_request / concurrency, 4) if concurrency > 0 else None,
        },
    }
    with open(timing_json_path, 'w') as f:
        json.dump(timing_data, f, indent=2)

    print(f"Timing summary saved to: {timing_summary_path}")
    print(f"Timing JSON saved to:    {timing_json_path}")


async def evaluate_batch(input_file, output_file, val_model, round2_model=None,
                         node_dict_path=None, max_concurrent=None):
    """Evaluate all rows in the TSV file with concurrent processing."""

    if max_concurrent:
        original_max = settings.max_concurrent_requests
        settings.max_concurrent_requests = max_concurrent
        print(f"Using max concurrent requests: {settings.max_concurrent_requests}")
    else:
        print(f"Using default max concurrent requests: {settings.max_concurrent_requests}")

    print(f"Reading input file: {input_file}")
    df = pd.read_csv(input_file, sep='\t')

    if 'Supported' in df.columns and 'ground_truth' not in df.columns:
        df = df.rename(columns={'Supported': 'ground_truth'})

    df = df.dropna(subset=['subject_curie', 'object_curie', 'PMID', 'predicate'])

    total_rows = len(df)
    print(f"Total rows to process: {total_rows}")
    print(f"Model: {val_model}")
    if round2_model:
        print(f"Round 2 model: {round2_model}")
    else:
        print("Round 2 model: disabled")

    # Load node_dict if provided
    node_dict = None
    if node_dict_path:
        all_curies = set(df['subject_curie'].dropna().unique()) | set(df['object_curie'].dropna().unique())
        print(f"Loading node_dict for {len(all_curies)} unique CURIEs from: {node_dict_path}")
        node_dict = NodeDictLoader.from_file(node_dict_path, target_curies=all_curies)
        print(f"  Loaded info for {len(node_dict)} CURIEs")
    
    print("=" * 60)

    evaluator = TripleEvaluatorSystem(
        llm_provider=val_model,
        round2_model=round2_model,
    )

    semaphore = asyncio.Semaphore(settings.max_concurrent_requests)

    async def evaluate_with_semaphore(row, idx):
        async with semaphore:
            return await evaluate_single_row(evaluator, row, idx, total_rows, node_dict)

    overall_start_time = time.time()

    tasks = [
        evaluate_with_semaphore(row, idx)
        for idx, row in df.iterrows()
    ]

    print(f"\nStarting batch evaluation with {settings.max_concurrent_requests} concurrent workers...")
    print("=" * 60)
    results = await asyncio.gather(*tasks)

    total_runtime = time.time() - overall_start_time

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, sep='\t', index=False)

    concurrency = settings.max_concurrent_requests
    save_timing_log(results_df, output_file, total_runtime, total_rows, concurrency, val_model)

    print("\n" + "=" * 60)
    print("Batch evaluation complete!")
    print("=" * 60)
    print(f"Total time: {total_runtime:.2f} seconds")
    print(f"Average time per row (wall-clock): {total_runtime / total_rows:.2f} seconds")
    print(f"Concurrency: {concurrency}")
    print(f"Results saved to: {output_file}")

    if 'ground_truth' in results_df.columns and 'predicted' in results_df.columns:
        results_df['ground_truth_bool'] = results_df['ground_truth'].astype(str).str.lower().isin(['true', '1', 'yes'])
        results_df['predicted_bool'] = results_df['predicted'].astype(str).str.lower().isin(['true', '1', 'yes'])

        correct = (results_df['ground_truth_bool'] == results_df['predicted_bool']).sum()
        accuracy = correct / total_rows * 100

        print(f"\nAccuracy: {correct}/{total_rows} ({accuracy:.1f}%)")

        tp = ((results_df['ground_truth_bool']) & (results_df['predicted_bool'])).sum()
        tn = ((~results_df['ground_truth_bool']) & (~results_df['predicted_bool'])).sum()
        fp = ((~results_df['ground_truth_bool']) & (results_df['predicted_bool'])).sum()
        fn = ((results_df['ground_truth_bool']) & (~results_df['predicted_bool'])).sum()

        print(f"\nConfusion Matrix:")
        print(f"  True Positives:  {tp}")
        print(f"  True Negatives:  {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        print(f"\nMetrics:")
        print(f"  Precision:   {precision:.3f}")
        print(f"  Recall:      {recall:.3f}")
        print(f"  Specificity: {specificity:.3f}")
        print(f"  F1 Score:    {f1:.3f}")

        if 'support' in results_df.columns:
            print(f"\nSupport distribution:")
            for val in ['yes', 'maybe', 'no']:
                count = (results_df['support'] == val).sum()
                print(f"  {val}: {count}")

    if max_concurrent:
        settings.max_concurrent_requests = original_max

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluation script for processing TSV files with concurrent evaluation"
    )
    parser.add_argument('--input', required=True, help='Input TSV file')
    parser.add_argument('--output', required=True, help='Output TSV file for results')
    parser.add_argument('--val_model', default=settings.default_model,
                        help=f'Validation model (default: {settings.default_model})')
    parser.add_argument('--round2_model', default=None,
                        help='Optional Round 2 model for re-evaluation of yes/maybe results')
    parser.add_argument('--node_dict', default=None,
                        help='Path to KG2 nodes file (.jsonl.gz) or pre-built dict (.json/.json.gz) '
                             'for enriching prompts with entity context')
    parser.add_argument('--max_concurrent', type=int, default=None,
                        help=f'Maximum concurrent requests (default: {settings.max_concurrent_requests})')

    args = parser.parse_args()

    asyncio.run(evaluate_batch(
        input_file=args.input,
        output_file=args.output,
        val_model=args.val_model,
        round2_model=args.round2_model,
        node_dict_path=args.node_dict,
        max_concurrent=args.max_concurrent,
    ))


if __name__ == "__main__":
    main()
