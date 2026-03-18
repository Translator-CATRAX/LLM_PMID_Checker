#!/usr/bin/env python3
"""Main interface for triple checking."""
import argparse
import asyncio
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from src.triple_evaluator import TripleEvaluatorSystem
from src.node_dict_loader import NodeDictLoader
from src.config import settings
from src import prompt_builder

logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    'predicted', 'support', 'subject_mentioned', 'object_mentioned',
    'supporting_sentences', 'reasoning', 'runtime_seconds',
]

POLARS_TO_SQLITE = {
    pl.Boolean: "INTEGER",
    pl.Int8: "INTEGER", pl.Int16: "INTEGER", pl.Int32: "INTEGER", pl.Int64: "INTEGER",
    pl.UInt8: "INTEGER", pl.UInt16: "INTEGER", pl.UInt32: "INTEGER", pl.UInt64: "INTEGER",
    pl.Float32: "REAL", pl.Float64: "REAL",
}

DB_EXTENSIONS = {'.db', '.sqlite', '.sqlite3'}
TSV_EXTENSIONS = {'.tsv', '.txt'}


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _detect_output_format(output_path: str) -> str:
    """Return 'db' or 'tsv' based on the file extension."""
    ext = Path(output_path).suffix.lower()
    if ext in DB_EXTENSIONS:
        return 'db'
    if ext in TSV_EXTENSIONS:
        return 'tsv'
    return 'db'


def _write_to_sqlite(df: pl.DataFrame, db_path: str, table_name: str) -> None:
    """Write a Polars DataFrame to a SQLite table (replace if exists)."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")

    col_defs = []
    for name, dtype in zip(df.columns, df.dtypes):
        sql_type = POLARS_TO_SQLITE.get(dtype, "TEXT")
        col_defs.append(f'"{name}" {sql_type}')

    conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    conn.execute(f'CREATE TABLE "{table_name}" ({", ".join(col_defs)})')

    placeholders = ", ".join("?" for _ in df.columns)
    conn.executemany(
        f'INSERT INTO "{table_name}" VALUES ({placeholders})',
        df.rows(),
    )
    conn.commit()
    conn.close()


def _write_to_tsv(df: pl.DataFrame, output_path: str) -> None:
    """Write a Polars DataFrame to a TSV file."""
    df.write_csv(output_path, separator='\t')


# ---------------------------------------------------------------------------
# Companion result files (timing + metrics)
# ---------------------------------------------------------------------------

def _companion_paths(output_path: str) -> dict[str, Path]:
    """Derive companion file paths from the main output path.

    Naming convention (matches evaluation/results_max_concurrent* format):
        {stem}_metrics.json / .txt          – classification metrics
        {base}_timing_results.txt           – per-row PMID runtimes
        {base}_timing_summary.json / .txt   – aggregate timing stats

    where *stem* is the output filename without extension and *base* is *stem*
    with a trailing ``_results`` removed (if present).
    """
    p = Path(output_path)
    parent = p.parent
    stem = p.stem                       # e.g. "gptoss_20b_r1_evaluation_only_results"

    base = stem.removesuffix("_results") if stem.endswith("_results") else stem

    return {
        "metrics_json":      parent / f"{stem}_metrics.json",
        "metrics_txt":       parent / f"{stem}_metrics.txt",
        "timing_results":    parent / f"{base}_timing_results.txt",
        "timing_json":       parent / f"{base}_timing_summary.json",
        "timing_txt":        parent / f"{base}_timing_summary.txt",
    }


def _write_timing_files(
    df: pl.DataFrame,
    total_runtime: float,
    concurrency: int,
    model: str,
    paths: dict[str, Path],
) -> None:
    """Write per-row timing TSV + aggregate timing summary (JSON + TXT)."""
    runtimes = df["runtime_seconds"].cast(pl.Float64).to_numpy()
    total_rows = len(runtimes)
    ts = datetime.now()

    # -- per-row timing TSV ------------------------------------------------
    pmids = df["PMID"].cast(pl.Utf8).to_list()
    with open(paths["timing_results"], "w") as fh:
        fh.write("PMID\truntime_seconds\n")
        for pmid, rt in zip(pmids, runtimes):
            fh.write(f"{pmid}\t{rt}\n")

    # -- aggregate stats ---------------------------------------------------
    rt_sum   = float(np.sum(runtimes))
    rt_mean  = float(np.mean(runtimes))
    rt_std   = float(np.std(runtimes, ddof=1)) if total_rows > 1 else 0.0
    rt_med   = float(np.median(runtimes))
    rt_min   = float(np.min(runtimes))
    rt_max   = float(np.max(runtimes))
    rt_p25   = float(np.percentile(runtimes, 25))
    rt_p75   = float(np.percentile(runtimes, 75))
    rt_p95   = float(np.percentile(runtimes, 95))
    throughput = total_rows / total_runtime if total_runtime else 0

    summary = {
        "timestamp": ts.isoformat(),
        "model": model,
        "total_rows": total_rows,
        "concurrency": concurrency,
        "wall_clock": {
            "total_seconds": round(total_runtime, 4),
            "avg_per_row_seconds": round(total_runtime / total_rows, 4) if total_rows else 0,
            "effective_throughput_rows_per_sec": round(throughput, 4),
        },
        "per_request": {
            "sum_seconds": round(rt_sum, 4),
            "mean_seconds": round(rt_mean, 4),
            "std_seconds": round(rt_std, 4),
            "median_seconds": round(rt_med, 4),
            "min_seconds": round(rt_min, 4),
            "max_seconds": round(rt_max, 4),
            "p25_seconds": round(rt_p25, 4),
            "p75_seconds": round(rt_p75, 4),
            "p95_seconds": round(rt_p95, 4),
        },
        "effective_throughput_estimate": {
            "approx_mean_per_row_seconds": round(rt_mean / concurrency, 4) if concurrency else 0,
            "approx_std_per_row_seconds": round(rt_std / concurrency, 4) if concurrency else 0,
        },
    }

    with open(paths["timing_json"], "w") as fh:
        json.dump(summary, fh, indent=2)

    wc = summary["wall_clock"]
    pr = summary["per_request"]
    et = summary["effective_throughput_estimate"]
    sep = "=" * 60
    lines = [
        sep,
        "  Timing Summary",
        sep,
        "",
        f"Timestamp:       {ts:%Y-%m-%d %H:%M:%S}",
        f"Model:           {model}",
        f"Total rows:      {total_rows}",
        f"Concurrency:     {concurrency}",
        "",
        "--- Wall-Clock Performance ---",
        f"Total wall-clock time:          {wc['total_seconds']:8.2f} s  ({wc['total_seconds']/60:.2f} min)",
        f"Avg wall-clock time per row:    {wc['avg_per_row_seconds']:8.2f} s",
        f"Effective throughput:            {wc['effective_throughput_rows_per_sec']:8.2f} rows/s",
        "",
        "--- Per-Request (Individual) Timing ---",
        f"Sum of all request times:       {pr['sum_seconds']:8.2f} s  ({pr['sum_seconds']/60:.2f} min)",
        f"Mean:                           {pr['mean_seconds']:8.2f} s",
        f"Std Dev:                        {pr['std_seconds']:8.2f} s",
        f"Median:                         {pr['median_seconds']:8.2f} s",
        f"Min:                            {pr['min_seconds']:8.2f} s",
        f"Max:                            {pr['max_seconds']:8.2f} s",
        f"25th percentile:                {pr['p25_seconds']:8.2f} s",
        f"75th percentile:                {pr['p75_seconds']:8.2f} s",
        f"95th percentile:                {pr['p95_seconds']:8.2f} s",
        "",
        "--- Estimated Effective Per-Row Throughput ---",
        f"Approx mean (request_mean / concurrency):  {et['approx_mean_per_row_seconds']:.2f} s",
        f"Approx std  (request_std  / concurrency):  {et['approx_std_per_row_seconds']:.2f} s",
        "",
        sep,
    ]
    with open(paths["timing_txt"], "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _write_metrics_files(
    df: pl.DataFrame,
    paths: dict[str, Path],
) -> None:
    """Compute classification metrics and write JSON + TXT reports."""
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score,
        average_precision_score, confusion_matrix, classification_report,
    )

    if "ground_truth" not in df.columns:
        return

    predicted_col = "predicted"
    gt_str  = df["ground_truth"].cast(pl.Utf8).str.strip_chars().str.to_lowercase()
    pred_str = df[predicted_col].cast(pl.Utf8).str.strip_chars().str.to_lowercase()

    gt_bin = gt_str.replace_strict(
        {"true": "1", "false": "0", "yes": "1", "no": "0"}, default=None
    ).cast(pl.Int8, strict=False)
    pred_bin = pred_str.replace_strict(
        {"true": "1", "false": "0", "yes": "1", "no": "0"}, default=None
    ).cast(pl.Int8, strict=False)

    mask = gt_bin.is_not_null() & pred_bin.is_not_null()
    gt_bin = gt_bin.filter(mask)
    pred_bin = pred_bin.filter(mask)

    if gt_bin.len() == 0:
        return

    y_true = gt_bin.to_numpy()
    y_pred = pred_bin.to_numpy()

    total_count = df.height
    valid_count = int(gt_bin.len())

    accuracy  = accuracy_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred, average="binary")

    try:
        auroc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auroc = None
    try:
        auprc = average_precision_score(y_true, y_pred)
    except ValueError:
        auprc = None

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    precision   = tp / (tp + fp) if (tp + fp) else 0
    recall      = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0

    # -- JSON --------------------------------------------------------------
    metrics_dict = {
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "auroc": float(auroc) if auroc is not None else None,
        "auprc": float(auprc) if auprc is not None else None,
        "total_samples": total_count,
        "valid_predictions": valid_count,
        "unknown_predictions": total_count - valid_count,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    with open(paths["metrics_json"], "w") as fh:
        json.dump(metrics_dict, fh, indent=2)

    # -- TXT ---------------------------------------------------------------
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    report = (
        "\n========================================\n"
        "Evaluation Metrics Report\n"
        "========================================\n\n"
        "Dataset Information:\n"
        "-------------------\n"
        f"Total samples: {total_count}\n"
        f"Valid predictions: {valid_count}\n"
        f"Unknown predictions: {total_count - valid_count}\n\n"
        "Class Distribution (Ground Truth):\n"
        "---------------------------------\n"
        f"True (Supported): {n_pos} ({n_pos/len(y_true)*100:.2f}%)\n"
        f"False (Not Supported): {n_neg} ({n_neg/len(y_true)*100:.2f}%)\n\n"
        "Performance Metrics:\n"
        "-------------------\n"
        f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\n"
        f"F1 Score:  {f1:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall:    {recall:.4f}\n"
        f"Specificity: {specificity:.4f}\n"
    )
    report += f"AUROC:     {auroc:.4f}\n" if auroc is not None else "AUROC:     N/A (requires varying prediction scores)\n"
    report += f"AUPRC:     {auprc:.4f}\n" if auprc is not None else "AUPRC:     N/A (requires varying prediction scores)\n"

    report += (
        f"\nConfusion Matrix:\n"
        "----------------\n"
        "                Predicted Negative  Predicted Positive\n"
        f"Actual Negative        {tn:6d}              {fp:6d}\n"
        f"Actual Positive        {fn:6d}              {tp:6d}\n\n"
        "Detailed Classification Report:\n"
        "------------------------------\n"
    )
    report += classification_report(y_true, y_pred,
                                    target_names=["Not Supported", "Supported"])
    report += "\n========================================\n"

    with open(paths["metrics_txt"], "w") as fh:
        fh.write(report)


# ---------------------------------------------------------------------------
# Row evaluation
# ---------------------------------------------------------------------------

async def evaluate_single_row(evaluator, row: dict, row_idx: int,
                              total_rows: int, node_dict=None):
    """Evaluate a single row and return a dict of output columns only."""
    try:
        subject_curie = row['subject_curie']
        predicate = row['predicate']
        object_curie = row['object_curie']
        pmid = str(row['PMID'])

        subject = row.get('subject') or subject_curie
        object_ = row.get('object') or object_curie

        print(f"[{row_idx + 1}/{total_rows}] {subject_curie} | {predicate} | {object_curie}  PMID:{pmid}")

        subject_info = node_dict.get_node_info(subject_curie) if node_dict else None
        object_info = node_dict.get_node_info(object_curie) if node_dict else None

        start = time.time()

        result = await evaluator.evaluate_triple_with_names(
            subject=subject,
            predicate=predicate,
            object_=object_,
            subject_names=[subject],
            object_names=[object_],
            pmids=[pmid],
            subject_info=subject_info,
            object_info=object_info,
        )

        runtime = time.time() - start

        if result.evaluations:
            ev = result.evaluations[0]
            sentences = " | ".join(ev.supporting_sentences) if ev.supporting_sentences else ""
            return {
                'predicted': ev.is_supported,
                'support': ev.support,
                'subject_mentioned': ev.subject_mentioned,
                'object_mentioned': ev.object_mentioned,
                'supporting_sentences': sentences,
                'reasoning': ev.reasoning or '',
                'runtime_seconds': runtime,
            }

        return _error_output('No evaluation result', 0.0)

    except Exception as e:
        print(f"    ERROR: {e}")
        return _error_output(f'Error: {e}', 0.0)


def _error_output(reasoning: str, runtime: float) -> dict:
    return {
        'predicted': False,
        'support': 'no',
        'subject_mentioned': False,
        'object_mentioned': False,
        'supporting_sentences': '',
        'reasoning': reasoning,
        'runtime_seconds': runtime,
    }


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------

async def run_batch(input_file: str, db_path: str, val_model: str, *,
                    table_name: str = "evaluations",
                    round2_model: str = None, node_dict_path: str = None,
                    max_concurrent: int = None,
                    predicate_file: str = None):
    """Read a TSV, evaluate every row concurrently, and write to SQLite."""
    if predicate_file:
        prompt_builder.load_predicate_descriptions(predicate_file)
    original_max = settings.max_concurrent_requests
    if max_concurrent:
        settings.max_concurrent_requests = max_concurrent

    concurrency = settings.max_concurrent_requests
    print(f"Max concurrent requests: {concurrency}")

    # ---- read input --------------------------------------------------
    df = pl.read_csv(input_file, separator='\t', infer_schema_length=0)

    required_cols = {'subject_curie', 'predicate', 'object_curie', 'PMID'}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Error: input TSV is missing required columns: {missing}",
              file=sys.stderr)
        return 1

    if 'Supported' in df.columns and 'ground_truth' not in df.columns:
        df = df.rename({'Supported': 'ground_truth'})

    df = df.drop_nulls(subset=list(required_cols))
    total_rows = df.height
    out_fmt = _detect_output_format(db_path)

    print(f"Input file:  {input_file}  ({total_rows} rows)")
    print(f"Model:       {val_model}")
    print(f"Round 2:     {round2_model or 'disabled'}")
    if out_fmt == 'db':
        print(f"Output DB:   {db_path}  (table: {table_name})")
    else:
        print(f"Output TSV:  {db_path}")

    # ---- optional node_dict ------------------------------------------
    node_dict = None
    if node_dict_path:
        all_curies = (
            set(df['subject_curie'].drop_nulls().unique().to_list())
            | set(df['object_curie'].drop_nulls().unique().to_list())
        )
        print(f"Loading node_dict for {len(all_curies)} CURIEs from: {node_dict_path}")
        node_dict = NodeDictLoader.from_file(node_dict_path, target_curies=all_curies)
        print(f"  Loaded info for {len(node_dict)} CURIEs")

    print("=" * 60)

    # ---- evaluate ----------------------------------------------------
    evaluator = TripleEvaluatorSystem(
        llm_provider=val_model,
        round2_model=round2_model,
    )
    semaphore = asyncio.Semaphore(concurrency)

    rows_as_dicts = df.to_dicts()

    async def _eval(row, idx):
        async with semaphore:
            return await evaluate_single_row(evaluator, row, idx, total_rows,
                                             node_dict)

    overall_start = time.time()
    tasks = [_eval(row, idx) for idx, row in enumerate(rows_as_dicts)]
    print(f"Starting batch evaluation with {concurrency} concurrent workers...")
    print("=" * 60)
    results = await asyncio.gather(*tasks)
    total_runtime = time.time() - overall_start

    # ---- merge input columns with output columns ---------------------
    output_df = pl.DataFrame(results)
    df = df.hstack(output_df.select(OUTPUT_COLUMNS))

    # ---- write output --------------------------------------------------
    if out_fmt == 'db':
        _write_to_sqlite(df, db_path, table_name)
    else:
        _write_to_tsv(df, db_path)

    # ---- companion files ------------------------------------------------
    cpaths = _companion_paths(db_path)

    _write_timing_files(df, total_runtime, concurrency, val_model, cpaths)
    _write_metrics_files(df, cpaths)

    # ---- console summary -------------------------------------------------
    print("\n" + "=" * 60)
    print("Batch evaluation complete!")
    print("=" * 60)
    print(f"Total time:    {total_runtime:.2f}s  ({total_runtime / 60:.1f} min)")
    print(f"Avg per row:   {total_runtime / total_rows:.2f}s")
    if out_fmt == 'db':
        print(f"Results in:    {db_path}  (table: {table_name})")
    else:
        print(f"Results in:    {db_path}")

    support_counts = dict(
        df['support'].value_counts()
        .sort('support')
        .iter_rows()
    )
    print(f"\nSupport distribution:")
    for val in ('yes', 'maybe', 'no'):
        print(f"  {val}: {support_counts.get(val, 0)}")

    if 'ground_truth' in df.columns:
        _print_metrics(df, total_rows)

    print(f"\nCompanion files:")
    print(f"  Timing results:  {cpaths['timing_results']}")
    print(f"  Timing summary:  {cpaths['timing_json']}")
    print(f"                   {cpaths['timing_txt']}")
    if cpaths["metrics_json"].exists():
        print(f"  Metrics:         {cpaths['metrics_json']}")
        print(f"                   {cpaths['metrics_txt']}")

    if max_concurrent:
        settings.max_concurrent_requests = original_max

    return 0


def _print_metrics(df: pl.DataFrame, total_rows: int):
    """Print accuracy / confusion matrix to console."""
    gt_bool = df['ground_truth'].cast(pl.Utf8).str.strip_chars().str.to_lowercase().is_in(['true', '1', 'yes'])
    pred_bool = df['predicted'].cast(pl.Utf8).str.strip_chars().str.to_lowercase().is_in(['true', '1', 'yes'])

    tp = int((gt_bool & pred_bool).sum())
    tn = int((~gt_bool & ~pred_bool).sum())
    fp = int((~gt_bool & pred_bool).sum())
    fn = int((gt_bool & ~pred_bool).sum())
    correct = tp + tn
    accuracy = correct / total_rows * 100 if total_rows else 0

    print(f"\nAccuracy: {correct}/{total_rows} ({accuracy:.1f}%)")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp}  TN: {tn}  FP: {fp}  FN: {fn}")

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0

    print(f"\n  Precision:   {precision:.3f}")
    print(f"  Recall:      {recall:.3f}")
    print(f"  Specificity: {specificity:.3f}")
    print(f"  F1 Score:    {f1:.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate research triples from a TSV file using vLLM, "
                    "saving results to a SQLite database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Required TSV columns:  subject_curie, predicate, object_curie, PMID
Optional TSV columns:  subject, object (entity names), ground_truth / Supported
                       (any extra columns are preserved in the output)

Output columns added:  predicted, support, subject_mentioned, object_mentioned,
                       supporting_sentences, reasoning, runtime_seconds

Output format is auto-detected from the --output file extension:
  .db / .sqlite / .sqlite3  →  SQLite database
  .tsv / .txt               →  Tab-separated values

Examples:
  python main.py --input data/test_data_biolink.tsv --output results.tsv --val_model gpt-oss-20b-vllm --predicate_file data/biolink_data/biolink_predicates.tsv --node_dict data/kg2_data/kg2c-2.10.2-v1.0-nodes.jsonl.gz
  python main.py --input data/test_data_biolink.tsv --output results.tsv --val_model gpt-oss-20b-vllm --predicate_file data/biolink_data/biolink_predicates.tsv --node_dict data/kg2_data/kg2c-2.10.2-v1.0-nodes.jsonl.gz
  python main.py --input data/test_data_biolink.tsv --output results.tsv --val_model gpt-oss-20b-vllm --predicate_file data/biolink_data/biolink_predicates.tsv --node_dict data/kg2_data/kg2c-2.10.2-v1.0-nodes.jsonl.gz --max_concurrent 16
  python main.py --input data/test_data_biolink.tsv --output results.tsv --val_model gpt-oss-20b-vllm --predicate_file data/biolink_data/biolink_predicates.tsv --node_dict data/kg2_data/kg2c-2.10.2-v1.0-nodes.jsonl.gz --round2_model gpt-oss-120b-vllm
  python main.py --input data/test_data_biolink.tsv --output results.db --val_model gpt-oss-20b-vllm --predicate_file data/biolink_data/biolink_predicates.tsv --node_dict data/kg2_data/kg2c-2.10.2-v1.0-nodes.jsonl.gz --table my_run_1
        """,
    )
    parser.add_argument('--input', required=True,
                        help='Input TSV file (must contain subject_curie, predicate, object_curie, PMID)')
    parser.add_argument('--output', required=True,
                        help='Output file. Use .db/.sqlite for SQLite or .tsv for TSV')
    parser.add_argument('--table', default='evaluations',
                        help='SQLite table name (only used for .db output; default: evaluations)')
    parser.add_argument('--val_model', default=settings.default_model,
                        help=f'Validation model (default: {settings.default_model})')
    parser.add_argument('--round2_model', default=None,
                        help='Optional Round 2 model for re-evaluation of yes/maybe results')
    parser.add_argument('--node_dict', default=None,
                        help='Path to KG2 nodes file (.jsonl.gz) or pre-built dict (.json/.json.gz)')
    parser.add_argument('--predicate_file', default=None,
                        help='Path to biolink predicates TSV (columns: predicate, description)')
    parser.add_argument('--max_concurrent', type=int, default=None,
                        help=f'Max concurrent requests (default: {settings.max_concurrent_requests})')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose (DEBUG) logging')

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    try:
        settings.validate_model(args.val_model)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.round2_model:
        try:
            settings.validate_model(args.round2_model)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    rc = asyncio.run(run_batch(
        input_file=args.input,
        db_path=args.output,
        val_model=args.val_model,
        table_name=args.table,
        round2_model=args.round2_model,
        node_dict_path=args.node_dict,
        max_concurrent=args.max_concurrent,
        predicate_file=args.predicate_file,
    ))
    sys.exit(rc)


if __name__ == "__main__":
    main()
