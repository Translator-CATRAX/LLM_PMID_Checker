#!/usr/bin/env python3
"""Main interface for triple checking."""
import argparse
import asyncio
import logging
import sqlite3
import sys
import time
from pathlib import Path

import polars as pl

from src.triple_evaluator import TripleEvaluatorSystem
from src.node_dict_loader import NodeDictLoader
from src.config import settings

logger = logging.getLogger(__name__)

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

        qualifiers = PREDICATE_QUALIFIERS.get(predicate, DEFAULT_QUALIFIERS)

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
            qualified_predicate=qualifiers.get('qualified_predicate'),
            qualified_object_aspect=qualifiers.get('qualified_object_aspect'),
            qualified_object_direction=qualifiers.get('qualified_object_direction'),
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
                    max_concurrent: int = None):
    """Read a TSV, evaluate every row concurrently, and write to SQLite."""
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

    # ---- summary -----------------------------------------------------
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

    if max_concurrent:
        settings.max_concurrent_requests = original_max

    return 0


def _print_metrics(df: pl.DataFrame, total_rows: int):
    """Print accuracy / confusion matrix when ground truth is available."""
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
  python main.py --input data/test_data.tsv --output results.db --val_model gpt-oss-20b-vllm
  python main.py --input data/test_data.tsv --output results.tsv --val_model gpt-oss-20b-vllm
  python main.py --input data/test_data.tsv --output results.db --val_model hermes4-vllm --max_concurrent 16
  python main.py --input data/test_data.tsv --output results.db --val_model gpt-oss-20b-vllm --round2_model gpt-oss-120b-vllm
  python main.py --input data/test_data.tsv --output results.db --val_model gpt-oss-20b-vllm --table my_run_1
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
    ))
    sys.exit(rc)


if __name__ == "__main__":
    main()
