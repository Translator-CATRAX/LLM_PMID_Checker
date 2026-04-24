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
from src.pmid_cache import PMIDCache
from src.config import settings
from src import prompt_builder

logger = logging.getLogger(__name__)


def _normalize_pmid(raw: str) -> str:
    """Strip 'PMID:' prefix to get the bare numeric ID used by the cache."""
    raw = raw.strip()
    if raw.upper().startswith("PMID:"):
        return raw[5:]
    return raw


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
# Resume helpers
# ---------------------------------------------------------------------------

_ROW_KEY_COLS = ('subject_curie', 'predicate', 'object_curie', 'PMID')


def _row_key(row: dict) -> tuple:
    return tuple(str(row.get(c, '')) for c in _ROW_KEY_COLS)


def _load_completed_keys(output_path: str, out_fmt: str,
                         table_name: str) -> set[tuple]:
    """Load the set of already-evaluated row keys from the output file."""
    p = Path(output_path)
    if not p.exists():
        return set()

    try:
        if out_fmt == 'tsv':
            df = pl.read_csv(output_path, separator='\t', infer_schema_length=0)
        else:
            conn = sqlite3.connect(output_path)
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            if table_name not in tables:
                conn.close()
                return set()
            df = pl.read_database(
                f'SELECT * FROM "{table_name}"', conn
            )
            conn.close()

        if not all(c in df.columns for c in _ROW_KEY_COLS):
            return set()

        keys = set()
        for row in df.select(list(_ROW_KEY_COLS)).iter_rows(named=True):
            keys.add(_row_key(row))
        return keys
    except Exception as e:
        print(f"Warning: could not load existing output for resume: {e}")
        return set()


class IncrementalWriter:
    """Writes evaluation results one row at a time so progress survives
    interruption.  Thread-safe via an asyncio.Lock."""

    def __init__(self, output_path: str, out_fmt: str, table_name: str,
                 columns: list[str], *, is_resume: bool):
        self.output_path = output_path
        self.out_fmt = out_fmt
        self.table_name = table_name
        self.columns = columns
        self.lock = asyncio.Lock()
        self.rows_written = 0

        if out_fmt == 'tsv':
            if is_resume and Path(output_path).exists():
                self._tsv_fh = open(output_path, 'a', newline='',
                                    encoding='utf-8')
            else:
                self._tsv_fh = open(output_path, 'w', newline='',
                                    encoding='utf-8')
                self._tsv_fh.write('\t'.join(columns) + '\n')
                self._tsv_fh.flush()
        else:
            self._db_conn = sqlite3.connect(output_path)
            self._db_conn.execute("PRAGMA journal_mode=WAL")
            if not is_resume:
                col_defs = ', '.join(f'"{c}" TEXT' for c in columns)
                self._db_conn.execute(
                    f'DROP TABLE IF EXISTS "{table_name}"')
                self._db_conn.execute(
                    f'CREATE TABLE IF NOT EXISTS "{table_name}" ({col_defs})')
                self._db_conn.commit()
            self._db_placeholders = ', '.join('?' for _ in columns)

    async def write_row(self, combined: dict):
        """Write one result row.  Called from async evaluation tasks."""
        async with self.lock:
            vals = [str(combined.get(c, '')) for c in self.columns]
            if self.out_fmt == 'tsv':
                self._tsv_fh.write('\t'.join(vals) + '\n')
                if self.rows_written % 50 == 0:
                    self._tsv_fh.flush()
            else:
                self._db_conn.execute(
                    f'INSERT INTO "{self.table_name}" '
                    f'VALUES ({self._db_placeholders})', vals)
                if self.rows_written % 50 == 0:
                    self._db_conn.commit()
            self.rows_written += 1

    def finalize(self):
        """Flush and close the output handle."""
        if self.out_fmt == 'tsv':
            self._tsv_fh.flush()
            self._tsv_fh.close()
        else:
            self._db_conn.commit()
            self._db_conn.close()


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

        subject_names = [subject]
        object_names = [object_]
        if node_dict:
            sn = node_dict.get_all_names(subject_curie)
            if sn:
                subject_names = sn
            on = node_dict.get_all_names(object_curie)
            if on:
                object_names = on

        start = time.time()

        result = await evaluator.evaluate_triple_with_names(
            subject=subject,
            predicate=predicate,
            object_=object_,
            subject_names=subject_names,
            object_names=object_names,
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
                    names_file: str = None,
                    max_concurrent: int = None,
                    predicate_file: str = None,
                    overwrite: bool = False):
    """Read a TSV, evaluate every row concurrently, and write results
    incrementally so the run can be stopped and resumed at any time."""
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
    input_total = df.height
    out_fmt = _detect_output_format(db_path)

    print(f"Input file:  {input_file}  ({input_total:,} rows)")
    print(f"Model:       {val_model}")
    print(f"Round 2:     {round2_model or 'disabled'}")
    if out_fmt == 'db':
        print(f"Output DB:   {db_path}  (table: {table_name})")
    else:
        print(f"Output TSV:  {db_path}")

    # ---- resume: detect already-evaluated rows -----------------------
    is_resume = False
    completed_keys: set[tuple] = set()
    if not overwrite:
        completed_keys = _load_completed_keys(db_path, out_fmt, table_name)
    if completed_keys:
        is_resume = True
        before = df.height
        pending_mask = pl.Series(
            "_pending",
            [_row_key(row) not in completed_keys for row in df.iter_rows(named=True)]
        )
        df = df.filter(pending_mask)
        print(f"Resume: {len(completed_keys):,} rows already done, "
              f"{df.height:,} remaining")
    else:
        if overwrite and Path(db_path).exists():
            Path(db_path).unlink()
            print("Overwrite: removed existing output file")

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
        if names_file:
            node_dict.merge_all_names_tsv(names_file)
            print(f"  Merged all_names from: {names_file}")

    # ---- filter out rows with no valid abstract in cache ---------------
    cache = PMIDCache()
    print(f"Checking PMID abstracts in cache ({cache.count():,} cached)...")

    raw_pmids = df['PMID'].to_list()
    bare_pmids = [_normalize_pmid(str(p)) for p in raw_pmids]
    cached_valid = cache.get_all_cached_pmids()

    has_abstract = [bp in cached_valid for bp in bare_pmids]
    has_abstract_series = pl.Series("_has_abstract", has_abstract)

    df_valid = df.filter(has_abstract_series)
    df_skipped = df.filter(~has_abstract_series)

    if df_skipped.height > 0:
        no_abs_path = Path(db_path).with_name(
            Path(db_path).stem + "_no_abstract.tsv"
        )
        df_skipped.write_csv(no_abs_path, separator="\t")
        print(f"Skipped {df_skipped.height:,} rows (no valid abstract) -> {no_abs_path}")

    df = df_valid
    total_rows = df.height
    print(f"Rows to evaluate this run: {total_rows:,}")
    print("=" * 60)

    if total_rows == 0:
        print("Nothing to evaluate (all rows done or have no abstract).")
        return 0

    # ---- prepare incremental writer ----------------------------------
    input_cols = [c for c in df.columns if c not in OUTPUT_COLUMNS]
    all_columns = input_cols + OUTPUT_COLUMNS

    writer = IncrementalWriter(
        db_path, out_fmt, table_name, all_columns, is_resume=is_resume,
    )

    # ---- evaluate with incremental writes ----------------------------
    evaluator = TripleEvaluatorSystem(
        llm_provider=val_model,
        round2_model=round2_model,
    )
    semaphore = asyncio.Semaphore(concurrency)
    rows_as_dicts = df.to_dicts()
    all_results = []

    async def _eval(row, idx):
        async with semaphore:
            result = await evaluate_single_row(
                evaluator, row, idx, total_rows, node_dict
            )
        combined = {**row, **result}
        await writer.write_row(combined)
        return result

    overall_start = time.time()
    tasks = [_eval(row, idx) for idx, row in enumerate(rows_as_dicts)]
    print(f"Starting batch evaluation with {concurrency} concurrent workers...")
    print(f"Results are written incrementally — safe to Ctrl+C and resume.")
    print("=" * 60)

    try:
        all_results = await asyncio.gather(*tasks)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print(f"\nInterrupted! {writer.rows_written:,} rows saved.")
        print("Re-run the same command to resume from where you stopped.")
    finally:
        writer.finalize()

    total_runtime = time.time() - overall_start

    # ---- companion files (based on full output) ----------------------
    # Re-read the full output for timing/metrics (includes resumed rows)
    try:
        if out_fmt == 'tsv':
            df_full = pl.read_csv(db_path, separator='\t', infer_schema_length=0)
        else:
            conn = sqlite3.connect(db_path)
            df_full = pl.read_database(
                f'SELECT * FROM "{table_name}"', conn)
            conn.close()

        cpaths = _companion_paths(db_path)
        _write_timing_files(df_full, total_runtime, concurrency, val_model, cpaths)
        _write_metrics_files(df_full, cpaths)
    except Exception as e:
        print(f"Warning: could not write companion files: {e}")
        df_full = None
        cpaths = _companion_paths(db_path)

    # ---- console summary ---------------------------------------------
    newly_done = writer.rows_written
    total_done = len(completed_keys) + newly_done

    print("\n" + "=" * 60)
    print("Batch evaluation complete!")
    print("=" * 60)
    print(f"Newly evaluated: {newly_done:,} rows in {total_runtime:.1f}s "
          f"({total_runtime / 60:.1f} min)")
    if is_resume:
        print(f"Previously done: {len(completed_keys):,} rows")
    print(f"Total evaluated: {total_done:,} / {input_total:,} input rows")
    if newly_done > 0:
        print(f"Avg per row:     {total_runtime / newly_done:.2f}s")
    if out_fmt == 'db':
        print(f"Results in:      {db_path}  (table: {table_name})")
    else:
        print(f"Results in:      {db_path}")

    if df_full is not None and 'support' in df_full.columns:
        support_counts = dict(
            df_full['support'].value_counts()
            .sort('support')
            .iter_rows()
        )
        print(f"\nSupport distribution (all {total_done:,} rows):")
        for val in ('yes', 'maybe', 'no'):
            print(f"  {val}: {support_counts.get(val, 0)}")

        if 'ground_truth' in df_full.columns:
            _print_metrics(df_full, df_full.height)

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
  python main.py --input data/semmedb_kgx/semmeddb_edges_extracted.tsv --output results.tsv --val_model gpt-oss-20b-vllm --predicate_file data/biolink_data/biolink_predicates.tsv --node_dict data/semmedb_kgx/normalized_nodes.jsonl --names_file data/semmedb_kgx/curie_all_names.tsv
  python main.py --input data/semmedb_kgx/semmeddb_edges_extracted.tsv --output results.tsv --val_model gpt-oss-20b-vllm --predicate_file data/biolink_data/biolink_predicates.tsv --node_dict data/semmedb_kgx/normalized_nodes.jsonl --names_file data/semmedb_kgx/curie_all_names.tsv --max_concurrent 16
  python main.py --input data/semmedb_kgx/semmeddb_edges_extracted.tsv --output results.tsv --val_model gpt-oss-20b-vllm --predicate_file data/biolink_data/biolink_predicates.tsv --node_dict data/semmedb_kgx/normalized_nodes.jsonl --names_file data/semmedb_kgx/curie_all_names.tsv --round2_model gpt-oss-120b-vllm
  python main.py --input data/semmedb_kgx/semmeddb_edges_extracted.tsv --output results.db --val_model gpt-oss-20b-vllm --predicate_file data/biolink_data/biolink_predicates.tsv --node_dict data/semmedb_kgx/normalized_nodes.jsonl --names_file data/semmedb_kgx/curie_all_names.tsv --table my_run_1
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
                        help='Path to nodes file (.jsonl, .jsonl.gz) or pre-built dict (.json/.json.gz)')
    parser.add_argument('--names_file', default=None,
                        help='Path to curie_all_names.tsv (columns: curie_id, all_names) '
                             'to supplement node_dict with richer equivalent names')
    parser.add_argument('--predicate_file', default=None,
                        help='Path to biolink predicates TSV (columns: predicate, description)')
    parser.add_argument('--max_concurrent', type=int, default=None,
                        help=f'Max concurrent requests (default: {settings.max_concurrent_requests})')
    parser.add_argument('--overwrite', action='store_true',
                        help='Discard existing output and start fresh (default: auto-resume)')
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
        names_file=args.names_file,
        max_concurrent=args.max_concurrent,
        predicate_file=args.predicate_file,
        overwrite=args.overwrite,
    ))
    sys.exit(rc)


if __name__ == "__main__":
    main()
