#!/usr/bin/env python3
"""Compare 4-key coverage between the extracted input and evaluation results.

Reads:
  - semmeddb_edges_extracted.parquet  (the input)
  - results.db  (evaluations + evaluations_no_abstract tables)

Reports duplicate counts, overlap between tables, coverage against the input,
and lists any missing or extra 4-keys.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import polars as pl


def main():
    parser = argparse.ArgumentParser(
        description="Compare 4-key coverage between extracted input and results."
    )
    parser.add_argument(
        "--extracted", "-e",
        default="data/semmedb_kgx/semmeddb_edges_extracted.parquet",
        help="Path to the extracted input Parquet file",
    )
    parser.add_argument(
        "--results-db", "-r",
        default="results.db",
        help="Path to the results SQLite database",
    )
    parser.add_argument(
        "--table", "-t",
        default="evaluations",
        help="Main evaluations table name (default: evaluations)",
    )
    args = parser.parse_args()

    ext_path = Path(args.extracted)
    db_path = Path(args.results_db)

    if not ext_path.exists():
        sys.exit(f"Error: extracted file not found: {ext_path}")
    if not db_path.exists():
        sys.exit(f"Error: results database not found: {db_path}")

    key_cols = ['subject_curie', 'predicate', 'object_curie', 'PMID']
    na_table = args.table + "_no_abstract"

    # ---- Extracted input ----
    print("Loading extracted input ...")
    df_ext = pl.read_parquet(str(ext_path), columns=key_cols)
    ext_keys = set(df_ext.iter_rows())
    ext_total = df_ext.height
    ext_unique = len(ext_keys)
    print(f"  Total rows:    {ext_total:,}")
    print(f"  Unique 4-keys: {ext_unique:,}")
    print(f"  Duplicates:    {ext_total - ext_unique:,}")
    del df_ext

    # ---- Evaluations table ----
    print(f"\nLoading {args.table} table ...")
    conn = sqlite3.connect(str(db_path))
    eval_count = conn.execute(
        f'SELECT COUNT(*) FROM "{args.table}"'
    ).fetchone()[0]
    eval_keys = set()
    for row in conn.execute(
        f'SELECT subject_curie, predicate, object_curie, PMID FROM "{args.table}"'
    ):
        eval_keys.add(row)
    print(f"  Total rows:    {eval_count:,}")
    print(f"  Unique 4-keys: {len(eval_keys):,}")
    print(f"  Duplicates:    {eval_count - len(eval_keys):,}")

    # ---- No-abstract table ----
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]

    na_keys = set()
    na_count = 0
    if na_table in tables:
        print(f"\nLoading {na_table} table ...")
        na_count = conn.execute(
            f'SELECT COUNT(*) FROM "{na_table}"'
        ).fetchone()[0]
        for row in conn.execute(
            f'SELECT subject_curie, predicate, object_curie, PMID FROM "{na_table}"'
        ):
            na_keys.add(row)
        print(f"  Total rows:    {na_count:,}")
        print(f"  Unique 4-keys: {len(na_keys):,}")
        print(f"  Duplicates:    {na_count - len(na_keys):,}")
    else:
        print(f"\nNo '{na_table}' table found in {db_path}")

    conn.close()

    # ---- Cross-check ----
    combined = eval_keys | na_keys
    overlap = eval_keys & na_keys

    print(f"\n{'=' * 60}")
    print("Cross-check Summary")
    print("=" * 60)
    print(f"  {args.table} rows:         {eval_count:,}")
    print(f"  {na_table} rows:  {na_count:,}")
    print(f"  Overlap between tables:       {len(overlap):,}")
    print(f"  Combined unique 4-keys:       {len(combined):,}")
    print(f"  Extracted unique 4-keys:      {ext_unique:,}")

    in_results_not_ext = combined - ext_keys
    in_ext_not_results = ext_keys - combined
    matched = combined & ext_keys

    print(f"\n  In results but NOT in extracted: {len(in_results_not_ext):,}")
    print(f"  In extracted but NOT in results: {len(in_ext_not_results):,}")
    print(f"  Matched:                         {len(matched):,}")

    if ext_unique > 0:
        print(f"  Coverage: {len(matched):,} / {ext_unique:,} = "
              f"{len(matched) / ext_unique * 100:.4f}%")

    if in_ext_not_results:
        print(f"\n  Sample missing keys (up to 10):")
        for k in list(in_ext_not_results)[:10]:
            print(f"    {k}")

    if in_results_not_ext:
        print(f"\n  Sample extra keys (up to 10):")
        for k in list(in_results_not_ext)[:10]:
            print(f"    {k}")

    if not in_ext_not_results and not in_results_not_ext and len(overlap) == 0:
        print(f"\n  *** PERFECT MATCH: All rows accounted for ***")

    return 0 if not in_ext_not_results and not in_results_not_ext else 1


if __name__ == "__main__":
    sys.exit(main())
