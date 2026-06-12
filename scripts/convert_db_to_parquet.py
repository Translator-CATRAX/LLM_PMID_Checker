#!/usr/bin/env python3
"""Convert results.db SQLite tables to Parquet files.

Reads the evaluations and evaluations_no_abstract tables from a SQLite
database and writes them as separate Parquet files.

"""

import argparse
import os
import sqlite3
import sys
from pathlib import Path

import polars as pl


def main():
    parser = argparse.ArgumentParser(
        description="Convert results.db SQLite tables to Parquet files."
    )
    parser.add_argument(
        "--db", "-d",
        default="results.db",
        help="Path to the SQLite database (default: results.db)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=".",
        help="Directory for output Parquet files (default: current directory)",
    )
    parser.add_argument(
        "--table", "-t",
        default="evaluations",
        help="Main evaluations table name (default: evaluations)",
    )
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=["runtime_seconds"],
        help="Columns to drop from the evaluations table (default: runtime_seconds)",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    out_dir = Path(args.output_dir)

    if not db_path.exists():
        sys.exit(f"Error: database not found: {db_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]

    na_table = args.table + "_no_abstract"

    # ---- Convert main evaluations table ----
    if args.table in tables:
        print(f"Converting '{args.table}' table ...")
        df = pl.read_database(f'SELECT * FROM "{args.table}"', conn)

        if args.drop_columns:
            cols_to_drop = [c for c in args.drop_columns if c in df.columns]
            if cols_to_drop:
                df = df.drop(cols_to_drop)
                print(f"  Dropped columns: {cols_to_drop}")

        out_path = out_dir / "results.parquet"
        df.write_parquet(str(out_path))
        size_gb = os.path.getsize(out_path) / (1024 ** 3)
        print(f"  Shape:   {df.shape}")
        print(f"  Columns: {df.columns}")
        print(f"  Output:  {out_path} ({size_gb:.2f} GB)")
        del df
    else:
        print(f"Warning: table '{args.table}' not found in {db_path}")

    # ---- Convert no-abstract table ----
    if na_table in tables:
        print(f"\nConverting '{na_table}' table ...")
        df_na = pl.read_database(f'SELECT * FROM "{na_table}"', conn)
        na_out = out_dir / "results_no_abstract.parquet"
        df_na.write_parquet(str(na_out))
        size_mb = os.path.getsize(na_out) / (1024 ** 2)
        print(f"  Shape:   {df_na.shape}")
        print(f"  Columns: {df_na.columns}")
        print(f"  Output:  {na_out} ({size_mb:.1f} MB)")
        del df_na
    else:
        print(f"\nNo '{na_table}' table found in {db_path}")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
