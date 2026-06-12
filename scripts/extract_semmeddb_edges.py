#!/usr/bin/env python3
"""
Extract subject_curie, predicate, object_curie, PMID, and SemMedDB sentences
from a SemMedDB KGX JSONL file and write to a Parquet file.

Each output row is keyed by (subject, predicate, object, PMID). 
"""

import json
import argparse
import sys
from pathlib import Path
from collections import OrderedDict

import polars as pl


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract fields from SemMedDB edges JSONL into a Parquet file."
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("data/semmedb_kgx/normalized_edges.jsonl"),
        help="Path to the input JSONL file (plain or .gz)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/semmedb_kgx/semmeddb_edges_extracted.parquet"),
        help="Path to the output Parquet file",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Process only the first N records (useful for testing)",
    )
    return parser.parse_args()


def open_input(path: Path):
    if path.suffix == ".gz":
        import gzip
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def build_pmid_text_map(record: dict) -> dict[str, list[str]]:
    """Build a PMID -> list of supporting_text mapping."""
    pmid_map: dict[str, list[str]] = {}

    # KGX normalized format: has_supporting_studies
    studies = record.get("has_supporting_studies", {})
    if studies:
        for study in studies.values():
            for result in study.get("has_study_results", []):
                xrefs = result.get("xref", [])
                texts = result.get("supporting_text", [])
                if xrefs and texts:
                    pmid_map.setdefault(xrefs[0], []).append(texts[0])
        return pmid_map

    # Legacy kg2 format: publications_info
    pub_info = record.get("publications_info", {})
    for pmid, info in pub_info.items():
        sentence = info.get("sentence", "")
        if sentence:
            pmid_map.setdefault(pmid, []).append(sentence)

    return pmid_map


def main():
    args = parse_args()

    if not args.input.exists():
        sys.exit(f"Error: input file not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["subject_curie", "predicate", "object_curie", "PMID", "SemMedDB_sentences"]

    records_processed = 0
    skipped_no_text = 0

    # Collect all sentences per (sub, pred, obj, PMID) key
    # Use OrderedDict to preserve insertion order
    rows: OrderedDict[tuple, list[str]] = OrderedDict()

    print("Pass 1: collecting sentences per (sub, pred, obj, PMID) ...")

    with open_input(args.input) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            subject = record.get("subject", "")
            predicate = record.get("predicate", "")
            obj = record.get("object", "")
            publications = record.get("publications", [])
            pmid_text = build_pmid_text_map(record)

            for pmid in publications:
                texts = pmid_text.get(pmid, [])
                key = (subject, predicate, obj, pmid)

                if not texts:
                    skipped_no_text += 1
                    if key not in rows:
                        rows[key] = []
                else:
                    existing = rows.setdefault(key, [])
                    for t in texts:
                        if t not in existing:
                            existing.append(t)

            records_processed += 1

            if records_processed % 500_000 == 0:
                print(f"  Processed {records_processed:,} records, "
                      f"{len(rows):,} unique keys ...", flush=True)

            if args.limit and records_processed >= args.limit:
                break

    print(f"\nPass 2: building DataFrame ({len(rows):,} rows) ...")

    multi_sentence = 0
    data = {col: [] for col in fieldnames}

    for (subject, predicate, obj, pmid), sentences in rows.items():
        combined = " | ".join(sentences) if sentences else ""
        if len(sentences) > 1:
            multi_sentence += 1
        data["subject_curie"].append(subject)
        data["predicate"].append(predicate)
        data["object_curie"].append(obj)
        data["PMID"].append(pmid)
        data["SemMedDB_sentences"].append(combined)

    df = pl.DataFrame(data)
    print(f"  DataFrame shape: {df.shape}")
    print(f"  Writing Parquet ...")
    df.write_parquet(args.output)

    print(f"\nDone.")
    print(f"  Records processed:      {records_processed:,}")
    print(f"  Unique (sub,pred,obj,PMID) rows: {df.height:,}")
    print(f"  Rows with multiple sentences:    {multi_sentence:,}")
    print(f"  PMIDs with no sentence:          {skipped_no_text:,}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
