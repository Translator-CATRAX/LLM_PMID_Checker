#!/usr/bin/env python3
"""
Extract subject_curie, predicate, object_curie, PMID, and SemMedDB sentence
from a SemMedDB KGX JSONL file and write to a TSV file.

When a record has multiple PMIDs, each PMID (with its corresponding sentence)
is written as a separate row.

Supports two input formats:
  - KGX normalized: supporting text lives in has_supporting_studies -> study_results
  - Legacy kg2:     supporting text lives in publications_info -> sentence
"""

import json
import csv
import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract fields from SemMedDB edges JSONL into a TSV."
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
        default=Path("data/semmedb_kgx/semmeddb_edges_extracted.tsv"),
        help="Path to the output TSV file",
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


def build_pmid_text_map(record: dict) -> dict[str, str]:
    """Build a PMID -> supporting_text mapping from whichever format is present."""
    pmid_map: dict[str, str] = {}

    # KGX normalized format: has_supporting_studies
    studies = record.get("has_supporting_studies", {})
    if studies:
        for study in studies.values():
            for result in study.get("has_study_results", []):
                xrefs = result.get("xref", [])
                texts = result.get("supporting_text", [])
                if xrefs and texts:
                    pmid_map[xrefs[0]] = texts[0]
        return pmid_map

    # Legacy kg2 format: publications_info
    pub_info = record.get("publications_info", {})
    for pmid, info in pub_info.items():
        sentence = info.get("sentence", "")
        if sentence:
            pmid_map[pmid] = sentence

    return pmid_map


def main():
    args = parse_args()

    if not args.input.exists():
        sys.exit(f"Error: input file not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["subject_curie", "predicate", "object_curie", "PMID", "SemMedDB_sentence"]
    rows_written = 0
    records_processed = 0
    skipped_no_text = 0

    with open_input(args.input) as fin, \
         open(args.output, "w", newline="", encoding="utf-8") as fout:

        writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for line in fin:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            subject = record.get("subject", "")
q            predicate = record.get("predicate", "")
            obj = record.get("object", "")
            publications = record.get("publications", [])
            pmid_text = build_pmid_text_map(record)

            for pmid in publications:
                sentence = pmid_text.get(pmid, "")
                if not sentence:
                    skipped_no_text += 1

                writer.writerow({
                    "subject_curie": subject,
                    "predicate": predicate,
                    "object_curie": obj,
                    "PMID": pmid,
                    "SemMedDB_sentence": sentence,
                })
                rows_written += 1

            records_processed += 1

            if records_processed % 500_000 == 0:
                print(f"  Processed {records_processed:,} records, "
                      f"wrote {rows_written:,} rows ...", flush=True)

            if args.limit and records_processed >= args.limit:
                break

    print(f"\nDone.")
    print(f"  Records processed: {records_processed:,}")
    print(f"  Rows written:      {rows_written:,}")
    print(f"  Skipped (no sentence): {skipped_no_text:,}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
