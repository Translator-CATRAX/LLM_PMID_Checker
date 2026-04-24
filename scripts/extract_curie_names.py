#!/usr/bin/env python3
"""
Extract all_names for unique CURIE IDs from semmeddb_edges_extracted.tsv
using the Node Normalization API (https://nodenormalization-sri.renci.org).

For each CURIE, queries the /get_normalized_nodes endpoint and collects
all labels from `id.label` and `equivalent_identifiers[*].label`.
Results are saved as a TSV: curie_id <tab> all_names (pipe-delimited).
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import aiohttp
from tqdm import tqdm

API_URL = "https://nodenormalization-sri.renci.org/get_normalized_nodes"
BATCH_SIZE = 500
MAX_CONCURRENT = 10
MAX_RETRIES = 3
RETRY_DELAY = 2.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def extract_all_names(node_info: dict) -> list[str]:
    """Extract all unique labels from a single node normalization result.

    Deduplicates case-insensitively: when multiple labels differ only by case
    (e.g. "Adenosine", "adenosine", "ADENOSINE"), only the lowercase form is kept.
    """
    if node_info is None:
        return []
    raw_labels: list[str] = []
    primary = node_info.get("id", {}).get("label")
    if primary:
        raw_labels.append(primary)
    for eq in node_info.get("equivalent_identifiers", []):
        lbl = eq.get("label")
        if lbl:
            raw_labels.append(lbl)

    seen_lower: dict[str, str] = {}
    for lbl in raw_labels:
        key = lbl.lower()
        if key not in seen_lower:
            seen_lower[key] = key
    return list(seen_lower.values())


def load_unique_curies(tsv_path: str) -> list[str]:
    """Read subject_curie and object_curie columns, return sorted unique list."""
    logger.info(f"Loading unique CURIEs from {tsv_path} ...")
    curies = set()
    with open(tsv_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            curies.add(row["subject_curie"])
            curies.add(row["object_curie"])
    result = sorted(curies)
    logger.info(f"Found {len(result)} unique CURIEs")
    return result


async def fetch_batch(
    session: aiohttp.ClientSession,
    curies: list[str],
    semaphore: asyncio.Semaphore,
) -> dict:
    """POST a batch of CURIEs to the API with retry logic."""
    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with session.post(
                    API_URL,
                    json={"curies": curies},
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    text = await resp.text()
                    logger.warning(
                        f"Batch returned status {resp.status} "
                        f"(attempt {attempt}/{MAX_RETRIES}): {text[:200]}"
                    )
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(
                    f"Request error (attempt {attempt}/{MAX_RETRIES}): {e}"
                )
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY * attempt)
        logger.error(f"Failed batch after {MAX_RETRIES} attempts, {len(curies)} CURIEs lost")
        return {}


async def process_all_curies(
    curies: list[str],
    output_path: str,
    batch_size: int = BATCH_SIZE,
    max_concurrent: int = MAX_CONCURRENT,
) -> None:
    """Fetch all_names for all CURIEs and write to TSV."""
    batches = [curies[i : i + batch_size] for i in range(0, len(curies), batch_size)]
    logger.info(
        f"Processing {len(curies)} CURIEs in {len(batches)} batches "
        f"(batch_size={batch_size}, concurrency={max_concurrent})"
    )

    results: dict[str, list[str]] = {}
    semaphore = asyncio.Semaphore(max_concurrent)
    pbar = tqdm(total=len(batches), desc="Batches", unit="batch")

    async with aiohttp.ClientSession() as session:
        tasks = []
        for batch in batches:
            tasks.append(fetch_batch(session, batch, semaphore))

        for coro in asyncio.as_completed(tasks):
            batch_result = await coro
            for curie, info in batch_result.items():
                results[curie] = extract_all_names(info)
            pbar.update(1)

    pbar.close()

    resolved = sum(1 for v in results.values() if v)
    logger.info(
        f"Resolved {resolved}/{len(curies)} CURIEs with at least one name"
    )

    for c in curies:
        if c not in results:
            results[c] = []

    logger.info(f"Writing results to {output_path} ...")
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["curie_id", "all_names"])
        for curie in sorted(results.keys()):
            names = results[curie]
            writer.writerow([curie, "|".join(names) if names else ""])

    logger.info("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract all_names for CURIEs via Node Normalization API"
    )
    parser.add_argument(
        "--input",
        default="data/semmedb_kgx/semmeddb_edges_extracted.tsv",
        help="Path to input TSV with subject_curie and object_curie columns",
    )
    parser.add_argument(
        "--output",
        default="data/semmedb_kgx/curie_all_names.tsv",
        help="Path to output TSV (curie_id, all_names)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"CURIEs per API request (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=MAX_CONCURRENT,
        help=f"Max concurrent requests (default: {MAX_CONCURRENT})",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    input_path = (
        args.input
        if os.path.isabs(args.input)
        else str(project_root / args.input)
    )
    output_path = (
        args.output
        if os.path.isabs(args.output)
        else str(project_root / args.output)
    )

    curies = load_unique_curies(input_path)
    asyncio.run(
        process_all_curies(curies, output_path, args.batch_size, args.max_concurrent)
    )


if __name__ == "__main__":
    main()
