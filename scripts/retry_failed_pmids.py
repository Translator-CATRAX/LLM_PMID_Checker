#!/usr/bin/env python3
"""
Retry fetching PMID abstracts that failed due to transient network errors.

Reads the SQLite cache for entries whose error column matches known retryable
patterns (timeouts, disconnects, rate-limiting, XML parse errors, PMID not
found due to batch failures).  Deletes those stale error rows and re-fetches
via the NCBI E-utilities API.

Non-retryable errors like "No abstract available" are left untouched.
"""

import asyncio
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pmid_cache import PMIDCache
from src.pmid_extractor import PMIDExtractor, AbstractData
from src.config import settings

import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

RETRYABLE_PATTERNS = [
    "Extraction failed:%",
    "PMID not found or could not be retrieved",
]


def load_retryable_pmids(db_path: str) -> List[str]:
    """Query the cache for PMIDs whose error matches a retryable pattern."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    where_clauses = " OR ".join(f"error LIKE ?" for _ in RETRYABLE_PATTERNS)
    cursor.execute(
        f"SELECT pmid FROM pmid_abstracts WHERE {where_clauses}",
        RETRYABLE_PATTERNS,
    )
    pmids = [row[0] for row in cursor.fetchall()]
    conn.close()
    return pmids


def delete_retryable_errors(db_path: str, pmids: List[str], chunk_size: int = 500):
    """Remove stale error rows so the fetcher can re-insert fresh results."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for i in range(0, len(pmids), chunk_size):
        chunk = pmids[i : i + chunk_size]
        placeholders = ",".join("?" * len(chunk))
        cursor.execute(
            f"DELETE FROM pmid_abstracts WHERE pmid IN ({placeholders})", chunk
        )

    conn.commit()
    conn.close()


class AsyncPMIDFetcher:
    """Re-uses the same fetch-and-cache logic from prefetch_pmid_abstracts."""

    def __init__(self, batch_size: int = 200, delay: float = 1.0):
        self.batch_size = batch_size
        self.delay = delay
        self.cache = PMIDCache()
        self.extractor = PMIDExtractor(
            api_key=settings.ncbi_api_key,
            email=settings.ncbi_email,
        )

    async def fetch_and_cache_batch(
        self, pmids: List[str], batch_num: int, total_batches: int
    ) -> int:
        if not pmids:
            return 0

        logger.info(
            f"Fetching batch {batch_num}/{total_batches} ({len(pmids)} PMIDs)..."
        )

        try:
            loop = asyncio.get_event_loop()
            abstracts = await loop.run_in_executor(
                None, self.extractor.extract_abstracts, pmids
            )

            fetch_date = datetime.utcnow().isoformat()
            cache_data = []
            successful = 0

            for pmid, ad in abstracts.items():
                cache_data.append(
                    {
                        "pmid": pmid,
                        "title": ad.title,
                        "abstract": ad.abstract,
                        "fetch_date": fetch_date,
                        "error": ad.error,
                    }
                )
                if not ad.error and ad.abstract:
                    successful += 1

            if cache_data:
                self.cache.put_many(cache_data)

            logger.info(
                f"Batch {batch_num}/{total_batches}: "
                f"{successful}/{len(pmids)} abstracts cached"
            )
            return successful

        except Exception as e:
            logger.error(f"Error in batch {batch_num}: {e}")
            return 0

    async def fetch_all(self, pmids: List[str]) -> dict:
        start = datetime.now()

        batches = [
            pmids[i : i + self.batch_size]
            for i in range(0, len(pmids), self.batch_size)
        ]
        total_batches = len(batches)
        logger.info(
            f"Retrying {len(pmids)} PMIDs in {total_batches} batches "
            f"of {self.batch_size}"
        )

        total_ok = 0
        for i, batch in enumerate(batches, 1):
            total_ok += await self.fetch_and_cache_batch(batch, i, total_batches)
            if i < total_batches:
                await asyncio.sleep(self.delay)

        duration = (datetime.now() - start).total_seconds()
        return {
            "retried": len(pmids),
            "successful": total_ok,
            "still_failed": len(pmids) - total_ok,
            "duration_seconds": duration,
            "cache_size": self.cache.count(),
        }


async def main():
    parser = argparse.ArgumentParser(
        description="Retry PMID fetches that failed due to transient errors"
    )
    parser.add_argument(
        "--db",
        default="data/pmid_cache.db",
        help="Path to the SQLite cache (default: data/pmid_cache.db)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="PMIDs per NCBI request (default: 200)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds between batches (default: 1.0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only retry the first N PMIDs (for testing)",
    )
    args = parser.parse_args()

    if not settings.ncbi_email:
        logger.error("NCBI_EMAIL is not set. Please configure your .env file.")
        return 1
    if not settings.ncbi_api_key:
        logger.warning("NCBI_API_KEY is not set. Rate limits will be lower.")

    logger.info("Loading retryable failed PMIDs from cache...")
    pmids = load_retryable_pmids(args.db)
    logger.info(f"Found {len(pmids):,} PMIDs with retryable errors")

    if not pmids:
        logger.info("Nothing to retry!")
        return 0

    if args.limit:
        pmids = pmids[: args.limit]
        logger.info(f"Limited to first {args.limit:,} PMIDs (--limit)")

    logger.info("Deleting stale error entries from cache...")
    delete_retryable_errors(args.db, pmids)
    logger.info("Stale entries removed")

    logger.info("=" * 80)
    logger.info("RETRY — PMID Abstract Fetching")
    logger.info("=" * 80)
    logger.info(f"PMIDs to retry: {len(pmids):,}")
    logger.info(f"Batch size:     {args.batch_size}")
    logger.info(f"Delay:          {args.delay}s")
    logger.info("=" * 80)

    fetcher = AsyncPMIDFetcher(batch_size=args.batch_size, delay=args.delay)

    try:
        stats = await fetcher.fetch_all(pmids)

        logger.info("=" * 80)
        logger.info("RETRY SUMMARY")
        logger.info("=" * 80)
        logger.info(f"PMIDs retried:       {stats['retried']:,}")
        logger.info(f"Now successful:      {stats['successful']:,}")
        logger.info(f"Still failed:        {stats['still_failed']:,}")
        logger.info(f"Total cache size:    {stats['cache_size']:,} abstracts")
        logger.info(f"Duration:            {stats['duration_seconds']:.2f}s")
        if stats["retried"] > 0 and stats["duration_seconds"] > 0:
            rate = stats["retried"] / stats["duration_seconds"]
            logger.info(f"Fetch rate:          {rate:.2f} PMIDs/s")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Retry failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
