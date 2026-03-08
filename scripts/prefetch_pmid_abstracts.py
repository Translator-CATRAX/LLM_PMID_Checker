#!/usr/bin/env python3
"""
Async script to pre-fetch PMID abstracts from a TSV file
and store them in the local SQLite cache.
"""
import asyncio
import polars as pl
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Set
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pmid_cache import PMIDCache
from src.pmid_extractor import PMIDExtractor, AbstractData
from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AsyncPMIDFetcher:
    """Async fetcher for PMID abstracts with rate limiting and progress tracking."""
    
    def __init__(self, batch_size: int = 100, delay_between_batches: float = 0.5):
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches
        self.cache = PMIDCache()
        self.extractor = PMIDExtractor(
            api_key=settings.ncbi_api_key,
            email=settings.ncbi_email
        )
    
    async def fetch_and_cache_batch(self, pmids: List[str], batch_num: int, total_batches: int) -> int:
        """Fetch a batch of PMIDs and cache them."""
        if not pmids:
            return 0
        
        logger.info(f"Fetching batch {batch_num}/{total_batches} ({len(pmids)} PMIDs)...")
        
        try:
            loop = asyncio.get_event_loop()
            abstracts = await loop.run_in_executor(
                None, 
                self.extractor.extract_abstracts,
                pmids
            )
            
            fetch_date = datetime.utcnow().isoformat()
            cache_data = []
            
            successful_count = 0
            for pmid, abstract_data in abstracts.items():
                cache_data.append({
                    'pmid': pmid,
                    'title': abstract_data.title,
                    'abstract': abstract_data.abstract,
                    'fetch_date': fetch_date,
                    'error': abstract_data.error
                })
                
                if not abstract_data.error and abstract_data.abstract:
                    successful_count += 1
            
            if cache_data:
                self.cache.put_many(cache_data)
            
            logger.info(f"Batch {batch_num}/{total_batches} completed: {successful_count}/{len(pmids)} abstracts cached")
            
            return successful_count
            
        except Exception as e:
            logger.error(f"Error fetching batch {batch_num}: {e}")
            return 0
    
    async def fetch_all(self, pmids: List[str], skip_cached: bool = True) -> dict:
        """Fetch all PMIDs with progress tracking."""
        start_time = datetime.now()
        
        if skip_cached:
            original_count = len(pmids)
            pmids_to_fetch = self.cache.get_missing_pmids(pmids)
            cached_count = original_count - len(pmids_to_fetch)
            logger.info(f"Found {cached_count} PMIDs already cached, fetching {len(pmids_to_fetch)} new PMIDs")
        else:
            pmids_to_fetch = pmids
            cached_count = 0
        
        if not pmids_to_fetch:
            logger.info("All PMIDs are already cached!")
            return {
                'total_pmids': len(pmids),
                'already_cached': cached_count,
                'newly_fetched': 0,
                'successful': 0,
                'failed': 0,
                'duration_seconds': 0
            }
        
        batches = [
            pmids_to_fetch[i:i + self.batch_size]
            for i in range(0, len(pmids_to_fetch), self.batch_size)
        ]
        
        total_batches = len(batches)
        logger.info(f"Fetching {len(pmids_to_fetch)} PMIDs in {total_batches} batches of {self.batch_size}")
        
        total_successful = 0
        for i, batch in enumerate(batches, 1):
            successful = await self.fetch_and_cache_batch(batch, i, total_batches)
            total_successful += successful
            
            if i < total_batches:
                await asyncio.sleep(self.delay_between_batches)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        stats = {
            'total_pmids': len(pmids),
            'already_cached': cached_count,
            'newly_fetched': len(pmids_to_fetch),
            'successful': total_successful,
            'failed': len(pmids_to_fetch) - total_successful,
            'duration_seconds': duration,
            'cache_size': self.cache.count()
        }
        
        return stats


def load_pmids_from_tsv(tsv_path: str) -> Set[str]:
    """Load unique PMIDs from the TSV file."""
    logger.info(f"Loading PMIDs from {tsv_path}...")
    
    try:
        df = pl.read_csv(tsv_path, separator='\t', infer_schema_length=0)
        
        if 'PMID' not in df.columns:
            raise ValueError("TSV file does not contain a 'PMID' column")
        
        pmids = set(
            df['PMID']
            .drop_nulls()
            .cast(pl.Utf8)
            .unique()
            .to_list()
        )
        
        pmids = {pmid for pmid in pmids if pmid.strip()}
        
        logger.info(f"Found {len(pmids)} unique PMIDs in the file")
        return pmids
        
    except Exception as e:
        logger.error(f"Error loading PMIDs from TSV: {e}")
        raise


async def main():
    """Main function to pre-fetch PMID abstracts."""
    parser = argparse.ArgumentParser(
        description="Pre-fetch PMID abstracts from a TSV file and cache them locally",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--tsv-file',
        default='data/andy_team_data/processed_mediK_results_v2.tsv',
        help='Path to the TSV file containing PMIDs (default: data/andy_team_data/processed_mediK_results_v2.tsv)'
    )
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Number of PMIDs to fetch per batch (default: 100)')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Delay in seconds between batches for rate limiting (default: 0.5)')
    parser.add_argument('--force', action='store_true',
                        help='Re-fetch all PMIDs, even if already cached')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not settings.ncbi_email:
        logger.error("NCBI_EMAIL environment variable is not set!")
        logger.error("Please set it in your .env file")
        return 1
    
    if not settings.ncbi_api_key:
        logger.warning("NCBI_API_KEY is not set. Rate limits will be lower.")
    
    try:
        pmids = load_pmids_from_tsv(args.tsv_file)
    except Exception as e:
        logger.error(f"Failed to load PMIDs: {e}")
        return 1
    
    if not pmids:
        logger.error("No PMIDs found in the TSV file")
        return 1
    
    pmids_list = sorted(pmids)
    
    logger.info("=" * 80)
    logger.info("PMID Abstract Pre-fetching")
    logger.info("=" * 80)
    logger.info(f"TSV file: {args.tsv_file}")
    logger.info(f"Total unique PMIDs: {len(pmids_list)}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Delay between batches: {args.delay}s")
    logger.info(f"Force re-fetch: {args.force}")
    logger.info("=" * 80)
    
    fetcher = AsyncPMIDFetcher(
        batch_size=args.batch_size,
        delay_between_batches=args.delay
    )
    
    try:
        stats = await fetcher.fetch_all(pmids_list, skip_cached=not args.force)
        
        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total PMIDs in file:     {stats['total_pmids']}")
        logger.info(f"Already cached:          {stats['already_cached']}")
        logger.info(f"Newly fetched:           {stats['newly_fetched']}")
        logger.info(f"Successfully cached:     {stats['successful']}")
        logger.info(f"Failed:                  {stats['failed']}")
        logger.info(f"Total cache size:        {stats['cache_size']} abstracts")
        logger.info(f"Duration:                {stats['duration_seconds']:.2f} seconds")
        
        if stats['newly_fetched'] > 0:
            rate = stats['newly_fetched'] / stats['duration_seconds']
            logger.info(f"Fetch rate:              {rate:.2f} PMIDs/second")
        
        logger.info("=" * 80)
        logger.info("Pre-fetching completed successfully!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pre-fetching failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
