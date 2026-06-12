#!/usr/bin/env python3
"""
Script to identify PMIDs that failed to cache properly.
"""
import sys
from pathlib import Path
import sqlite3

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pmid_cache import PMIDCache

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


def get_failed_pmids(cache: PMIDCache):
    """Get PMIDs that have errors or no abstract."""
    conn = cache._get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT pmid, error, abstract, title
        FROM pmid_abstracts
        WHERE error IS NOT NULL 
           OR abstract IS NULL 
           OR abstract = ''
        ORDER BY pmid
    """)
    
    failed = []
    for row in cursor.fetchall():
        pmid = row['pmid']
        error = row['error']
        abstract = row['abstract']
        title = row['title']
        
        if error:
            failed.append((pmid, error, title))
        elif not abstract or abstract.strip() == '':
            failed.append((pmid, "No abstract available", title))
    
    return failed


def get_missing_pmids_from_tsv(tsv_path: str, cache: PMIDCache):
    """Find PMIDs from a Parquet or TSV file that are not in cache."""
    if not HAS_POLARS:
        print("Note: polars not available, skipping file comparison")
        return set()
    
    try:
        from pathlib import Path as _Path
        ext = _Path(tsv_path).suffix.lower()
        if ext in ('.parquet', '.pq'):
            df = pl.read_parquet(tsv_path, columns=['PMID'])
        else:
            df = pl.read_csv(tsv_path, separator='\t', infer_schema_length=0)
        all_pmids = set(
            df['PMID']
            .drop_nulls()
            .cast(pl.Utf8)
            .unique()
            .to_list()
        )
        all_pmids = {pmid for pmid in all_pmids if pmid.strip()}
        
        conn = cache._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT pmid FROM pmid_abstracts")
        cached_pmids = {row['pmid'] for row in cursor.fetchall()}
        
        missing = all_pmids - cached_pmids
        return missing
        
    except Exception as e:
        print(f"Error checking TSV: {e}")
        return set()


def main():
    """Display failed PMIDs."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Identify PMIDs that failed to cache"
    )
    parser.add_argument(
        '--tsv-file',
        default='data/semmedb_kgx/semmeddb_edges_extracted.parquet',
        help='Path to the input file (.parquet or .tsv; to check for missing PMIDs)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed information including titles'
    )
    
    args = parser.parse_args()
    
    cache = PMIDCache()
    
    print("=" * 80)
    print("Failed PMID Analysis")
    print("=" * 80)
    
    failed_pmids = get_failed_pmids(cache)
    
    if failed_pmids:
        print(f"\nFound {len(failed_pmids)} PMIDs with errors or no abstract:\n")
        
        for i, (pmid, error, title) in enumerate(failed_pmids, 1):
            print(f"{i}. PMID: {pmid}")
            print(f"   Error: {error}")
            if args.verbose and title:
                print(f"   Title: {title}")
            print(f"   Link: https://pubmed.ncbi.nlm.nih.gov/{pmid}")
            print()
    else:
        print("\nNo PMIDs with errors found in cache!")
    
    print("-" * 80)
    missing_pmids = get_missing_pmids_from_tsv(args.tsv_file, cache)
    
    if missing_pmids:
        print(f"\nFound {len(missing_pmids)} PMIDs from TSV not in cache at all:\n")
        for i, pmid in enumerate(sorted(missing_pmids), 1):
            print(f"{i}. PMID: {pmid}")
            print(f"   Link: https://pubmed.ncbi.nlm.nih.gov/{pmid}")
            print()
    else:
        print("\nAll PMIDs from TSV are in cache!")
    
    print("=" * 80)
    
    total_issues = len(failed_pmids) + len(missing_pmids)
    if total_issues > 0:
        print(f"\nSummary: {total_issues} total issues")
        print(f"   - {len(failed_pmids)} with errors/no abstract")
        print(f"   - {len(missing_pmids)} completely missing from cache")
        print("\nTip: You can manually check these PMIDs at PubMed to see if they have abstracts.")
        print("   Some PMIDs may legitimately have no abstract available.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
