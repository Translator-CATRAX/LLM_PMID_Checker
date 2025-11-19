#!/usr/bin/env python3
"""
Script to identify PMIDs that failed to cache properly.
"""
import sys
from pathlib import Path
import sqlite3

# Add parent directory to path to import project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pmid_cache import PMIDCache

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def get_failed_pmids(cache: PMIDCache):
    """Get PMIDs that have errors or no abstract.
    
    Args:
        cache: PMIDCache instance
        
    Returns:
        List of tuples (pmid, error_message)
    """
    conn = cache._get_connection()
    cursor = conn.cursor()
    
    # Query for PMIDs with errors or empty abstracts
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
    """Find PMIDs from TSV that are not in cache at all.
    
    Args:
        tsv_path: Path to TSV file
        cache: PMIDCache instance
        
    Returns:
        Set of missing PMIDs
    """
    if not HAS_PANDAS:
        print("Note: pandas not available, skipping TSV comparison")
        return set()
    
    try:
        df = pd.read_csv(tsv_path, sep='\t')
        all_pmids = set(df['PMID'].dropna().astype(str).unique())
        all_pmids = {pmid for pmid in all_pmids if pmid.strip()}
        
        # Get all PMIDs in cache (including failed ones)
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
        default='data/andy_team_data/processed_mediK_results_v2.tsv',
        help='Path to the TSV file (to check for completely missing PMIDs)'
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
    
    # Get PMIDs with errors or no abstract
    failed_pmids = get_failed_pmids(cache)
    
    if failed_pmids:
        print(f"\n📋 Found {len(failed_pmids)} PMIDs with errors or no abstract:\n")
        
        for i, (pmid, error, title) in enumerate(failed_pmids, 1):
            print(f"{i}. PMID: {pmid}")
            print(f"   Error: {error}")
            if args.verbose and title:
                print(f"   Title: {title}")
            print(f"   Link: https://pubmed.ncbi.nlm.nih.gov/{pmid}")
            print()
    else:
        print("\n✅ No PMIDs with errors found in cache!")
    
    # Check for completely missing PMIDs
    print("-" * 80)
    missing_pmids = get_missing_pmids_from_tsv(args.tsv_file, cache)
    
    if missing_pmids:
        print(f"\n⚠️  Found {len(missing_pmids)} PMIDs from TSV not in cache at all:\n")
        for i, pmid in enumerate(sorted(missing_pmids), 1):
            print(f"{i}. PMID: {pmid}")
            print(f"   Link: https://pubmed.ncbi.nlm.nih.gov/{pmid}")
            print()
    else:
        print("\n✅ All PMIDs from TSV are in cache!")
    
    print("=" * 80)
    
    # Summary
    total_issues = len(failed_pmids) + len(missing_pmids)
    if total_issues > 0:
        print(f"\n📊 Summary: {total_issues} total issues")
        print(f"   - {len(failed_pmids)} with errors/no abstract")
        print(f"   - {len(missing_pmids)} completely missing from cache")
        print("\n💡 Tip: You can manually check these PMIDs at PubMed to see if they have abstracts.")
        print("   Some PMIDs may legitimately have no abstract available.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

