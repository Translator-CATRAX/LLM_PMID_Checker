#!/usr/bin/env python3
"""
Utility script to check the status of the PMID cache.
"""
import sys
from pathlib import Path

# Add parent directory to path to import project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pmid_cache import PMIDCache


def main():
    """Display cache statistics."""
    cache = PMIDCache()
    
    print("=" * 60)
    print("PMID Cache Status")
    print("=" * 60)
    print(f"Database location: {cache.db_path}")
    print(f"Total cached abstracts: {cache.count()}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

