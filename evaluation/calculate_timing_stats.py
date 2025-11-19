#!/usr/bin/env python3
"""
Calculate timing statistics from timing results file.
Computes average runtime and standard deviation.
"""

import sys
import pandas as pd
import numpy as np


def calculate_timing_stats(timing_file):
    """
    Calculate timing statistics from timing file.
    
    Args:
        timing_file: Path to timing file with PMID and runtime_seconds columns
    """
    try:
        # Read timing data
        df = pd.read_csv(timing_file, sep='\t')
        
        if df.empty:
            print("No timing data available.")
            return
        
        # Calculate statistics
        runtimes = df['runtime_seconds'].values
        mean_time = np.mean(runtimes)
        std_time = np.std(runtimes, ddof=1)  # Sample standard deviation
        median_time = np.median(runtimes)
        min_time = np.min(runtimes)
        max_time = np.max(runtimes)
        total_time = np.sum(runtimes)
        count = len(runtimes)
        
        # Print statistics
        print(f"Number of queries: {count}")
        print(f"Total runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"Average runtime: {mean_time:.2f} seconds")
        print(f"Standard deviation: {std_time:.2f} seconds")
        print(f"Median runtime: {median_time:.2f} seconds")
        print(f"Min runtime: {min_time:.2f} seconds")
        print(f"Max runtime: {max_time:.2f} seconds")
        
    except FileNotFoundError:
        print(f"Error: Timing file '{timing_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error calculating timing statistics: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python calculate_timing_stats.py <timing_file>", file=sys.stderr)
        sys.exit(1)
    
    timing_file = sys.argv[1]
    calculate_timing_stats(timing_file)

