#!/usr/bin/env python3
"""Calculate token usage statistics from token usage file."""
import sys
import statistics

def calculate_token_stats(tokens_file):
    """Calculate and print token usage statistics.
    
    Args:
        tokens_file: Path to file containing token usage data
    """
    prompt_tokens_list = []
    completion_tokens_list = []
    total_tokens_list = []
    cached_tokens_list = []
    uncached_tokens_list = []
    
    # Read token usage data
    with open(tokens_file, 'r') as f:
        lines = f.readlines()
    
    # Skip header
    for line in lines[1:]:
        parts = line.strip().split('\t')
        if len(parts) >= 6:
            try:
                prompt_tokens = int(parts[1])
                completion_tokens = int(parts[2])
                total_tokens = int(parts[3])
                cached_tokens = int(parts[4])
                uncached_tokens = int(parts[5])
                
                prompt_tokens_list.append(prompt_tokens)
                completion_tokens_list.append(completion_tokens)
                total_tokens_list.append(total_tokens)
                cached_tokens_list.append(cached_tokens)
                uncached_tokens_list.append(uncached_tokens)
            except (ValueError, IndexError):
                continue
    
    if not prompt_tokens_list:
        print("No token usage data found!")
        return
    
    # Calculate statistics
    num_requests = len(prompt_tokens_list)
    
    # Prompt tokens (input)
    avg_prompt = statistics.mean(prompt_tokens_list)
    min_prompt = min(prompt_tokens_list)
    max_prompt = max(prompt_tokens_list)
    total_prompt = sum(prompt_tokens_list)
    
    # Completion tokens (output)
    avg_completion = statistics.mean(completion_tokens_list)
    min_completion = min(completion_tokens_list)
    max_completion = max(completion_tokens_list)
    total_completion = sum(completion_tokens_list)
    
    # Total tokens
    avg_total = statistics.mean(total_tokens_list)
    total_all = sum(total_tokens_list)
    
    # Cached tokens
    avg_cached = statistics.mean(cached_tokens_list)
    total_cached = sum(cached_tokens_list)
    cache_hit_rate = (sum(1 for x in cached_tokens_list if x > 0) / num_requests * 100) if num_requests > 0 else 0
    
    # Uncached tokens
    avg_uncached = statistics.mean(uncached_tokens_list)
    total_uncached = sum(uncached_tokens_list)
    
    # Print statistics
    print(f"Number of requests: {num_requests}")
    print("")
    
    print("INPUT TOKENS (Prompt):")
    print(f"  Average:  {avg_prompt:.1f} tokens/request")
    print(f"  Min:      {min_prompt} tokens")
    print(f"  Max:      {max_prompt} tokens")
    print(f"  Total:    {total_prompt:,} tokens")
    print("")
    
    print("OUTPUT TOKENS (Completion):")
    print(f"  Average:  {avg_completion:.1f} tokens/request")
    print(f"  Min:      {min_completion} tokens")
    print(f"  Max:      {max_completion} tokens")
    print(f"  Total:    {total_completion:,} tokens")
    print("")
    
    print("TOTAL TOKENS:")
    print(f"  Average:  {avg_total:.1f} tokens/request")
    print(f"  Total:    {total_all:,} tokens")
    print("")
    
    print("CACHED TOKENS (Prompt Caching):")
    print(f"  Average:  {avg_cached:.1f} tokens/request")
    print(f"  Total:    {total_cached:,} tokens")
    print(f"  Cache hit rate: {cache_hit_rate:.1f}% of requests")
    if total_prompt > 0:
        print(f"  Cached proportion: {(total_cached/total_prompt*100):.1f}% of prompt tokens")
    else:
        print(f"  Cached proportion: N/A (no prompt tokens recorded)")
    print("")
    
    print("UNCACHED PROMPT TOKENS:")
    print(f"  Average:  {avg_uncached:.1f} tokens/request")
    print(f"  Total:    {total_uncached:,} tokens")
    print("")
    
    # Cost estimation (GPT-5 nano pricing - adjust as needed)
    # Typical pricing: Input $0.05/1M, Cached $0.025/1M, Output $0.15/1M
    input_cost_per_1m = 0.05
    cached_cost_per_1m = 0.025
    output_cost_per_1m = 0.15
    
    uncached_input_cost = (total_uncached / 1_000_000) * input_cost_per_1m
    cached_input_cost = (total_cached / 1_000_000) * cached_cost_per_1m
    output_cost = (total_completion / 1_000_000) * output_cost_per_1m
    total_cost = uncached_input_cost + cached_input_cost + output_cost
    
    # Calculate what cost would be without caching
    cost_without_cache = (total_prompt / 1_000_000) * input_cost_per_1m + output_cost
    savings = cost_without_cache - total_cost
    savings_percent = (savings / cost_without_cache * 100) if cost_without_cache > 0 else 0
    
    print("ESTIMATED COST (GPT-5 Nano):")
    print(f"  Uncached input: ${uncached_input_cost:.6f} ({total_uncached:,} tokens @ ${input_cost_per_1m}/1M)")
    print(f"  Cached input:   ${cached_input_cost:.6f} ({total_cached:,} tokens @ ${cached_cost_per_1m}/1M)")
    print(f"  Output:         ${output_cost:.6f} ({total_completion:,} tokens @ ${output_cost_per_1m}/1M)")
    print(f"  Total cost:     ${total_cost:.6f}")
    print("")
    print(f"  Cost without caching: ${cost_without_cache:.6f}")
    print(f"  Savings from caching: ${savings:.6f} ({savings_percent:.1f}%)")
    print("")
    
    print(f"AVERAGE COST PER REQUEST: ${total_cost/num_requests:.6f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python calculate_token_stats.py <tokens_file>")
        sys.exit(1)
    
    tokens_file = sys.argv[1]
    calculate_token_stats(tokens_file)

