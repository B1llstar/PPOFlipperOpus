#!/usr/bin/env python3
"""
Test script to verify shared cache memory across multiple processes.
This verifies that cache is loaded once and shared, not duplicated.
"""

import multiprocessing
import psutil
import os
from training.cached_market_loader import load_cache, get_cache


def worker_process(worker_id):
    """Worker that accesses the cache."""
    process = psutil.Process(os.getpid())
    
    # Access the cache
    load_cache("training_cache.json")
    cache = get_cache()
    
    # Get memory usage
    mem_mb = process.memory_info().rss / (1024 * 1024)
    
    print(f"Worker {worker_id}: Memory usage = {mem_mb:.1f} MB")
    print(f"Worker {worker_id}: Cache has {len(cache.items)} items")
    
    return mem_mb


if __name__ == "__main__":
    print("Testing shared cache memory...\n")
    
    # Get baseline memory
    process = psutil.Process(os.getpid())
    baseline_mb = process.memory_info().rss / (1024 * 1024)
    print(f"Baseline memory (parent): {baseline_mb:.1f} MB")
    
    # Load cache in parent process
    print("\nLoading cache in parent process...")
    load_cache("training_cache.json")
    cache = get_cache()
    
    after_load_mb = process.memory_info().rss / (1024 * 1024)
    cache_size_mb = after_load_mb - baseline_mb
    print(f"Parent memory after load: {after_load_mb:.1f} MB")
    print(f"Cache size: ~{cache_size_mb:.1f} MB")
    print(f"Cache has {len(cache.items)} items")
    
    # Spawn multiple workers
    print("\nSpawning 5 worker processes...")
    num_workers = 5
    
    with multiprocessing.Pool(num_workers) as pool:
        worker_mems = pool.starmap(worker_process, [(i,) for i in range(num_workers)])
    
    avg_worker_mem = sum(worker_mems) / len(worker_mems)
    
    print("\n=== RESULTS ===")
    print(f"Cache size in parent: ~{cache_size_mb:.1f} MB")
    print(f"Average worker memory: {avg_worker_mem:.1f} MB")
    print(f"Expected if NOT shared: ~{cache_size_mb * num_workers:.1f} MB total for workers")
    print(f"Actual worker total: ~{avg_worker_mem * num_workers:.1f} MB")
    
    if avg_worker_mem < cache_size_mb * 0.5:
        print("\n✓ SHARED MEMORY IS WORKING! Workers use minimal memory.")
    else:
        print("\n✗ WARNING: Workers may be duplicating cache. Memory usage high.")
