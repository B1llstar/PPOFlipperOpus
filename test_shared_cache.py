#!/usr/bin/env python3
"""
Test script to verify TRUE shared cache memory across multiple processes.
This verifies that cache is loaded once in shared memory and accessed by all agents.
Works on Windows and Unix systems.
"""

import multiprocessing
import psutil
import os
from training.cached_market_loader import load_cache, get_shared_memory_name, load_cache_from_shared_memory, cleanup_shared_memory


def worker_process(worker_id, shm_name):
    """Worker that accesses the cache from shared memory."""
    process = psutil.Process(os.getpid())
    
    # Get baseline
    baseline_mb = process.memory_info().rss / (1024 * 1024)
    
    # Load from shared memory
    if shm_name:
        load_cache_from_shared_memory(shm_name)
        print(f"Worker {worker_id}: Loaded from shared memory: {shm_name}")
    else:
        print(f"Worker {worker_id}: ERROR - No shared memory name provided")
        return baseline_mb
    
    from training.cached_market_loader import get_cache
    cache = get_cache()
    
    # Get memory usage after load
    mem_mb = process.memory_info().rss / (1024 * 1024)
    cache_overhead = mem_mb - baseline_mb
    
    print(f"Worker {worker_id}: Memory usage = {mem_mb:.1f} MB (overhead: {cache_overhead:.1f} MB)")
    print(f"Worker {worker_id}: Cache has {len(cache.items)} items")
    
    return mem_mb


if __name__ == "__main__":
    print("Testing TRUE shared cache memory (works on Windows)...\n")
    
    # Get baseline memory
    process = psutil.Process(os.getpid())
    baseline_mb = process.memory_info().rss / (1024 * 1024)
    print(f"Baseline memory (parent): {baseline_mb:.1f} MB")
    
    # Load cache in parent process WITH SHARED MEMORY
    print("\nLoading cache in parent process with shared memory...")
    load_cache("training_cache.json", use_shared_memory=True)
    shm_name = get_shared_memory_name()
    
    if not shm_name:
        print("ERROR: Shared memory not created!")
        exit(1)
    
    print(f"✓ Shared memory created: {shm_name}")
    
    from training.cached_market_loader import get_cache
    cache = get_cache()
    
    after_load_mb = process.memory_info().rss / (1024 * 1024)
    cache_size_mb = after_load_mb - baseline_mb
    print(f"Parent memory after load: {after_load_mb:.1f} MB")
    print(f"Cache size: ~{cache_size_mb:.1f} MB")
    print(f"Cache has {len(cache.items)} items")
    
    # Spawn multiple workers
    print(f"\nSpawning 5 worker processes (passing shared memory name: {shm_name})...")
    num_workers = 5
    
    try:
        with multiprocessing.Pool(num_workers) as pool:
            worker_mems = pool.starmap(worker_process, [(i, shm_name) for i in range(num_workers)])
        
        avg_worker_mem = sum(worker_mems) / len(worker_mems)
        avg_overhead = avg_worker_mem - baseline_mb
        
        print("\n=== RESULTS ===")
        print(f"Cache size in parent: ~{cache_size_mb:.1f} MB")
        print(f"Average worker memory: {avg_worker_mem:.1f} MB")
        print(f"Average worker overhead: ~{avg_overhead:.1f} MB")
        print(f"Expected if NOT shared: ~{cache_size_mb:.1f} MB per worker")
        print(f"Actual overhead per worker: ~{avg_overhead:.1f} MB")
        
        savings_pct = (1 - avg_overhead / cache_size_mb) * 100 if cache_size_mb > 0 else 0
        
        if avg_overhead < cache_size_mb * 0.2:
            print(f"\n✓ SHARED MEMORY IS WORKING! Saving ~{savings_pct:.0f}% memory per worker")
            print(f"  With 45 agents: ~{cache_size_mb:.0f} MB total instead of {cache_size_mb * 45:.0f} MB")
        else:
            print(f"\n✗ WARNING: Overhead is {avg_overhead:.1f} MB (expected < {cache_size_mb * 0.2:.1f} MB)")
    
    finally:
        # Cleanup
        cleanup_shared_memory()
        print("\nShared memory cleaned up")

