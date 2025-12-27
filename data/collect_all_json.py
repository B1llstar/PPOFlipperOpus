#!/usr/bin/env python3
"""
MongoDB Data Collection Script

Collects GE price data and stores it directly in MongoDB for optimal performance.
Eliminates file I/O bottlenecks by writing directly to database.

Usage:
    python data/collect_all_json.py --email your@email.com

    # Use 5 parallel workers (default: 3)
    python data/collect_all_json.py --email your@email.com --workers 5

    # Resume from checkpoint
    python data/collect_all_json.py --email your@email.com --resume

    # Aggressive mode (10 workers, faster but riskier)
    python data/collect_all_json.py --email your@email.com --workers 10 --rps 15
"""

import argparse
import json
import logging
import os
import queue
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.ge_rest_client import GrandExchangeClient, GrandExchangeAPIError, RateLimitError
from data.mongo_data_store import MongoDataStore

# Rich console for pretty output (fallback to basic if not installed)
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn, TaskProgressColumn
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.layout import Layout
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Note: Install 'rich' for better progress display: uv pip install rich")

# Configure logging
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('collect_all_json.log')
file_handler.setFormatter(log_formatter)

logger = logging.getLogger("CollectAllJSON")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

# Only add stream handler if rich is not available
if not RICH_AVAILABLE:
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)


@dataclass
class CollectionStats:
    """Thread-safe collection statistics."""
    started_at: float = field(default_factory=time.time)
    items_total: int = 0
    items_completed: int = 0
    timeseries_24h_collected: int = 0
    timeseries_1h_collected: int = 0
    hourly_snapshots: int = 0
    five_min_snapshots: int = 0
    errors: int = 0
    rate_limits_hit: int = 0
    total_records: int = 0
    last_save: float = field(default_factory=time.time)
    active_workers: int = 0
    requests_made: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def increment(self, **kwargs):
        """Thread-safe increment of stats."""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, getattr(self, key) + value)

    def get_snapshot(self) -> Dict:
        """Get thread-safe snapshot of current stats."""
        with self._lock:
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class RateLimiter:
    """Thread-safe global rate limiter."""

    def __init__(self, requests_per_second: float = 10.0):
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0
        self.lock = threading.Lock()
        self.backoff_until = 0
        self.backoff_multiplier = 1.0

    def wait(self):
        """Wait until we're allowed to make another request."""
        with self.lock:
            now = time.time()
            
            # Check if we're in backoff period
            if now < self.backoff_until:
                wait_time = self.backoff_until - now
                logger.debug(f"In backoff, waiting {wait_time:.2f}s")
                time.sleep(wait_time)
                now = time.time()

            # Enforce minimum interval between requests
            time_since_last = now - self.last_request_time
            if time_since_last < self.min_interval * self.backoff_multiplier:
                wait_time = (self.min_interval * self.backoff_multiplier) - time_since_last
                time.sleep(wait_time)
            
            self.last_request_time = time.time()

    def trigger_backoff(self, duration: float = 5.0):
        """Trigger rate limit backoff."""
        with self.lock:
            self.backoff_until = time.time() + duration
            self.backoff_multiplier = min(self.backoff_multiplier * 1.5, 3.0)
            logger.warning(f"Rate limit triggered, backing off for {duration}s, multiplier: {self.backoff_multiplier:.2f}")

    def reset_backoff(self):
        """Reset backoff on successful requests."""
        with self.lock:
            if self.backoff_multiplier > 1.0:
                self.backoff_multiplier = max(1.0, self.backoff_multiplier * 0.9)


class CheckpointManager:
    """Manages collection checkpoints for resume capability."""

    def __init__(self, checkpoint_file: str = "collection_checkpoint.json"):
        self.checkpoint_file = checkpoint_file
        self.lock = threading.Lock()

    def save(self, completed_items: Set[int], stats: Dict):
        """Save checkpoint."""
        with self.lock:
            checkpoint = {
                "completed_items": list(completed_items),
                "stats": stats,
                "timestamp": datetime.now().isoformat()
            }
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)

    def load(self) -> Optional[Tuple[Set[int], Dict]]:
        """Load checkpoint if it exists."""
        if not os.path.exists(self.checkpoint_file):
            return None

        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            
            completed_items = set(checkpoint.get("completed_items", []))
            stats = checkpoint.get("stats", {})
            timestamp = checkpoint.get("timestamp")
            
            logger.info(f"Loaded checkpoint from {timestamp}")
            logger.info(f"  Completed items: {len(completed_items)}")
            
            return completed_items, stats
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def clear(self):
        """Remove checkpoint file."""
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
            logger.info("Checkpoint cleared")


def collect_item_data(
    item_id: int,
    client: GrandExchangeClient,
    data_store: MongoDataStore,
    stats: CollectionStats,
    rate_limiter: RateLimiter
) -> bool:
    """
    Collect all available data for a single item.
    
    Returns True if successful, False otherwise.
    """
    try:
        stats.increment(active_workers=1)
        
        # Collect 24h timeseries (most comprehensive)
        rate_limiter.wait()
        try:
            timeseries_24h = client.get_timeseries(item_id, "24h")
            if timeseries_24h:
                count = data_store.save_timeseries(item_id, "24h", timeseries_24h)
                stats.increment(timeseries_24h_collected=count, requests_made=1, total_records=count)
                rate_limiter.reset_backoff()
        except RateLimitError as e:
            logger.warning(f"Rate limit on item {item_id} (24h)")
            stats.increment(rate_limits_hit=1)
            rate_limiter.trigger_backoff(10.0)
            raise
        
        # Collect 1h timeseries
        rate_limiter.wait()
        try:
            timeseries_1h = client.get_timeseries(item_id, "1h")
            if timeseries_1h:
                count = data_store.save_timeseries(item_id, "1h", timeseries_1h)
                stats.increment(timeseries_1h_collected=count, requests_made=1, total_records=count)
                rate_limiter.reset_backoff()
        except RateLimitError as e:
            logger.warning(f"Rate limit on item {item_id} (1h)")
            stats.increment(rate_limits_hit=1)
            rate_limiter.trigger_backoff(10.0)
            raise
        
        stats.increment(items_completed=1, active_workers=-1)
        return True
        
    except RateLimitError:
        stats.increment(active_workers=-1)
        raise
    except Exception as e:
        logger.error(f"Error collecting item {item_id}: {e}")
        stats.increment(errors=1, active_workers=-1)
        return False


def main():
    parser = argparse.ArgumentParser(description="Collect GE data to JSON files for MongoDB")
    parser.add_argument("--email", required=True, help="Contact email (required by API)")
    parser.add_argument("--workers", type=int, default=3, help="Number of parallel workers (default: 3)")
    parser.add_argument("--rps", type=float, default=10.0, help="Requests per second limit (default: 10)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--output-dir", default="mongo_data", help="Output directory for JSON files")
    parser.add_argument("--backfill-5m-days", type=int, default=0, help="Backfill 5m data for N days (recommended: 30 for training)")
    parser.add_argument("--backfill-5m-items", type=int, nargs='+', help="Specific item IDs to backfill 5m data for")
    parser.add_argument("--backfill-5m-top-items", type=int, help="Auto-select top N most traded items for 5m backfill")
    parser.add_argument("--skip-timeseries", action="store_true", help="Skip 24h/1h timeseries collection (for 5m-only runs)")
    args = parser.parse_args()

    # Initialize components
    client = GrandExchangeClient(contact_email=args.email)
    data_store = MongoDataStore(
        connection_string="mongodb://localhost:27017/",
        database="ppoflipper",
        collection="GEData"
    )
    stats = CollectionStats()
    rate_limiter = RateLimiter(requests_per_second=args.rps)
    checkpoint_mgr = CheckpointManager()

    # Setup graceful shutdown
    shutdown_event = threading.Event()

    def signal_handler(signum, frame):
        logger.info("Shutdown signal received, finishing current work...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    console = Console() if RICH_AVAILABLE else None

    try:
        # Step 1: Get item mapping
        if console:
            console.print("[bold cyan]Fetching item mapping...[/bold cyan]")
        else:
            print("Fetching item mapping...")

        mapping = client.get_mapping()
        data_store.save_mapping(mapping)
        all_item_ids = [item["id"] for item in mapping]
        stats.items_total = len(all_item_ids)
        
        logger.info(f"Found {len(all_item_ids)} items")

        # Step 2: Collect latest prices
        if console:
            console.print("[bold cyan]Collecting latest prices...[/bold cyan]")
        else:
            print("Collecting latest prices...")

        rate_limiter.wait()
        latest = client.get_latest()
        data_store.save_latest_prices(latest)
        logger.info(f"Collected latest prices for {len(latest)} items")

        # Step 3: Collect 5m snapshot
        if console:
            console.print("[bold cyan]Collecting 5-minute prices...[/bold cyan]")
        else:
            print("Collecting 5-minute prices...")

        rate_limiter.wait()
        prices_5m = client.get_5m()
        count = data_store.save_5m_prices(prices_5m)
        stats.increment(five_min_snapshots=count)
        logger.info(f"Collected 5m prices for {count} items")

        # Step 4: Collect 1h snapshot
        if console:
            console.print("[bold cyan]Collecting 1-hour prices...[/bold cyan]")
        else:
            print("Collecting 1-hour prices...")

        rate_limiter.wait()
        prices_1h = client.get_1h()
        count = data_store.save_1h_prices(prices_1h)
        stats.increment(hourly_snapshots=count)
        logger.info(f"Collected 1h prices for {count} items")

        # Step 4.5: Backfill 5m historical data if requested
        if args.backfill_5m_days > 0:
            if console:
                console.print(f"[bold cyan]Backfilling {args.backfill_5m_days} days of 5-minute data...[/bold cyan]")
            else:
                print(f"Backfilling {args.backfill_5m_days} days of 5-minute data...")
            
            # Determine which items to collect
            target_items = []
            if args.backfill_5m_items:
                target_items = args.backfill_5m_items
                if console:
                    console.print(f"[yellow]Collecting for {len(target_items)} specified items[/yellow]")
                else:
                    print(f"Collecting for {len(target_items)} specified items")
            elif args.backfill_5m_top_items:
                # Get top items by volume from latest 1h prices
                if console:
                    console.print(f"[yellow]Auto-selecting top {args.backfill_5m_top_items} most traded items...[/yellow]")
                else:
                    print(f"Auto-selecting top {args.backfill_5m_top_items} most traded items...")
                
                # Calculate volumes
                item_volumes = {}
                for item_id_str, price_data in prices_1h.items():
                    item_id = int(item_id_str)
                    volume = (price_data.get('highPriceVolume') or 0) + (price_data.get('lowPriceVolume') or 0)
                    item_volumes[item_id] = volume
                
                # Sort by volume and take top N
                sorted_items = sorted(item_volumes.items(), key=lambda x: x[1], reverse=True)
                target_items = [item_id for item_id, vol in sorted_items[:args.backfill_5m_top_items]]
                
                if console:
                    console.print(f"[green]Selected {len(target_items)} top items[/green]")
                else:
                    print(f"Selected {len(target_items)} top items")
            
            # Calculate timestamps (every 5 minutes, going back N days)
            now = int(time.time())
            current_5m = (now // 300) * 300  # Round to nearest 5min
            seconds_back = args.backfill_5m_days * 86400
            timestamps = []
            
            for i in range(0, seconds_back, 300):  # Every 5 minutes
                timestamps.append(current_5m - i)
            
            timestamps.reverse()  # Oldest first
            
            total_requests = len(timestamps)
            if target_items:
                estimated_records = total_requests * len(target_items)
                if console:
                    console.print(f"[yellow]Will make {total_requests} requests for {len(target_items)} items[/yellow]")
                    console.print(f"[yellow]Estimated records: {estimated_records:,}[/yellow]")
                    console.print(f"[yellow]Estimated time: {total_requests / args.rps / 60:.1f} minutes[/yellow]")
                else:
                    print(f"Will make {total_requests} requests. Estimated time: {total_requests / args.rps / 60:.1f} minutes")
            else:
                if console:
                    console.print(f"[yellow]Will make {total_requests} requests for ALL items. This may take a while...[/yellow]")
                else:
                    print(f"Will make {total_requests} requests for ALL items. This may take a while...")
            
            collected_5m = 0
            for idx, timestamp in enumerate(timestamps):
                if shutdown_event.is_set():
                    break
                
                try:
                    rate_limiter.wait()
                    prices_5m_hist = client.get_5m(timestamp=timestamp)
                    
                    # Filter by target items if specified
                    if target_items:
                        prices_5m_hist = {
                            k: v for k, v in prices_5m_hist.items() 
                            if int(k) in target_items
                        }
                    
                    count = data_store.save_5m_prices(prices_5m_hist, timestamp=timestamp)
                    collected_5m += count
                    
                    # Progress update every 100 requests
                    if (idx + 1) % 100 == 0:
                        progress = (idx + 1) / total_requests * 100
                        if console:
                            console.print(f"Progress: {idx + 1}/{total_requests} ({progress:.1f}%) | Collected: {collected_5m:,} records")
                        else:
                            print(f"Progress: {idx + 1}/{total_requests} ({progress:.1f}%)")
                
                except RateLimitError:
                    logger.warning("Rate limited during 5m backfill, backing off...")
                    rate_limiter.trigger_backoff(30.0)
                except Exception as e:
                    logger.error(f"Error backfilling 5m data for timestamp {timestamp}: {e}")
            
            logger.info(f"Backfilled {collected_5m:,} 5m records from {len(timestamps)} timestamps")
            if console:
                console.print(f"[bold green]✓ Backfilled {collected_5m:,} 5m records[/bold green]")

        # Step 5: Load checkpoint if resuming
        completed_items = set()
        if args.resume:
            checkpoint_data = checkpoint_mgr.load()
            if checkpoint_data:
                completed_items, old_stats = checkpoint_data
                if console:
                    console.print(f"[green]Resuming from checkpoint: {len(completed_items)} items already completed[/green]")
                else:
                    print(f"Resuming from checkpoint: {len(completed_items)} items already completed")

        # Filter out completed items
        remaining_items = [item_id for item_id in all_item_ids if item_id not in completed_items]
        stats.items_completed = len(completed_items)

        # Skip timeseries collection if requested (for 5m-only runs)
        if args.skip_timeseries:
            if console:
                console.print("[yellow]Skipping timeseries collection (--skip-timeseries)[/yellow]")
            else:
                print("Skipping timeseries collection")
            remaining_items = []

        if console:
            console.print(f"[bold]Collecting timeseries for {len(remaining_items)} items with {args.workers} workers[/bold]")
        else:
            print(f"Collecting timeseries for {len(remaining_items)} items with {args.workers} workers")

        # Step 6: Parallel timeseries collection
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            
            for item_id in remaining_items:
                if shutdown_event.is_set():
                    break
                
                future = executor.submit(
                    collect_item_data,
                    item_id,
                    client,
                    data_store,
                    stats,
                    rate_limiter
                )
                futures[future] = item_id

            # Process results
            for future in as_completed(futures):
                item_id = futures[future]
                
                try:
                    success = future.result()
                    if success:
                        completed_items.add(item_id)
                        
                        # Periodic checkpoint save
                        if time.time() - stats.last_save > 30:
                            checkpoint_mgr.save(completed_items, stats.get_snapshot())
                            stats.last_save = time.time()
                        
                        # Progress update
                        if stats.items_completed % 50 == 0:
                            elapsed = time.time() - stats.started_at
                            rate = stats.items_completed / elapsed if elapsed > 0 else 0
                            remaining = stats.items_total - stats.items_completed
                            eta = remaining / rate if rate > 0 else 0
                            
                            if console:
                                console.print(
                                    f"Progress: {stats.items_completed}/{stats.items_total} items "
                                    f"({stats.items_completed/stats.items_total*100:.1f}%) | "
                                    f"Rate: {rate:.1f} items/s | ETA: {eta/60:.1f}min"
                                )
                            else:
                                print(
                                    f"Progress: {stats.items_completed}/{stats.items_total} items "
                                    f"({stats.items_completed/stats.items_total*100:.1f}%)"
                                )
                
                except Exception as e:
                    logger.error(f"Failed to process item {item_id}: {e}")
                
                if shutdown_event.is_set():
                    logger.info("Shutdown requested, stopping collection...")
                    break

        # Final save
        checkpoint_mgr.save(completed_items, stats.get_snapshot())
        data_store.close()

        # Merge timeseries if completed
        if stats.items_completed >= stats.items_total:
            if console:
                console.print("[bold cyan]Merging timeseries files...[/bold cyan]")
            else:
                print("Merging timeseries files...")
            
            data_store.merge_all_timeseries_to_single_collection()
            data_store.generate_import_commands()
            checkpoint_mgr.clear()

        # Final stats
        elapsed = time.time() - stats.started_at
        snapshot = stats.get_snapshot()
        
        if console:
            table = Table(title="Collection Complete")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Items Completed", f"{snapshot['items_completed']}/{snapshot['items_total']}")
            table.add_row("24h Timeseries", str(snapshot['timeseries_24h_collected']))
            table.add_row("1h Timeseries", str(snapshot['timeseries_1h_collected']))
            table.add_row("Total Records", str(snapshot['total_records']))
            table.add_row("Errors", str(snapshot['errors']))
            table.add_row("Rate Limits Hit", str(snapshot['rate_limits_hit']))
            table.add_row("Elapsed Time", f"{elapsed/60:.1f} minutes")
            
            console.print(table)
        else:
            print("\n=== Collection Complete ===")
            print(f"Items: {snapshot['items_completed']}/{snapshot['items_total']}")
            print(f"24h Timeseries: {snapshot['timeseries_24h_collected']}")
            print(f"1h Timeseries: {snapshot['timeseries_1h_collected']}")
            print(f"Total Records: {snapshot['total_records']}")
            print(f"Elapsed Time: {elapsed/60:.1f} minutes")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Collection failed: {e}", exc_info=True)
    finally:
        data_store.close()


if __name__ == "__main__":
    main()
