#!/usr/bin/env python3
"""
Multithreaded Data Collection Script

Collects maximum historical data with:
- Multithreaded parallel requests (configurable workers)
- Global rate limiter shared across all threads
- Automatic rate limit detection and backoff
- Progress checkpoints (resume on crash/restart)
- Real-time progress indicators
- Thread-safe database writes

Usage:
    python data/collect_all.py --email your@email.com

    # Use 5 parallel workers (default: 3)
    python data/collect_all.py --email your@email.com --workers 5

    # Resume from checkpoint
    python data/collect_all.py --email your@email.com --resume

    # Aggressive mode (10 workers, faster but riskier)
    python data/collect_all.py --email your@email.com --workers 10 --rps 15
"""

import argparse
import json
import logging
import os
import queue
import signal
import sqlite3
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
file_handler = logging.FileHandler('collect_all.log')
file_handler.setFormatter(log_formatter)

logger = logging.getLogger("CollectAll")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

# Only add stream handler if rich is not available (rich handles console output)
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

    def elapsed(self) -> str:
        elapsed = time.time() - self.started_at
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def rate(self) -> float:
        elapsed = time.time() - self.started_at
        if elapsed > 0:
            return self.items_completed / elapsed * 60
        return 0

    def requests_per_sec(self) -> float:
        elapsed = time.time() - self.started_at
        if elapsed > 0:
            return self.requests_made / elapsed
        return 0


@dataclass
class Checkpoint:
    """Checkpoint for resumable collection."""
    phase: str = "init"
    completed_items: Set[int] = field(default_factory=set)
    last_hourly_collection: float = 0
    last_5m_collection: float = 0
    stats: Dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add_completed(self, item_id: int):
        """Thread-safe add completed item."""
        with self._lock:
            self.completed_items.add(item_id)

    def save(self, path: str = "collection_checkpoint.json"):
        with self._lock:
            data = {
                "phase": self.phase,
                "completed_items": list(self.completed_items),
                "last_hourly_collection": self.last_hourly_collection,
                "last_5m_collection": self.last_5m_collection,
                "stats": self.stats,
                "saved_at": datetime.now().isoformat()
            }
        temp_path = path + ".tmp"
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
        os.rename(temp_path, path)
        logger.debug(f"Checkpoint saved: {len(self.completed_items)} items completed")

    @classmethod
    def load(cls, path: str = "collection_checkpoint.json") -> "Checkpoint":
        if not os.path.exists(path):
            return cls()
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            checkpoint = cls(
                phase=data.get("phase", "init"),
                completed_items=set(data.get("completed_items", [])),
                last_hourly_collection=data.get("last_hourly_collection", 0),
                last_5m_collection=data.get("last_5m_collection", 0),
                stats=data.get("stats", {})
            )
            logger.info(f"Loaded checkpoint: {len(checkpoint.completed_items)} items already completed")
            return checkpoint
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return cls()


class GlobalRateLimiter:
    """Thread-safe global rate limiter shared across all workers."""

    def __init__(self, requests_per_second: float = 10.0):
        self.min_interval = 1.0 / requests_per_second
        self.lock = threading.Lock()
        self.last_request = 0
        self.backoff_until = 0
        self.consecutive_errors = 0
        self.total_requests = 0

        # Adaptive rate limiting
        self.current_interval = self.min_interval
        self.max_interval = 10.0

    def acquire(self) -> float:
        """
        Acquire permission to make a request. Returns wait time.
        Blocks until it's safe to make a request.
        """
        with self.lock:
            now = time.time()

            # Check backoff
            if now < self.backoff_until:
                wait_time = self.backoff_until - now
            else:
                # Calculate wait based on interval
                elapsed = now - self.last_request
                wait_time = max(0, self.current_interval - elapsed)

            # Update last request time (projected)
            self.last_request = now + wait_time
            self.total_requests += 1

            return wait_time

    def wait(self):
        """Acquire and wait."""
        wait_time = self.acquire()
        if wait_time > 0:
            time.sleep(wait_time)

    def report_success(self):
        """Report successful request - can speed up."""
        with self.lock:
            self.consecutive_errors = 0
            self.current_interval = max(
                self.min_interval,
                self.current_interval * 0.95
            )

    def report_error(self, is_rate_limit: bool = False):
        """Report failed request - slow down."""
        with self.lock:
            self.consecutive_errors += 1

            if is_rate_limit:
                backoff_time = min(120, 5 * (2 ** min(self.consecutive_errors, 5)))
                self.backoff_until = time.time() + backoff_time
                self.current_interval = min(self.max_interval, self.current_interval * 3)
                logger.warning(f"Rate limit! All workers backing off for {backoff_time:.1f}s")
            else:
                self.current_interval = min(self.max_interval, self.current_interval * 1.2)

    def get_stats(self) -> Dict:
        """Get rate limiter statistics."""
        with self.lock:
            return {
                "total_requests": self.total_requests,
                "current_interval": self.current_interval,
                "consecutive_errors": self.consecutive_errors,
                "is_backing_off": time.time() < self.backoff_until
            }


class ThreadSafeDB:
    """Thread-safe SQLite database wrapper."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.local = threading.local()
        self.write_lock = threading.Lock()
        self._create_tables()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self.local, 'conn'):
            self.local.conn = sqlite3.connect(self.db_path, timeout=30)
        return self.local.conn

    def _create_tables(self):
        """Create database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                examine TEXT,
                members INTEGER,
                lowalch INTEGER,
                highalch INTEGER,
                ge_limit INTEGER,
                value INTEGER,
                icon TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS latest_prices (
                item_id INTEGER PRIMARY KEY,
                high_price INTEGER,
                high_time INTEGER,
                low_price INTEGER,
                low_time INTEGER,
                collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prices_5m (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                avg_high_price INTEGER,
                high_price_volume INTEGER,
                avg_low_price INTEGER,
                low_price_volume INTEGER,
                collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(item_id, timestamp)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prices_1h (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                avg_high_price INTEGER,
                high_price_volume INTEGER,
                avg_low_price INTEGER,
                low_price_volume INTEGER,
                collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(item_id, timestamp)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS timeseries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id INTEGER NOT NULL,
                timestep TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                avg_high_price INTEGER,
                high_price_volume INTEGER,
                avg_low_price INTEGER,
                low_price_volume INTEGER,
                collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(item_id, timestep, timestamp)
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ts_item ON timeseries(item_id, timestep)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_5m_item ON prices_5m(item_id, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_1h_item ON prices_1h(item_id, timestamp)")

        conn.commit()
        conn.close()

    def save_mapping(self, items: List[Dict]):
        """Save item mapping (thread-safe)."""
        with self.write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            for item in items:
                cursor.execute("""
                    INSERT OR REPLACE INTO items
                    (id, name, examine, members, lowalch, highalch, ge_limit, value, icon, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    item.get("id"),
                    item.get("name"),
                    item.get("examine"),
                    1 if item.get("members") else 0,
                    item.get("lowalch"),
                    item.get("highalch"),
                    item.get("limit"),
                    item.get("value"),
                    item.get("icon")
                ))
            conn.commit()

    def save_timeseries(self, item_id: int, timestep: str, data: List[Dict]) -> int:
        """Save timeseries data (thread-safe). Returns new record count."""
        if not data:
            return 0

        with self.write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            new_records = 0

            for point in data:
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO timeseries
                        (item_id, timestep, timestamp, avg_high_price, high_price_volume, avg_low_price, low_price_volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        item_id,
                        timestep,
                        point.get("timestamp"),
                        point.get("avgHighPrice"),
                        point.get("highPriceVolume"),
                        point.get("avgLowPrice"),
                        point.get("lowPriceVolume")
                    ))
                    if cursor.rowcount > 0:
                        new_records += 1
                except sqlite3.IntegrityError:
                    pass

            conn.commit()
            return new_records

    def save_bulk_prices(self, data: Dict, table: str, timestamp: int) -> int:
        """Save bulk price data (thread-safe)."""
        with self.write_lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            new_records = 0

            for item_id, prices in data.items():
                if prices.get("avgHighPrice") is None and prices.get("avgLowPrice") is None:
                    continue

                try:
                    cursor.execute(f"""
                        INSERT OR IGNORE INTO {table}
                        (item_id, timestamp, avg_high_price, high_price_volume, avg_low_price, low_price_volume)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        int(item_id),
                        timestamp,
                        prices.get("avgHighPrice"),
                        prices.get("highPriceVolume"),
                        prices.get("avgLowPrice"),
                        prices.get("lowPriceVolume")
                    ))
                    if cursor.rowcount > 0:
                        new_records += 1
                except sqlite3.IntegrityError:
                    pass

            conn.commit()
            return new_records

    def get_stats(self) -> Dict:
        """Get database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        stats = {}

        cursor.execute("SELECT COUNT(*) FROM items")
        stats["items"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM timeseries WHERE timestep='24h'")
        stats["timeseries_24h"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM timeseries WHERE timestep='1h'")
        stats["timeseries_1h"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM prices_1h")
        stats["prices_1h"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM prices_5m")
        stats["prices_5m"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT item_id) FROM timeseries")
        stats["items_with_timeseries"] = cursor.fetchone()[0]

        stats["db_size_mb"] = os.path.getsize(self.db_path) / (1024 * 1024)

        conn.close()
        return stats


class Worker:
    """Worker thread for collecting data."""

    def __init__(
        self,
        worker_id: int,
        email: str,
        rate_limiter: GlobalRateLimiter,
        db: ThreadSafeDB,
        stats: CollectionStats
    ):
        self.worker_id = worker_id
        self.email = email
        self.rate_limiter = rate_limiter
        self.db = db
        self.stats = stats
        self.client = GrandExchangeClient(contact_email=f"{email}")

    def collect_item(self, item_id: int) -> Tuple[int, int, Optional[str]]:
        """Collect timeseries for a single item."""
        records_24h = 0
        records_1h = 0
        error = None

        try:
            # Get 24h timeseries
            self.rate_limiter.wait()
            data_24h = self.client.get_timeseries(item_id, "24h")
            self.rate_limiter.report_success()
            self.stats.increment(requests_made=1)
            records_24h = self.db.save_timeseries(item_id, "24h", data_24h)

            # Get 1h timeseries
            self.rate_limiter.wait()
            data_1h = self.client.get_timeseries(item_id, "1h")
            self.rate_limiter.report_success()
            self.stats.increment(requests_made=1)
            records_1h = self.db.save_timeseries(item_id, "1h", data_1h)

        except Exception as e:
            error = str(e)
            is_rate_limit = "429" in error or "rate" in error.lower()
            self.rate_limiter.report_error(is_rate_limit)
            if is_rate_limit:
                self.stats.increment(rate_limits_hit=1)
            logger.debug(f"Worker {self.worker_id}: Error on item {item_id}: {e}")

        return records_24h, records_1h, error


class MultithreadedCollector:
    """Multithreaded data collector."""

    def __init__(
        self,
        email: str,
        db_path: str = "ge_prices.db",
        checkpoint_path: str = "collection_checkpoint.json",
        workers: int = 3,
        requests_per_second: float = 10.0
    ):
        self.email = email
        self.db_path = db_path
        self.checkpoint_path = checkpoint_path
        self.num_workers = workers

        # Shared components
        self.rate_limiter = GlobalRateLimiter(requests_per_second)
        self.db = ThreadSafeDB(db_path)
        self.checkpoint = Checkpoint.load(checkpoint_path)
        self.stats = CollectionStats()

        # Control
        self.running = True
        self.executor: Optional[ThreadPoolExecutor] = None

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"Initialized with {workers} workers, {requests_per_second} RPS limit")

    def _signal_handler(self, signum, frame):
        """Graceful shutdown."""
        logger.info("Shutdown signal received...")
        self.running = False
        self._save_checkpoint()

    def _save_checkpoint(self):
        """Save current progress."""
        self.checkpoint.stats = {
            "items_completed": self.stats.items_completed,
            "timeseries_24h": self.stats.timeseries_24h_collected,
            "timeseries_1h": self.stats.timeseries_1h_collected,
            "errors": self.stats.errors,
            "requests": self.stats.requests_made
        }
        self.checkpoint.save(self.checkpoint_path)

    def _get_mapping(self) -> List[Dict]:
        """Fetch and save item mapping."""
        client = GrandExchangeClient(contact_email=self.email)
        self.rate_limiter.wait()
        mapping = client.get_mapping()
        self.rate_limiter.report_success()
        self.db.save_mapping(mapping)
        return mapping

    def run(self, backfill_only: bool = False):
        """Run collection."""
        if RICH_AVAILABLE:
            self._run_with_rich(backfill_only)
        else:
            self._run_basic(backfill_only)

    def _run_basic(self, backfill_only: bool = False):
        """Run with basic console output."""
        print("\n=== Multithreaded Data Collection ===")
        print(f"Workers: {self.num_workers}")

        # Get mapping
        print("\nFetching item mapping...")
        mapping = self._get_mapping()
        self.stats.items_total = len(mapping)
        print(f"Loaded {len(mapping)} items")

        # Get pending items
        all_item_ids = [item["id"] for item in mapping]
        pending_items = [i for i in all_item_ids if i not in self.checkpoint.completed_items]
        print(f"Pending: {len(pending_items)} items ({len(self.checkpoint.completed_items)} already done)")

        if not pending_items:
            print("All items already collected!")
            self.checkpoint.phase = "continuous"
        else:
            # Create workers
            workers = [
                Worker(i, self.email, self.rate_limiter, self.db, self.stats)
                for i in range(self.num_workers)
            ]

            # Process items with thread pool
            print(f"\nStarting backfill with {self.num_workers} workers...")
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                self.executor = executor

                # Submit all tasks
                future_to_item = {
                    executor.submit(workers[i % self.num_workers].collect_item, item_id): item_id
                    for i, item_id in enumerate(pending_items)
                }

                completed = 0
                for future in as_completed(future_to_item):
                    if not self.running:
                        break

                    item_id = future_to_item[future]
                    try:
                        records_24h, records_1h, error = future.result()

                        if error:
                            self.stats.increment(errors=1)
                        else:
                            self.checkpoint.add_completed(item_id)
                            self.stats.increment(
                                items_completed=1,
                                timeseries_24h_collected=records_24h,
                                timeseries_1h_collected=records_1h,
                                total_records=records_24h + records_1h
                            )

                        completed += 1
                        if completed % 100 == 0:
                            self._save_checkpoint()
                            print(f"Progress: {completed}/{len(pending_items)} | "
                                  f"Records: {self.stats.total_records:,} | "
                                  f"Rate: {self.stats.requests_per_sec():.1f} req/s | "
                                  f"Errors: {self.stats.errors}")

                    except Exception as e:
                        logger.error(f"Future error for item {item_id}: {e}")
                        self.stats.increment(errors=1)

            self.checkpoint.phase = "continuous"
            self._save_checkpoint()
            print(f"\nBackfill complete! {self.stats.items_completed} items, {self.stats.total_records:,} records")

        # Exit if backfill only
        if backfill_only:
            print("\n=== Backfill Only Mode - Exiting ===")
            self._print_summary()
            return

        # Phase 2: Continuous collection
        print("\n=== Continuous Collection ===")
        print("Press Ctrl+C to stop\n")

        client = GrandExchangeClient(contact_email=self.email)

        while self.running:
            now = time.time()

            # 1h collection
            if now - self.checkpoint.last_hourly_collection >= 3600:
                try:
                    self.rate_limiter.wait()
                    data = client.get_1h()
                    self.rate_limiter.report_success()
                    timestamp = int(now // 3600) * 3600
                    records = self.db.save_bulk_prices(data, "prices_1h", timestamp)
                    self.stats.increment(hourly_snapshots=1, total_records=records, requests_made=1)
                    self.checkpoint.last_hourly_collection = now
                    print(f"[1h] +{records:,} records | Total: {self.stats.total_records:,}")
                except Exception as e:
                    logger.error(f"1h collection error: {e}")
                    self.stats.increment(errors=1)

            # 5m collection
            if now - self.checkpoint.last_5m_collection >= 300:
                try:
                    self.rate_limiter.wait()
                    data = client.get_5m()
                    self.rate_limiter.report_success()
                    timestamp = int(now // 300) * 300
                    records = self.db.save_bulk_prices(data, "prices_5m", timestamp)
                    self.stats.increment(five_min_snapshots=1, total_records=records, requests_made=1)
                    self.checkpoint.last_5m_collection = now
                    print(f"[5m] +{records:,} records | Total: {self.stats.total_records:,}")
                except Exception as e:
                    logger.error(f"5m collection error: {e}")
                    self.stats.increment(errors=1)

            # Checkpoint
            if now - self.stats.last_save >= 60:
                self._save_checkpoint()
                self.stats.last_save = now

            time.sleep(10)

        print("\nShutdown complete.")
        self._print_summary()

    def _run_with_rich(self, backfill_only: bool = False):
        """Run with rich console output."""
        console = Console()

        console.print("\n[bold cyan]Multithreaded Data Collection[/bold cyan]")
        console.print(f"Workers: {self.num_workers} | RPS Limit: {self.rate_limiter.min_interval**-1:.1f}")
        console.print("=" * 50)

        # Get mapping
        with console.status("[bold green]Fetching item mapping..."):
            mapping = self._get_mapping()
            self.stats.items_total = len(mapping)

        console.print(f"[green]✓[/green] Loaded {len(mapping):,} items")

        # Get pending
        all_item_ids = [item["id"] for item in mapping]
        pending_items = [i for i in all_item_ids if i not in self.checkpoint.completed_items]
        console.print(f"[yellow]→[/yellow] Pending: {len(pending_items):,} ({len(self.checkpoint.completed_items):,} done)")

        if pending_items:
            console.print(f"\n[bold cyan]Phase 1: Backfill ({self.num_workers} workers)[/bold cyan]")

            # Create workers
            workers = [
                Worker(i, self.email, self.rate_limiter, self.db, self.stats)
                for i in range(self.num_workers)
            ]

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("•"),
                TextColumn("[cyan]{task.fields[rps]:.1f} req/s"),
                TextColumn("•"),
                TextColumn("[green]{task.fields[records]:,} records"),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("→"),
                TimeRemainingColumn(),
                console=console,
                refresh_per_second=2
            ) as progress:
                task = progress.add_task(
                    "[cyan]Collecting...",
                    total=len(pending_items),
                    rps=0,
                    records=0
                )

                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    self.executor = executor

                    future_to_item = {
                        executor.submit(workers[i % self.num_workers].collect_item, item_id): item_id
                        for i, item_id in enumerate(pending_items)
                    }

                    for future in as_completed(future_to_item):
                        if not self.running:
                            executor.shutdown(wait=False, cancel_futures=True)
                            break

                        item_id = future_to_item[future]
                        try:
                            records_24h, records_1h, error = future.result()

                            if error:
                                self.stats.increment(errors=1)
                            else:
                                self.checkpoint.add_completed(item_id)
                                self.stats.increment(
                                    items_completed=1,
                                    timeseries_24h_collected=records_24h,
                                    timeseries_1h_collected=records_1h,
                                    total_records=records_24h + records_1h
                                )

                            progress.update(
                                task,
                                advance=1,
                                rps=self.stats.requests_per_sec(),
                                records=self.stats.total_records
                            )

                            # Checkpoint every 100 items
                            if self.stats.items_completed % 100 == 0:
                                self._save_checkpoint()

                        except Exception as e:
                            self.stats.increment(errors=1)

            self.checkpoint.phase = "continuous"
            self._save_checkpoint()

            console.print(f"\n[green]✓[/green] Backfill complete!")
            console.print(f"  • Items: {self.stats.items_completed:,}")
            console.print(f"  • Records: {self.stats.total_records:,}")
            console.print(f"  • Errors: {self.stats.errors}")
            console.print(f"  • Avg speed: {self.stats.requests_per_sec():.1f} req/s")

        # Exit if backfill only
        if backfill_only:
            console.print(f"\n[bold yellow]Backfill Only Mode - Exiting[/bold yellow]")
            self._print_summary_rich(console)
            return

        # Phase 2
        console.print(f"\n[bold cyan]Phase 2: Continuous Collection[/bold cyan]")
        console.print("[dim]Press Ctrl+C to stop[/dim]\n")

        client = GrandExchangeClient(contact_email=self.email)

        while self.running:
            now = time.time()

            if now - self.checkpoint.last_hourly_collection >= 3600:
                with console.status("[green]Collecting 1h data..."):
                    try:
                        self.rate_limiter.wait()
                        data = client.get_1h()
                        self.rate_limiter.report_success()
                        timestamp = int(now // 3600) * 3600
                        records = self.db.save_bulk_prices(data, "prices_1h", timestamp)
                        self.stats.increment(hourly_snapshots=1, total_records=records, requests_made=1)
                        self.checkpoint.last_hourly_collection = now
                        console.print(f"[green]✓[/green] [1h] +{records:,} records")
                    except Exception as e:
                        console.print(f"[red]✗[/red] [1h] Error: {e}")
                        self.stats.increment(errors=1)

            if now - self.checkpoint.last_5m_collection >= 300:
                with console.status("[green]Collecting 5m data..."):
                    try:
                        self.rate_limiter.wait()
                        data = client.get_5m()
                        self.rate_limiter.report_success()
                        timestamp = int(now // 300) * 300
                        records = self.db.save_bulk_prices(data, "prices_5m", timestamp)
                        self.stats.increment(five_min_snapshots=1, total_records=records, requests_made=1)
                        self.checkpoint.last_5m_collection = now
                        console.print(f"[green]✓[/green] [5m] +{records:,} records")
                    except Exception as e:
                        console.print(f"[red]✗[/red] [5m] Error: {e}")
                        self.stats.increment(errors=1)

            if now - self.stats.last_save >= 60:
                self._save_checkpoint()
                self.stats.last_save = now

            time.sleep(10)

        self._print_summary_rich(console)

    def _print_summary(self):
        """Print final summary."""
        print("\n=== Collection Summary ===")
        print(f"Runtime: {self.stats.elapsed()}")
        print(f"Items: {self.stats.items_completed:,}")
        print(f"24h timeseries: {self.stats.timeseries_24h_collected:,}")
        print(f"1h timeseries: {self.stats.timeseries_1h_collected:,}")
        print(f"Total records: {self.stats.total_records:,}")
        print(f"Requests: {self.stats.requests_made:,}")
        print(f"Avg speed: {self.stats.requests_per_sec():.2f} req/s")
        print(f"Errors: {self.stats.errors}")
        print(f"Rate limits: {self.stats.rate_limits_hit}")

        db_stats = self.db.get_stats()
        print(f"Database size: {db_stats['db_size_mb']:.1f} MB")

    def _print_summary_rich(self, console):
        """Print rich summary."""
        table = Table(title="Collection Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Runtime", self.stats.elapsed())
        table.add_row("Items Processed", f"{self.stats.items_completed:,}")
        table.add_row("24h Timeseries", f"{self.stats.timeseries_24h_collected:,}")
        table.add_row("1h Timeseries", f"{self.stats.timeseries_1h_collected:,}")
        table.add_row("Total Records", f"{self.stats.total_records:,}")
        table.add_row("Requests Made", f"{self.stats.requests_made:,}")
        table.add_row("Avg Speed", f"{self.stats.requests_per_sec():.2f} req/s")
        table.add_row("Errors", f"{self.stats.errors}")
        table.add_row("Rate Limits Hit", f"{self.stats.rate_limits_hit}")

        db_stats = self.db.get_stats()
        table.add_row("Database Size", f"{db_stats['db_size_mb']:.1f} MB")

        console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="Multithreaded OSRS GE Data Collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard collection (3 workers)
    python data/collect_all.py --email your@email.com

    # Fast collection (5 workers, 15 RPS)
    python data/collect_all.py --email your@email.com --workers 5 --rps 15

    # Conservative (2 workers, 5 RPS)
    python data/collect_all.py --email your@email.com --workers 2 --rps 5

    # Check database stats
    python data/collect_all.py --email your@email.com --stats
        """
    )

    parser.add_argument("--email", "-e", required=True, help="Contact email (REQUIRED)")
    parser.add_argument("--db", "-d", default="ge_prices.db", help="Database path")
    parser.add_argument("--checkpoint", "-c", default="collection_checkpoint.json", help="Checkpoint path")
    parser.add_argument("--workers", "-w", type=int, default=8, help="Number of workers (default: 8)")
    parser.add_argument("--rps", type=float, default=20.0, help="Max requests per second (default: 20)")
    parser.add_argument("--fresh", action="store_true", help="Start fresh, ignore checkpoint")
    parser.add_argument("--stats", action="store_true", help="Show stats and exit")
    parser.add_argument("--backfill-only", action="store_true", help="Exit after backfill (skip Phase 2)")

    args = parser.parse_args()

    if args.fresh and os.path.exists(args.checkpoint):
        os.remove(args.checkpoint)
        print("Checkpoint removed, starting fresh")

    collector = MultithreadedCollector(
        email=args.email,
        db_path=args.db,
        checkpoint_path=args.checkpoint,
        workers=args.workers,
        requests_per_second=args.rps
    )

    if args.stats:
        stats = collector.db.get_stats()
        print("\n=== Database Statistics ===")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v:,}")
        return

    try:
        collector.run(backfill_only=args.backfill_only)
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        collector._save_checkpoint()
        print("Checkpoint saved.")


if __name__ == "__main__":
    main()
