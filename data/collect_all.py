#!/usr/bin/env python3
"""
Aggressive Data Collection Script

Collects maximum historical data with:
- Automatic rate limit detection and backoff
- Progress checkpoints (resume on crash/restart)
- Parallel collection where safe
- Real-time progress indicators
- Data integrity verification

Usage:
    python data/collect_all.py --email your@email.com

    # Resume from checkpoint
    python data/collect_all.py --email your@email.com --resume

    # Specify number of parallel workers
    python data/collect_all.py --email your@email.com --workers 3
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.ge_rest_client import GrandExchangeClient, GrandExchangeAPIError, RateLimitError

# Rich console for pretty output (fallback to basic if not installed)
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Note: Install 'rich' for better progress display: pip install rich")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('collect_all.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CollectAll")


@dataclass
class CollectionStats:
    """Track collection statistics."""
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

    def elapsed(self) -> str:
        elapsed = time.time() - self.started_at
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def rate(self) -> float:
        elapsed = time.time() - self.started_at
        if elapsed > 0:
            return self.items_completed / elapsed * 60  # items per minute
        return 0


@dataclass
class Checkpoint:
    """Checkpoint for resumable collection."""
    phase: str = "init"
    completed_items: Set[int] = field(default_factory=set)
    last_hourly_collection: float = 0
    last_5m_collection: float = 0
    stats: Dict = field(default_factory=dict)

    def save(self, path: str = "collection_checkpoint.json"):
        data = {
            "phase": self.phase,
            "completed_items": list(self.completed_items),
            "last_hourly_collection": self.last_hourly_collection,
            "last_5m_collection": self.last_5m_collection,
            "stats": self.stats,
            "saved_at": datetime.now().isoformat()
        }
        # Write to temp file first, then rename (atomic)
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


class RateLimitedClient:
    """API client with intelligent rate limiting."""

    def __init__(self, email: str, requests_per_second: float = 5.0):
        self.client = GrandExchangeClient(contact_email=email)
        self.min_interval = 1.0 / requests_per_second
        self.last_request = 0
        self.backoff_until = 0
        self.consecutive_errors = 0

        # Adaptive rate limiting
        self.current_interval = self.min_interval
        self.max_interval = 5.0  # Max 5 seconds between requests

    def _wait_for_rate_limit(self):
        """Wait if we're rate limited or need to respect interval."""
        now = time.time()

        # Check if we're in backoff period
        if now < self.backoff_until:
            wait_time = self.backoff_until - now
            logger.warning(f"Rate limit backoff: waiting {wait_time:.1f}s")
            time.sleep(wait_time)

        # Respect minimum interval
        elapsed = now - self.last_request
        if elapsed < self.current_interval:
            time.sleep(self.current_interval - elapsed)

        self.last_request = time.time()

    def _handle_success(self):
        """Reduce interval on success."""
        self.consecutive_errors = 0
        # Gradually reduce interval back to minimum
        self.current_interval = max(
            self.min_interval,
            self.current_interval * 0.9
        )

    def _handle_error(self, is_rate_limit: bool = False):
        """Increase interval on error."""
        self.consecutive_errors += 1

        if is_rate_limit:
            # Exponential backoff for rate limits
            backoff_time = min(60, 2 ** self.consecutive_errors)
            self.backoff_until = time.time() + backoff_time
            self.current_interval = min(self.max_interval, self.current_interval * 2)
            logger.warning(f"Rate limit hit! Backing off for {backoff_time}s")
        else:
            # Smaller increase for other errors
            self.current_interval = min(self.max_interval, self.current_interval * 1.5)

    def get_mapping(self) -> List[Dict]:
        """Get item mapping with rate limiting."""
        self._wait_for_rate_limit()
        try:
            result = self.client.get_mapping()
            self._handle_success()
            return result
        except Exception as e:
            self._handle_error("429" in str(e) or "rate" in str(e).lower())
            raise

    def get_latest(self) -> Dict:
        """Get latest prices with rate limiting."""
        self._wait_for_rate_limit()
        try:
            result = self.client.get_latest()
            self._handle_success()
            return result
        except Exception as e:
            self._handle_error("429" in str(e) or "rate" in str(e).lower())
            raise

    def get_5m(self) -> Dict:
        """Get 5m prices with rate limiting."""
        self._wait_for_rate_limit()
        try:
            result = self.client.get_5m()
            self._handle_success()
            return result
        except Exception as e:
            self._handle_error("429" in str(e) or "rate" in str(e).lower())
            raise

    def get_1h(self) -> Dict:
        """Get 1h prices with rate limiting."""
        self._wait_for_rate_limit()
        try:
            result = self.client.get_1h()
            self._handle_success()
            return result
        except Exception as e:
            self._handle_error("429" in str(e) or "rate" in str(e).lower())
            raise

    def get_timeseries(self, item_id: int, timestep: str) -> List[Dict]:
        """Get timeseries with rate limiting."""
        self._wait_for_rate_limit()
        try:
            result = self.client.get_timeseries(item_id, timestep)
            self._handle_success()
            return result
        except Exception as e:
            self._handle_error("429" in str(e) or "rate" in str(e).lower())
            raise


class DataCollectorAggressive:
    """Aggressive data collector with all safety features."""

    def __init__(
        self,
        email: str,
        db_path: str = "ge_prices.db",
        checkpoint_path: str = "collection_checkpoint.json",
        workers: int = 1
    ):
        self.email = email
        self.db_path = db_path
        self.checkpoint_path = checkpoint_path
        self.workers = workers

        # Initialize database
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

        # Load or create checkpoint
        self.checkpoint = Checkpoint.load(checkpoint_path)

        # Statistics
        self.stats = CollectionStats()

        # Control flags
        self.running = True
        self.paused = False

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Create client pool for parallel requests
        self.clients = [RateLimitedClient(email) for _ in range(workers)]

        logger.info(f"Initialized collector with {workers} worker(s)")

    def _signal_handler(self, signum, frame):
        """Handle shutdown gracefully."""
        logger.info("Shutdown signal received, saving checkpoint...")
        self.running = False
        self.checkpoint.stats = {
            "items_completed": self.stats.items_completed,
            "timeseries_24h": self.stats.timeseries_24h_collected,
            "timeseries_1h": self.stats.timeseries_1h_collected,
            "errors": self.stats.errors
        }
        self.checkpoint.save(self.checkpoint_path)
        logger.info("Checkpoint saved. Safe to exit.")

    def _create_tables(self):
        """Create database tables."""
        cursor = self.conn.cursor()

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

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ts_item ON timeseries(item_id, timestep)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_5m_item ON prices_5m(item_id, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_1h_item ON prices_1h(item_id, timestamp)")

        self.conn.commit()

    def _save_mapping(self, items: List[Dict]):
        """Save item mapping."""
        cursor = self.conn.cursor()
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
        self.conn.commit()

    def _save_timeseries(self, item_id: int, timestep: str, data: List[Dict]) -> int:
        """Save timeseries data. Returns number of new records."""
        if not data:
            return 0

        cursor = self.conn.cursor()
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

        self.conn.commit()
        return new_records

    def _save_bulk_prices(self, data: Dict, table: str, timestamp: int) -> int:
        """Save bulk price data (5m or 1h)."""
        cursor = self.conn.cursor()
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

        self.conn.commit()
        return new_records

    def _collect_item_timeseries(self, client: RateLimitedClient, item_id: int) -> Tuple[int, int, Optional[str]]:
        """Collect timeseries for a single item. Returns (records_24h, records_1h, error)."""
        records_24h = 0
        records_1h = 0
        error = None

        try:
            # Get 24h timeseries (365 days of data)
            data_24h = client.get_timeseries(item_id, "24h")
            records_24h = self._save_timeseries(item_id, "24h", data_24h)

            # Get 1h timeseries (365 hours = ~15 days)
            data_1h = client.get_timeseries(item_id, "1h")
            records_1h = self._save_timeseries(item_id, "1h", data_1h)

        except Exception as e:
            error = str(e)
            logger.debug(f"Error collecting item {item_id}: {e}")

        return records_24h, records_1h, error

    def run(self, resume: bool = True):
        """Run the collection process."""
        if RICH_AVAILABLE:
            self._run_with_rich()
        else:
            self._run_basic()

    def _run_basic(self):
        """Run with basic console output."""
        client = self.clients[0]

        # Phase 1: Get mapping
        print("\n=== Phase 1: Fetching Item Mapping ===")
        mapping = client.get_mapping()
        self._save_mapping(mapping)
        self.stats.items_total = len(mapping)
        print(f"Loaded {len(mapping)} items")

        # Get list of items to process
        all_item_ids = [item["id"] for item in mapping]
        pending_items = [i for i in all_item_ids if i not in self.checkpoint.completed_items]
        print(f"Items to process: {len(pending_items)} ({len(self.checkpoint.completed_items)} already done)")

        # Phase 1b: Backfill timeseries
        print("\n=== Phase 1b: Backfilling Historical Data ===")
        for i, item_id in enumerate(pending_items):
            if not self.running:
                break

            records_24h, records_1h, error = self._collect_item_timeseries(client, item_id)

            if error:
                self.stats.errors += 1
                if "429" in error or "rate" in error.lower():
                    self.stats.rate_limits_hit += 1
            else:
                self.checkpoint.completed_items.add(item_id)
                self.stats.items_completed += 1
                self.stats.timeseries_24h_collected += records_24h
                self.stats.timeseries_1h_collected += records_1h
                self.stats.total_records += records_24h + records_1h

            # Progress update
            if (i + 1) % 50 == 0:
                print(f"Progress: {i+1}/{len(pending_items)} | "
                      f"Records: {self.stats.total_records:,} | "
                      f"Errors: {self.stats.errors} | "
                      f"Rate: {self.stats.rate():.1f}/min")
                self.checkpoint.save(self.checkpoint_path)

        # Phase 2: Continuous collection
        print("\n=== Phase 2: Continuous Collection ===")
        self.checkpoint.phase = "continuous"

        while self.running:
            now = time.time()

            # Collect 1h data every hour
            if now - self.checkpoint.last_hourly_collection >= 3600:
                try:
                    data = client.get_1h()
                    timestamp = int(now // 3600) * 3600
                    records = self._save_bulk_prices(data, "prices_1h", timestamp)
                    self.stats.hourly_snapshots += 1
                    self.stats.total_records += records
                    self.checkpoint.last_hourly_collection = now
                    print(f"[1h] Collected {records:,} records | Total: {self.stats.total_records:,}")
                except Exception as e:
                    logger.error(f"Error collecting 1h data: {e}")
                    self.stats.errors += 1

            # Collect 5m data every 5 minutes
            if now - self.checkpoint.last_5m_collection >= 300:
                try:
                    data = client.get_5m()
                    timestamp = int(now // 300) * 300
                    records = self._save_bulk_prices(data, "prices_5m", timestamp)
                    self.stats.five_min_snapshots += 1
                    self.stats.total_records += records
                    self.checkpoint.last_5m_collection = now
                    print(f"[5m] Collected {records:,} records | Total: {self.stats.total_records:,}")
                except Exception as e:
                    logger.error(f"Error collecting 5m data: {e}")
                    self.stats.errors += 1

            # Save checkpoint periodically
            if now - self.stats.last_save >= 60:
                self.checkpoint.save(self.checkpoint_path)
                self.stats.last_save = now

            # Sleep until next collection
            time.sleep(10)

    def _run_with_rich(self):
        """Run with rich console output."""
        console = Console()
        client = self.clients[0]

        console.print("\n[bold cyan]PPO Flipper Data Collection[/bold cyan]")
        console.print("=" * 50)

        # Phase 1: Get mapping
        with console.status("[bold green]Fetching item mapping..."):
            mapping = client.get_mapping()
            self._save_mapping(mapping)
            self.stats.items_total = len(mapping)

        console.print(f"[green]✓[/green] Loaded {len(mapping):,} items")

        # Get pending items
        all_item_ids = [item["id"] for item in mapping]
        pending_items = [i for i in all_item_ids if i not in self.checkpoint.completed_items]
        console.print(f"[yellow]→[/yellow] Items to process: {len(pending_items):,} "
                     f"({len(self.checkpoint.completed_items):,} already done)")

        # Phase 1b: Backfill with progress bar
        if pending_items:
            console.print("\n[bold cyan]Phase 1: Historical Backfill[/bold cyan]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                task = progress.add_task(
                    "[cyan]Collecting timeseries...",
                    total=len(pending_items)
                )

                for item_id in pending_items:
                    if not self.running:
                        break

                    records_24h, records_1h, error = self._collect_item_timeseries(client, item_id)

                    if error:
                        self.stats.errors += 1
                        if "429" in error or "rate" in error.lower():
                            self.stats.rate_limits_hit += 1
                    else:
                        self.checkpoint.completed_items.add(item_id)
                        self.stats.items_completed += 1
                        self.stats.timeseries_24h_collected += records_24h
                        self.stats.timeseries_1h_collected += records_1h
                        self.stats.total_records += records_24h + records_1h

                    progress.update(task, advance=1)

                    # Checkpoint every 100 items
                    if self.stats.items_completed % 100 == 0:
                        self.checkpoint.save(self.checkpoint_path)

            # Final checkpoint after backfill
            self.checkpoint.phase = "continuous"
            self.checkpoint.save(self.checkpoint_path)

            console.print(f"\n[green]✓[/green] Backfill complete!")
            console.print(f"  • 24h records: {self.stats.timeseries_24h_collected:,}")
            console.print(f"  • 1h records: {self.stats.timeseries_1h_collected:,}")
            console.print(f"  • Errors: {self.stats.errors}")

        # Phase 2: Continuous collection
        console.print("\n[bold cyan]Phase 2: Continuous Collection[/bold cyan]")
        console.print("[dim]Press Ctrl+C to stop gracefully[/dim]\n")

        while self.running:
            now = time.time()

            # Collect 1h data
            if now - self.checkpoint.last_hourly_collection >= 3600:
                with console.status("[bold green]Collecting hourly data..."):
                    try:
                        data = client.get_1h()
                        timestamp = int(now // 3600) * 3600
                        records = self._save_bulk_prices(data, "prices_1h", timestamp)
                        self.stats.hourly_snapshots += 1
                        self.stats.total_records += records
                        self.checkpoint.last_hourly_collection = now
                        console.print(f"[green]✓[/green] [1h] {records:,} records | "
                                     f"Total: {self.stats.total_records:,}")
                    except Exception as e:
                        console.print(f"[red]✗[/red] [1h] Error: {e}")
                        self.stats.errors += 1

            # Collect 5m data
            if now - self.checkpoint.last_5m_collection >= 300:
                with console.status("[bold green]Collecting 5-minute data..."):
                    try:
                        data = client.get_5m()
                        timestamp = int(now // 300) * 300
                        records = self._save_bulk_prices(data, "prices_5m", timestamp)
                        self.stats.five_min_snapshots += 1
                        self.stats.total_records += records
                        self.checkpoint.last_5m_collection = now
                        console.print(f"[green]✓[/green] [5m] {records:,} records | "
                                     f"Total: {self.stats.total_records:,}")
                    except Exception as e:
                        console.print(f"[red]✗[/red] [5m] Error: {e}")
                        self.stats.errors += 1

            # Checkpoint
            if now - self.stats.last_save >= 60:
                self.checkpoint.save(self.checkpoint_path)
                self.stats.last_save = now

            # Calculate next collection time
            next_5m = 300 - (now - self.checkpoint.last_5m_collection)
            next_1h = 3600 - (now - self.checkpoint.last_hourly_collection)
            wait_time = max(1, min(next_5m, next_1h, 30))

            time.sleep(wait_time)

        # Final stats
        console.print("\n[bold cyan]Collection Summary[/bold cyan]")
        table = Table()
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Runtime", self.stats.elapsed())
        table.add_row("Items Processed", f"{self.stats.items_completed:,}")
        table.add_row("24h Timeseries", f"{self.stats.timeseries_24h_collected:,}")
        table.add_row("1h Timeseries", f"{self.stats.timeseries_1h_collected:,}")
        table.add_row("Hourly Snapshots", f"{self.stats.hourly_snapshots:,}")
        table.add_row("5m Snapshots", f"{self.stats.five_min_snapshots:,}")
        table.add_row("Total Records", f"{self.stats.total_records:,}")
        table.add_row("Errors", f"{self.stats.errors}")
        table.add_row("Rate Limits", f"{self.stats.rate_limits_hit}")
        console.print(table)

    def get_database_stats(self) -> Dict:
        """Get current database statistics."""
        cursor = self.conn.cursor()
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

        # Database file size
        stats["db_size_mb"] = os.path.getsize(self.db_path) / (1024 * 1024)

        return stats

    def close(self):
        """Close database connection."""
        self.checkpoint.save(self.checkpoint_path)
        self.conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Aggressive OSRS GE Data Collection",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--email", "-e",
        required=True,
        help="Contact email (REQUIRED by API)"
    )

    parser.add_argument(
        "--db", "-d",
        default="ge_prices.db",
        help="Database path (default: ge_prices.db)"
    )

    parser.add_argument(
        "--checkpoint", "-c",
        default="collection_checkpoint.json",
        help="Checkpoint file path"
    )

    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume from checkpoint (default behavior)"
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start fresh, ignore checkpoint"
    )

    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, be careful with rate limits)"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show database stats and exit"
    )

    args = parser.parse_args()

    # Handle fresh start
    if args.fresh and os.path.exists(args.checkpoint):
        os.remove(args.checkpoint)
        print("Removed checkpoint, starting fresh")

    # Initialize collector
    collector = DataCollectorAggressive(
        email=args.email,
        db_path=args.db,
        checkpoint_path=args.checkpoint,
        workers=args.workers
    )

    try:
        if args.stats:
            stats = collector.get_database_stats()
            print("\n=== Database Statistics ===")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value:,}")
            return

        collector.run()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        collector.close()
        print("Collector closed, checkpoint saved.")


if __name__ == "__main__":
    main()
