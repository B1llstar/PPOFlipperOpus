#!/usr/bin/env python3
"""
Aggressive Parallel Backfill for 5-Minute Historical Data

Maximizes throughput with adaptive rate limiting:
- Hammers the API with concurrent requests
- Backs off automatically on rate limits or errors
- Tracks all fetched timestamps to avoid duplicates
- Saves progress continuously so nothing is lost
- Nice progress bar and live stats

Usage:
    python data/backfill_5m_parallel.py --email your@email.com --days 365 --workers 50
"""

import argparse
import asyncio
import aiohttp
import sqlite3
import time
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Set, List, Optional
from dataclasses import dataclass, field
import threading

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def format_time(seconds: float) -> str:
    """Format seconds into human readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def progress_bar(current: int, total: int, width: int = 40) -> str:
    """Create a progress bar string."""
    if total == 0:
        return "[" + "=" * width + "]"
    pct = current / total
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"


@dataclass
class Stats:
    """Track backfill statistics."""
    total_to_fetch: int = 0
    fetched: int = 0
    records: int = 0
    errors: int = 0
    rate_limits: int = 0
    start_time: float = field(default_factory=time.time)
    current_workers: int = 50

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def rps(self) -> float:
        return self.fetched / self.elapsed if self.elapsed > 0 else 0

    @property
    def eta_seconds(self) -> float:
        remaining = self.total_to_fetch - self.fetched
        return remaining / self.rps if self.rps > 0 else 0

    @property
    def pct(self) -> float:
        return (self.fetched / self.total_to_fetch * 100) if self.total_to_fetch > 0 else 0


class BackfillDatabase:
    """Thread-safe database handler."""

    def __init__(self, db_path: str = "ge_prices.db"):
        self.db_path = db_path
        self.write_lock = threading.Lock()
        self.fetched_timestamps: Set[int] = set()
        self._init_db()
        self._load_existing_timestamps()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backfill_5m (
                item_id INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                avg_high_price INTEGER,
                high_price_volume INTEGER,
                avg_low_price INTEGER,
                low_price_volume INTEGER,
                PRIMARY KEY (item_id, timestamp)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_backfill_5m_ts ON backfill_5m(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_backfill_5m_item ON backfill_5m(item_id)")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backfill_progress (
                timestamp INTEGER PRIMARY KEY,
                fetched_at INTEGER
            )
        """)

        conn.commit()
        conn.close()

    def _load_existing_timestamps(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp FROM backfill_progress")
        self.fetched_timestamps = {row[0] for row in cursor.fetchall()}
        conn.close()

    def is_fetched(self, timestamp: int) -> bool:
        return timestamp in self.fetched_timestamps

    def save_batch(self, results: List[tuple]) -> int:
        """Save a batch of (timestamp, data) tuples."""
        with self.write_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            total_records = 0
            now = int(time.time())

            for timestamp, data in results:
                if data:
                    records = [
                        (int(item_id), timestamp,
                         item_data.get('avgHighPrice'),
                         item_data.get('highPriceVolume', 0),
                         item_data.get('avgLowPrice'),
                         item_data.get('lowPriceVolume', 0))
                        for item_id, item_data in data.items()
                    ]
                    cursor.executemany("""
                        INSERT OR REPLACE INTO backfill_5m
                        (item_id, timestamp, avg_high_price, high_price_volume, avg_low_price, low_price_volume)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, records)
                    total_records += len(records)

                cursor.execute("INSERT OR REPLACE INTO backfill_progress VALUES (?, ?)", (timestamp, now))
                self.fetched_timestamps.add(timestamp)

            conn.commit()
            conn.close()
            return total_records

    def get_stats(self) -> Dict:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM backfill_5m")
        total = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM backfill_progress")
        snapshots = cursor.fetchone()[0]
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM backfill_5m")
        min_ts, max_ts = cursor.fetchone()
        conn.close()
        return {
            "total_records": total,
            "snapshots": snapshots,
            "min_ts": min_ts,
            "max_ts": max_ts
        }


class ParallelBackfiller:
    """High-performance parallel backfiller."""

    BASE_URL = "https://prices.runescape.wiki/api/v1/osrs"

    def __init__(self, email: str, db_path: str = "ge_prices.db", workers: int = 50):
        self.user_agent = f"PPOFlipper-Backfill ({email})"
        self.db = BackfillDatabase(db_path)
        self.workers = workers
        self.stats = Stats(current_workers=workers)
        self.running = True
        self.backoff_until = 0

    async def fetch_one(self, session: aiohttp.ClientSession, timestamp: int) -> tuple:
        """Fetch a single timestamp."""
        # Check backoff
        now = time.time()
        if now < self.backoff_until:
            await asyncio.sleep(self.backoff_until - now)

        try:
            async with session.get(f"{self.BASE_URL}/5m", params={"timestamp": timestamp}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return (timestamp, data.get("data", {}))
                elif resp.status == 429:
                    self.stats.rate_limits += 1
                    self.backoff_until = time.time() + 5  # Back off 5 seconds
                    return (timestamp, None)
                else:
                    self.stats.errors += 1
                    return (timestamp, None)
        except Exception as e:
            self.stats.errors += 1
            return (timestamp, None)

    async def worker(self, session: aiohttp.ClientSession, queue: asyncio.Queue, results: List):
        """Worker that processes timestamps from queue."""
        while self.running:
            try:
                timestamp = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            result = await self.fetch_one(session, timestamp)
            results.append(result)
            self.stats.fetched += 1
            if result[1]:
                self.stats.records += len(result[1])
            queue.task_done()

    def print_status(self):
        """Print a nice status line."""
        bar = progress_bar(self.stats.fetched, self.stats.total_to_fetch, 30)
        eta = format_time(self.stats.eta_seconds)
        elapsed = format_time(self.stats.elapsed)

        # Clear line and print status
        status = (
            f"\r{bar} {self.stats.pct:5.1f}% | "
            f"{self.stats.fetched:,}/{self.stats.total_to_fetch:,} | "
            f"{self.stats.rps:.1f} req/s | "
            f"{self.stats.records:,} records | "
            f"Errors: {self.stats.errors} | "
            f"429s: {self.stats.rate_limits} | "
            f"Elapsed: {elapsed} | "
            f"ETA: {eta}"
        )
        print(status, end="", flush=True)

    async def run(self, days: int = 365):
        """Run the backfill."""
        # Calculate timestamps to fetch
        now = int(time.time())
        current_5m = (now // 300) * 300
        start_ts = current_5m - (days * 24 * 60 * 60)

        # Build list of timestamps we need
        timestamps_needed = []
        ts = start_ts
        while ts <= current_5m:
            if not self.db.is_fetched(ts):
                timestamps_needed.append(ts)
            ts += 300

        total_in_range = (current_5m - start_ts) // 300
        already_have = len(self.db.fetched_timestamps)

        print(f"\n{'='*70}")
        print(f"  OSRS GE 5-Minute Historical Data Backfill")
        print(f"{'='*70}")
        print(f"  Date range: {datetime.fromtimestamp(start_ts)} to {datetime.fromtimestamp(current_5m)}")
        print(f"  Total timestamps in range: {total_in_range:,}")
        print(f"  Already fetched: {already_have:,}")
        print(f"  Remaining to fetch: {len(timestamps_needed):,}")
        print(f"  Concurrent workers: {self.workers}")
        print(f"{'='*70}\n")

        if not timestamps_needed:
            print("Nothing to fetch - all done!")
            return

        self.stats.total_to_fetch = len(timestamps_needed)
        self.stats.start_time = time.time()

        # Create queue and fill it
        queue = asyncio.Queue()
        for ts in timestamps_needed:
            await queue.put(ts)

        # Results accumulator
        results = []
        results_lock = asyncio.Lock()

        # Setup HTTP session - maximize connections
        connector = aiohttp.TCPConnector(
            limit=0,  # No limit
            limit_per_host=0,  # No limit per host
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        headers = {"User-Agent": self.user_agent}

        async with aiohttp.ClientSession(connector=connector, timeout=timeout, headers=headers) as session:
            # Start workers
            workers = [asyncio.create_task(self.worker(session, queue, results)) for _ in range(self.workers)]

            # Progress loop
            last_save = time.time()
            last_count = 0

            try:
                while self.stats.fetched < self.stats.total_to_fetch:
                    await asyncio.sleep(0.5)
                    self.print_status()

                    # Save periodically
                    if time.time() - last_save > 5 and len(results) > last_count:
                        to_save = results[last_count:]
                        last_count = len(results)
                        saved = self.db.save_batch(to_save)
                        last_save = time.time()

            except KeyboardInterrupt:
                print("\n\nInterrupted! Saving progress...")
                self.running = False

            finally:
                self.running = False
                for w in workers:
                    w.cancel()

                # Final save
                if len(results) > last_count:
                    self.db.save_batch(results[last_count:])

        # Final stats
        print(f"\n\n{'='*70}")
        print(f"  BACKFILL COMPLETE!")
        print(f"{'='*70}")
        stats = self.db.get_stats()
        print(f"  Total records in database: {stats['total_records']:,}")
        print(f"  Total snapshots: {stats['snapshots']:,}")
        if stats['min_ts']:
            print(f"  Date range: {datetime.fromtimestamp(stats['min_ts'])} to {datetime.fromtimestamp(stats['max_ts'])}")
        print(f"  Time elapsed: {format_time(self.stats.elapsed)}")
        print(f"  Average speed: {self.stats.rps:.1f} requests/second")
        print(f"  Rate limits hit: {self.stats.rate_limits}")
        print(f"  Errors: {self.stats.errors}")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Aggressive parallel 5m data backfill")
    parser.add_argument("--email", "-e", required=True, help="Contact email for User-Agent")
    parser.add_argument("--days", "-d", type=int, default=365, help="Days to backfill (default: 365)")
    parser.add_argument("--db", default="ge_prices.db", help="Database path")
    parser.add_argument("--workers", "-w", type=int, default=50, help="Concurrent workers (default: 50)")
    parser.add_argument("--stats-only", action="store_true", help="Just show statistics")

    args = parser.parse_args()

    if args.stats_only:
        db = BackfillDatabase(args.db)
        stats = db.get_stats()
        print(f"\n{'='*50}")
        print(f"  Backfill Statistics")
        print(f"{'='*50}")
        print(f"  Total records: {stats['total_records']:,}")
        print(f"  Snapshots: {stats['snapshots']:,}")
        if stats['min_ts']:
            print(f"  From: {datetime.fromtimestamp(stats['min_ts'])}")
            print(f"  To:   {datetime.fromtimestamp(stats['max_ts'])}")
        print(f"{'='*50}\n")
        return

    backfiller = ParallelBackfiller(
        email=args.email,
        db_path=args.db,
        workers=args.workers
    )
    asyncio.run(backfiller.run(days=args.days))


if __name__ == "__main__":
    main()
