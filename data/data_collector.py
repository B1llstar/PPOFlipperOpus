#!/usr/bin/env python3
"""
OSRS Grand Exchange Data Collection Pipeline

Collects price data from the Real-time Prices API and stores it in SQLite
for PPO flipper bot training.

Usage:
    # One-time collection
    python data_collector.py --email your@email.com

    # Continuous collection (daemon mode)
    python data_collector.py --email your@email.com --daemon

    # Collect specific timesteps
    python data_collector.py --email your@email.com --timesteps 5m 1h

    # Backfill historical data for specific items
    python data_collector.py --email your@email.com --backfill --items 2 554 555

Requirements:
    - Contact email is REQUIRED (API policy)
    - SQLite3 (included with Python)
    - requests library
"""

import argparse
import logging
import os
import signal
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.ge_rest_client import GrandExchangeClient, GrandExchangeAPIError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_collector.log')
    ]
)
logger = logging.getLogger("DataCollector")


class PriceDatabase:
    """SQLite database for storing GE price data."""

    def __init__(self, db_path: str = "ge_prices.db"):
        """
        Initialize the database connection.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info(f"Database initialized at {db_path}")

    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()

        # Item metadata table
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

        # Latest prices table (most recent snapshot)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS latest_prices (
                item_id INTEGER PRIMARY KEY,
                high_price INTEGER,
                high_time INTEGER,
                low_price INTEGER,
                low_time INTEGER,
                collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (item_id) REFERENCES items(id)
            )
        """)

        # 5-minute price history
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
                FOREIGN KEY (item_id) REFERENCES items(id),
                UNIQUE(item_id, timestamp)
            )
        """)

        # 1-hour price history
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
                FOREIGN KEY (item_id) REFERENCES items(id),
                UNIQUE(item_id, timestamp)
            )
        """)

        # Timeseries data (for individual item deep history)
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
                FOREIGN KEY (item_id) REFERENCES items(id),
                UNIQUE(item_id, timestep, timestamp)
            )
        """)

        # Collection metadata (for tracking what we've collected)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS collection_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_type TEXT NOT NULL,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                items_collected INTEGER,
                errors INTEGER DEFAULT 0,
                notes TEXT
            )
        """)

        # Create indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prices_5m_item_ts ON prices_5m(item_id, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prices_1h_item_ts ON prices_1h(item_id, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timeseries_item_ts ON timeseries(item_id, timestep, timestamp)")

        self.conn.commit()
        logger.info("Database tables created/verified")

    def save_mapping(self, items: List[Dict]):
        """
        Save item mapping data.

        Args:
            items: List of item metadata dictionaries from API
        """
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
        logger.info(f"Saved mapping for {len(items)} items")

    def save_latest_prices(self, prices: Dict):
        """
        Save latest price snapshot.

        Args:
            prices: Dictionary of item_id -> price data
        """
        cursor = self.conn.cursor()
        count = 0

        for item_id, data in prices.items():
            if data.get("high") is None and data.get("low") is None:
                continue

            cursor.execute("""
                INSERT OR REPLACE INTO latest_prices
                (item_id, high_price, high_time, low_price, low_time, collected_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                int(item_id),
                data.get("high"),
                data.get("highTime"),
                data.get("low"),
                data.get("lowTime")
            ))
            count += 1

        self.conn.commit()
        logger.info(f"Saved latest prices for {count} items")
        return count

    def save_5m_prices(self, prices: Dict, timestamp: Optional[int] = None):
        """
        Save 5-minute price data.

        Args:
            prices: Dictionary of item_id -> 5m price data
            timestamp: Unix timestamp for this data point
        """
        if timestamp is None:
            timestamp = int(time.time())
            # Round to nearest 5 minutes
            timestamp = (timestamp // 300) * 300

        cursor = self.conn.cursor()
        count = 0

        for item_id, data in prices.items():
            if data.get("avgHighPrice") is None and data.get("avgLowPrice") is None:
                continue

            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO prices_5m
                    (item_id, timestamp, avg_high_price, high_price_volume, avg_low_price, low_price_volume)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    int(item_id),
                    timestamp,
                    data.get("avgHighPrice"),
                    data.get("highPriceVolume"),
                    data.get("avgLowPrice"),
                    data.get("lowPriceVolume")
                ))
                if cursor.rowcount > 0:
                    count += 1
            except sqlite3.IntegrityError:
                pass  # Duplicate, skip

        self.conn.commit()
        logger.info(f"Saved 5m prices for {count} items at timestamp {timestamp}")
        return count

    def save_1h_prices(self, prices: Dict, timestamp: Optional[int] = None):
        """
        Save 1-hour price data.

        Args:
            prices: Dictionary of item_id -> 1h price data
            timestamp: Unix timestamp for this data point
        """
        if timestamp is None:
            timestamp = int(time.time())
            # Round to nearest hour
            timestamp = (timestamp // 3600) * 3600

        cursor = self.conn.cursor()
        count = 0

        for item_id, data in prices.items():
            if data.get("avgHighPrice") is None and data.get("avgLowPrice") is None:
                continue

            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO prices_1h
                    (item_id, timestamp, avg_high_price, high_price_volume, avg_low_price, low_price_volume)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    int(item_id),
                    timestamp,
                    data.get("avgHighPrice"),
                    data.get("highPriceVolume"),
                    data.get("avgLowPrice"),
                    data.get("lowPriceVolume")
                ))
                if cursor.rowcount > 0:
                    count += 1
            except sqlite3.IntegrityError:
                pass  # Duplicate, skip

        self.conn.commit()
        logger.info(f"Saved 1h prices for {count} items at timestamp {timestamp}")
        return count

    def save_timeseries(self, item_id: int, timestep: str, data: List[Dict]):
        """
        Save timeseries data for an item.

        Args:
            item_id: Item ID
            timestep: Timestep ("5m", "1h", "6h", "24h")
            data: List of timeseries data points
        """
        cursor = self.conn.cursor()
        count = 0

        for point in data:
            if point.get("avgHighPrice") is None and point.get("avgLowPrice") is None:
                continue

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
                    count += 1
            except sqlite3.IntegrityError:
                pass  # Duplicate, skip

        self.conn.commit()
        logger.debug(f"Saved {count} timeseries points for item {item_id} ({timestep})")
        return count

    def get_item_ids(self, min_volume: int = 0) -> List[int]:
        """
        Get list of item IDs, optionally filtered by volume.

        Args:
            min_volume: Minimum total volume to include

        Returns:
            List of item IDs
        """
        cursor = self.conn.cursor()

        if min_volume > 0:
            # Get items that have had recent trading volume
            cursor.execute("""
                SELECT DISTINCT item_id FROM prices_1h
                WHERE high_price_volume + low_price_volume >= ?
                ORDER BY item_id
            """, (min_volume,))
        else:
            cursor.execute("SELECT id FROM items ORDER BY id")

        return [row[0] for row in cursor.fetchall()]

    def get_collection_stats(self) -> Dict:
        """Get statistics about collected data."""
        cursor = self.conn.cursor()

        stats = {}

        cursor.execute("SELECT COUNT(*) FROM items")
        stats["total_items"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM prices_5m")
        stats["total_5m_records"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM prices_1h")
        stats["total_1h_records"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM timeseries")
        stats["total_timeseries_records"] = cursor.fetchone()[0]

        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM prices_5m")
        row = cursor.fetchone()
        if row[0]:
            stats["5m_date_range"] = (
                datetime.fromtimestamp(row[0]).isoformat(),
                datetime.fromtimestamp(row[1]).isoformat()
            )

        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM prices_1h")
        row = cursor.fetchone()
        if row[0]:
            stats["1h_date_range"] = (
                datetime.fromtimestamp(row[0]).isoformat(),
                datetime.fromtimestamp(row[1]).isoformat()
            )

        return stats

    def close(self):
        """Close database connection."""
        self.conn.close()


class DataCollector:
    """Orchestrates data collection from the API to the database."""

    def __init__(
        self,
        email: str,
        db_path: str = "ge_prices.db",
        project_name: str = "PPOFlipper"
    ):
        """
        Initialize the data collector.

        Args:
            email: Contact email (REQUIRED by API)
            db_path: Path to SQLite database
            project_name: Project name for User-Agent
        """
        self.client = GrandExchangeClient(
            contact_email=email,
            project_name=project_name
        )
        self.db = PriceDatabase(db_path)
        self._running = False
        self._stop_requested = False

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, requesting stop...")
        self._stop_requested = True

    def collect_mapping(self) -> int:
        """
        Collect and save item mapping data.

        Returns:
            Number of items saved
        """
        logger.info("Collecting item mapping...")
        mapping = self.client.get_mapping()
        self.db.save_mapping(mapping)
        return len(mapping)

    def collect_latest(self) -> int:
        """
        Collect and save latest prices.

        Returns:
            Number of items saved
        """
        logger.info("Collecting latest prices...")
        latest = self.client.get_latest()
        return self.db.save_latest_prices(latest)

    def collect_5m(self) -> int:
        """
        Collect and save 5-minute price data.

        Returns:
            Number of items saved
        """
        logger.info("Collecting 5m prices...")
        data = self.client.get_5m()
        return self.db.save_5m_prices(data)

    def collect_1h(self) -> int:
        """
        Collect and save 1-hour price data.

        Returns:
            Number of items saved
        """
        logger.info("Collecting 1h prices...")
        data = self.client.get_1h()
        return self.db.save_1h_prices(data)

    def backfill_timeseries(
        self,
        item_ids: Optional[List[int]] = None,
        timestep: str = "1h"
    ) -> int:
        """
        Backfill historical timeseries data for items.

        Args:
            item_ids: List of item IDs to backfill. If None, uses all items.
            timestep: Timestep to fetch ("5m", "1h", "6h", "24h")

        Returns:
            Total number of data points saved
        """
        if item_ids is None:
            item_ids = self.db.get_item_ids()

        total_saved = 0
        errors = 0

        logger.info(f"Backfilling timeseries ({timestep}) for {len(item_ids)} items...")

        for i, item_id in enumerate(item_ids):
            if self._stop_requested:
                logger.info("Stop requested, aborting backfill...")
                break

            try:
                data = self.client.get_timeseries(item_id, timestep)
                saved = self.db.save_timeseries(item_id, timestep, data)
                total_saved += saved

                if (i + 1) % 100 == 0:
                    logger.info(f"Progress: {i + 1}/{len(item_ids)} items processed")

                # Be nice to the API - small delay between requests
                time.sleep(0.2)

            except GrandExchangeAPIError as e:
                logger.warning(f"Error fetching timeseries for item {item_id}: {e}")
                errors += 1
                if errors > 10:
                    logger.error("Too many errors, aborting backfill")
                    break

        logger.info(f"Backfill complete. Saved {total_saved} data points, {errors} errors")
        return total_saved

    def run_collection_cycle(self, collect_5m: bool = True, collect_1h: bool = True):
        """
        Run a single collection cycle.

        Args:
            collect_5m: Whether to collect 5m data
            collect_1h: Whether to collect 1h data
        """
        try:
            # Always collect latest prices
            self.collect_latest()

            if collect_5m:
                self.collect_5m()

            if collect_1h:
                self.collect_1h()

        except GrandExchangeAPIError as e:
            logger.error(f"Collection error: {e}")

    def run_daemon(
        self,
        interval_5m: int = 300,
        interval_1h: int = 3600
    ):
        """
        Run continuous data collection.

        Args:
            interval_5m: Seconds between 5m collections (default: 5 minutes)
            interval_1h: Seconds between 1h collections (default: 1 hour)
        """
        logger.info("Starting daemon mode...")
        self._running = True

        last_5m = 0
        last_1h = 0

        # Initial collection
        self.collect_mapping()
        self.run_collection_cycle()
        last_5m = last_1h = time.time()

        while self._running and not self._stop_requested:
            current_time = time.time()

            collect_5m = (current_time - last_5m) >= interval_5m
            collect_1h = (current_time - last_1h) >= interval_1h

            if collect_5m or collect_1h:
                self.run_collection_cycle(collect_5m, collect_1h)

                if collect_5m:
                    last_5m = current_time
                if collect_1h:
                    last_1h = current_time

            # Sleep until next collection (check every 30 seconds for stop signal)
            next_5m = last_5m + interval_5m - current_time
            next_1h = last_1h + interval_1h - current_time
            sleep_time = min(30, max(1, min(next_5m, next_1h)))

            time.sleep(sleep_time)

        logger.info("Daemon stopped")
        self._running = False

    def get_stats(self) -> Dict:
        """Get collection statistics."""
        return self.db.get_collection_stats()

    def close(self):
        """Clean up resources."""
        self.db.close()


def main():
    parser = argparse.ArgumentParser(
        description="OSRS Grand Exchange Data Collector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # One-time collection of all data
    python data_collector.py --email your@email.com

    # Run as daemon for continuous collection
    python data_collector.py --email your@email.com --daemon

    # Backfill historical data for high-volume items
    python data_collector.py --email your@email.com --backfill --timestep 24h

    # Show collection statistics
    python data_collector.py --email your@email.com --stats
        """
    )

    parser.add_argument(
        "--email", "-e",
        required=True,
        help="Contact email (REQUIRED by API policy)"
    )

    parser.add_argument(
        "--db", "-d",
        default="ge_prices.db",
        help="Path to SQLite database (default: ge_prices.db)"
    )

    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run in daemon mode for continuous collection"
    )

    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Backfill historical timeseries data"
    )

    parser.add_argument(
        "--timestep",
        choices=["5m", "1h", "6h", "24h"],
        default="1h",
        help="Timestep for backfill (default: 1h)"
    )

    parser.add_argument(
        "--items",
        nargs="+",
        type=int,
        help="Specific item IDs to backfill"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show collection statistics and exit"
    )

    parser.add_argument(
        "--project",
        default="PPOFlipper",
        help="Project name for User-Agent (default: PPOFlipper)"
    )

    args = parser.parse_args()

    # Initialize collector
    collector = DataCollector(
        email=args.email,
        db_path=args.db,
        project_name=args.project
    )

    try:
        if args.stats:
            stats = collector.get_stats()
            print("\n=== Collection Statistics ===")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            return

        # Always collect mapping first
        collector.collect_mapping()

        if args.backfill:
            collector.backfill_timeseries(
                item_ids=args.items,
                timestep=args.timestep
            )
        elif args.daemon:
            collector.run_daemon()
        else:
            # One-time collection
            collector.run_collection_cycle()

        # Show final stats
        stats = collector.get_stats()
        print("\n=== Final Statistics ===")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        collector.close()


if __name__ == "__main__":
    main()
