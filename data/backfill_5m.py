#!/usr/bin/env python3
"""
Backfill Historical 5-Minute Data

Fetches historical 5-minute price data from the OSRS Wiki API going back up to 1 year.
This creates a rich dataset for training the PPO agent with high-frequency data.

Usage:
    python data/backfill_5m.py --email your@email.com --days 365

The /5m endpoint accepts a timestamp parameter to fetch historical snapshots.
Each call returns all items for that specific 5-minute window.
"""

import argparse
import sqlite3
import time
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.ge_rest_client import GrandExchangeClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BackfillDatabase:
    """Database handler for backfill data."""

    def __init__(self, db_path: str = "ge_prices.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database tables for 5m backfill data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create backfill_5m table (separate from live prices_5m)
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

        # Create indexes for fast queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_backfill_5m_ts
            ON backfill_5m(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_backfill_5m_item
            ON backfill_5m(item_id)
        """)

        # Create checkpoint table to track progress
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backfill_checkpoint (
                id INTEGER PRIMARY KEY,
                last_timestamp INTEGER,
                total_records INTEGER,
                started_at INTEGER,
                updated_at INTEGER
            )
        """)

        conn.commit()
        conn.close()
        logger.info(f"Database initialized: {self.db_path}")

    def save_5m_snapshot(self, data: Dict, timestamp: int) -> int:
        """Save a 5-minute snapshot to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        count = 0
        for item_id, item_data in data.items():
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO backfill_5m
                    (item_id, timestamp, avg_high_price, high_price_volume,
                     avg_low_price, low_price_volume)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    int(item_id),
                    timestamp,
                    item_data.get('avgHighPrice'),
                    item_data.get('highPriceVolume', 0),
                    item_data.get('avgLowPrice'),
                    item_data.get('lowPriceVolume', 0)
                ))
                count += 1
            except Exception as e:
                logger.warning(f"Error saving item {item_id}: {e}")

        conn.commit()
        conn.close()
        return count

    def get_checkpoint(self) -> Optional[int]:
        """Get the last processed timestamp."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT last_timestamp FROM backfill_checkpoint WHERE id = 1")
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    def save_checkpoint(self, timestamp: int, total_records: int):
        """Save progress checkpoint."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = int(time.time())

        cursor.execute("""
            INSERT OR REPLACE INTO backfill_checkpoint
            (id, last_timestamp, total_records, started_at, updated_at)
            VALUES (1, ?, ?, COALESCE(
                (SELECT started_at FROM backfill_checkpoint WHERE id = 1), ?
            ), ?)
        """, (timestamp, total_records, now, now))

        conn.commit()
        conn.close()

    def get_stats(self) -> Dict:
        """Get backfill statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM backfill_5m")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM backfill_5m")
        min_ts, max_ts = cursor.fetchone()

        cursor.execute("SELECT COUNT(DISTINCT item_id) FROM backfill_5m")
        items = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT timestamp) FROM backfill_5m")
        snapshots = cursor.fetchone()[0]

        conn.close()

        return {
            "total_records": total,
            "unique_items": items,
            "snapshots": snapshots,
            "min_timestamp": min_ts,
            "max_timestamp": max_ts,
            "date_range": (
                datetime.fromtimestamp(min_ts).isoformat() if min_ts else None,
                datetime.fromtimestamp(max_ts).isoformat() if max_ts else None
            )
        }


def backfill_5m_data(
    email: str,
    days: int = 365,
    db_path: str = "ge_prices.db",
    interval_seconds: float = 0.2,
    resume: bool = True
):
    """
    Backfill historical 5-minute data.

    Args:
        email: Contact email for API User-Agent
        days: Number of days to backfill
        db_path: Path to database
        interval_seconds: Delay between API requests
        resume: Whether to resume from checkpoint
    """
    client = GrandExchangeClient(contact_email=email)
    db = BackfillDatabase(db_path)

    # Calculate time range
    now = int(time.time())
    current_5m = (now // 300) * 300  # Round to 5 minutes
    start_ts = current_5m - (days * 24 * 60 * 60)

    # Resume from checkpoint if available
    if resume:
        checkpoint = db.get_checkpoint()
        if checkpoint and checkpoint > start_ts:
            start_ts = checkpoint + 300  # Start from next interval
            logger.info(f"Resuming from checkpoint: {datetime.fromtimestamp(start_ts)}")

    # Calculate total intervals
    total_intervals = (current_5m - start_ts) // 300

    logger.info(f"Backfilling 5m data from {datetime.fromtimestamp(start_ts)} to {datetime.fromtimestamp(current_5m)}")
    logger.info(f"Total intervals to fetch: {total_intervals:,}")
    logger.info(f"Estimated time: {total_intervals * interval_seconds / 60:.1f} minutes")

    # Fetch data for each 5-minute interval
    processed = 0
    total_records = 0
    errors = 0

    timestamp = start_ts
    last_stats_time = time.time()

    while timestamp <= current_5m:
        try:
            # Fetch 5m data for this timestamp
            data = client.get_5m(timestamp=timestamp)

            if data:
                count = db.save_5m_snapshot(data, timestamp)
                total_records += count
            else:
                logger.debug(f"No data for timestamp {timestamp}")

            processed += 1

            # Save checkpoint periodically
            if processed % 100 == 0:
                db.save_checkpoint(timestamp, total_records)

            # Print progress
            if processed % 50 == 0 or time.time() - last_stats_time > 30:
                progress = processed / total_intervals * 100
                eta_seconds = (total_intervals - processed) * interval_seconds
                eta_minutes = eta_seconds / 60

                logger.info(
                    f"Progress: {progress:.1f}% ({processed:,}/{total_intervals:,}) | "
                    f"Records: {total_records:,} | "
                    f"ETA: {eta_minutes:.1f} min | "
                    f"Date: {datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')}"
                )
                last_stats_time = time.time()

            # Rate limiting
            time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("Interrupted by user. Saving checkpoint...")
            db.save_checkpoint(timestamp, total_records)
            break

        except Exception as e:
            errors += 1
            logger.warning(f"Error at timestamp {timestamp}: {e}")
            if errors > 10:
                logger.error("Too many errors, stopping")
                break
            time.sleep(1)  # Extra delay on error

        timestamp += 300  # Next 5-minute interval

    # Final checkpoint and stats
    db.save_checkpoint(timestamp, total_records)
    stats = db.get_stats()

    logger.info("=" * 50)
    logger.info("Backfill Complete!")
    logger.info(f"Total records: {stats['total_records']:,}")
    logger.info(f"Unique items: {stats['unique_items']:,}")
    logger.info(f"Snapshots: {stats['snapshots']:,}")
    logger.info(f"Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")
    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Backfill historical 5-minute GE price data"
    )
    parser.add_argument(
        "--email", "-e",
        required=True,
        help="Contact email for API User-Agent (required)"
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=365,
        help="Number of days to backfill (default: 365)"
    )
    parser.add_argument(
        "--db",
        default="ge_prices.db",
        help="Database path (default: ge_prices.db)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.2,
        help="Seconds between API requests (default: 0.2)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh instead of resuming from checkpoint"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Just show current statistics"
    )

    args = parser.parse_args()

    if args.stats_only:
        db = BackfillDatabase(args.db)
        stats = db.get_stats()
        print("\nBackfill Statistics:")
        print(f"  Total records: {stats['total_records']:,}")
        print(f"  Unique items: {stats['unique_items']:,}")
        print(f"  Snapshots: {stats['snapshots']:,}")
        if stats['date_range'][0]:
            print(f"  Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")
        return

    backfill_5m_data(
        email=args.email,
        days=args.days,
        db_path=args.db,
        interval_seconds=args.interval,
        resume=not args.no_resume
    )


if __name__ == "__main__":
    main()
