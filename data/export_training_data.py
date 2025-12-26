#!/usr/bin/env python3
"""
Training Data Export Utility

Exports collected GE price data into formats suitable for PPO training.

Supports multiple export formats:
- NumPy arrays (.npz) - For direct use with PyTorch/TensorFlow
- Parquet files - For large datasets with efficient compression
- CSV files - For inspection and external tools
- JSON files - For compatibility with existing codebase

Usage:
    # Export all data as NumPy arrays
    python export_training_data.py --db ge_prices.db --format numpy --output training_data.npz

    # Export specific items as Parquet
    python export_training_data.py --db ge_prices.db --format parquet --items 2 554 555

    # Export with normalization for training
    python export_training_data.py --db ge_prices.db --format numpy --normalize
"""

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TrainingDataExport")


class TrainingDataExporter:
    """Exports collected price data for PPO training."""

    # Feature columns for training (matches compute_features output order)
    PRICE_FEATURES = [
        "timestamp",        # Unix timestamp (column 0, excluded from training)
        "avg_high_price",   # Average instant-buy price
        "avg_low_price",    # Average instant-sell price
        "high_price_volume",# Volume of instant-buys
        "low_price_volume", # Volume of instant-sells
        "spread",           # high - low
        "spread_pct",       # spread / avg_price
        "volume_ratio",     # high_vol / low_vol
        "total_volume"      # high_vol + low_vol
    ]

    # Features used for training (excludes timestamp)
    TRAINING_FEATURES = PRICE_FEATURES[1:]

    def __init__(self, db_path: str):
        """
        Initialize the exporter.

        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        logger.info(f"Connected to database: {db_path}")

    def get_item_list(self, min_data_points: int = 100) -> List[Dict]:
        """
        Get list of items with sufficient data.

        Args:
            min_data_points: Minimum number of price records required

        Returns:
            List of item dictionaries with id, name, and data_count
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT
                i.id,
                i.name,
                i.ge_limit,
                COUNT(p.id) as data_count
            FROM items i
            LEFT JOIN prices_1h p ON i.id = p.item_id
            GROUP BY i.id
            HAVING data_count >= ?
            ORDER BY data_count DESC
        """, (min_data_points,))

        items = []
        for row in cursor.fetchall():
            items.append({
                "id": row["id"],
                "name": row["name"],
                "ge_limit": row["ge_limit"],
                "data_count": row["data_count"]
            })

        logger.info(f"Found {len(items)} items with >= {min_data_points} data points")
        return items

    def get_price_history(
        self,
        item_id: int,
        table: str = "prices_1h",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> np.ndarray:
        """
        Get price history for an item.

        Args:
            item_id: Item ID to fetch
            table: Table to fetch from (prices_5m, prices_1h, timeseries)
            start_time: Optional start timestamp
            end_time: Optional end timestamp

        Returns:
            NumPy array with columns: [timestamp, avg_high, avg_low, high_vol, low_vol]
        """
        cursor = self.conn.cursor()

        query = f"""
            SELECT
                timestamp,
                avg_high_price,
                avg_low_price,
                high_price_volume,
                low_price_volume
            FROM {table}
            WHERE item_id = ?
        """
        params = [item_id]

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp ASC"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        if not rows:
            return np.array([])

        # Convert to numpy array, handling None values
        data = []
        for row in rows:
            data.append([
                row["timestamp"] or 0,
                row["avg_high_price"] or 0,
                row["avg_low_price"] or 0,
                row["high_price_volume"] or 0,
                row["low_price_volume"] or 0
            ])

        return np.array(data, dtype=np.float32)

    def compute_features(self, price_data: np.ndarray) -> np.ndarray:
        """
        Compute training features from raw price data.

        Args:
            price_data: Raw price data [timestamp, high, low, high_vol, low_vol]

        Returns:
            Feature array with derived features
        """
        if len(price_data) == 0:
            return np.array([])

        # Extract columns
        timestamps = price_data[:, 0]
        high_prices = price_data[:, 1]
        low_prices = price_data[:, 2]
        high_volumes = price_data[:, 3]
        low_volumes = price_data[:, 4]

        # Compute derived features
        spread = high_prices - low_prices
        avg_price = (high_prices + low_prices) / 2
        spread_pct = np.where(avg_price > 0, spread / avg_price, 0)

        total_volume = high_volumes + low_volumes
        volume_ratio = np.where(
            low_volumes > 0,
            high_volumes / low_volumes,
            np.where(high_volumes > 0, 10.0, 1.0)  # Cap at 10 if low_vol is 0
        )

        # Stack features
        features = np.column_stack([
            timestamps,
            high_prices,
            low_prices,
            high_volumes,
            low_volumes,
            spread,
            spread_pct,
            volume_ratio,
            total_volume
        ])

        return features

    def normalize_features(
        self,
        features: np.ndarray,
        method: str = "zscore"
    ) -> Tuple[np.ndarray, Dict]:
        """
        Normalize features for training.

        Args:
            features: Feature array
            method: Normalization method ("zscore", "minmax", "log")

        Returns:
            Tuple of (normalized_features, normalization_params)
        """
        if len(features) == 0:
            return features, {}

        params = {}

        if method == "zscore":
            # Z-score normalization (skip timestamp column)
            means = np.mean(features[:, 1:], axis=0)
            stds = np.std(features[:, 1:], axis=0)
            stds = np.where(stds == 0, 1, stds)  # Avoid division by zero

            normalized = features.copy()
            normalized[:, 1:] = (features[:, 1:] - means) / stds

            params = {"method": "zscore", "means": means.tolist(), "stds": stds.tolist()}

        elif method == "minmax":
            # Min-max normalization
            mins = np.min(features[:, 1:], axis=0)
            maxs = np.max(features[:, 1:], axis=0)
            ranges = maxs - mins
            ranges = np.where(ranges == 0, 1, ranges)

            normalized = features.copy()
            normalized[:, 1:] = (features[:, 1:] - mins) / ranges

            params = {"method": "minmax", "mins": mins.tolist(), "maxs": maxs.tolist()}

        elif method == "log":
            # Log normalization (useful for prices/volumes)
            normalized = features.copy()
            # Add 1 to avoid log(0)
            normalized[:, 1:] = np.log1p(np.abs(features[:, 1:]))

            params = {"method": "log"}

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return normalized, params

    def create_sequences(
        self,
        features: np.ndarray,
        sequence_length: int = 24,
        prediction_horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time-series training.

        Args:
            features: Feature array
            sequence_length: Number of timesteps in each sequence
            prediction_horizon: How many steps ahead to predict

        Returns:
            Tuple of (X sequences, y targets)
        """
        if len(features) < sequence_length + prediction_horizon:
            return np.array([]), np.array([])

        X = []
        y = []

        for i in range(len(features) - sequence_length - prediction_horizon + 1):
            X.append(features[i:i + sequence_length, 1:])  # Skip timestamp
            # Target: future spread percentage (for flip profit prediction)
            future_spread = features[i + sequence_length + prediction_horizon - 1, 6]
            y.append(future_spread)

        return np.array(X), np.array(y)

    def export_numpy(
        self,
        output_path: str,
        item_ids: Optional[List[int]] = None,
        table: str = "prices_1h",
        normalize: bool = True,
        sequence_length: int = 24
    ):
        """
        Export data as NumPy arrays.

        Args:
            output_path: Output .npz file path
            item_ids: Optional list of item IDs to export
            table: Source table
            normalize: Whether to normalize features
            sequence_length: Sequence length for time-series data
        """
        if item_ids is None:
            items = self.get_item_list()
            item_ids = [item["id"] for item in items]

        all_X = []
        all_y = []
        item_metadata = []
        norm_params = {}

        logger.info(f"Exporting {len(item_ids)} items to NumPy format...")

        for item_id in item_ids:
            # Get raw price history
            raw_data = self.get_price_history(item_id, table)
            if len(raw_data) < sequence_length + 1:
                continue

            # Compute features
            features = self.compute_features(raw_data)

            # Normalize if requested
            if normalize:
                features, params = self.normalize_features(features)
                norm_params[str(item_id)] = params

            # Create sequences
            X, y = self.create_sequences(features, sequence_length)
            if len(X) == 0:
                continue

            all_X.append(X)
            all_y.append(y)
            item_metadata.append({
                "item_id": item_id,
                "num_sequences": len(X),
                "start_idx": sum(len(x) for x in all_X[:-1]),
                "end_idx": sum(len(x) for x in all_X)
            })

        if not all_X:
            logger.warning("No data to export!")
            return

        # Concatenate all data
        X_combined = np.concatenate(all_X, axis=0)
        y_combined = np.concatenate(all_y, axis=0)

        # Save to npz
        np.savez_compressed(
            output_path,
            X=X_combined,
            y=y_combined,
            feature_names=self.TRAINING_FEATURES,
            item_metadata=json.dumps(item_metadata),
            normalization_params=json.dumps(norm_params),
            sequence_length=sequence_length,
            export_timestamp=datetime.now().isoformat()
        )

        logger.info(f"Exported {len(X_combined)} sequences to {output_path}")
        logger.info(f"  Shape: X={X_combined.shape}, y={y_combined.shape}")
        logger.info(f"  Items: {len(item_metadata)}")

    def export_csv(
        self,
        output_dir: str,
        item_ids: Optional[List[int]] = None,
        table: str = "prices_1h"
    ):
        """
        Export data as CSV files (one per item).

        Args:
            output_dir: Output directory
            item_ids: Optional list of item IDs to export
            table: Source table
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if item_ids is None:
            items = self.get_item_list()
            item_ids = [item["id"] for item in items]

        logger.info(f"Exporting {len(item_ids)} items to CSV format...")

        for item_id in item_ids:
            raw_data = self.get_price_history(item_id, table)
            if len(raw_data) == 0:
                continue

            features = self.compute_features(raw_data)

            # Create header
            header = "timestamp," + ",".join(self.PRICE_FEATURES[:-1])  # Exclude duplicate timestamp

            # Save to CSV
            item_path = output_path / f"item_{item_id}.csv"
            np.savetxt(
                item_path,
                features,
                delimiter=",",
                header=header,
                comments=""
            )

        logger.info(f"Exported CSVs to {output_dir}")

    def export_json(
        self,
        output_path: str,
        item_ids: Optional[List[int]] = None,
        table: str = "prices_1h"
    ):
        """
        Export data as JSON (compatible with existing historical client).

        Args:
            output_path: Output JSON file path
            item_ids: Optional list of item IDs to export
            table: Source table
        """
        if item_ids is None:
            items = self.get_item_list()
            item_ids = [item["id"] for item in items]

        logger.info(f"Exporting {len(item_ids)} items to JSON format...")

        # Group by timestamp
        cursor = self.conn.cursor()
        cursor.execute(f"""
            SELECT DISTINCT timestamp FROM {table}
            ORDER BY timestamp ASC
        """)
        timestamps = [row[0] for row in cursor.fetchall()]

        output_data = []
        for ts in timestamps:
            cursor.execute(f"""
                SELECT
                    item_id,
                    avg_high_price,
                    avg_low_price,
                    high_price_volume,
                    low_price_volume
                FROM {table}
                WHERE timestamp = ?
            """, (ts,))

            snapshot = {
                "timestamp": ts,
                "data": {}
            }

            for row in cursor.fetchall():
                if item_ids is None or row["item_id"] in item_ids:
                    snapshot["data"][str(row["item_id"])] = {
                        "high": row["avg_high_price"],
                        "low": row["avg_low_price"],
                        "highPriceVolume": row["high_price_volume"],
                        "lowPriceVolume": row["low_price_volume"]
                    }

            if snapshot["data"]:
                output_data.append(snapshot)

        with open(output_path, "w") as f:
            json.dump(output_data, f)

        logger.info(f"Exported {len(output_data)} snapshots to {output_path}")

    def get_stats(self) -> Dict:
        """Get export statistics."""
        cursor = self.conn.cursor()

        stats = {}

        cursor.execute("SELECT COUNT(*) FROM items")
        stats["total_items"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT item_id) FROM prices_1h")
        stats["items_with_1h_data"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM prices_1h")
        stats["total_1h_records"] = cursor.fetchone()[0]

        cursor.execute("""
            SELECT item_id, COUNT(*) as cnt
            FROM prices_1h
            GROUP BY item_id
            ORDER BY cnt DESC
            LIMIT 10
        """)
        stats["top_10_items_by_data"] = [
            {"item_id": row[0], "records": row[1]}
            for row in cursor.fetchall()
        ]

        return stats

    def close(self):
        """Close database connection."""
        self.conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Export GE price data for PPO training",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--db", "-d",
        default="ge_prices.db",
        help="Path to SQLite database (default: ge_prices.db)"
    )

    parser.add_argument(
        "--format", "-f",
        choices=["numpy", "csv", "json"],
        default="numpy",
        help="Export format (default: numpy)"
    )

    parser.add_argument(
        "--output", "-o",
        help="Output path (file for numpy/json, directory for csv)"
    )

    parser.add_argument(
        "--items",
        nargs="+",
        type=int,
        help="Specific item IDs to export"
    )

    parser.add_argument(
        "--table", "-t",
        choices=["prices_5m", "prices_1h"],
        default="prices_1h",
        help="Source table (default: prices_1h)"
    )

    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize features (for numpy export)"
    )

    parser.add_argument(
        "--sequence-length", "-s",
        type=int,
        default=24,
        help="Sequence length for time-series (default: 24)"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show export statistics and exit"
    )

    args = parser.parse_args()

    # Set default output paths
    if args.output is None:
        if args.format == "numpy":
            args.output = "training_data.npz"
        elif args.format == "csv":
            args.output = "training_csv"
        elif args.format == "json":
            args.output = "training_data.json"

    exporter = TrainingDataExporter(args.db)

    try:
        if args.stats:
            stats = exporter.get_stats()
            print("\n=== Export Statistics ===")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            return

        if args.format == "numpy":
            exporter.export_numpy(
                args.output,
                item_ids=args.items,
                table=args.table,
                normalize=args.normalize,
                sequence_length=args.sequence_length
            )
        elif args.format == "csv":
            exporter.export_csv(
                args.output,
                item_ids=args.items,
                table=args.table
            )
        elif args.format == "json":
            exporter.export_json(
                args.output,
                item_ids=args.items,
                table=args.table
            )

    finally:
        exporter.close()


if __name__ == "__main__":
    main()
