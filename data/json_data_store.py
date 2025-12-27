#!/usr/bin/env python3
"""
JSON Data Store for MongoDB Import

Stores GE price data as JSON files suitable for direct MongoDB import.
Creates separate JSON files for each collection type.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger("JSONDataStore")


class JSONDataStore:
    """JSON-based storage for GE price data - MongoDB ready."""

    def __init__(self, output_dir: str = "mongo_data"):
        """
        Initialize the JSON data store.

        Args:
            output_dir: Directory to store JSON files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # In-memory buffers for batch writes
        self.items_buffer = {}
        self.latest_prices_buffer = {}
        self.prices_5m_buffer = []
        self.prices_1h_buffer = []
        self.timeseries_buffer = defaultdict(lambda: defaultdict(list))
        
        # File paths
        self.items_file = self.output_dir / "items.json"
        self.latest_prices_file = self.output_dir / "latest_prices.json"
        self.prices_5m_file = self.output_dir / "prices_5m.json"
        self.prices_1h_file = self.output_dir / "prices_1h.json"
        self.timeseries_dir = self.output_dir / "timeseries"
        self.timeseries_dir.mkdir(exist_ok=True)
        
        # Load existing data if present
        self._load_existing_data()
        
        logger.info(f"JSON data store initialized at {output_dir}")

    def _load_existing_data(self):
        """Load existing JSON files into memory buffers."""
        try:
            if self.items_file.exists():
                with open(self.items_file, 'r') as f:
                    items = json.load(f)
                    self.items_buffer = {item['id']: item for item in items}
                logger.info(f"Loaded {len(self.items_buffer)} existing items")
        except Exception as e:
            logger.warning(f"Could not load existing items: {e}")

        try:
            if self.latest_prices_file.exists():
                with open(self.latest_prices_file, 'r') as f:
                    prices = json.load(f)
                    self.latest_prices_buffer = {p['item_id']: p for p in prices}
                logger.info(f"Loaded {len(self.latest_prices_buffer)} existing latest prices")
        except Exception as e:
            logger.warning(f"Could not load existing latest prices: {e}")

        try:
            if self.prices_5m_file.exists():
                with open(self.prices_5m_file, 'r') as f:
                    self.prices_5m_buffer = json.load(f)
                logger.info(f"Loaded {len(self.prices_5m_buffer)} existing 5m price records")
        except Exception as e:
            logger.warning(f"Could not load existing 5m prices: {e}")

        try:
            if self.prices_1h_file.exists():
                with open(self.prices_1h_file, 'r') as f:
                    self.prices_1h_buffer = json.load(f)
                logger.info(f"Loaded {len(self.prices_1h_buffer)} existing 1h price records")
        except Exception as e:
            logger.warning(f"Could not load existing 1h prices: {e}")

    def save_mapping(self, items: List[Dict]):
        """
        Save item mapping data.

        Args:
            items: List of item metadata dictionaries from API
        """
        for item in items:
            item_doc = {
                "id": item.get("id"),
                "name": item.get("name"),
                "examine": item.get("examine"),
                "members": bool(item.get("members")),
                "lowalch": item.get("lowalch"),
                "highalch": item.get("highalch"),
                "ge_limit": item.get("limit"),
                "value": item.get("value"),
                "icon": item.get("icon"),
                "updated_at": datetime.utcnow().isoformat()
            }
            self.items_buffer[item_doc["id"]] = item_doc

        self._write_items()
        logger.info(f"Saved mapping for {len(items)} items")

    def save_latest_prices(self, prices: Dict):
        """
        Save latest price snapshot.

        Args:
            prices: Dictionary of item_id -> price data
        """
        count = 0
        for item_id, data in prices.items():
            if data.get("high") is None and data.get("low") is None:
                continue

            price_doc = {
                "item_id": int(item_id),
                "high_price": data.get("high"),
                "high_time": data.get("highTime"),
                "low_price": data.get("low"),
                "low_time": data.get("lowTime"),
                "collected_at": datetime.utcnow().isoformat()
            }
            self.latest_prices_buffer[int(item_id)] = price_doc
            count += 1

        self._write_latest_prices()
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

        count = 0
        existing_keys = {(p['item_id'], p['timestamp']) for p in self.prices_5m_buffer}

        for item_id, data in prices.items():
            if data.get("avgHighPrice") is None and data.get("avgLowPrice") is None:
                continue

            key = (int(item_id), timestamp)
            if key in existing_keys:
                continue  # Skip duplicates

            price_doc = {
                "item_id": int(item_id),
                "timestamp": timestamp,
                "avg_high_price": data.get("avgHighPrice"),
                "high_price_volume": data.get("highPriceVolume"),
                "avg_low_price": data.get("avgLowPrice"),
                "low_price_volume": data.get("lowPriceVolume"),
                "collected_at": datetime.utcnow().isoformat()
            }
            self.prices_5m_buffer.append(price_doc)
            existing_keys.add(key)
            count += 1

        self._write_5m_prices()
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

        count = 0
        existing_keys = {(p['item_id'], p['timestamp']) for p in self.prices_1h_buffer}

        for item_id, data in prices.items():
            if data.get("avgHighPrice") is None and data.get("avgLowPrice") is None:
                continue

            key = (int(item_id), timestamp)
            if key in existing_keys:
                continue  # Skip duplicates

            price_doc = {
                "item_id": int(item_id),
                "timestamp": timestamp,
                "avg_high_price": data.get("avgHighPrice"),
                "high_price_volume": data.get("highPriceVolume"),
                "avg_low_price": data.get("avgLowPrice"),
                "low_price_volume": data.get("lowPriceVolume"),
                "collected_at": datetime.utcnow().isoformat()
            }
            self.prices_1h_buffer.append(price_doc)
            existing_keys.add(key)
            count += 1

        self._write_1h_prices()
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
        count = 0
        
        # Load existing data for this item/timestep
        timeseries_file = self.timeseries_dir / f"{item_id}_{timestep}.json"
        existing_data = []
        existing_timestamps = set()
        
        if timeseries_file.exists():
            try:
                with open(timeseries_file, 'r') as f:
                    existing_data = json.load(f)
                    existing_timestamps = {p['timestamp'] for p in existing_data}
            except Exception as e:
                logger.warning(f"Could not load existing timeseries for {item_id}/{timestep}: {e}")

        # Add new data points
        for point in data:
            if point.get("avgHighPrice") is None and point.get("avgLowPrice") is None:
                continue

            timestamp = point.get("timestamp")
            if timestamp in existing_timestamps:
                continue  # Skip duplicates

            point_doc = {
                "item_id": item_id,
                "timestep": timestep,
                "timestamp": timestamp,
                "avg_high_price": point.get("avgHighPrice"),
                "high_price_volume": point.get("highPriceVolume"),
                "avg_low_price": point.get("avgLowPrice"),
                "low_price_volume": point.get("lowPriceVolume"),
                "collected_at": datetime.utcnow().isoformat()
            }
            existing_data.append(point_doc)
            existing_timestamps.add(timestamp)
            count += 1

        # Write back to file
        if count > 0:
            with open(timeseries_file, 'w') as f:
                json.dump(existing_data, f, indent=2)

        logger.debug(f"Saved {count} timeseries points for item {item_id} ({timestep})")
        return count

    def _write_items(self):
        """Write items buffer to file."""
        items_list = list(self.items_buffer.values())
        with open(self.items_file, 'w') as f:
            json.dump(items_list, f, indent=2)

    def _write_latest_prices(self):
        """Write latest prices buffer to file."""
        prices_list = list(self.latest_prices_buffer.values())
        with open(self.latest_prices_file, 'w') as f:
            json.dump(prices_list, f, indent=2)

    def _write_5m_prices(self):
        """Write 5m prices buffer to file."""
        with open(self.prices_5m_file, 'w') as f:
            json.dump(self.prices_5m_buffer, f, indent=2)

    def _write_1h_prices(self):
        """Write 1h prices buffer to file."""
        with open(self.prices_1h_file, 'w') as f:
            json.dump(self.prices_1h_buffer, f, indent=2)

    def get_item_ids(self, min_volume: int = 0) -> List[int]:
        """
        Get list of item IDs, optionally filtered by volume.

        Args:
            min_volume: Minimum total volume to include

        Returns:
            List of item IDs
        """
        if min_volume > 0:
            # Filter items by volume from 1h prices
            item_volumes = defaultdict(int)
            for price in self.prices_1h_buffer:
                item_id = price['item_id']
                volume = (price.get('high_price_volume') or 0) + (price.get('low_price_volume') or 0)
                item_volumes[item_id] += volume
            
            return sorted([item_id for item_id, vol in item_volumes.items() if vol >= min_volume])
        else:
            return sorted(self.items_buffer.keys())

    def get_collection_stats(self) -> Dict:
        """Get statistics about collected data."""
        stats = {
            "total_items": len(self.items_buffer),
            "total_latest_prices": len(self.latest_prices_buffer),
            "total_5m_records": len(self.prices_5m_buffer),
            "total_1h_records": len(self.prices_1h_buffer),
            "total_timeseries_files": len(list(self.timeseries_dir.glob("*.json")))
        }
        return stats

    def close(self):
        """Flush all buffers and close."""
        logger.info("Flushing all data to disk...")
        self._write_items()
        self._write_latest_prices()
        self._write_5m_prices()
        self._write_1h_prices()
        logger.info("JSON data store closed")

    def merge_all_timeseries_to_single_collection(self):
        """
        Merge all individual timeseries files into a single collection file.
        Useful for MongoDB import.
        """
        logger.info("Merging timeseries files...")
        all_timeseries = []
        
        for timeseries_file in self.timeseries_dir.glob("*.json"):
            try:
                with open(timeseries_file, 'r') as f:
                    data = json.load(f)
                    all_timeseries.extend(data)
            except Exception as e:
                logger.warning(f"Could not load {timeseries_file}: {e}")
        
        output_file = self.output_dir / "timeseries_all.json"
        with open(output_file, 'w') as f:
            json.dump(all_timeseries, f, indent=2)
        
        logger.info(f"Merged {len(all_timeseries)} timeseries records to {output_file}")
        return len(all_timeseries)

    def generate_import_commands(self):
        """Generate MongoDB import commands for the collected data."""
        commands = [
            "# MongoDB Import Commands",
            "# Replace 'your_database' with your actual database name",
            "",
            f"mongoimport --db your_database --collection items --file {self.items_file} --jsonArray",
            f"mongoimport --db your_database --collection latest_prices --file {self.latest_prices_file} --jsonArray",
            f"mongoimport --db your_database --collection prices_5m --file {self.prices_5m_file} --jsonArray",
            f"mongoimport --db your_database --collection prices_1h --file {self.prices_1h_file} --jsonArray",
            "",
            "# After merging timeseries:",
            f"mongoimport --db your_database --collection timeseries --file {self.output_dir / 'timeseries_all.json'} --jsonArray"
        ]
        
        import_file = self.output_dir / "IMPORT_COMMANDS.txt"
        with open(import_file, 'w') as f:
            f.write('\n'.join(commands))
        
        print('\n'.join(commands))
        logger.info(f"Import commands saved to {import_file}")
