#!/usr/bin/env python3
"""
MongoDB Data Store for GE Price Data

Stores GE price data directly in MongoDB with optimized indexing.
Eliminates file I/O bottlenecks for better performance.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict

try:
    from pymongo import MongoClient, UpdateOne, ASCENDING, DESCENDING
    from pymongo.errors import BulkWriteError, ConnectionFailure
except ImportError:
    raise ImportError("pymongo is required. Install it with: pip install pymongo")

logger = logging.getLogger("MongoDataStore")


class MongoDataStore:
    """MongoDB-based storage for GE price data - high performance."""

    def __init__(self, connection_string: str = "mongodb://localhost:27017/", 
                 database: str = "ppoflipper",
                 collection: str = "GEData"):
        """
        Initialize the MongoDB data store.

        Args:
            connection_string: MongoDB connection string
            database: Database name
            collection: Collection name for price data
        """
        try:
            self.client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ping')
            logger.info(f"Connected to MongoDB at {connection_string}")
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
        
        self.db = self.client[database]
        
        # Collections
        self.items_collection = self.db["items"]
        self.latest_prices_collection = self.db["latest_prices"]
        self.prices_5m_collection = self.db[collection]
        self.prices_1h_collection = self.db["prices_1h"]
        
        # Create indexes for optimal query performance
        self._ensure_indexes()
        
        logger.info(f"MongoDB data store initialized - database: {database}, collection: {collection}")

    def _ensure_indexes(self):
        """Create indexes for efficient querying."""
        try:
            # Items collection - unique on item id
            self.items_collection.create_index([("id", ASCENDING)], unique=True)
            
            # Latest prices - unique on item_id
            self.latest_prices_collection.create_index([("item_id", ASCENDING)], unique=True)
            
            # 5m prices - compound index on item_id and timestamp for fast lookups
            self.prices_5m_collection.create_index([
                ("item_id", ASCENDING),
                ("timestamp", DESCENDING)
            ])
            self.prices_5m_collection.create_index([("timestamp", DESCENDING)])
            
            # 1h prices - similar indexing
            self.prices_1h_collection.create_index([
                ("item_id", ASCENDING),
                ("timestamp", DESCENDING)
            ])
            
            logger.info("MongoDB indexes created successfully")
        except Exception as e:
            logger.warning(f"Could not create indexes (may already exist): {e}")

    def save_mapping(self, items: List[Dict]):
        """
        Save item mapping data.

        Args:
            items: List of item metadata dictionaries from API
        """
        if not items:
            return
        
        operations = []
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
                "updated_at": datetime.utcnow()
            }
            
            # Upsert: update if exists, insert if not
            operations.append(
                UpdateOne(
                    {"id": item_doc["id"]},
                    {"$set": item_doc},
                    upsert=True
                )
            )
        
        try:
            result = self.items_collection.bulk_write(operations, ordered=False)
            logger.info(f"Saved mapping for {len(items)} items "
                       f"(inserted: {result.upserted_count}, modified: {result.modified_count})")
        except BulkWriteError as e:
            logger.error(f"Error saving item mappings: {e.details}")

    def save_latest_prices(self, prices: Dict):
        """
        Save latest price snapshot.

        Args:
            prices: Dictionary of item_id -> price data
        """
        if not prices:
            return
        
        operations = []
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
                "collected_at": datetime.utcnow()
            }
            
            operations.append(
                UpdateOne(
                    {"item_id": price_doc["item_id"]},
                    {"$set": price_doc},
                    upsert=True
                )
            )
            count += 1

        if operations:
            try:
                result = self.latest_prices_collection.bulk_write(operations, ordered=False)
                logger.info(f"Saved latest prices for {count} items")
            except BulkWriteError as e:
                logger.error(f"Error saving latest prices: {e.details}")
        
        return count

    def save_5m_prices(self, prices: Dict, timestamp: Optional[int] = None):
        """
        Save 5-minute price data using bulk insert for maximum performance.

        Args:
            prices: Dictionary of item_id -> 5m price data
            timestamp: Unix timestamp for this data point
        """
        if not prices:
            return 0
        
        if timestamp is None:
            timestamp = int(time.time())
            # Round to nearest 5 minutes
            timestamp = (timestamp // 300) * 300

        documents = []
        for item_id, data in prices.items():
            if data.get("avgHighPrice") is None and data.get("avgLowPrice") is None:
                continue

            price_doc = {
                "item_id": int(item_id),
                "timestamp": timestamp,
                "avg_high_price": data.get("avgHighPrice"),
                "high_price_volume": data.get("highPriceVolume"),
                "avg_low_price": data.get("avgLowPrice"),
                "low_price_volume": data.get("lowPriceVolume"),
                "collected_at": datetime.utcnow()
            }
            documents.append(price_doc)

        if documents:
            try:
                # Use bulk insert with ordered=False for better performance
                # Ignore duplicate key errors (if timestamp already exists)
                result = self.prices_5m_collection.insert_many(documents, ordered=False)
                count = len(result.inserted_ids)
                logger.info(f"Saved 5m prices for {count} items at timestamp {timestamp}")
                return count
            except BulkWriteError as e:
                # Count successful inserts even if some failed due to duplicates
                count = e.details.get('nInserted', 0)
                logger.info(f"Saved 5m prices for {count} items at timestamp {timestamp} (some duplicates skipped)")
                return count
            except Exception as e:
                logger.error(f"Error saving 5m prices: {e}")
                return 0
        
        return 0

    def save_1h_prices(self, prices: Dict, timestamp: Optional[int] = None):
        """
        Save 1-hour price data.

        Args:
            prices: Dictionary of item_id -> 1h price data
            timestamp: Unix timestamp for this data point
        """
        if not prices:
            return 0
        
        if timestamp is None:
            timestamp = int(time.time())
            # Round to nearest hour
            timestamp = (timestamp // 3600) * 3600

        documents = []
        for item_id, data in prices.items():
            if data.get("avgHighPrice") is None and data.get("avgLowPrice") is None:
                continue

            price_doc = {
                "item_id": int(item_id),
                "timestamp": timestamp,
                "avg_high_price": data.get("avgHighPrice"),
                "high_price_volume": data.get("highPriceVolume"),
                "avg_low_price": data.get("avgLowPrice"),
                "low_price_volume": data.get("lowPriceVolume"),
                "collected_at": datetime.utcnow()
            }
            documents.append(price_doc)

        if documents:
            try:
                result = self.prices_1h_collection.insert_many(documents, ordered=False)
                count = len(result.inserted_ids)
                logger.info(f"Saved 1h prices for {count} items at timestamp {timestamp}")
                return count
            except BulkWriteError as e:
                count = e.details.get('nInserted', 0)
                logger.info(f"Saved 1h prices for {count} items at timestamp {timestamp} (some duplicates skipped)")
                return count
            except Exception as e:
                logger.error(f"Error saving 1h prices: {e}")
                return 0
        
        return 0

    def save_timeseries(self, item_id: int, timestep: str, data: List[Dict]):
        """
        Save timeseries data (if needed for compatibility).

        Args:
            item_id: Item ID
            timestep: Timestep ('5m' or '1h')
            data: List of price data points
        """
        # Can be implemented if timeseries storage is needed
        pass

    def get_item_ids(self, min_volume: int = 0) -> List[int]:
        """
        Get list of item IDs, optionally filtered by volume.

        Args:
            min_volume: Minimum total volume to include

        Returns:
            List of item IDs
        """
        if min_volume > 0:
            # Aggregate to calculate total volume
            pipeline = [
                {
                    "$group": {
                        "_id": "$item_id",
                        "total_volume": {
                            "$sum": {
                                "$add": [
                                    {"$ifNull": ["$high_price_volume", 0]},
                                    {"$ifNull": ["$low_price_volume", 0]}
                                ]
                            }
                        }
                    }
                },
                {"$match": {"total_volume": {"$gte": min_volume}}},
                {"$project": {"_id": 1}}
            ]
            results = list(self.prices_5m_collection.aggregate(pipeline))
            return [r["_id"] for r in results]
        else:
            # Return all unique item IDs
            return self.prices_5m_collection.distinct("item_id")

    def get_stats(self) -> Dict:
        """Get statistics about stored data."""
        stats = {
            "items_count": self.items_collection.count_documents({}),
            "latest_prices_count": self.latest_prices_collection.count_documents({}),
            "prices_5m_count": self.prices_5m_collection.count_documents({}),
            "prices_1h_count": self.prices_1h_collection.count_documents({})
        }
        return stats

    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
