#!/usr/bin/env python3
"""
Export MongoDB data to optimized JSON cache for training

Creates a single JSON file with all market data pre-loaded for fast training.
This eliminates database queries during training entirely.
"""

import json
import logging
from pathlib import Path 
from typing import Dict, List
from collections import defaultdict
from mongo_data_store import MongoDataStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ExportTrainingCache")


def export_training_cache(
    output_file: str = "training_cache.json",
    min_volume: int = 1000,
    top_n_items: int = 200,
    min_data_points: int = 200  # ~8 days of hourly data minimum
):
    """
    Export MongoDB data to optimized JSON format for training.
    
    Args:
        output_file: Output JSON file path
        min_volume: Minimum volume threshold
        top_n_items: Number of top items by volume to include
        min_data_points: Minimum data points required per item
    """
    logger.info("Connecting to MongoDB...")
    store = MongoDataStore()
    
    # Get item metadata
    logger.info("Fetching item metadata...")
    items_cursor = store.items_collection.find(
        {"ge_limit": {"$exists": True, "$ne": None, "$gt": 0}}
    )
    items_data = {}
    for item in items_cursor:
        items_data[item["id"]] = {
            "id": item["id"],
            "name": item.get("name"),
            "ge_limit": item.get("ge_limit"),
            "highalch": item.get("highalch"),
            "lowalch": item.get("lowalch"),
            "value": item.get("value")
        }
    logger.info(f"Loaded {len(items_data)} items with GE limits")
    
    # Get top items by volume from 5m data
    logger.info(f"Finding top {top_n_items} items by volume...")
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
                },
                "data_points": {"$sum": 1}
            }
        },
        {"$match": {
            "total_volume": {"$gte": min_volume},
            "data_points": {"$gte": min_data_points}
        }},
        {"$sort": {"total_volume": -1}},
        {"$limit": top_n_items}
    ]
    
    top_items_cursor = store.prices_5m_collection.aggregate(pipeline)
    top_items = []
    for doc in top_items_cursor:
        item_id = doc["_id"]
        if item_id in items_data:  # Must have metadata
            top_items.append({
                "item_id": item_id,
                "total_volume": doc["total_volume"],
                "data_points": doc["data_points"]
            })
    
    logger.info(f"Found {len(top_items)} items meeting criteria")
    item_ids = [item["item_id"] for item in top_items]
    
    # Get all 5m price data for these items
    logger.info("Fetching 5m price history...")
    prices_cursor = store.prices_5m_collection.find(
        {"item_id": {"$in": item_ids}},
        {"_id": 0, "collected_at": 0}  # Exclude unnecessary fields
    ).sort([("timestamp", 1)])
    
    # Organize by item_id
    market_history = defaultdict(list)
    all_timestamps = set()
    
    for price in prices_cursor:
        item_id = price["item_id"]
        timestamp = price["timestamp"]
        
        # Skip if missing critical data
        if not price.get("avg_high_price") or not price.get("avg_low_price"):
            continue
            
        market_history[item_id].append({
            "timestamp": timestamp,
            "high_price": float(price["avg_high_price"]),
            "low_price": float(price["avg_low_price"]),
            "high_volume": int(price.get("high_price_volume", 0)),
            "low_volume": int(price.get("low_price_volume", 0))
        })
        all_timestamps.add(timestamp)
    
    # Sort timestamps
    all_timestamps = sorted(list(all_timestamps))
    
    logger.info(f"Loaded {len(all_timestamps)} unique timestamps")
    logger.info(f"Time range: {min(all_timestamps)} to {max(all_timestamps)}")
    
    # Filter items that have enough data
    filtered_items = []
    final_market_history = {}
    
    for item_id in item_ids:
        if item_id in market_history and len(market_history[item_id]) >= min_data_points:
            filtered_items.append(item_id)
            final_market_history[str(item_id)] = market_history[item_id]  # JSON needs string keys
            logger.info(f"  Item {item_id} ({items_data[item_id]['name']}): {len(market_history[item_id])} data points")
    
    logger.info(f"Final item count: {len(filtered_items)}")
    
    # Build final cache structure
    cache = {
        "metadata": {
            "created_at": store.db.command("serverStatus")["localTime"].isoformat(),
            "num_items": len(filtered_items),
            "num_timestamps": len(all_timestamps),
            "min_data_points": min_data_points,
            "time_range": {
                "start": min(all_timestamps),
                "end": max(all_timestamps)
            }
        },
        "items": {str(item_id): items_data[item_id] for item_id in filtered_items},
        "market_history": final_market_history,
        "timestamps": all_timestamps,
        "item_ids": filtered_items
    }
    
    # Write to file
    output_path = Path(output_file)
    logger.info(f"Writing cache to {output_path}...")
    
    with open(output_path, 'w') as f:
        json.dump(cache, f, indent=2)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"✓ Export complete: {file_size_mb:.2f} MB")
    logger.info(f"  Items: {len(filtered_items)}")
    logger.info(f"  Timestamps: {len(all_timestamps)}")
    logger.info(f"  Total data points: {sum(len(h) for h in final_market_history.values())}")
    
    store.close()
    return str(output_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export MongoDB data to training cache")
    parser.add_argument("--output", "-o", default="training_cache.json",
                       help="Output JSON file path")
    parser.add_argument("--top-items", "-n", type=int, default=200,
                       help="Number of top items to include")
    parser.add_argument("--min-volume", type=int, default=1000,
                       help="Minimum volume threshold")
    parser.add_argument("--min-data-points", type=int, default=200,
                       help="Minimum data points required per item")
    
    args = parser.parse_args()
    
    export_training_cache(
        output_file=args.output,
        min_volume=args.min_volume,
        top_n_items=args.top_items,
        min_data_points=args.min_data_points
    )
