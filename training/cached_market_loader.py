#!/usr/bin/env python3
"""
Cached Market Data Loader

Loads market data from JSON cache once and shares across all environments.
Provides 10-100x faster environment initialization compared to database queries.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger("CachedMarketLoader")


@dataclass
class CachedMarketState:
    """Lightweight market state from cache."""
    item_id: int
    high_price: float
    low_price: float
    high_volume: int
    low_volume: int
    timestamp: int

    @property
    def spread(self) -> float:
        return self.high_price - self.low_price

    @property
    def spread_pct(self) -> float:
        mid = (self.high_price + self.low_price) / 2
        return self.spread / mid if mid > 0 else 0

    @property
    def total_volume(self) -> int:
        return self.high_volume + self.low_volume

    @property
    def mid_price(self) -> float:
        return (self.high_price + self.low_price) / 2


class CachedMarketData:
    """
    Global market data cache loaded once and shared across all environments.
    Thread-safe for read-only access.
    """
    _instance: Optional['CachedMarketData'] = None
    _loaded: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize empty cache. Call load() to populate."""
        if not self._loaded:
            self.items: Dict[int, Dict] = {}
            self.market_history: Dict[int, List[CachedMarketState]] = {}
            self.timestamps: List[int] = []
            self.item_ids: List[int] = []
            self.metadata: Dict = {}
    
    def load(self, cache_file: str = "training_cache.json", force_reload: bool = False):
        """
        Load market data from JSON cache file.
        
        Args:
            cache_file: Path to JSON cache file
            force_reload: Force reload even if already loaded
        """
        if self._loaded and not force_reload:
            logger.info("Cache already loaded, skipping reload")
            return
        
        cache_path = Path(cache_file)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Cache file not found: {cache_path}\n"
                f"Run 'python data/export_training_cache.py' to create it."
            )
        
        logger.info(f"Loading market data cache from {cache_path}...")
        
        with open(cache_path, 'r') as f:
            cache = json.load(f)
        
        # Load metadata
        self.metadata = cache.get("metadata", {})
        
        # Load items (convert string keys back to int)
        self.items = {int(k): v for k, v in cache["items"].items()}
        
        # Load timestamps
        self.timestamps = cache["timestamps"]
        
        # Load item IDs
        self.item_ids = cache["item_ids"]
        
        # Load market history (convert to dataclass objects for efficiency)
        self.market_history = {}
        for item_id_str, history in cache["market_history"].items():
            item_id = int(item_id_str)
            self.market_history[item_id] = [
                CachedMarketState(
                    item_id=item_id,
                    high_price=point["high_price"],
                    low_price=point["low_price"],
                    high_volume=point["high_volume"],
                    low_volume=point["low_volume"],
                    timestamp=point["timestamp"]
                )
                for point in history
            ]
        
        self._loaded = True
        
        file_size_mb = cache_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Cache loaded: {file_size_mb:.2f} MB")
        logger.info(f"  Items: {len(self.items)}")
        logger.info(f"  Timestamps: {len(self.timestamps)}")
        logger.info(f"  Time range: {self.metadata.get('time_range', 'unknown')}")
        logger.info(f"  Total data points: {sum(len(h) for h in self.market_history.values())}")
    
    def get_item_metadata(self, item_id: int) -> Optional[Dict]:
        """Get metadata for an item."""
        return self.items.get(item_id)
    
    def get_market_history(self, item_id: int) -> List[CachedMarketState]:
        """Get full market history for an item."""
        return self.market_history.get(item_id, [])
    
    def get_tradeable_items(self, min_data_points: int = 0) -> List[int]:
        """
        Get list of tradeable item IDs.
        
        Args:
            min_data_points: Minimum data points required
            
        Returns:
            List of item IDs meeting criteria
        """
        if min_data_points == 0:
            return self.item_ids.copy()
        
        return [
            item_id for item_id in self.item_ids
            if len(self.market_history.get(item_id, [])) >= min_data_points
        ]
    
    def is_loaded(self) -> bool:
        """Check if cache is loaded."""
        return self._loaded
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "loaded": self._loaded,
            "num_items": len(self.items),
            "num_timestamps": len(self.timestamps),
            "total_data_points": sum(len(h) for h in self.market_history.values()),
            "metadata": self.metadata
        }


# Global singleton instance
_CACHE = CachedMarketData()


def load_cache(cache_file: str = "training_cache.json", force_reload: bool = False):
    """
    Load the global market data cache.
    
    Args:
        cache_file: Path to JSON cache file
        force_reload: Force reload even if already loaded
    """
    _CACHE.load(cache_file, force_reload)


def get_cache() -> CachedMarketData:
    """Get the global cache instance."""
    return _CACHE


if __name__ == "__main__":
    # Test loading
    import time
    
    print("Testing cache loading...")
    start = time.time()
    
    load_cache("training_cache.json")
    cache = get_cache()
    
    elapsed = time.time() - start
    print(f"\nLoad time: {elapsed:.3f}s")
    print(f"Stats: {cache.get_stats()}")
    
    # Test access speed
    if cache.item_ids:
        test_item = cache.item_ids[0]
        start = time.time()
        
        for _ in range(10000):
            history = cache.get_market_history(test_item)
            _ = len(history)
        
        elapsed = time.time() - start
        print(f"\n10,000 history lookups: {elapsed:.3f}s ({elapsed/10000*1000:.4f}ms each)")
