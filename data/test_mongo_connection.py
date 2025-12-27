#!/usr/bin/env python3
"""Quick test to verify MongoDB connection"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.mongo_data_store import MongoDataStore

def test_connection():
    print("Testing MongoDB connection...")
    try:
        store = MongoDataStore(
            connection_string="mongodb://localhost:27017/",
            database="ppoflipper",
            collection="GEData"
        )
        print("✓ Connected successfully!")
        
        stats = store.get_stats()
        print(f"\nDatabase stats:")
        print(f"  Items: {stats['items_count']}")
        print(f"  Latest prices: {stats['latest_prices_count']}")
        print(f"  5m prices: {stats['prices_5m_count']}")
        print(f"  1h prices: {stats['prices_1h_count']}")
        
        store.close()
        print("\n✓ MongoDB connection test passed!")
        return True
    except Exception as e:
        print(f"\n✗ MongoDB connection test failed: {e}")
        print("\nMake sure MongoDB is running on localhost:27017")
        return False

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
