# MongoDB JSON Data Collection

This module provides JSON-based data collection for direct MongoDB import, as an alternative to SQLite storage.

## Quick Start

### Collect Data as JSON

```bash
# Run JSON-based collection
python data/collect_all_json.py --email your@email.com

# With custom settings
python data/collect_all_json.py --email your@email.com --workers 5 --rps 12

# Resume from checkpoint
python data/collect_all_json.py --email your@email.com --resume

# Custom output directory
python data/collect_all_json.py --email your@email.com --output-dir my_data
```

### Output Structure

The script creates a directory (default: `mongo_data/`) with these files:

```
mongo_data/
├── items.json              # Item metadata (4500+ items)
├── latest_prices.json      # Most recent prices
├── prices_5m.json          # 5-minute price history
├── prices_1h.json          # 1-hour price history
├── timeseries_all.json     # All timeseries data merged
├── IMPORT_COMMANDS.txt     # MongoDB import commands
└── timeseries/             # Individual item timeseries
    ├── 2_24h.json
    ├── 2_1h.json
    ├── 554_24h.json
    └── ...
```

## Import to MongoDB

After collection completes, import commands are generated in `IMPORT_COMMANDS.txt`:

```bash
# Import items
mongoimport --db osrs_ge --collection items --file mongo_data/items.json --jsonArray

# Import latest prices
mongoimport --db osrs_ge --collection latest_prices --file mongo_data/latest_prices.json --jsonArray

# Import 5-minute prices
mongoimport --db osrs_ge --collection prices_5m --file mongo_data/prices_5m.json --jsonArray

# Import 1-hour prices
mongoimport --db osrs_ge --collection prices_1h --file mongo_data/prices_1h.json --jsonArray

# Import all timeseries
mongoimport --db osrs_ge --collection timeseries --file mongo_data/timeseries_all.json --jsonArray
```

### MongoDB Compass Import

Alternatively, use MongoDB Compass:
1. Open MongoDB Compass
2. Connect to your database
3. Create collections (items, latest_prices, prices_5m, prices_1h, timeseries)
4. For each collection, click "Add Data" → "Import JSON or CSV file"
5. Select the corresponding `.json` file
6. Ensure "Array" format is selected
7. Click "Import"

## Data Schema

### Items Collection
```json
{
  "id": 2,
  "name": "Cannonball",
  "examine": "Ammo for the Dwarf Cannon.",
  "members": true,
  "lowalch": 2,
  "highalch": 3,
  "ge_limit": 11000,
  "value": 5,
  "icon": "https://...",
  "updated_at": "2025-12-26T10:30:00"
}
```

### Latest Prices Collection
```json
{
  "item_id": 2,
  "high_price": 245,
  "high_time": 1735210800,
  "low_price": 243,
  "low_time": 1735210920,
  "collected_at": "2025-12-26T10:30:00"
}
```

### Prices (5m/1h) Collections
```json
{
  "item_id": 2,
  "timestamp": 1735210500,
  "avg_high_price": 245,
  "high_price_volume": 15000,
  "avg_low_price": 243,
  "low_price_volume": 12000,
  "collected_at": "2025-12-26T10:30:00"
}
```

### Timeseries Collection
```json
{
  "item_id": 2,
  "timestep": "24h",
  "timestamp": 1735123200,
  "avg_high_price": 244,
  "high_price_volume": 150000,
  "avg_low_price": 242,
  "low_price_volume": 145000,
  "collected_at": "2025-12-26T10:30:00"
}
```

## Features

- **JSON Format**: All data stored as JSON arrays, ready for MongoDB import
- **Automatic Merging**: Timeseries files automatically merged on completion
- **Resume Support**: Checkpoint system allows resuming interrupted collections
- **Thread-Safe**: Parallel collection with thread-safe writes
- **Rate Limiting**: Built-in rate limiting to respect API limits
- **Import Commands**: Auto-generated import commands for easy setup

## Advanced Usage

### Programmatic Usage

```python
from data.json_data_store import JSONDataStore
from api.ge_rest_client import GrandExchangeClient

# Initialize
client = GrandExchangeClient(user_agent_email="your@email.com")
store = JSONDataStore(output_dir="my_data")

# Collect data
mapping = client.get_mapping()
store.save_mapping(mapping)

latest = client.get_latest_prices()
store.save_latest_prices(latest)

# Get stats
stats = store.get_collection_stats()
print(f"Collected {stats['total_items']} items")

# Generate import commands
store.generate_import_commands()

# Close and flush
store.close()
```

### Merge Existing Timeseries

If you already have timeseries files and want to merge them:

```python
from data.json_data_store import JSONDataStore

store = JSONDataStore(output_dir="mongo_data")
count = store.merge_all_timeseries_to_single_collection()
print(f"Merged {count} records")
store.generate_import_commands()
```

## Comparison: JSON vs SQLite

| Feature | JSON (json_data_store.py) | SQLite (data_collector.py) |
|---------|---------------------------|----------------------------|
| Storage | JSON files | SQLite database |
| MongoDB Import | Direct | Requires export |
| Query Capability | None (file-based) | SQL queries |
| Human Readable | Yes | No (binary) |
| Size | Larger (~2-3x) | Smaller |
| Portability | Excellent | Good |
| Best For | MongoDB import, sharing | Training, querying |

## Troubleshooting

**Issue**: Import fails with "unexpected token"
- Ensure `--jsonArray` flag is used
- Verify JSON files are valid with: `python -m json.tool < items.json`

**Issue**: Duplicate key errors on import
- Drop existing collections first
- Or use `--mode upsert` with mongoimport

**Issue**: Out of memory during merge
- Import timeseries files individually from `timeseries/` directory
- Skip the merge step

## Converting Existing SQLite Data

To convert your existing `ge_prices.db` to JSON:

```bash
python export_to_json.py
```

This will create the same JSON structure from your SQLite database.
