# PPOFlipper Opus

A PPO (Proximal Policy Optimization) reinforcement learning bot for flipping items on the Old School RuneScape Grand Exchange.

## Project Structure

```
PPOFlipperOpus/
├── api/                    # API clients for OSRS Real-time Prices
│   ├── ge_rest_client.py   # Base REST client with User-Agent compliance
│   └── real_time_ge_client.py  # Real-time trading client
├── data/                   # Data collection and storage
│   ├── data_collector.py   # Continuous data collection pipeline
│   └── export_training_data.py  # Export data for training
├── training/               # PPO training code
│   ├── train_ppo.py        # Main training script
│   ├── volume_analysis.py  # Volume analysis utilities
│   └── shared_knowledge.py # Shared utilities
├── inference/              # Real-time inference and trading
│   ├── run_ppo_websocket.py
│   ├── run_real_trading.py
│   └── run_real_trading_enhanced.py
├── tests/                  # Test suite
├── config/                 # Configuration files
│   ├── volume_blacklist.txt
│   └── real_trading_config.py
└── docs/                   # Documentation
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect Training Data

**IMPORTANT**: The API requires a valid User-Agent with contact email.

```bash
# Start continuous data collection (daemon mode)
python data/data_collector.py --email your@email.com --daemon

# Or one-time collection
python data/data_collector.py --email your@email.com

# Backfill historical data (up to 1 year with 24h timestep)
python data/data_collector.py --email your@email.com --backfill --timestep 24h
```

### 3. Export Training Data

```bash
# Export as NumPy arrays for training
python data/export_training_data.py --db ge_prices.db --format numpy --normalize

# View statistics
python data/export_training_data.py --db ge_prices.db --stats
```

### 4. Train the Model

```bash
python training/train_ppo.py
```

## API Compliance

This project uses the [OSRS Real-time Prices API](https://oldschool.runescape.wiki/w/RuneScape:Real-time_Prices).

**Requirements:**
- All requests MUST include a descriptive User-Agent header
- Generic user agents (python-requests, curl, etc.) are blocked
- Be respectful of rate limits (no explicit limit, but don't abuse)

Example User-Agent:
```
PPOFlipper - OSRS GE Flipper Training Bot (your@email.com)
```

## Data Collection

### Available Endpoints

| Endpoint | Description | Max History |
|----------|-------------|-------------|
| `/latest` | Current instant-buy/sell prices | Current only |
| `/5m` | 5-minute averaged prices | ~30 hours |
| `/1h` | 1-hour averaged prices | ~15 days |
| `/timeseries?timestep=24h` | Historical data | ~1 year |

### Recommended Training Data

For PPO training, we recommend:
- **Minimum**: 6 months of hourly data
- **Optimal**: 12 months of hourly data
- Start collecting 5m data now for future fine-tuning

## Database Schema

The SQLite database stores:
- `items` - Item metadata (ID, name, GE limit, alch values)
- `latest_prices` - Most recent price snapshot
- `prices_5m` - 5-minute aggregated prices
- `prices_1h` - 1-hour aggregated prices
- `timeseries` - Historical timeseries data

## Training Features

The export utility computes these features for training:
- `avg_high_price` - Average instant-buy price
- `avg_low_price` - Average instant-sell price
- `high_price_volume` - Volume of instant-buys
- `low_price_volume` - Volume of instant-sells
- `spread` - Price spread (high - low)
- `spread_pct` - Spread as percentage of price
- `volume_ratio` - Ratio of buy/sell volume
- `total_volume` - Total trading volume

## License

Private project - not for public distribution.
