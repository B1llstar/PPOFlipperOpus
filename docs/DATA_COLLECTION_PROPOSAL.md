# PPO Flipper Data Collection Proposal

## Executive Summary

This document outlines a phased approach to data collection for training and deploying a PPO-based GE flipping bot. The strategy uses multiple API endpoints at different resolutions to capture market dynamics at various timescales.

---

## Phase Overview

| Phase | Duration | Focus | Endpoints Used |
|-------|----------|-------|----------------|
| **Phase 1: Bootstrap** | Day 1 | Historical backfill | `/mapping`, `/timeseries` |
| **Phase 2: Accumulate** | Weeks 1-4 | Build training dataset | `/1h`, `/5m` |
| **Phase 3: Train** | Week 4+ | Model training | Collected data |
| **Phase 4: Deploy** | Ongoing | Live inference | `/latest`, `/5m` |

---

## Phase 1: Bootstrap (Day 1)

### Objective
Get maximum historical data immediately using `/timeseries` endpoint.

### Data Collection

```
1. Fetch /mapping → Save all 4500 items with metadata
2. For each tradeable item:
   - Fetch /timeseries?timestep=24h → 365 days of daily data
   - Fetch /timeseries?timestep=1h  → 365 hours (~15 days) of hourly data
```

### Priority Items
Focus on high-volume, flippable items first:

| Category | Example Items | Priority |
|----------|---------------|----------|
| Runes | Fire, Nature, Blood, Death | P0 |
| Ammunition | Cannonballs, Dragon arrows | P0 |
| Potions | Prayer, Super restore, Saradomin brew | P0 |
| Food | Shark, Anglerfish, Manta ray | P1 |
| Bones | Dragon bones, Superior dragon bones | P1 |
| Herbs | Ranarr, Snapdragon, Torstol | P1 |
| Ores/Bars | Runite ore, Dragon bars | P2 |
| High-value | Twisted bow, Scythe (for learning volatility) | P2 |

### Expected Output
```
Database after Phase 1:
├── items: 4,500 records
├── timeseries (24h): ~1.6M records (4500 items × 365 days)
└── timeseries (1h): ~1.6M records (4500 items × 365 hours)
```

### Estimated Time
- ~4500 API calls for 24h timeseries
- ~4500 API calls for 1h timeseries
- At 0.2s delay between calls: ~30 minutes total

---

## Phase 2: Accumulate (Weeks 1-4)

### Objective
Build a rich, continuous dataset with hourly granularity for training.

### Collection Schedule

| Endpoint | Frequency | Purpose |
|----------|-----------|---------|
| `/1h` | Every hour | Primary training data |
| `/5m` | Every 5 min | High-frequency patterns |
| `/latest` | Every hour | Snapshot prices |
| `/mapping` | Daily | Catch new items |

### Daemon Configuration

```python
COLLECTION_CONFIG = {
    "training_mode": {
        "1h_interval": 3600,      # Every hour
        "5m_interval": 300,       # Every 5 minutes
        "latest_interval": 3600,  # Every hour (for snapshots)
        "mapping_interval": 86400 # Daily
    }
}
```

### Storage Estimates

| Timeframe | 1h Records | 5m Records | Storage |
|-----------|------------|------------|---------|
| 1 week | ~750K | ~9M | ~500 MB |
| 1 month | ~3M | ~36M | ~2 GB |
| 3 months | ~9M | ~108M | ~6 GB |

### Data Quality Checks
Run daily validation:
- Check for gaps in timestamps
- Verify price sanity (no negative, no extreme outliers)
- Ensure volume data is present
- Flag items with insufficient data

---

## Phase 3: Training Data Preparation

### Multi-Resolution Feature Computation

Combine data from multiple timescales to create rich feature vectors.

#### Feature Hierarchy

```
Level 1: Raw Features (per timestep)
├── avg_high_price
├── avg_low_price
├── high_price_volume
├── low_price_volume
├── spread
├── spread_pct
├── volume_ratio
└── total_volume

Level 2: Derived Features (rolling windows)
├── price_sma_6h      (6-hour simple moving average)
├── price_sma_24h     (24-hour SMA)
├── price_ema_6h      (6-hour exponential MA)
├── volatility_6h     (6-hour std dev)
├── volatility_24h    (24-hour std dev)
├── volume_sma_6h     (volume moving average)
├── rsi_14            (14-period RSI)
├── spread_sma_6h     (spread moving average)
└── spread_percentile (where current spread ranks historically)

Level 3: Cross-Resolution Features
├── trend_1h_vs_24h   (short-term vs long-term trend)
├── volume_surge      (current vs average volume ratio)
├── spread_expansion  (current vs historical spread)
└── mean_reversion_signal (distance from 24h mean)

Level 4: Market Context Features
├── hour_of_day       (0-23, cyclical encoded)
├── day_of_week       (0-6, cyclical encoded)
├── is_weekend        (binary)
├── time_since_update (minutes since last game update)
└── market_wide_volume (total GE volume proxy)
```

#### Feature Computation Pipeline

```python
class MultiResolutionFeatures:
    """Compute features from multiple time resolutions."""

    def compute(self, item_id: int) -> np.ndarray:
        # Get data at different resolutions
        data_1h = self.db.get_prices_1h(item_id, hours=168)  # 7 days
        data_24h = self.db.get_timeseries(item_id, "24h", days=30)

        features = []
        for t in range(len(data_1h)):
            row = self._compute_row(
                current=data_1h[t],
                history_1h=data_1h[max(0,t-24):t],  # Last 24 hours
                history_24h=data_24h  # Monthly context
            )
            features.append(row)

        return np.array(features)
```

### Training Dataset Structure

```
training_data.npz
├── X: (N, 24, 32)     # N sequences, 24 timesteps, 32 features
├── y: (N,)            # Target: future spread_pct or profit
├── item_ids: (N,)     # Which item each sequence belongs to
├── timestamps: (N,)   # Start timestamp of each sequence
├── feature_names: [32 strings]
├── normalization_params: {...}
└── metadata: {
      "collection_period": "2024-01-01 to 2024-12-25",
      "num_items": 500,
      "resolution": "1h"
    }
```

### Train/Validation/Test Split

```
Timeline-based split (no leakage):

|-------- Train --------|--- Val ---|--- Test ---|
   Months 1-9 (75%)      Month 10    Months 11-12
                          (8%)          (17%)
```

---

## Phase 4: Inference System

### Dedicated Inference Client

The inference client operates differently from training collection:

#### Polling Strategy

```python
INFERENCE_CONFIG = {
    # Price updates
    "latest_interval": 30,        # Poll /latest every 30 seconds
    "5m_interval": 60,            # Poll /5m every minute

    # Active trade monitoring
    "active_order_interval": 10,  # Check order status every 10s

    # Feature refresh
    "feature_recompute": 300,     # Recompute features every 5 min
}
```

#### Real-Time Feature Pipeline

```
/latest (30s) ─────────────────────────────────┐
                                               │
/5m (60s) ──────────┐                          │
                    ▼                          ▼
              ┌─────────────┐           ┌─────────────┐
              │ 5m Feature  │           │ Price Delta │
              │   Buffer    │           │  Calculator │
              └─────────────┘           └─────────────┘
                    │                          │
                    ▼                          ▼
              ┌─────────────────────────────────────┐
              │      Multi-Resolution Merger        │
              │  (combines 5m + 1h + 24h features)  │
              └─────────────────────────────────────┘
                              │
                              ▼
              ┌─────────────────────────────────────┐
              │         Feature Normalizer          │
              │   (uses training normalization)     │
              └─────────────────────────────────────┘
                              │
                              ▼
              ┌─────────────────────────────────────┐
              │           PPO Policy Net            │
              │     action = policy(features)       │
              └─────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Action: BUY    │
                    │  Item: 554      │
                    │  Price: 5       │
                    │  Qty: 10000     │
                    └─────────────────┘
```

#### Inference Client Architecture

```python
class InferenceClient:
    """Real-time inference client for PPO flipper."""

    def __init__(self, model_path: str, db_path: str, email: str):
        self.api = GrandExchangeClient(contact_email=email)
        self.db = PriceDatabase(db_path)
        self.model = self._load_model(model_path)
        self.feature_computer = MultiResolutionFeatures(self.db)

        # Caches
        self.latest_cache = {}
        self.feature_cache = {}

    async def run(self):
        """Main inference loop."""
        while True:
            # Update market data
            await self._update_latest()

            # For each item we're tracking
            for item_id in self.watchlist:
                # Get current features
                features = self._get_features(item_id)

                # Get model prediction
                action = self.model.predict(features)

                # Execute if confident
                if action.confidence > 0.7:
                    await self._execute_action(action)

            await asyncio.sleep(30)

    async def _update_latest(self):
        """Fetch latest prices."""
        self.latest_cache = self.api.get_latest()
        self.db.save_latest_prices(self.latest_cache)
```

---

## Data Collection Modes

### Mode 1: Training Collection (Default)

```bash
python data/data_collector.py \
    --email your@email.com \
    --mode training \
    --daemon
```

Behavior:
- Collects `/1h` every hour
- Collects `/5m` every 5 minutes
- Runs continuously in background
- Stores to `ge_prices_training.db`

### Mode 2: Backfill Collection

```bash
python data/data_collector.py \
    --email your@email.com \
    --mode backfill \
    --timesteps 24h 1h \
    --priority-items config/priority_items.txt
```

Behavior:
- One-time historical backfill
- Fetches `/timeseries` for all items
- Prioritizes high-volume items first
- Progress checkpoint for resume

### Mode 3: Inference Collection

```bash
python inference/inference_client.py \
    --email your@email.com \
    --model models/ppo_flipper_v1.pt \
    --mode inference
```

Behavior:
- Polls `/latest` every 30 seconds
- Polls `/5m` every minute
- Computes real-time features
- Feeds to model for decisions

---

## Implementation Roadmap

### Week 1: Foundation
- [ ] Implement Phase 1 backfill script
- [ ] Create priority items list
- [ ] Set up daemon for Phase 2 collection
- [ ] Create database backup strategy

### Week 2: Features
- [ ] Implement multi-resolution feature computation
- [ ] Add rolling window calculations
- [ ] Add market context features
- [ ] Create feature validation tests

### Week 3: Training Pipeline
- [ ] Implement train/val/test split logic
- [ ] Create data loader for PPO training
- [ ] Add data augmentation (if needed)
- [ ] Benchmark training throughput

### Week 4: Inference
- [ ] Build inference client
- [ ] Implement real-time feature pipeline
- [ ] Add order execution integration
- [ ] Create monitoring dashboard

---

## Appendix: API Rate Limiting

The API states no explicit rate limit, but we should be respectful:

| Collection Mode | Requests/Hour | Requests/Day |
|-----------------|---------------|--------------|
| Training | ~720 | ~17,280 |
| Backfill | ~3,600 (burst) | One-time |
| Inference | ~240 | ~5,760 |

**Safety margins:**
- Add 100-200ms delay between requests
- Implement exponential backoff on errors
- Cache mapping data (refresh daily)
- Batch requests where possible

---

## Appendix: Database Maintenance

### Daily Tasks
- Vacuum database (reclaim space)
- Check for data gaps
- Verify latest collection timestamp

### Weekly Tasks
- Export backup to cloud storage
- Analyze data quality metrics
- Prune old 5m data (keep 30 days)

### Monthly Tasks
- Archive old data to parquet
- Regenerate training dataset
- Review collection performance
