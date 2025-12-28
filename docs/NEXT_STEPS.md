# Next Steps

This document outlines the remaining work to complete the Firebase-based PPO inference system.

## Immediate Next Steps

### 1. Test the Plugin-Side Firebase Integration

The Java Firebase integration was added to GE Auto but hasn't been tested yet.

**Prerequisites:**
- Firebase service account JSON in one of these locations:
  - `RuneLite__Star/runelite_star_new/runelite-client/src/main/java/net/runelite/client/plugins/geauto/ppoflipperopus-firebase-adminsdk-*.json`
  - Environment variable `FIREBASE_SERVICE_ACCOUNT`
  - `~/.config/ppoflipperopus/service_account.json`

**Steps:**
1. Start RuneLite with the GE Auto plugin enabled
2. Check logs for Firebase connection success
3. Verify plugin starts listening for orders in Firestore

### 2. Test the Python-Side Firebase Integration

```bash
cd /Users/b1llstar/IdeaProjects/PPOFlipperOpus

# Install dependencies
pip install -e .

# Run the inference server
python inference/run_firebase_inference.py
```

**Expected output:**
- Firebase connection established
- Listeners started for trades, portfolio, orders
- Main inference loop running

### 3. End-to-End Integration Test

With both PPO inference and GE Auto plugin running:

| Step | Component | Action |
|------|-----------|--------|
| 1 | PPO | Submits buy order to Firestore |
| 2 | Firestore | Order appears in `orders` collection with status `pending` |
| 3 | Plugin | Picks up order, updates status to `received` |
| 4 | Plugin | Places order in GE, updates status to `placed` |
| 5 | Plugin | Trade completes, writes to `trades` collection |
| 6 | PPO | Receives trade callback, updates P&L |

---

## Implementation Gaps (TODOs)

### GE Auto Plugin (Java)

#### Cancel Handler
**File:** `firebase/FirebaseCancelHandler.java`

The in-game order cancellation is stubbed with a TODO. Needs implementation to:
- Interact with GE interface to cancel an active offer
- Update order status in Firestore to `cancelled`
- Handle partial fills before cancellation

#### Bank Scanner
**File:** `firebase/FirebaseInventorySync.java`

Bank state scanning is stubbed with a TODO. Needs implementation to:
- Scan bank contents when bank interface opens
- Sync full bank inventory to Firestore
- Track bank value over time

### PPO Inference (Python)

#### Load Trained Model
**File:** `inference/run_firebase_inference.py`

Currently uses a mock decision maker. Needs to:
- Load trained PPO model from `models/ppo_agent.pt`
- Initialize with correct observation/action spaces
- Use model predictions for buy/sell decisions

```python
# TODO: Replace mock decision maker with:
model_path = "models/ppo_agent.pt"
if os.path.exists(model_path):
    self.agent = PPOAgent.load(model_path)
```

#### Real PPO Integration
**File:** `inference/run_firebase_inference.py`

Connect actual `PPOAgent` class to decision loop:
- Build observation vector from portfolio/market state
- Get action from agent.predict()
- Map action to buy/sell/hold with item, quantity, price

---

## Optional Enhancements

### High Priority

#### Real-Time Dashboard
Build a web UI to monitor the system in real-time.

**Features:**
- Current portfolio (gold, holdings, total value)
- Active orders with status
- Recent trades
- P&L chart over time
- System health (plugin online, errors)

**Tech options:**
- Next.js + Firebase client SDK
- Streamlit (quick Python option)
- Grafana + Firestore export

### Medium Priority

#### Market Data Sync
Populate Firestore with OSRS Wiki price data for plugin access.

**Benefits:**
- Plugin can make price-informed decisions locally
- Reduces API calls from plugin
- Historical price tracking

**Implementation:**
- Scheduled job to fetch OSRS Wiki prices
- Write to `market_data` collection
- Plugin reads from Firestore instead of API

#### Alerting
Add notifications for significant events.

**Events to alert on:**
- Large profit/loss trades
- Order failures
- Plugin offline
- Consecutive errors

**Channels:**
- Discord webhook
- Slack webhook
- Email (via Firebase Functions)

### Low Priority

#### Multi-Account Support
Run inference for multiple accounts simultaneously.

**Changes needed:**
- Account ID as runtime parameter
- Separate Firestore paths per account
- Independent inference loops

#### Performance Metrics
Track and analyze trading performance.

**Metrics:**
- Win rate (profitable trades / total trades)
- Average profit per trade
- Best/worst performing items
- Sharpe ratio of returns

---

## File Reference

### Python (PPOFlipperOpus)

| File | Purpose |
|------|---------|
| `firebase/__init__.py` | Package exports |
| `firebase/firebase_client.py` | Firestore connection singleton |
| `firebase/order_manager.py` | Create/manage orders |
| `firebase/trade_monitor.py` | Listen for completed trades |
| `firebase/portfolio_tracker.py` | Track portfolio state |
| `firebase/inference_bridge.py` | Main orchestrator |
| `config/firebase_config.py` | Configuration settings |
| `inference/run_firebase_inference.py` | Entry point |

### Java (GE Auto Plugin)

| File | Purpose |
|------|---------|
| `firebase/FirebaseConfig.java` | Constants and field names |
| `firebase/FirebaseManager.java` | Firestore connection singleton |
| `firebase/FirebaseOrder.java` | Order data class |
| `firebase/FirebaseOrderListener.java` | Listen for PPO orders |
| `firebase/FirebaseTradeReporter.java` | Report trades to Firestore |
| `firebase/FirebaseInventorySync.java` | Sync inventory/portfolio |
| `firebase/FirebaseCancelHandler.java` | Handle order cancellation |
| `firebase/GEAutoFirebaseIntegration.java` | Main orchestrator |

### Documentation

| File | Purpose |
|------|---------|
| `docs/FIREBASE_INFERENCE_ARCHITECTURE.md` | System architecture |
| `docs/FIREBASE_INFERENCE_INTEGRATION.md` | Integration outline |
| `docs/NEXT_STEPS.md` | This document |
