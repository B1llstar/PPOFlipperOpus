# Firebase Inference Architecture

This document explains how the PPO inference system communicates with the GE Auto plugin via Firebase Firestore.

## Overview

The Firebase-based inference system replaces the previous WebSocket architecture with a real-time database approach. This provides:

- **Persistence**: Orders and trades survive server restarts
- **Decoupling**: PPO and plugin don't need direct network connection
- **Real-time sync**: Firestore listeners provide instant updates
- **Auditability**: Complete history of all orders and trades

## System Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Firebase Firestore                                 │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   orders    │  │   trades    │  │  portfolio  │  │   account   │        │
│  │ collection  │  │ collection  │  │  document   │  │  document   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│        ▲               │                 │                 │                │
│        │               ▼                 ▼                 ▼                │
└────────┼───────────────┼─────────────────┼─────────────────┼────────────────┘
         │               │                 │                 │
    ┌────┴────┐     ┌────┴────┐       ┌────┴────┐       ┌────┴────┐
    │  WRITE  │     │  LISTEN │       │  LISTEN │       │  LISTEN │
    └────┬────┘     └────┬────┘       └────┬────┘       └────┬────┘
         │               │                 │                 │
┌────────┴───────────────┴─────────────────┴─────────────────┴────────────────┐
│                         PPO Inference Server                                 │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │  InferenceBridge │  │  OrderManager   │  │  TradeMonitor   │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
│  ┌─────────────────┐  ┌─────────────────┐                                   │
│  │PortfolioTracker │  │  FirebaseClient │                                   │
│  └─────────────────┘  └─────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          GE Auto Plugin (Java)                               │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │FirebaseOrderList│  │FirebaseTradeRep │  │FirebaseInventory│             │
│  │     ener        │  │     orter       │  │      Sync       │             │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘             │
│           │                    │                    │                       │
│      ┌────┴────┐          ┌────┴────┐          ┌────┴────┐                 │
│      │  LISTEN │          │  WRITE  │          │  WRITE  │                 │
│      └────┬────┘          └────┬────┘          └────┬────┘                 │
└───────────┼────────────────────┼────────────────────┼───────────────────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                           Firebase Firestore                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   orders    │  │   trades    │  │  portfolio  │  │   account   │          │
│  │ collection  │  │ collection  │  │  document   │  │  document   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘          │
└───────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Order Submission (PPO → Plugin)

```
PPO Decision → InferenceBridge → OrderManager → Firestore orders collection
                                                        │
                                                        ▼
                                              FirebaseOrderListener (Java)
                                                        │
                                                        ▼
                                              GE Auto executes order
```

**Order Document Structure:**
```json
{
  "order_id": "ord_abc123def456",
  "item_id": 4151,
  "item_name": "Abyssal whip",
  "action": "buy",
  "quantity": 10,
  "price": 2500000,
  "status": "pending",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z",
  "ge_slot": null,
  "filled_quantity": 0,
  "metadata": {
    "confidence": 0.85,
    "strategy": "ppo_firebase"
  }
}
```

### 2. Order Status Updates (Plugin → PPO)

```
GE Auto updates order status → Firestore orders collection
                                        │
                                        ▼
                              OrderManager listener (Python)
                                        │
                                        ▼
                              InferenceBridge callback
                                        │
                                        ▼
                              PPO tracks order progress
```

**Order Status Flow:**
```
pending → received → placed → partial → completed
                         │         │
                         ▼         ▼
                    cancelled   failed
```

### 3. Trade Reporting (Plugin → PPO)

```
GE Auto trade completes → FirebaseTradeReporter → Firestore trades collection
                                                          │
                                                          ▼
                                                TradeMonitor listener (Python)
                                                          │
                                                          ▼
                                                InferenceBridge callback
                                                          │
                                                          ▼
                                                PPO updates reward/P&L
```

**Trade Document Structure:**
```json
{
  "trade_id": "trd_xyz789",
  "order_id": "ord_abc123def456",
  "item_id": 4151,
  "item_name": "Abyssal whip",
  "action": "buy",
  "quantity": 10,
  "price": 2450000,
  "total_cost": 24500000,
  "tax_paid": 0,
  "completed_at": "2024-01-15T10:35:00Z",
  "ge_slot": 0
}
```

### 4. Portfolio Sync (Plugin → PPO)

```
GE Auto inventory change → FirebaseInventorySync → Firestore portfolio document
                                                           │
                                                           ▼
                                                 PortfolioTracker listener (Python)
                                                           │
                                                           ▼
                                                 PPO knows current holdings
```

**Portfolio Document Structure:**
```json
{
  "gold": 50000000,
  "items": {
    "4151": {
      "item_id": 4151,
      "item_name": "Abyssal whip",
      "quantity": 10,
      "avg_price": 2450000
    }
  },
  "total_value": 74500000,
  "updated_at": "2024-01-15T10:35:00Z"
}
```

## Inference Loop

The main inference loop runs continuously, making decisions at configured intervals:

```python
while running:
    # 1. Check if plugin is online
    if not bridge.is_plugin_online():
        wait(5 seconds)
        continue

    # 2. Get current state
    portfolio = bridge.get_portfolio_summary()
    gold = portfolio["gold"]
    holdings = portfolio["holdings"]
    active_orders = bridge.get_active_orders()

    # 3. Check capacity
    if len(active_orders) >= MAX_ACTIVE_ORDERS:
        wait(decision_interval)
        continue

    # 4. Get market data
    market_data = fetch_osrs_wiki_prices()

    # 5. Make PPO decision
    decision = ppo_agent.predict(
        gold=gold,
        holdings=holdings,
        active_orders=active_orders,
        market_data=market_data
    )

    # 6. Execute if confident
    if decision.confidence >= MIN_CONFIDENCE:
        if decision.action == "buy":
            bridge.submit_buy_order(...)
        elif decision.action == "sell":
            bridge.submit_sell_order(...)

    # 7. Wait for next cycle
    wait(decision_interval)
```

## Firestore Collections

### `accounts/{account_id}/orders`
- Contains all orders (pending, active, completed, failed)
- Indexed by `status` and `created_at`
- Plugin listens for `status == "pending"` orders

### `accounts/{account_id}/trades`
- Contains completed trade records
- Indexed by `completed_at` and `item_id`
- Used for P&L calculations

### `accounts/{account_id}/portfolio`
- Single document with current portfolio state
- Updated by plugin on inventory changes
- Contains gold, items, total value

### `accounts/{account_id}/account`
- Single document with account metadata
- Contains status, last_heartbeat, ge_slots_available
- Used to verify plugin is online

## Configuration

Key settings in `config/firebase_config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `DECISION_INTERVAL` | 5 sec | Time between inference decisions |
| `MIN_CONFIDENCE_THRESHOLD` | 0.7 | Minimum confidence to execute |
| `MAX_ACTIVE_ORDERS` | 8 | Maximum concurrent GE orders |
| `PLUGIN_ONLINE_THRESHOLD` | 120 sec | Max heartbeat age |
| `ORDER_TIMEOUT` | 600 sec | Cancel orders after this time |

## Error Handling

### Consecutive Errors
If the inference loop encounters multiple errors:
1. After 5 consecutive errors, pause for 60 seconds
2. Log error details for debugging
3. Auto-resume after pause period

### Plugin Offline
If plugin heartbeat is stale:
1. Skip inference decisions
2. Continue monitoring portfolio
3. Resume when plugin comes back online

### Order Failures
If order submission fails:
1. Log failure reason
2. Don't count against error limit
3. Retry on next decision cycle

## Monitoring

### Status Check
```python
runner.get_status()
# Returns:
{
    "running": True,
    "paused": False,
    "plugin_online": True,
    "gold": 50000000,
    "holdings_count": 5,
    "active_orders": 2,
    "decisions_made": 150,
    "orders_submitted": 25,
    "trades_completed": 20,
    "net_profit": 5000000
}
```

### Logs
- Console output with real-time status
- `firebase_inference.log` for persistent logs
- Includes all decisions, orders, and trades

## Security

### Service Account
- Firebase Admin SDK requires service account JSON
- Store securely, never commit to git
- Searched in multiple locations (see `firebase_config.py`)

### Firestore Rules
Recommended security rules:
```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /accounts/{accountId}/{document=**} {
      // Only authenticated admin SDK can access
      allow read, write: if false;
    }
  }
}
```

## Usage

### Starting Inference
```bash
cd /Users/b1llstar/IdeaProjects/PPOFlipperOpus
python inference/run_firebase_inference.py
```

### Programmatic Usage
```python
from firebase.inference_bridge import InferenceBridge
from config.firebase_config import SERVICE_ACCOUNT_PATH

bridge = InferenceBridge(
    service_account_path=SERVICE_ACCOUNT_PATH,
    account_id="my_account"
)

bridge.start()

# Submit an order
order_id = bridge.submit_buy_order(
    item_id=4151,
    item_name="Abyssal whip",
    quantity=10,
    price=2500000,
    confidence=0.85
)

# Check status
status = bridge.get_status()
print(f"Gold: {status['gold']:,}")
print(f"Active orders: {status['active_orders']}")

bridge.shutdown()
```
