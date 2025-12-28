# PPO Inference Firebase Integration

## Overview

This document outlines the Firebase integration for PPO inference, replacing the WebSocket-based communication system. Firebase Firestore provides real-time synchronization between the PPO inference server and the GE Auto RuneLite plugin.

---

## Files to Create

```
PPOFlipperOpus/
├── firebase/                          # NEW: Firebase integration module
│   ├── __init__.py                    # Exports main classes
│   ├── firebase_client.py             # Core Firestore connection (singleton)
│   ├── order_manager.py               # Create/manage orders to GEAuto plugin
│   ├── trade_monitor.py               # Listen for trade completions from plugin
│   ├── portfolio_tracker.py           # Track account portfolio state
│   └── inference_bridge.py            # Main orchestrator connecting PPO → Firebase → Plugin
│
├── config/
│   └── firebase_config.py             # NEW: Firebase-specific configuration
│
└── inference/
    └── run_firebase_inference.py      # NEW: Main entry point (replaces WebSocket)
```

---

## Module Responsibilities

| Module | Purpose | Key Methods |
|--------|---------|-------------|
| `firebase_client.py` | Singleton Firestore connection | `initialize()`, `get_db()`, `collection()`, `shutdown()` |
| `order_manager.py` | Send buy/sell orders to plugin | `create_buy_order()`, `create_sell_order()`, `cancel_order()`, `get_pending_orders()`, `listen_for_status_updates()` |
| `trade_monitor.py` | Receive trade completions | `start_listening()`, `on_trade_completed()`, `get_recent_trades()`, `calculate_pnl()` |
| `portfolio_tracker.py` | Track gold/inventory state | `get_portfolio()`, `listen_for_updates()`, `get_available_gold()`, `get_holdings()` |
| `inference_bridge.py` | Main orchestrator | `start()`, `stop()`, `submit_decision()`, `on_order_update()`, `sync_state()` |

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PPO INFERENCE SERVER                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐       │
│   │  PPO Agent   │────▶│  Inference   │────▶│    Order     │       │
│   │  (trained)   │     │   Bridge     │     │   Manager    │       │
│   └──────────────┘     └──────────────┘     └──────┬───────┘       │
│                               ▲                     │               │
│                               │                     ▼               │
│   ┌──────────────┐     ┌──────┴───────┐     ┌──────────────┐       │
│   │   Portfolio  │◀────│    Trade     │◀────│   Firebase   │       │
│   │   Tracker    │     │   Monitor    │     │    Client    │       │
│   └──────────────┘     └──────────────┘     └──────┬───────┘       │
│                                                     │               │
└─────────────────────────────────────────────────────┼───────────────┘
                                                      │
                                                      ▼
                                          ┌──────────────────┐
                                          │    FIRESTORE     │
                                          │                  │
                                          │  - orders        │
                                          │  - trades        │
                                          │  - portfolio     │
                                          │  - accounts      │
                                          │  - items         │
                                          │  - itemNames     │
                                          └────────┬─────────┘
                                                   │
                                                   ▼
                                          ┌──────────────────┐
                                          │  GE AUTO PLUGIN  │
                                          │   (RuneLite)     │
                                          └──────────────────┘
```

---

## Key Integration Points

### 1. Order Creation (PPO → Plugin)
- PPO decides to buy/sell → `order_manager.create_buy_order()`
- Writes to Firestore `/accounts/{id}/orders`
- Plugin listener picks up pending orders
- Plugin executes and updates status

### 2. Trade Completion (Plugin → PPO)
- Plugin completes trade → writes to `/accounts/{id}/trades`
- `trade_monitor` receives real-time update
- Updates PPO's internal state for next decision
- Triggers reward calculation for online learning

### 3. Portfolio Sync (Bidirectional)
- Plugin syncs inventory/gold to Firestore
- PPO reads current portfolio before decisions
- Ensures PPO knows actual game state

### 4. Order Status Updates (Plugin → PPO)
- Listen for status changes: pending → placed → partial → completed/failed
- Update internal order tracking
- Handle failures/retries

---

## Configuration (`firebase_config.py`)

```python
FIREBASE_CONFIG = {
    # Connection
    "project_id": "ppoflipperopus",
    "service_account_path": "ppoflipperopus-firebase-adminsdk-fbsvc-25f1af05a0.json",
    "account_id": "default_account",  # or player name

    # Intervals (seconds)
    "heartbeat_interval": 30,
    "portfolio_sync_interval": 60,
    "decision_interval": 5,  # How often to run inference

    # Trading limits
    "max_pending_orders": 8,
    "min_confidence_threshold": 0.7,
    "max_order_value": 10_000_000,
    "min_order_value": 10_000,

    # Retry settings
    "max_order_retries": 3,
    "order_timeout_sec": 600,  # 10 minutes
    "stale_order_threshold_sec": 300,  # 5 minutes
}
```

---

## Entry Point (`run_firebase_inference.py`)

```python
# Pseudocode outline

async def main():
    # 1. Load configuration
    config = load_firebase_config()

    # 2. Load trained PPO model
    agent = load_ppo_agent(checkpoint_path)

    # 3. Initialize Firebase client
    firebase = FirebaseClient()
    firebase.initialize(config.service_account_path, config.account_id)

    # 4. Create inference bridge
    bridge = InferenceBridge(firebase, agent, config)

    # 5. Start listeners
    bridge.start_trade_monitor()      # Listen for completed trades
    bridge.start_portfolio_tracker()  # Listen for portfolio updates
    bridge.start_order_status_listener()  # Listen for order status changes

    # 6. Initialize state from Firestore
    await bridge.sync_initial_state()

    # 7. Main inference loop
    while running:
        try:
            # Get current state
            portfolio = await bridge.get_portfolio()
            market_data = await bridge.get_market_data()
            pending_orders = await bridge.get_pending_orders()

            # Skip if too many pending orders
            if len(pending_orders) >= config.max_pending_orders:
                await asyncio.sleep(config.decision_interval)
                continue

            # Run PPO inference
            action, confidence, item_id, quantity, price = agent.predict(
                portfolio, market_data, pending_orders
            )

            # Execute if confident
            if action != HOLD and confidence >= config.min_confidence_threshold:
                if action == BUY:
                    await bridge.submit_buy_order(item_id, quantity, price, confidence)
                elif action == SELL:
                    await bridge.submit_sell_order(item_id, quantity, price, confidence)

            # Wait for next decision cycle
            await asyncio.sleep(config.decision_interval)

        except Exception as e:
            logger.error(f"Inference error: {e}")
            await asyncio.sleep(config.decision_interval)

    # 8. Graceful shutdown
    bridge.shutdown()
    firebase.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Firestore Collections Reference

### `/accounts/{accountId}`
Account-level state and settings.

### `/accounts/{accountId}/orders/{orderId}`
Trading orders from PPO to plugin.
```json
{
  "order_id": "ord_abc123",
  "item_id": 4151,
  "item_name": "Abyssal whip",
  "action": "buy",
  "quantity": 10,
  "price": 2500000,
  "status": "pending",
  "created_at": "2025-12-27T...",
  "updated_at": "2025-12-27T...",
  "ge_slot": null,
  "filled_quantity": 0,
  "metadata": {
    "confidence": 0.85,
    "strategy": "momentum"
  }
}
```

### `/accounts/{accountId}/trades/{tradeId}`
Completed trade history (written by plugin).

### `/accounts/{accountId}/portfolio/current`
Current holdings snapshot (written by plugin).

### `/items/{itemId}`
Item data (id, name, ge_limit, etc.) - already populated.

### `/itemNames/{itemName}`
Fast name→ID lookup - already populated.

---

## Estimated File Sizes

| File | Lines | Complexity |
|------|-------|------------|
| `firebase_client.py` | ~100 | Low |
| `order_manager.py` | ~250 | Medium |
| `trade_monitor.py` | ~200 | Medium |
| `portfolio_tracker.py` | ~150 | Low |
| `inference_bridge.py` | ~350 | High |
| `firebase_config.py` | ~60 | Low |
| `run_firebase_inference.py` | ~300 | High |

**Total: ~1,400 lines**

---

## Dependencies

Add to `pyproject.toml`:
```toml
[project.dependencies]
firebase-admin = ">=6.0.0"
google-cloud-firestore = ">=2.0.0"
```

Or `requirements.txt`:
```
firebase-admin>=6.0.0
google-cloud-firestore>=2.0.0
```

---

## Migration from WebSocket

The following files become **obsolete** after Firebase integration:
- `inference/run_ppo_websocket.py`
- `inference/run_ppo_websocket_test.py`
- Any `websocket_server.py` or `ppo_websocket_integration.py`

The new `run_firebase_inference.py` replaces all WebSocket-based inference.

---

## Service Account

The service account JSON is already available at:
- `/Users/b1llstar/IdeaProjects/PPOFlipperOpus/ppoflipperopus-firebase-adminsdk-fbsvc-25f1af05a0.json`

This same service account is used by both:
- PPO Inference (Python)
- GE Auto Plugin (Java)
