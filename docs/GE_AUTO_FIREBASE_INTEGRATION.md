# GE Auto Plugin - Firebase Integration

This document explains how the GE Auto RuneLite plugin coordinates with the PPO inference system via Firebase Firestore.

## Overview

```
┌─────────────────────┐         ┌─────────────────────┐         ┌─────────────────────┐
│   PPO Inference     │         │      Firestore      │         │   GE Auto Plugin    │
│   (Python)          │◄───────►│     (Database)      │◄───────►│   (RuneLite/Java)   │
└─────────────────────┘         └─────────────────────┘         └─────────────────────┘
        │                               │                               │
        │  1. Read state                │                               │
        │  2. Submit orders             │                               │
        │                               │  - accounts/{id}              │
        │                               │  - items                      │
        │                               │  - itemNames                  │
        │                               │                               │
        │                               │                               │  3. Sync state
        │                               │                               │  4. Execute orders
        │                               │                               │  5. Update status
```

## Data Flow

### 1. Startup Sync (Plugin → Firestore)

When the player logs in, the plugin waits 5 game ticks then performs a one-time sync:

```
Login → Wait 5 ticks → Sync inventory → Sync GE slots → Update gold balance
```

**Why wait 5 ticks?** The game state needs time to settle after login. This pattern is borrowed from the AutoReconnect plugin's camera pitch adjustment.

**Collections updated:**
- `/accounts/{id}/inventory/current` - Player's 28 inventory slots
- `/accounts/{id}/ge_slots/current` - 8 GE slot states
- `/accounts/{id}` - Account document with `current_gold` field

**Bank sync:** Only happens when bank is opened (not at login since bank isn't open).

### 2. PPO Reads State (Firestore → Python)

The Python inference system reads the current state:

```python
from firebase.inventory_scanner import InventoryScanner

scanner = InventoryScanner(account_id="b1llstar")

# Get everything needed for inference
state = scanner.get_inference_state()
# Returns:
# {
#   "account": { "gold": 900000, "is_online": true, "ge_slots_available": 8 },
#   "inventory": { "items": {...}, "free_slots": 25 },
#   "bank": { "total_value": 5000000, "tradeable_items": {...} },
#   "ge_slots": { "active_orders": [...], "slots_available": 6 }
# }

# Or get specific data
gold = scanner.get_gold()
bank = scanner.get_bank_accounting()
```

### 3. PPO Submits Orders (Python → Firestore)

When PPO decides to trade:

```python
from firebase.order_manager import OrderManager

orders = OrderManager(account_id="b1llstar")

# Submit a buy order
order_id = orders.submit_buy_order(
    item_name="Cannonball",
    quantity=10000,
    price=150
)

# Submit a sell order
order_id = orders.submit_sell_order(
    item_name="Cannonball",
    quantity=10000,
    price=155
)
```

**Order document created in `/accounts/{id}/orders`:**
```json
{
  "order_id": "ord_abc123",
  "action": "buy",
  "item_name": "Cannonball",
  "quantity": 10000,
  "price": 150,
  "status": "pending",
  "created_at": "2025-01-01T00:00:00Z"
}
```

### 4. Plugin Executes Orders (Firestore → Plugin)

The plugin listens for orders with `status: "pending"`:

```
FirebaseOrderListener detects new order
    ↓
Updates status to "received"
    ↓
Queues order in GEQueueManager
    ↓
State machine executes:
    - Opens GE if needed
    - Clicks buy/sell button
    - Searches for item
    - Sets quantity and price
    - Confirms offer
    ↓
Updates status to "placed"
```

### 5. Trade Completion (Plugin → Firestore)

When an offer fills (BOUGHT or SOLD state):

```
onGrandExchangeOfferChanged event
    ↓
Detects BOUGHT or SOLD state
    ↓
Calls onTradeComplete()
    ↓
Syncs inventory to Firestore
Updates gold balance
    ↓
If inventory full after buy:
    Opens bank
    Deposits items
    Syncs bank to Firestore
```

## Firestore Collections

### `/accounts/{accountId}`

Main account document:

```json
{
  "account_id": "b1llstar",
  "status": "active",
  "current_gold": 900000,
  "ge_slots_available": 8,
  "last_heartbeat": "2025-01-01T00:00:00Z",
  "plugin_version": "2.0.0",
  "settings": {
    "auto_collect": true,
    "max_risk_per_flip": 500000
  }
}
```

### `/accounts/{accountId}/inventory/current`

```json
{
  "updated_at": "2025-01-01T00:00:00Z",
  "gold": 900000,
  "free_slots": 25,
  "items": {
    "2": {
      "item_id": 2,
      "item_name": "Cannonball",
      "quantity": 1000,
      "price_each": 150,
      "total_value": 150000
    }
  }
}
```

### `/accounts/{accountId}/bank/current`

```json
{
  "updated_at": "2025-01-01T00:00:00Z",
  "total_value": 5000000,
  "item_count": 150,
  "items": {
    "2": {
      "item_id": 2,
      "item_name": "Cannonball",
      "quantity": 50000,
      "price_each": 150,
      "total_value": 7500000,
      "is_tradeable": true
    }
  }
}
```

### `/accounts/{accountId}/ge_slots/current`

```json
{
  "updated_at": "2025-01-01T00:00:00Z",
  "slots_available": 6,
  "buy_slots_used": 1,
  "sell_slots_used": 1,
  "slots": {
    "1": {
      "item_id": 2,
      "item_name": "Cannonball",
      "type": "buy",
      "status": "active",
      "quantity": 10000,
      "filled_quantity": 5000,
      "price": 150
    },
    "2": null,
    "3": null
  }
}
```

### `/accounts/{accountId}/orders`

Orders submitted by PPO:

```json
{
  "order_id": "ord_abc123",
  "action": "buy",
  "item_name": "Cannonball",
  "quantity": 10000,
  "price": 150,
  "status": "completed",
  "created_at": "2025-01-01T00:00:00Z",
  "updated_at": "2025-01-01T00:01:00Z",
  "ge_slot": 1,
  "filled_quantity": 10000
}
```

**Order status progression:**
```
pending → received → placed → completed
                  ↘ partial ↗
                  ↘ cancelled
                  ↘ failed
```

### `/items`

Item master data (document ID = item ID):

```json
{
  "id": 2,
  "name": "Cannonball",
  "value": 5,
  "highalch": 3,
  "lowalch": 2,
  "ge_limit": 11000,
  "members": true,
  "examine": "Ammo for the dwarf cannon."
}
```

### `/itemNames`

Fast name → ID lookup (document ID = sanitized item name):

```json
{
  "id": 2
}
```

## Key Files

### Java (RuneLite Plugin)

| File | Purpose |
|------|---------|
| `GEAutoPlugin.java` | Main plugin, state machine, event handlers |
| `GEAutoFirebaseIntegration.java` | Coordinates Firebase components |
| `FirebaseManager.java` | Firebase connection singleton |
| `FirebaseOrderListener.java` | Listens for orders from PPO |
| `FirebaseTradeReporter.java` | Reports trades back to Firestore |
| `FirebaseInventorySync.java` | Syncs inventory/bank/GE slots |

### Python (PPO Inference)

| File | Purpose |
|------|---------|
| `firebase/inventory_scanner.py` | Read inventory/bank/gold state |
| `firebase/order_manager.py` | Submit buy/sell orders |
| `firebase/firebase_client.py` | Firebase connection |
| `firebase/inference_bridge.py` | Orchestrates inference ↔ Firebase |
| `firebase/position_tracker.py` | Track active positions (PPO-acquired items) |
| `firebase/position_sizer.py` | Enforce position limits (max % per item) |

## Simplified Design

The plugin was simplified from the original design:

**Removed:**
- WebSocket HTTP API server (redundant with Firebase)
- FirebaseCommandListener (bank commands from Python)
- FirebaseCancelHandler (order cancellation)
- Periodic inventory sync (replaced with one-time + on-trade)

**Kept:**
- One-time startup sync after login
- Order listener (for buy/sell from PPO)
- Trade reporter (updates Firestore on completion)
- Inventory sync after each trade

**Philosophy:**
- Plugin handles all GE and bank operations internally
- Python only submits orders and reads state
- No need for Python to control banking directly

## Heartbeat

The plugin sends a heartbeat every 30 seconds:

```json
{
  "last_heartbeat": "2025-01-01T00:00:00Z"
}
```

Python can check if plugin is online:

```python
scanner = InventoryScanner()
if scanner.is_plugin_online(max_age_seconds=120):
    print("Plugin is online")
```

## Error Handling

**Order failures:**
- Status set to `failed` with `error_message` field
- Plugin retries up to 3 times before failing

**Connection issues:**
- Plugin continues working offline
- Syncs resume when connection restored
- Orders queue locally if Firebase unavailable

## Usage Example

### Python: Monitor and Trade

```python
from firebase.inventory_scanner import InventoryScanner
from firebase.order_manager import OrderManager

scanner = InventoryScanner(account_id="b1llstar")
orders = OrderManager(account_id="b1llstar")

# Check if plugin is online
if not scanner.is_plugin_online():
    print("Plugin offline, waiting...")
    exit()

# Get current state
state = scanner.get_inference_state()
gold = state["account"]["gold"]
slots_available = state["ge_slots"]["slots_available"]

print(f"Gold: {gold:,}")
print(f"Slots available: {slots_available}")

# Make trading decision (PPO would do this)
if gold > 100000 and slots_available > 0:
    order_id = orders.submit_buy_order(
        item_name="Cannonball",
        quantity=1000,
        price=150
    )
    print(f"Submitted order: {order_id}")
```

### CLI: Check State

```bash
# Full summary
python3 scripts/scan_inventory.py

# Just gold
python3 scripts/scan_inventory.py --gold

# Bank contents
python3 scripts/scan_inventory.py --bank

# Active PPO positions
python3 scripts/scan_inventory.py --positions

# Portfolio utilization
python3 scripts/scan_inventory.py --portfolio

# Position limits status
python3 scripts/scan_inventory.py --limits

# Watch for changes
python3 scripts/scan_inventory.py --watch
```

## Position Tracking

The system differentiates between:
- **Active positions**: Items acquired by PPO inference (fair game for trading)
- **Pre-existing items**: Bank items that existed before PPO started (off limits)

### How It Works

1. **When PPO buys an item** → Added to active positions
2. **When PPO sells an item** → Reduced/removed from active positions
3. **Bank items** → Only tradeable if in active positions list
4. **Inventory/GE items** → Assumed fair game during sync

### Firestore Structure

```
/accounts/{id}/positions/active
{
    "items": {
        "2": {
            "item_id": 2,
            "item_name": "Cannonball",
            "quantity": 10000,
            "avg_cost": 150,
            "total_invested": 1500000,
            "first_acquired": "2025-01-01T00:00:00Z",
            "last_updated": "2025-01-01T00:00:00Z",
            "source": "ppo",     // or "manual"
            "locked": false      // if true, won't be sold
        }
    },
    "updated_at": "..."
}
```

### Vue Dashboard Controls

From the Vue dashboard, you can:
- **Lock a position**: Prevents PPO from selling it
- **Unlock a position**: Allows PPO to sell it again
- **Remove a position**: Removes item from active positions entirely
- **Add manual position**: Mark a pre-existing bank item as tradeable

### Python API

```python
from firebase.position_tracker import PositionTracker

tracker = PositionTracker(account_id="b1llstar")

# Check if item is tradeable
if tracker.is_tradeable(item_id=2):
    print("Cannonballs can be sold")

# Get all tradeable positions
positions = tracker.get_tradeable_positions()

# Lock a position (from Vue dashboard)
tracker.lock_position(item_id=2)

# Manually add a bank item to positions
tracker.add_manual_position(
    item_id=2,
    item_name="Cannonball",
    quantity=5000,
    avg_cost=150
)

# Get sellable items from bank
bank_items = {2: 10000, 3: 5000}  # item_id -> quantity
sellable = tracker.get_sellable_items(bank_items)
# Returns only items that are in active positions
```
