# GEAuto + PPOFlipperOpus Integration Documentation

## Overview

This document details the refactoring work performed to create a clean, synchronized integration between the GEAuto RuneLite plugin (Java) and the PPOFlipperOpus inference system (Python). The goal is to enable automated Grand Exchange trading where PPO makes intelligent buy/sell decisions and the plugin executes them.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FIRESTORE (Cloud)                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ /accounts/{accountId}                                                │   │
│  │   ├── /orders/{orderId}      ← PPO writes, Plugin reads & updates   │   │
│  │   ├── /portfolio/{itemId}    ← Plugin writes, PPO reads             │   │
│  │   ├── /inventory/{itemId}    ← Plugin writes, PPO reads             │   │
│  │   ├── /bank/{itemId}         ← Plugin writes, PPO reads             │   │
│  │   ├── /trades/{tradeId}      ← Plugin writes (completed trades)     │   │
│  │   └── /ge_slots/current      ← Plugin writes (GE slot states)       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
           ▲                                              ▲
           │ Firestore SDK                                │ Firestore SDK
           │                                              │
┌──────────┴──────────┐                      ┌────────────┴────────────┐
│   PPOFlipperOpus    │                      │      GEAuto Plugin      │
│      (Python)       │                      │         (Java)          │
├─────────────────────┤                      ├─────────────────────────┤
│ • InferenceBridge   │                      │ • GEAutoPlugin          │
│ • OrderManager      │   ──────────────►    │ • FirebaseOrderListener │
│ • PortfolioManager  │   Creates Orders     │ • FirebaseTradeReporter │
│ • PortfolioTracker  │   ◄──────────────    │ • PortfolioManager      │
│                     │   Reports Status     │ • FirebaseInventorySync │
└─────────────────────┘                      └─────────────────────────┘
```

---

## Firestore Schema

### Account Document
```
/accounts/{accountId}
  ├─ display_name: string
  ├─ last_login: timestamp
  ├─ gold: number
  ├─ heartbeat: timestamp
  ├─ status: string (online/offline)
  ├─ plugin_online: boolean
  ├─ ge_slots_available: number
  └─ queue_size: number
```

### Orders Subcollection
```
/accounts/{accountId}/orders/{orderId}
  ├─ order_id: string (UUID, e.g., "ord_abc123def456")
  ├─ action: string ("buy" | "sell")
  ├─ item_id: number
  ├─ item_name: string
  ├─ quantity: number
  ├─ price: number (price per item)
  ├─ status: string (see Order Lifecycle below)
  ├─ ge_slot: number (1-8, null until placed)
  ├─ filled_quantity: number
  ├─ gold_exchanged: number (net GP change: negative for buys, positive for sells)
  ├─ tax_paid: number (1% on sells)
  ├─ source: string ("ppo" | "manual")
  ├─ error_message: string | null
  ├─ created_at: timestamp
  ├─ updated_at: timestamp
  ├─ completed_at: timestamp | null
  └─ metadata: {
       confidence: number,
       strategy: string,
       ...
     }
```

### Portfolio Subcollection
```
/accounts/{accountId}/portfolio/{itemId}
  ├─ item_id: number
  ├─ item_name: string
  ├─ quantity: number (total owned = bought - sold)
  ├─ avg_cost: number (weighted average purchase price)
  ├─ total_invested: number (total GP spent acquiring this item)
  ├─ location: string ("inventory" | "bank" | "mixed")
  ├─ created_at: timestamp
  ├─ updated_at: timestamp
  └─ trades: [
       { order_id, action, quantity, price, tax_paid?, timestamp },
       ...
     ]
```

### Inventory Subcollection
```
/accounts/{accountId}/inventory/current
  ├─ updated_at: timestamp
  ├─ gold: number
  ├─ item_count: number
  ├─ total_value: number
  ├─ free_slots: number
  └─ items: {
       "{itemId}": {
         item_id: number,
         item_name: string,
         quantity: number,
         price_each: number,
         total_value: number,
         is_portfolio_item: boolean
       },
       ...
     }
```

### Bank Subcollection
```
/accounts/{accountId}/bank/current
  ├─ updated_at: timestamp
  ├─ item_count: number
  ├─ total_value: number
  ├─ tradeable_count: number
  └─ items: {
       "{itemId}": {
         item_id: number,
         item_name: string,
         quantity: number,
         price_each: number,
         total_value: number,
         is_portfolio_item: boolean
       },
       ...
     }
```

---

## Order Lifecycle

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   PENDING   │ ──► │  RECEIVED   │ ──► │   PLACED    │ ──► │  COMPLETED  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
      │                   │                   │
      │                   │                   ▼
      │                   │            ┌─────────────┐
      │                   │            │   PARTIAL   │ ──► (back to PLACED or COMPLETED)
      │                   │            └─────────────┘
      │                   │
      ▼                   ▼
┌─────────────┐     ┌─────────────┐
│  CANCELLED  │     │   FAILED    │
└─────────────┘     └─────────────┘
```

| Status | Set By | Description |
|--------|--------|-------------|
| `pending` | PPO | Order created, waiting for plugin to pick up |
| `received` | Plugin | Plugin has received and queued the order |
| `placed` | Plugin | Order placed in a GE slot |
| `partial` | Plugin | Order partially filled |
| `completed` | Plugin | Order fully filled, items collected |
| `cancelled` | Either | Order was cancelled |
| `failed` | Plugin | Order failed (error details in error_message) |

---

## Completed Work

### Phase 1: Schema Alignment ✅

#### Java (GEAuto Plugin)
**File: `FirebaseConfig.java`**
- Added new field constants:
  - `FIELD_SOURCE` - tracks who created the order ("ppo" or "manual")
  - `FIELD_GOLD_EXCHANGED` - net GP change from order
  - `FIELD_TAX_PAID` - 1% tax on sell orders
  - `FIELD_LOCATION` - where portfolio items are stored
  - `FIELD_IS_PORTFOLIO_ITEM` - flag for inventory/bank items
  - `FIELD_TOTAL_INVESTED` - cost basis tracking
- Added location constants: `LOCATION_INVENTORY`, `LOCATION_BANK`, `LOCATION_MIXED`
- Added source constants: `SOURCE_PPO`, `SOURCE_MANUAL`
- Added command types: `CMD_SYNC_PORTFOLIO`, `CMD_SYNC_ORDERS`
- Added `GE_TAX_RATE = 0.01` (1%)

#### Python (PPOFlipperOpus)
**File: `config/firebase_config.py`**
- Added matching constants for all Java fields
- Full schema documentation in docstring
- Constants used throughout codebase for consistency

### Phase 2: GEAuto Plugin Refactoring ✅

#### PortfolioManager.java (NEW)
**Location:** `plugins/geauto/firebase/PortfolioManager.java`

Core class for managing the portfolio subcollection. Key features:
- **`addToPortfolio()`** - Called when buy orders complete
  - Calculates weighted average cost
  - Tracks total invested
  - Records trade in history array
- **`removeFromPortfolio()`** - Called when sell orders complete
  - Proportionally reduces cost basis
  - Records trade with tax paid
  - Deletes document if position fully closed
- **`canSell()`** - Validates portfolio ownership before sells
- **`getPortfolioQuantity()`** - Returns quantity owned
- **`verifyPortfolio()`** - Compares portfolio to actual inventory+bank
- **`syncLocations()`** - Updates location field based on where items are
- **Cache system** - 5-second TTL cache for performance

Inner classes:
- `PortfolioItem` - Data class representing a portfolio position
- `PortfolioDiscrepancy` - Represents mismatch between portfolio and reality

#### FirebaseTradeReporter.java (UPDATED)
- Now accepts `PortfolioManager` in constructor
- `handleOfferFilled()` now:
  - Calculates `goldExchanged` and `taxPaid`
  - Calls `PortfolioManager.addToPortfolio()` for buys
  - Calls `PortfolioManager.removeFromPortfolio()` for sells
  - Records trade with full financial details
- `recordTrade()` now includes:
  - `gold_exchanged` field
  - `tax_paid` field
  - `source` field

#### FirebaseOrderListener.java (UPDATED)
- Added `markOrderCompletedWithDetails()` method
- Updates order with:
  - `filled_quantity`
  - `gold_exchanged`
  - `tax_paid`
  - `completed_at`

#### GEOrder.java (UPDATED)
- Added `source` field with getter/setter
- Tracks whether order was created by PPO or manually

### Phase 3: PPOFlipperOpus Refactoring ✅

#### portfolio_manager.py (NEW)
**Location:** `firebase/portfolio_manager.py`

Python equivalent of Java PortfolioManager:
- `PortfolioItem` dataclass with P&L calculation methods
- `PortfolioDiscrepancy` dataclass for verification
- `PortfolioManager` class with:
  - `add_to_portfolio()` / `remove_from_portfolio()`
  - `can_sell()` - validates before sell orders
  - `get_portfolio_quantity()` - returns owned quantity
  - `verify_portfolio()` - finds discrepancies
  - `sync_locations()` - updates location fields
  - `get_portfolio_summary()` - returns full state
  - Cache with 5-second TTL

#### order_manager.py (UPDATED)
- Imports config constants
- `_create_order()` now includes:
  - `source: SOURCE_PPO` - marks as PPO-created
  - `gold_exchanged: 0` - initialized
  - `tax_paid: 0` - initialized
- Uses field constants instead of string literals

#### portfolio_tracker.py (UPDATED)
- Now imports and integrates `PortfolioManager`
- Added `ppo_portfolio` attribute for PPO-owned items
- Provides unified access to:
  - Plugin-synced data (inventory, bank, GE slots)
  - PPO portfolio data

#### inference_bridge.py (UPDATED)
- `submit_sell_order()` now validates:
  1. **Portfolio ownership** - PPO can only sell items it acquired
  2. **Physical availability** - Item must exist in inventory or bank
- Logs detailed warnings when sells are rejected

---

## Remaining Work

### Phase 4: Sync Flow 🔄

#### Manual Sync Commands
Need to implement in GEAuto plugin:
- Button in panel to trigger full inventory+bank scan
- Command listener for `CMD_SYNC_PORTFOLIO`
- Reconciliation logic to fix discrepancies

#### Startup Sync
When plugin starts:
1. Load active orders from Firestore
2. Verify against current GE slots
3. Resume any in-progress orders
4. Verify portfolio consistency

#### Order Queue Persistence
- Save pending orders on logout
- Restore on login
- Handle orders that completed while offline

### Phase 5: Inventory Management 🔄

#### Auto-Banking
When inventory is full during collection:
1. Detect inventory full condition
2. Open bank and deposit items
3. Close bank, resume GE operations

#### Bank Withdrawal for Sells
When sell order needs item from bank:
1. Check if item is in inventory
2. If not, check bank
3. If in bank, withdraw to inventory
4. Proceed with sell order

#### Location Tracking
- Update portfolio location when items move
- Sync locations on bank open/close
- Track "mixed" state when item in both places

### Phase 6: Testing 📋

#### Integration Tests Needed
1. **Buy Order Flow**
   - PPO creates order → Plugin executes → Portfolio updated
   - Verify gold_exchanged is negative
   - Verify portfolio quantity increases

2. **Sell Order Flow**
   - PPO creates sell → Portfolio validation → Plugin executes
   - Verify gold_exchanged is positive (minus tax)
   - Verify portfolio quantity decreases
   - Verify tax_paid is 1% of gross

3. **Portfolio Validation**
   - Attempt to sell item not in portfolio → Rejected
   - Attempt to sell more than owned → Rejected
   - Verify portfolio matches inventory+bank

4. **Session Restart**
   - Create orders → Restart plugin → Orders resume
   - Complete trade offline → Sync on restart

5. **Edge Cases**
   - Partial fills
   - Cancelled orders
   - Failed orders
   - Network disconnection

---

## Key Files Reference

### Java (GEAuto Plugin)
```
runelite-client/src/main/java/net/runelite/client/plugins/geauto/
├── GEOrder.java                    # Order data class (updated)
├── GEAutoPlugin.java               # Main plugin
├── GEQueueManager.java             # Order queue management
└── firebase/
    ├── FirebaseConfig.java         # Constants (updated)
    ├── FirebaseManager.java        # Firebase connection
    ├── FirebaseOrderListener.java  # Listens for orders (updated)
    ├── FirebaseTradeReporter.java  # Reports trades (updated)
    ├── FirebaseInventorySync.java  # Syncs inventory/bank
    └── PortfolioManager.java       # NEW - Portfolio tracking
```

### Python (PPOFlipperOpus)
```
PPOFlipperOpus/
├── config/
│   └── firebase_config.py          # Constants (updated)
├── firebase/
│   ├── firebase_client.py          # Firebase connection
│   ├── order_manager.py            # Creates orders (updated)
│   ├── portfolio_manager.py        # NEW - Portfolio tracking
│   ├── portfolio_tracker.py        # Tracks state (updated)
│   ├── inference_bridge.py         # Main orchestrator (updated)
│   └── trade_monitor.py            # Monitors trades
└── inference/
    └── run_firebase_inference.py   # Main inference loop
```

---

## Usage Examples

### Creating a Buy Order (Python)
```python
from firebase.inference_bridge import InferenceBridge

bridge = InferenceBridge(
    service_account_path="path/to/service_account.json",
    account_id="b1llstar"
)
bridge.start()

# This will be picked up by the plugin and executed
order_id = bridge.submit_buy_order(
    item_id=4151,
    item_name="Abyssal whip",
    quantity=10,
    price=2500000,
    confidence=0.85,
    strategy="ppo"
)
```

### Creating a Sell Order (Python)
```python
# This will validate portfolio ownership first
order_id = bridge.submit_sell_order(
    item_id=4151,
    item_name="Abyssal whip",
    quantity=5,
    price=2600000,
    confidence=0.90,
    strategy="ppo"
)

# If not in portfolio, returns None with warning:
# "Cannot sell: item Abyssal whip (ID: 4151) - portfolio has 0, trying to sell 5"
```

### Checking Portfolio (Python)
```python
# Get quantity of an item
qty = bridge.portfolio_tracker.ppo_portfolio.get_portfolio_quantity(4151)
print(f"Own {qty} Abyssal whips")

# Check if can sell
can_sell = bridge.portfolio_tracker.ppo_portfolio.can_sell(4151, 5)
print(f"Can sell 5: {can_sell}")

# Get full portfolio summary
summary = bridge.portfolio_tracker.ppo_portfolio.get_portfolio_summary()
print(f"Portfolio: {summary['item_count']} items, {summary['total_invested']} GP invested")
```

### Handling Trades (Java Plugin)
```java
// In FirebaseTradeReporter.handleOfferFilled()
if (portfolioManager != null) {
    if (FirebaseConfig.ACTION_BUY.equals(action)) {
        portfolioManager.addToPortfolio(
            order.getItemId(),
            order.getItemName(),
            quantity,
            pricePerItem,
            order.getOrderId()
        );
    }
}
```

---

## Configuration

### Environment Variables
```bash
# Python side
export PPO_ACCOUNT_ID="b1llstar"
export FIREBASE_SERVICE_ACCOUNT="/path/to/service_account.json"
```

### Key Settings
| Setting | Value | Description |
|---------|-------|-------------|
| `GE_TAX_RATE` | 0.01 (1%) | Tax on sell orders |
| `HEARTBEAT_INTERVAL` | 30 seconds | Plugin heartbeat frequency |
| `PLUGIN_ONLINE_THRESHOLD` | 120 seconds | Max heartbeat age to consider online |
| `ORDER_STALE_TIMEOUT` | 600 seconds | Cancel orders after 10 minutes |
| `CACHE_TTL` | 5 seconds | Portfolio cache lifetime |

---

## Troubleshooting

### Order Not Being Picked Up
1. Check plugin is running and connected to Firebase
2. Verify account_id matches between PPO and plugin
3. Check order status is "pending"
4. Look for errors in plugin logs

### Sell Order Rejected
1. Check portfolio has the item: `ppo_portfolio.get_portfolio_quantity(item_id)`
2. Check physical availability in inventory/bank
3. Verify quantity doesn't exceed portfolio amount

### Portfolio Mismatch
1. Run `ppo_portfolio.verify_portfolio(inventory, bank)`
2. Check for discrepancies
3. May need manual reconciliation if trades completed while offline

### Plugin Appears Offline
1. Check heartbeat timestamp: `portfolio_tracker.get_last_heartbeat()`
2. Verify Firebase connection on plugin side
3. Check for network issues

---

## Future Enhancements

1. **Multi-Account Support** - Run PPO for multiple accounts simultaneously
2. **Trade Analytics** - Track P&L, win rate, average hold time
3. **Risk Management** - Position sizing based on portfolio value
4. **Alert System** - Notify on large P&L swings or errors
5. **Backup/Restore** - Export/import portfolio state
6. **Web Dashboard** - Real-time monitoring interface
