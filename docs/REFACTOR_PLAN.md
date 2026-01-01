# GEAuto + PPOFlipperOpus Refactoring Plan

## Current State Analysis

### GEAuto Plugin (Java/RuneLite)
- **Location**: `/Users/b1llstar/IdeaProjects/RuneLite__Star/runelite_star_new/runelite-client/src/main/java/net/runelite/client/plugins/geauto`
- Already has comprehensive Firebase integration
- 40+ state machine states for GE automation
- Has listeners for order status, trades, inventory sync
- Smart slot management and priority queue system

### PPOFlipperOpus (Python)
- **Location**: `/Users/b1llstar/IdeaProjects/PPOFlipperOpus`
- Has inference bridge, order manager, portfolio tracker
- Sends buy/sell orders via Firestore
- Tracks positions and portfolio state

---

## Target Firestore Schema

```
/items/{itemId}
  └─ item_name: string
  └─ price: number
  └─ tradeable: boolean

/itemNames/{itemNameKey}  (document ID = normalized item name)
  └─ item_id: number
  └─ display_name: string

/accounts/{accountId}   (Parent - represents each character)
  ├─ display_name: string
  ├─ last_login: timestamp
  ├─ gold: number
  ├─ heartbeat: timestamp
  ├─ status: string (online/offline)
  │
  ├─ /orders/{orderId}   (Subcollection - buy/sell order tracking)
  │  ├─ order_id: string (UUID)
  │  ├─ action: string (buy/sell)
  │  ├─ item_name: string
  │  ├─ item_id: number (reference to /items collection)
  │  ├─ quantity: number
  │  ├─ price: number
  │  ├─ status: string (pending|received|placed|partial|completed|cancelled|failed)
  │  ├─ ge_slot: number (1-8, null until placed)
  │  ├─ filled_quantity: number
  │  ├─ gold_exchanged: number (total GP spent or received)
  │  ├─ tax_paid: number (1% on sells)
  │  ├─ error_message: string
  │  ├─ source: string ("ppo" or "manual")
  │  ├─ created_at: timestamp
  │  ├─ updated_at: timestamp
  │  └─ completed_at: timestamp
  │
  ├─ /portfolio/{itemId}   (Subcollection - accumulated items from orders)
  │  ├─ item_id: number
  │  ├─ item_name: string
  │  ├─ quantity: number (bought - sold, should match inventory+bank)
  │  ├─ avg_cost: number (weighted average purchase price)
  │  ├─ total_invested: number (total GP spent acquiring)
  │  ├─ location: string (inventory|bank|mixed)
  │  ├─ updated_at: timestamp
  │  └─ trades: array [{ order_id, action, qty, price, timestamp }]
  │
  ├─ /inventory/{itemId}   (Subcollection - current inventory items)
  │  ├─ item_id: number
  │  ├─ item_name: string
  │  ├─ quantity: number
  │  ├─ slot: number (0-27)
  │  ├─ is_portfolio_item: boolean (true if part of PPO portfolio)
  │  ├─ noted: boolean
  │  └─ updated_at: timestamp
  │
  └─ /bank/{itemId}   (Subcollection - current bank items)
     ├─ item_id: number
     ├─ item_name: string
     ├─ quantity: number
     ├─ tab: number (bank tab)
     ├─ is_portfolio_item: boolean (true if part of PPO portfolio)
     └─ updated_at: timestamp
```

---

## Refactoring Tasks

### Phase 1: Schema & Constants Alignment
- [ ] Update `FirebaseConfig.java` in GEAuto with new schema
- [ ] Update `firebase_config.py` in PPOFlipperOpus with matching constants
- [ ] Create shared schema documentation

### Phase 2: GEAuto Plugin Refactoring
- [ ] Refactor `FirebaseInventorySync.java` to use new inventory/bank subcollections
- [ ] Add `is_portfolio_item` flag tracking
- [ ] Update `FirebaseTradeReporter.java` to include gold_exchanged, tax_paid
- [ ] Create `PortfolioManager.java` for portfolio subcollection management
- [ ] Add manual sync methods for user-triggered syncs
- [ ] Add order persistence across sessions (completed orders should be queryable)

### Phase 3: PPOFlipperOpus Refactoring
- [ ] Update `order_manager.py` to match new schema
- [ ] Update `portfolio_tracker.py` to read from new portfolio subcollection
- [ ] Add methods to check if item is in portfolio before selling
- [ ] Update `position_tracker.py` to sync with portfolio collection
- [ ] Ensure only portfolio items can be sold

### Phase 4: Sync & Communication Flow
- [ ] Implement startup sync workflow in GEAuto
- [ ] Add manual "Sync Portfolio" command to verify inventory+bank matches portfolio
- [ ] Add order queue persistence (pending orders survive restarts)
- [ ] Add heartbeat/status checking on both sides

### Phase 5: Inventory Management
- [ ] Implement auto-banking logic in GEAuto when inventory full
- [ ] Add withdraw-from-bank logic for sell orders
- [ ] Track item locations (inventory vs bank) in portfolio

### Phase 6: Testing & Validation
- [ ] Write integration tests for order flow
- [ ] Verify portfolio consistency after buy/sell cycles
- [ ] Test session restart scenarios

---

## Progress Tracker

| Task | Status | Notes |
|------|--------|-------|
| Phase 1: Schema Alignment | ✅ Complete | FirebaseConfig.java & firebase_config.py updated |
| Phase 2: GEAuto Refactor | ✅ Complete | PortfolioManager.java, FirebaseTradeReporter, GEOrder updated |
| Phase 3: PPO Refactor | ✅ Complete | order_manager.py, portfolio_manager.py, inference_bridge.py updated |
| Phase 4: Sync Flow | ✅ Complete | Manual "Sync Portfolio" button added to panel |
| Phase 5: Inventory Mgmt | 🔄 Partial | Auto-banking logic not yet implemented |
| Phase 6: Testing | Not Started | End-to-end flow testing needed |

### Completed Tasks:

#### Java/GEAuto Plugin:
1. ✅ Updated `FirebaseConfig.java` with new schema constants (source, gold_exchanged, tax_paid, location fields)
2. ✅ Created `PortfolioManager.java` - manages portfolio subcollection with:
   - `addToPortfolio()` - tracks buys with weighted average cost
   - `removeFromPortfolio()` - tracks sells with tax
   - `canSell()` - validates portfolio ownership before selling
   - `verifyPortfolio()` - reconciles with inventory/bank
   - `syncLocations()` - updates item location (inventory/bank/mixed)
3. ✅ Updated `FirebaseTradeReporter.java` with:
   - gold_exchanged and tax_paid on order completion
   - Integration with PortfolioManager
   - Source field tracking
4. ✅ Updated `GEOrder.java` with source field
5. ✅ Updated `FirebaseOrderListener.java` with `markOrderCompletedWithDetails()` method

#### Python/PPOFlipperOpus:
6. ✅ Updated `firebase_config.py` with matching constants
7. ✅ Updated `order_manager.py` with:
   - source field (SOURCE_PPO)
   - gold_exchanged and tax_paid fields
   - Using config constants
8. ✅ Created `portfolio_manager.py` - Python equivalent of Java PortfolioManager with:
   - `PortfolioItem` dataclass
   - `add_to_portfolio()` / `remove_from_portfolio()`
   - `can_sell()` - validates portfolio ownership
   - `verify_portfolio()` - finds discrepancies
   - Caching for performance
9. ✅ Updated `portfolio_tracker.py` with:
   - Integration with PortfolioManager
   - `ppo_portfolio` attribute for accessing PPO-owned items
10. ✅ Updated `inference_bridge.py` with:
    - Portfolio validation before sells
    - Checks both portfolio ownership AND physical availability

#### Phase 4: Manual Sync Commands (NEW):
11. ✅ Added `reconcilePortfolio()` method to `PortfolioManager.java`:
    - Removes stale items from portfolio that no longer exist in inventory/bank
    - Adjusts portfolio quantities to match actual holdings
    - Adds reconciliation record to trade history
    - Returns `ReconciliationResult` with summary of changes
12. ✅ Integrated `PortfolioManager` into `GEAutoFirebaseIntegration.java`:
    - Added `portfolioManager` field
    - Connected to `FirebaseTradeReporter` for automatic portfolio updates
    - Added `syncPortfolio()` method for manual reconciliation
13. ✅ Added `manualPortfolioSync()` method to `GEAutoPlugin.java`:
    - Syncs inventory and bank state first
    - Calls portfolio reconciliation
    - Returns summary message for UI display
14. ✅ Added "Sync Portfolio" button to `GEAutoPanel.java`:
    - Warns user if bank is not open (items in bank will appear missing)
    - Confirmation dialog before proceeding without bank
    - Shows reconciliation result summary

---

## Key Design Decisions

### 1. Portfolio as Source of Truth
- Portfolio collection = sum of all completed buy orders - all completed sell orders
- Portfolio should match inventory + bank quantities for portfolio items
- Non-portfolio items (player's existing items) are excluded from PPO trading

### 2. Order Lifecycle
```
PPO creates order (status=pending)
  → Plugin receives (status=received)
  → Plugin places in GE (status=placed)
  → Partial fills (status=partial, filled_quantity updated)
  → Complete fill (status=completed, gold_exchanged, tax_paid calculated)
  → Portfolio updated (buy adds, sell subtracts)
```

### 3. Inventory/Bank Management
- GEAuto manages space automatically
- If inventory full during collection, deposit to bank first
- Sell orders: withdraw from bank if not in inventory
- Track location in portfolio (inventory|bank|mixed)

### 4. Session Persistence
- All orders persist in Firestore
- On plugin startup: sync active orders, verify portfolio consistency
- Manual sync: user can trigger full inventory+bank scan to reconcile portfolio

---

## Remaining Work

### Phase 5: Inventory Management (Partial)
- [ ] Implement auto-banking logic when inventory is full during collection
- [ ] Add withdraw-from-bank logic for sell orders when item not in inventory
- [ ] Automatic location tracking when items move between inventory/bank

### Phase 6: Testing & Validation
- [ ] End-to-end testing of the complete flow
- [ ] Verify portfolio consistency after buy/sell cycles
- [ ] Test session restart scenarios
- [ ] Test portfolio reconciliation with edge cases

### Future Enhancements
- [ ] Refactor FirebaseInventorySync.java to use per-item subcollections (optional optimization)
- [ ] Add `is_portfolio_item` flag to inventory/bank sync
- [ ] Add startup portfolio verification