# GE Auto V2 + Inferencing V2 - Complete Redesign Proposal

## Executive Summary

This proposal outlines a complete redesign of the Firebase infrastructure between the PPO Inferencing system and the RuneLite GE Auto plugin. The new architecture focuses on **simplicity**, **clarity**, and **synchronization** with three core data flows:

1. **Portfolio Tracking** - Plugin maintains complete state of bank, inventory, and GE slots
2. **Order Execution** - Inferencing creates orders, plugin executes them
3. **Position Management** - Track what we own and what we're trading

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           FIRESTORE                                      │
│                                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────┐  │
│  │   items     │    │  itemNames  │    │         accounts            │  │
│  │             │    │             │    │                             │  │
│  │ {id: props} │    │ {name: id}  │    │  /{accountId}/              │  │
│  │             │    │             │    │     ├── portfolio (doc)     │  │
│  └─────────────┘    └─────────────┘    │     ├── bank (doc)          │  │
│                                        │     ├── inventory (doc)     │  │
│                                        │     ├── ge_state (doc)      │  │
│                                        │     └── orders/ (collection)│  │
│                                        └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                          │                           │
                          ▼                           ▼
        ┌─────────────────────────┐     ┌─────────────────────────────┐
        │     INFERENCING         │     │       GE AUTO V2 PLUGIN     │
        │                         │     │                             │
        │  • Reads portfolio      │     │  • Scans bank/inventory     │
        │  • Creates BUY orders   │     │  • Syncs GE slot state      │
        │  • Creates SELL orders  │     │  • Listens for new orders   │
        │  • Monitors completions │     │  • Executes trades          │
        │                         │     │  • Reports completions      │
        └─────────────────────────┘     └─────────────────────────────┘
```

---

## Firestore Schema

### Collection: `items`
Static item database. Pre-populated with all tradeable items.

```typescript
// Document ID: item_id (e.g., "2" for Cannonball)
{
  id: 2,
  name: "Cannonball",
  members: true,
  limit: 15000,           // GE buy limit per 4 hours
  high_alch: 2,
  low_alch: 1,
  tradeable: true,
  stackable: true
}
```

### Collection: `itemNames`
Reverse lookup: name → id. Pre-populated.

```typescript
// Document ID: normalized item name (e.g., "cannonball")
{
  id: 2,
  name: "Cannonball"
}
```

### Collection: `accounts`
Per-user data. Document ID = lowercased player name (spaces → underscores).

#### Document: `accounts/{accountId}/portfolio`
**Single source of truth for account state.** Updated by plugin.

```typescript
{
  gold: 5000000,
  total_value: 7500000,        // gold + holdings value
  holdings_count: 15,          // distinct items we own
  active_order_count: 3,       // orders currently in GE

  last_updated: Timestamp,
  plugin_online: true,
  plugin_version: "2.0.0"
}
```

#### Document: `accounts/{accountId}/bank`
Complete bank contents. Plugin scans and syncs.

```typescript
{
  items: {
    "2": { name: "Cannonball", quantity: 5000 },
    "4": { name: "Iron ore", quantity: 10000 },
    // ... all bank items
  },
  total_items: 150,
  scanned_at: Timestamp
}
```

#### Document: `accounts/{accountId}/inventory`
Current inventory contents. Plugin syncs on change.

```typescript
{
  items: {
    "2": { name: "Cannonball", quantity: 1000 },
    "995": { name: "Coins", quantity: 5000000 }
  },
  empty_slots: 20,
  scanned_at: Timestamp
}
```

#### Document: `accounts/{accountId}/ge_state`
Current state of all 8 GE slots. Plugin syncs on any GE change.

```typescript
{
  slots: {
    "1": {
      status: "active",        // empty | active | complete
      type: "buy",             // buy | sell
      item_id: 2,
      item_name: "Cannonball",
      quantity: 1000,
      price: 150,
      filled: 500,
      order_id: "ord_abc123"   // Links to our order (if we placed it)
    },
    "2": {
      status: "empty"
    },
    // ... slots 1-8
  },
  free_slots: 5,
  synced_at: Timestamp
}
```

#### Subcollection: `accounts/{accountId}/orders`
Orders created by inferencing, executed by plugin.

```typescript
// Document ID: auto-generated or "ord_{uuid}"
{
  // Identity
  order_id: "ord_abc123",

  // Order Details
  action: "buy",               // "buy" | "sell"
  item_id: 2,
  item_name: "Cannonball",
  quantity: 1000,
  price: 150,                  // Price per item

  // Status Tracking
  status: "pending",           // See status lifecycle below
  ge_slot: null,               // Assigned when placed (1-8)
  filled_quantity: 0,
  total_cost: 0,

  // Timestamps
  created_at: Timestamp,
  received_at: Timestamp,      // When plugin picked it up
  placed_at: Timestamp,        // When placed in GE
  completed_at: Timestamp,

  // Error Handling
  error: null,
  retry_count: 0,

  // Metadata (from inferencing)
  confidence: 0.85,
  strategy: "ppo_v2"
}
```

**Order Status Lifecycle:**
```
INFERENCING                           PLUGIN
    │                                   │
    ├─ Creates order ──────────────────►│ (status: "pending")
    │                                   │
    │◄─────────────────── Receives ─────┤ (status: "received")
    │                                   │
    │                      Places in GE─┤ (status: "placed", ge_slot: 3)
    │                                   │
    │                      Partial fill─┤ (status: "partial", filled: 500)
    │                                   │
    │◄──────────────────── Complete ────┤ (status: "completed")
    │                                   │
    ▼                                   ▼
```

Valid statuses: `pending` → `received` → `placed` → `partial` → `completed`
Error statuses: `failed`, `cancelled`

---

## Plugin Responsibilities (GE Auto V2)

### 1. Startup Sync

On plugin start or login:

```java
// Step 1: Scan and sync GE state
syncGESlots();
  - Read all 8 GE slots via built-in RuneLite APIs
  - For each slot, determine: empty, active buy, active sell, ready to collect
  - Update Firestore: accounts/{id}/ge_state
  - Match existing orders by item_id/quantity/price to link order_ids

// Step 2: Scan bank (if bank is open or can open)
scanBank();
  - Read entire bank contents
  - Update Firestore: accounts/{id}/bank

// Step 3: Sync inventory
syncInventory();
  - Read inventory contents
  - Update Firestore: accounts/{id}/inventory

// Step 4: Update portfolio summary
updatePortfolio();
  - Calculate totals from bank + inventory + ge_state
  - Update Firestore: accounts/{id}/portfolio
```

### 2. Continuous Sync

```java
// On ANY inventory change
@Subscribe
public void onItemContainerChanged(ItemContainerChanged event) {
    if (event.getContainerId() == InventoryID.INVENTORY.getId()) {
        syncInventory();
    }
    if (event.getContainerId() == InventoryID.BANK.getId()) {
        scanBank();
    }
}

// On ANY GE change (using built-in RuneLite)
@Subscribe
public void onGrandExchangeOfferChanged(GrandExchangeOfferChanged event) {
    syncGESlots();
    checkForCompletedOrders();
}
```

### 3. Order Listening & Execution

```java
// Real-time listener on accounts/{id}/orders
// WHERE status == "pending"
firestoreListener = ordersRef
    .whereEqualTo("status", "pending")
    .addSnapshotListener((snapshots, error) -> {
        for (DocumentChange change : snapshots.getDocumentChanges()) {
            if (change.getType() == ADDED) {
                Order order = parseOrder(change.getDocument());
                queueOrder(order);
            }
        }
    });

// Order execution queue (processes one at a time)
void executeOrder(Order order) {
    // Update status: pending → received
    updateOrderStatus(order.id, "received");

    if (order.action.equals("buy")) {
        executeBuyOrder(order);
    } else {
        executeSellOrder(order);
    }
}

void executeBuyOrder(Order order) {
    // Find free GE slot
    int slot = findFreeSlot();
    if (slot == -1) {
        updateOrderStatus(order.id, "failed", "No free GE slots");
        return;
    }

    // Place the buy offer (using existing GE utilities)
    placeBuyOffer(slot, order.itemId, order.quantity, order.price);

    // Update status: received → placed
    updateOrderStatus(order.id, "placed", slot);
}

void executeSellOrder(Order order) {
    // Verify we have the items in inventory
    int available = getInventoryQuantity(order.itemId);
    if (available < order.quantity) {
        // Try to withdraw from bank first
        if (!withdrawFromBank(order.itemId, order.quantity - available)) {
            updateOrderStatus(order.id, "failed", "Insufficient items");
            return;
        }
    }

    // Find free GE slot
    int slot = findFreeSlot();
    if (slot == -1) {
        updateOrderStatus(order.id, "failed", "No free GE slots");
        return;
    }

    // Place the sell offer
    placeSellOffer(slot, order.itemId, order.quantity, order.price);

    // Update status: received → placed
    updateOrderStatus(order.id, "placed", slot);
}
```

### 4. Completion Detection

```java
void checkForCompletedOrders() {
    // For each GE slot that is "complete" (ready to collect)
    for (GESlot slot : getCompleteSlots()) {
        // Find the matching order by order_id stored in ge_state
        String orderId = getOrderIdForSlot(slot.number);
        if (orderId != null) {
            // Update order: placed/partial → completed
            updateOrderCompletion(orderId, slot.filledQuantity, slot.totalCost);
        }

        // Collect the items/gold
        collectSlot(slot.number);
    }
}

void updateOrderCompletion(String orderId, int filled, int cost) {
    ordersRef.document(orderId).update(
        "status", "completed",
        "filled_quantity", filled,
        "total_cost", cost,
        "completed_at", FieldValue.serverTimestamp()
    );
}
```

### 5. Heartbeat

```java
// Every 30 seconds
void sendHeartbeat() {
    portfolioRef.update(
        "plugin_online", true,
        "last_updated", FieldValue.serverTimestamp()
    );
}
```

---

## Inferencing Responsibilities

### 1. Startup

```python
async def initialize():
    # Connect to Firestore
    firebase_client.initialize()

    # Start listening to portfolio state
    start_portfolio_listener()

    # Start listening to order completions
    start_order_listener()

    # Wait for plugin to be online
    await wait_for_plugin_online()

    # Initial state sync
    await sync_state()
```

### 2. State Access

```python
class InferenceBridge:
    """Clean interface for PPO inference to access state and place orders."""

    # Portfolio State
    def get_gold(self) -> int:
        """Current gold from portfolio doc."""

    def get_bank_items(self) -> Dict[int, BankItem]:
        """All items in bank."""

    def get_inventory_items(self) -> Dict[int, InventoryItem]:
        """All items in inventory."""

    def get_holdings(self) -> Dict[int, Holding]:
        """Combined bank + inventory items."""

    def get_ge_state(self) -> GEState:
        """Current GE slots state."""

    def get_free_slots(self) -> int:
        """Number of empty GE slots."""

    def is_plugin_online(self) -> bool:
        """Check if plugin is active (heartbeat within 2 min)."""
```

### 3. Order Submission

```python
class InferenceBridge:

    def submit_buy_order(
        self,
        item_id: int,
        item_name: str,
        quantity: int,
        price: int,
        confidence: float = 0.0,
        strategy: str = "ppo_v2"
    ) -> Optional[str]:
        """
        Submit a buy order for execution.

        Returns order_id if successful, None if validation fails.
        """
        # Validation
        if not self.is_plugin_online():
            logger.warning("Plugin offline, cannot submit order")
            return None

        if self.get_free_slots() < 1:
            logger.warning("No free GE slots")
            return None

        total_cost = quantity * price
        if total_cost > self.get_gold():
            logger.warning(f"Insufficient gold: need {total_cost}, have {self.get_gold()}")
            return None

        # Create order document
        order_id = f"ord_{uuid4().hex[:12]}"
        order = {
            "order_id": order_id,
            "action": "buy",
            "item_id": item_id,
            "item_name": item_name,
            "quantity": quantity,
            "price": price,
            "status": "pending",
            "ge_slot": None,
            "filled_quantity": 0,
            "total_cost": 0,
            "created_at": firestore.SERVER_TIMESTAMP,
            "confidence": confidence,
            "strategy": strategy
        }

        self.orders_ref.document(order_id).set(order)
        logger.info(f"Created buy order: {order_id} for {quantity}x {item_name} @ {price}")

        return order_id

    def submit_sell_order(
        self,
        item_id: int,
        item_name: str,
        quantity: int,
        price: int,
        confidence: float = 0.0,
        strategy: str = "ppo_v2"
    ) -> Optional[str]:
        """
        Submit a sell order for execution.

        Validates that we have the items before creating order.
        """
        # Validation
        if not self.is_plugin_online():
            return None

        if self.get_free_slots() < 1:
            return None

        # Check holdings (bank + inventory)
        holdings = self.get_holdings()
        available = holdings.get(item_id, Holding(0, 0)).quantity

        if available < quantity:
            logger.warning(f"Insufficient items: need {quantity}, have {available}")
            return None

        # Create order (same as buy, with action="sell")
        order_id = f"ord_{uuid4().hex[:12]}"
        order = {
            "order_id": order_id,
            "action": "sell",
            "item_id": item_id,
            "item_name": item_name,
            "quantity": quantity,
            "price": price,
            "status": "pending",
            "ge_slot": None,
            "filled_quantity": 0,
            "total_cost": 0,
            "created_at": firestore.SERVER_TIMESTAMP,
            "confidence": confidence,
            "strategy": strategy
        }

        self.orders_ref.document(order_id).set(order)
        logger.info(f"Created sell order: {order_id} for {quantity}x {item_name} @ {price}")

        return order_id
```

### 4. Order Monitoring

```python
class InferenceBridge:

    def start_order_listener(self):
        """Listen for order status changes."""

        def on_order_change(doc_snapshot, changes, read_time):
            for change in changes:
                order = change.document.to_dict()
                order_id = order["order_id"]
                status = order["status"]

                if status == "completed":
                    self._on_order_completed(order)
                elif status == "failed":
                    self._on_order_failed(order)
                elif status == "cancelled":
                    self._on_order_cancelled(order)

        # Listen to all our orders
        self.orders_ref.on_snapshot(on_order_change)

    def _on_order_completed(self, order: dict):
        """Handle order completion - update positions, log trade."""
        logger.info(f"Order completed: {order['order_id']} - "
                    f"{order['action']} {order['filled_quantity']}x {order['item_name']}")

        # Trade tracking/P&L can be calculated from completed orders
        # No need for separate trades collection

    def get_pending_orders(self) -> List[Order]:
        """Get all orders we're waiting on."""
        docs = self.orders_ref.where("status", "in",
            ["pending", "received", "placed", "partial"]).get()
        return [Order.from_dict(d.to_dict()) for d in docs]

    def get_active_order_count(self) -> int:
        """Orders currently using GE slots."""
        docs = self.orders_ref.where("status", "in",
            ["placed", "partial"]).get()
        return len(list(docs))
```

---

## Simplified Inference Script

```python
# inference/run_inference_v2.py

class InferenceRunner:
    """Simplified inference runner using V2 architecture."""

    def __init__(self):
        self.bridge = InferenceBridgeV2()
        self.agent = load_ppo_agent()
        self.items = load_item_data()

    async def run(self):
        """Main inference loop."""
        await self.bridge.initialize()

        while True:
            try:
                await self.make_decision()
            except Exception as e:
                logger.error(f"Decision error: {e}")

            await asyncio.sleep(DECISION_INTERVAL)

    async def make_decision(self):
        """Single decision cycle."""
        # Check plugin is online
        if not self.bridge.is_plugin_online():
            logger.debug("Plugin offline, skipping")
            return

        # Get current state
        gold = self.bridge.get_gold()
        holdings = self.bridge.get_holdings()
        ge_state = self.bridge.get_ge_state()
        active_orders = self.bridge.get_active_order_count()

        # Skip if all slots are full
        if ge_state.free_slots == 0:
            logger.debug("No free GE slots, waiting")
            return

        # Get market data
        market_data = await self.get_market_data()

        # Build observation for PPO
        obs = self.build_observation(gold, holdings, ge_state, market_data)

        # Get PPO decision
        action = self.agent.get_action(obs)

        # Execute decision
        if action.type == ActionType.BUY:
            await self.execute_buy(action)
        elif action.type == ActionType.SELL:
            await self.execute_sell(action)
        # else: HOLD - do nothing

    async def execute_buy(self, action: Action):
        """Execute a buy decision."""
        item = self.items[action.item_idx]

        # Calculate quantity (position sizing)
        max_value = self.bridge.get_gold() * 0.1  # 10% of gold max
        quantity = min(
            action.quantity,
            int(max_value / action.price),
            item.ge_limit
        )

        if quantity < 1:
            logger.debug(f"Buy quantity too low for {item.name}")
            return

        order_id = self.bridge.submit_buy_order(
            item_id=item.id,
            item_name=item.name,
            quantity=quantity,
            price=action.price,
            confidence=action.confidence,
            strategy="ppo_v2"
        )

        if order_id:
            logger.info(f"Submitted buy: {quantity}x {item.name} @ {action.price}")

    async def execute_sell(self, action: Action):
        """Execute a sell decision."""
        item = self.items[action.item_idx]

        # Can only sell what we have
        holdings = self.bridge.get_holdings()
        available = holdings.get(item.id, Holding(0, 0)).quantity

        if available < 1:
            logger.debug(f"No {item.name} to sell")
            return

        quantity = min(action.quantity, available)

        order_id = self.bridge.submit_sell_order(
            item_id=item.id,
            item_name=item.name,
            quantity=quantity,
            price=action.price,
            confidence=action.confidence,
            strategy="ppo_v2"
        )

        if order_id:
            logger.info(f"Submitted sell: {quantity}x {item.name} @ {action.price}")
```

---

## Key Simplifications from V1

| Aspect | V1 (Current) | V2 (New) |
|--------|--------------|----------|
| **Collections** | 7+ subcollections (orders, trades, inventory, bank, portfolio, ge_slots, positions, commands) | 4 documents + 1 subcollection (portfolio, bank, inventory, ge_state, orders) |
| **Order Status** | Complex with retry logic | Simple linear: pending → received → placed → completed |
| **Trade Recording** | Separate trades collection | Calculated from completed orders |
| **Position Tracking** | Separate positions system | Derived from holdings (bank + inventory) |
| **Commands** | Separate commands collection for withdraw/deposit | Plugin handles automatically (withdraws for sell orders) |
| **Bank/Inventory Sync** | Periodic with debouncing | Event-driven, always current |
| **GE Slot Tracking** | Complex slot management | Single ge_state document with slot linkage |

---

## File Structure

### Plugin (Java)
```
runelite-client/src/main/java/net/runelite/client/plugins/geautov2/
├── GEAutoV2Plugin.java          # Main plugin, lifecycle, event handling
├── GEAutoV2Config.java          # Configuration
├── GEAutoV2State.java           # State machine enum
├── GEAutoV2Overlay.java         # In-game overlay
├── GEAutoV2Panel.java           # Side panel UI
│
├── firebase/
│   ├── FirebaseManager.java     # Firestore connection singleton
│   ├── FirebaseConfig.java      # Collection names, field names
│   ├── PortfolioSync.java       # Syncs portfolio/bank/inventory/ge_state
│   └── OrderExecutor.java       # Listens for orders, executes them
│
├── model/
│   ├── Order.java               # Order POJO
│   ├── GESlotState.java         # GE slot representation
│   └── Holding.java             # Item holding (bank/inventory)
│
└── util/
    ├── GEInteraction.java       # GE widget interaction utilities
    └── BankInteraction.java     # Bank widget interaction utilities
```

### Inferencing (Python)
```
PPOFlipperOpus/
├── inference_v2/
│   ├── run_inference.py         # Main inference loop
│   ├── inference_bridge.py      # Clean Firestore interface
│   └── decision_maker.py        # PPO decision wrapper
│
├── firebase_v2/
│   ├── firebase_client.py       # Firestore connection
│   ├── state_listener.py        # Real-time state updates
│   └── order_manager.py         # Order creation and monitoring
│
├── config/
│   └── firebase_config.py       # Configuration
│
└── model/
    └── [existing PPO model files]
```

---

## Migration Path

1. **Create V2 collections** in Firestore (can coexist with V1)
2. **Build GE Auto V2 plugin** as new plugin (doesn't replace V1)
3. **Build inference_v2** module alongside existing inference
4. **Test with dry-run** (EXECUTE_TRADES = False)
5. **Gradually switch over** once stable
6. **Deprecate V1** after validation

---

## Vue Dashboard

A web-based dashboard for real-time monitoring, debugging, and manual intervention.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         VUE DASHBOARD                                    │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │   Portfolio  │  │    Orders    │  │  GE Slots    │  │   Actions   │  │
│  │    Panel     │  │    Panel     │  │    Panel     │  │    Panel    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘  │
│         │                │                 │                  │          │
│         └────────────────┴─────────────────┴──────────────────┘          │
│                                    │                                     │
│                          Firestore Real-time                             │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                              ┌─────────────┐
                              │  FIRESTORE  │
                              └─────────────┘
```

### Tech Stack

- **Vue 3** with Composition API
- **Vite** for build tooling
- **Firebase JS SDK** for real-time Firestore listeners
- **TailwindCSS** for styling
- **Pinia** for state management

### Dashboard Views

#### 1. Portfolio Overview (Main Dashboard)

```
┌─────────────────────────────────────────────────────────────────────┐
│  PPO Flipper Dashboard                        🟢 Plugin Online      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │   5.2M GP   │  │  7.8M Total │  │   15 Items  │  │  3/8 Slots │ │
│  │    Gold     │  │    Value    │  │   Holdings  │  │   Active   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┤
│  │  GE SLOTS                                                       │
│  ├─────────────────────────────────────────────────────────────────┤
│  │  [1] 🟢 BUY  Cannonball    500/1000 @ 150gp   [Cancel]         │
│  │  [2] 🟡 SELL Iron ore      250/500  @ 125gp   [Cancel]         │
│  │  [3] 🔵 BUY  Nature rune   1000/1000 READY    [Collect]        │
│  │  [4] ⚫ Empty                                                   │
│  │  [5] ⚫ Empty                                                   │
│  │  [6] ⚫ Empty                                                   │
│  │  [7] ⚫ Empty                                                   │
│  │  [8] ⚫ Empty                                                   │
│  └─────────────────────────────────────────────────────────────────┘
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### 2. Orders Panel

```
┌─────────────────────────────────────────────────────────────────────┐
│  ORDERS                                    [Clear Completed] [⟳]   │
├─────────────────────────────────────────────────────────────────────┤
│  Filter: [All ▼]  [Pending] [Placed] [Completed] [Failed]          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ord_abc123  BUY   Cannonball  1000 @ 150   PLACED   Slot 1        │
│              Confidence: 0.85  Strategy: ppo_v2                     │
│              Created: 2 min ago                    [Cancel Order]   │
│  ─────────────────────────────────────────────────────────────────  │
│  ord_def456  SELL  Iron ore    500 @ 125    PARTIAL  Slot 2        │
│              Filled: 250/500   Confidence: 0.72                     │
│              Created: 5 min ago                    [Cancel Order]   │
│  ─────────────────────────────────────────────────────────────────  │
│  ord_ghi789  BUY   Nature rune 1000 @ 180   COMPLETED               │
│              Filled: 1000/1000  Total: 180,000gp                    │
│              Completed: 1 min ago                  [Delete Record]  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### 3. Inventory Panel

```
┌─────────────────────────────────────────────────────────────────────┐
│  INVENTORY (8/28 slots used)               [Sync] [Deposit All]    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐       │
│  │ 💰    │ │ 🔵    │ │ ⚫    │ │ 🟤    │ │       │ │       │       │
│  │5.2M GP│ │ x1000 │ │ x500  │ │ x250  │ │       │ │       │       │
│  │       │ │Cannon │ │Iron   │ │Nature │ │       │ │       │       │
│  │[-----]│ │[Sell] │ │[Sell] │ │[Sell] │ │       │ │       │       │
│  │       │ │[Dep]  │ │[Dep]  │ │[Dep]  │ │       │ │       │       │
│  └───────┘ └───────┘ └───────┘ └───────┘ └───────┘ └───────┘       │
│  ... (more slots)                                                   │
│                                                                     │
│  Quick Actions:                                                     │
│  [Remove Item from Firestore ▼]  [Remove from Inventory ▼]         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### 4. Bank Panel

```
┌─────────────────────────────────────────────────────────────────────┐
│  BANK (150 unique items)                   [Sync] [Search: ____]   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Item Name          Quantity    Value Each    Total Value   Actions │
│  ─────────────────────────────────────────────────────────────────  │
│  Cannonball         50,000      150gp         7.5M          [Withdraw] [Remove] │
│  Iron ore           25,000      125gp         3.1M          [Withdraw] [Remove] │
│  Nature rune        10,000      180gp         1.8M          [Withdraw] [Remove] │
│  ...                                                                │
│                                                                     │
│  Debug Actions:                                                     │
│  [Remove Item from Bank Doc ▼]                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### 5. Debug Panel

```
┌─────────────────────────────────────────────────────────────────────┐
│  DEBUG & MANUAL ACTIONS                              [Danger Zone]  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┤
│  │  CREATE MANUAL ORDER                                            │
│  │  Type: [Buy ▼]  Item: [__________]  Qty: [____]  Price: [____] │
│  │                                           [Submit Order]        │
│  └─────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┤
│  │  FIRESTORE DOCUMENT ACTIONS                                     │
│  │                                                                 │
│  │  Delete Order:     [order_id: ___________]  [Delete]            │
│  │  Clear All Pending Orders:                  [Clear Pending]     │
│  │  Clear All Completed Orders:                [Clear Completed]   │
│  │                                                                 │
│  │  Remove from Inventory Doc:                                     │
│  │    Item ID: [____]                          [Remove]            │
│  │                                                                 │
│  │  Remove from Bank Doc:                                          │
│  │    Item ID: [____]                          [Remove]            │
│  │                                                                 │
│  │  Force Portfolio Refresh:                   [Trigger Sync]      │
│  │                                                                 │
│  └─────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┤
│  │  RAW FIRESTORE DATA                                             │
│  │  Document: [portfolio ▼]                    [View JSON]         │
│  │  ┌─────────────────────────────────────────────────────────────┐│
│  │  │ {                                                           ││
│  │  │   "gold": 5200000,                                          ││
│  │  │   "total_value": 7800000,                                   ││
│  │  │   "plugin_online": true,                                    ││
│  │  │   "last_updated": "2025-12-31T12:00:00Z"                    ││
│  │  │ }                                                           ││
│  │  └─────────────────────────────────────────────────────────────┘│
│  └─────────────────────────────────────────────────────────────────┤
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Features

#### Real-time Updates
- All panels use Firestore `onSnapshot()` listeners
- Instant updates when plugin syncs state
- Visual indicators for stale data (>2 min since last update)

#### Order Management
- **Cancel Order**: Sets order status to `cancelled` (plugin will abort if not yet placed)
- **Delete Record**: Removes completed/failed order documents from Firestore
- **Clear Completed**: Bulk delete all completed orders
- **Clear Pending**: Cancel and delete all pending orders

#### Inventory/Bank Debugging
- **Remove from Firestore**: Delete item entry from inventory/bank document
  - Does NOT affect actual game state
  - Forces discrepancy that plugin will detect and correct on next sync
- **Withdraw**: Creates a withdraw command for plugin (if V2 supports commands)
- **Deposit**: Creates a deposit command

#### Manual Order Creation
- Bypass inference to manually submit buy/sell orders
- Useful for testing or manual intervention
- Orders appear in normal order flow

### File Structure

```
dashboard/
├── index.html
├── package.json
├── vite.config.ts
├── tailwind.config.js
│
├── src/
│   ├── main.ts
│   ├── App.vue
│   │
│   ├── components/
│   │   ├── PortfolioSummary.vue      # Gold, total value, holdings count
│   │   ├── GESlotsPanel.vue          # 8 GE slots with status
│   │   ├── OrdersPanel.vue           # Order list with filters
│   │   ├── InventoryPanel.vue        # Inventory grid
│   │   ├── BankPanel.vue             # Bank table
│   │   ├── DebugPanel.vue            # Manual actions
│   │   └── RawDataViewer.vue         # JSON viewer for documents
│   │
│   ├── composables/
│   │   ├── useFirestore.ts           # Firestore connection
│   │   ├── usePortfolio.ts           # Portfolio state
│   │   ├── useOrders.ts              # Orders state + actions
│   │   ├── useInventory.ts           # Inventory state + actions
│   │   ├── useBank.ts                # Bank state + actions
│   │   └── useGEState.ts             # GE slots state
│   │
│   ├── stores/
│   │   └── appStore.ts               # Pinia store for global state
│   │
│   ├── types/
│   │   ├── order.ts                  # Order type definitions
│   │   ├── portfolio.ts              # Portfolio types
│   │   └── geState.ts                # GE state types
│   │
│   └── utils/
│       ├── formatters.ts             # Gold formatting (M/K), time ago
│       └── firebase.ts               # Firebase initialization
│
└── firebase.json                     # Firebase hosting config (optional)
```

### Core Composables

```typescript
// composables/useOrders.ts
import { ref, onMounted, onUnmounted } from 'vue'
import { collection, query, onSnapshot, doc, updateDoc, deleteDoc, addDoc, serverTimestamp } from 'firebase/firestore'
import { db } from '@/utils/firebase'

export function useOrders(accountId: string) {
  const orders = ref<Order[]>([])
  const loading = ref(true)
  let unsubscribe: () => void

  onMounted(() => {
    const ordersRef = collection(db, 'accounts', accountId, 'orders')
    const q = query(ordersRef)

    unsubscribe = onSnapshot(q, (snapshot) => {
      orders.value = snapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      } as Order))
      loading.value = false
    })
  })

  onUnmounted(() => unsubscribe?.())

  // Actions
  async function cancelOrder(orderId: string) {
    const orderRef = doc(db, 'accounts', accountId, 'orders', orderId)
    await updateDoc(orderRef, { status: 'cancelled' })
  }

  async function deleteOrder(orderId: string) {
    const orderRef = doc(db, 'accounts', accountId, 'orders', orderId)
    await deleteDoc(orderRef)
  }

  async function clearCompleted() {
    const completed = orders.value.filter(o => o.status === 'completed')
    await Promise.all(completed.map(o => deleteOrder(o.order_id)))
  }

  async function createManualOrder(
    action: 'buy' | 'sell',
    itemId: number,
    itemName: string,
    quantity: number,
    price: number
  ) {
    const ordersRef = collection(db, 'accounts', accountId, 'orders')
    const orderId = `ord_manual_${Date.now()}`

    await addDoc(ordersRef, {
      order_id: orderId,
      action,
      item_id: itemId,
      item_name: itemName,
      quantity,
      price,
      status: 'pending',
      ge_slot: null,
      filled_quantity: 0,
      total_cost: 0,
      created_at: serverTimestamp(),
      confidence: 1.0,
      strategy: 'manual'
    })
  }

  return {
    orders,
    loading,
    cancelOrder,
    deleteOrder,
    clearCompleted,
    createManualOrder
  }
}
```

```typescript
// composables/useInventory.ts
import { ref, onMounted, onUnmounted } from 'vue'
import { doc, onSnapshot, updateDoc, deleteField } from 'firebase/firestore'
import { db } from '@/utils/firebase'

export function useInventory(accountId: string) {
  const inventory = ref<InventoryState | null>(null)
  const loading = ref(true)
  let unsubscribe: () => void

  onMounted(() => {
    const inventoryRef = doc(db, 'accounts', accountId, 'inventory')

    unsubscribe = onSnapshot(inventoryRef, (snapshot) => {
      inventory.value = snapshot.data() as InventoryState
      loading.value = false
    })
  })

  onUnmounted(() => unsubscribe?.())

  // Remove item from Firestore inventory document (debugging)
  async function removeItemFromFirestore(itemId: string) {
    const inventoryRef = doc(db, 'accounts', accountId, 'inventory')
    await updateDoc(inventoryRef, {
      [`items.${itemId}`]: deleteField()
    })
  }

  return {
    inventory,
    loading,
    removeItemFromFirestore
  }
}
```

### Dashboard Commands to Plugin

For debugging actions that need plugin execution (like actual inventory manipulation), we can add an optional `commands` subcollection:

```typescript
// accounts/{accountId}/commands/{commandId}
{
  command_id: "cmd_abc123",
  type: "withdraw" | "deposit" | "deposit_all" | "force_sync",
  item_id: 2,            // optional
  item_name: "Cannonball", // optional
  quantity: 1000,        // optional, -1 for all
  status: "pending",     // pending | received | completed | failed
  created_at: Timestamp,
  completed_at: Timestamp
}
```

The plugin can optionally listen to this collection and execute commands.

### Deployment Options

1. **Local Development**: `npm run dev` - runs on localhost:5173
2. **Firebase Hosting**: Deploy to `ppoflipperopus.web.app`
3. **GitHub Pages**: Static hosting

---

## Summary

The V2 architecture:

- **Reduces Firestore complexity** from 7+ collections to 4 documents + 1 subcollection
- **Eliminates redundant tracking** (trades, positions, commands)
- **Uses event-driven sync** instead of polling/debouncing
- **Simplifies order lifecycle** to a linear state machine
- **Enables discrepancy detection** through bank/inventory/GE scanning
- **Maintains clear separation** between Plugin (execution) and Inference (decision)

The plugin becomes the **single source of truth** for game state, while inference focuses purely on **decision making** based on that state.
