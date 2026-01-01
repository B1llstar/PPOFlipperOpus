# PPOFlipperOpus + GEAuto System Flow

This document details the complete end-to-end flow of the automated Grand Exchange trading system, from AI inference to trade execution.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              PPOFlipperOpus (Python)                                │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                 │
│  │   PPO Model     │───►│ InferenceBridge │───►│  OrderManager   │                 │
│  │  (AI Decision)  │    │  (Orchestrator) │    │ (Create Orders) │                 │
│  └─────────────────┘    └────────┬────────┘    └────────┬────────┘                 │
│                                  │                      │                           │
│                         ┌────────▼────────┐             │                           │
│                         │PortfolioManager │             │                           │
│                         │  (Validate)     │             │                           │
│                         └─────────────────┘             │                           │
└─────────────────────────────────────────────────────────┼───────────────────────────┘
                                                          │
                                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              FIRESTORE (Cloud)                                       │
│                                                                                      │
│   /accounts/{accountId}/orders/{orderId}     ◄─── PPO writes pending orders         │
│   /accounts/{accountId}/portfolio/{itemId}   ◄─── Plugin writes, PPO reads          │
│   /accounts/{accountId}/inventory/current    ◄─── Plugin writes                     │
│   /accounts/{accountId}/trades/{tradeId}     ◄─── Plugin writes completed trades    │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                                          │
                                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              GEAuto Plugin (Java/RuneLite)                          │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                 │
│  │FirebaseOrder    │───►│ GEQueueManager  │───►│  GEAutoPlugin   │                 │
│  │   Listener      │    │ (Queue Orders)  │    │ (Execute in GE) │                 │
│  └─────────────────┘    └─────────────────┘    └────────┬────────┘                 │
│                                                         │                           │
│                         ┌─────────────────┐    ┌────────▼────────┐                 │
│                         │PortfolioManager │◄───│FirebaseTrade    │                 │
│                         │ (Update)        │    │   Reporter      │                 │
│                         └─────────────────┘    └─────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                                          │
                                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              Old School RuneScape                                    │
│                                                                                      │
│                              Grand Exchange Interface                                │
│                              8 Trading Slots Available                               │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Inference Phase

### 1.1 Market Data Collection

The PPO model requires market data to make trading decisions:

```python
# Market data sources
- OSRS Wiki API: Real-time prices, volume, price history
- Historical trades: Past trade performance from /trades collection
- Portfolio state: Current holdings from /portfolio collection
- Available capital: Gold balance from /accounts document
```

### 1.2 PPO Model Inference

The Proximal Policy Optimization model evaluates potential trades:

```python
# inference/run_firebase_inference.py
class PPOInferenceRunner:
    def run_inference_cycle(self):
        # 1. Get current state
        portfolio = self.bridge.portfolio_tracker.ppo_portfolio.get_all_portfolio_items()
        gold = self.bridge.get_current_gold()
        market_data = self.fetch_market_data()

        # 2. Build observation tensor
        observation = self.build_observation(portfolio, gold, market_data)

        # 3. Run PPO model
        action, value, confidence = self.model.predict(observation)

        # 4. Decode action
        if action.type == ActionType.BUY:
            item_id, quantity, price = self.decode_buy_action(action)
            self.bridge.submit_buy_order(item_id, item_name, quantity, price, confidence)
        elif action.type == ActionType.SELL:
            item_id, quantity, price = self.decode_sell_action(action)
            self.bridge.submit_sell_order(item_id, item_name, quantity, price, confidence)
        elif action.type == ActionType.HOLD:
            pass  # No action this cycle
```

### 1.3 Decision Factors

The model considers:

| Factor | Weight | Description |
|--------|--------|-------------|
| Price Momentum | High | Recent price direction and velocity |
| Volume | Medium | Trading volume indicates liquidity |
| Spread | High | Buy/sell spread affects profitability |
| Portfolio Exposure | Medium | Avoid over-concentration |
| Historical P&L | Medium | Past performance on this item |
| Tax Impact | Low | 1% sell tax consideration |
| Capital Utilization | Medium | Efficient use of available gold |

---

## 2. Order Creation Phase

### 2.1 Buy Order Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         BUY ORDER CREATION                          │
└─────────────────────────────────────────────────────────────────────┘

PPO Model Decision
       │
       ▼
┌──────────────────┐
│ InferenceBridge  │
│ submit_buy_order │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     ┌─────────────────────────────────────────┐
│ Validate:        │     │ Checks:                                 │
│ - Has enough GP  │────►│ - gold >= quantity * price              │
│ - Plugin online  │     │ - plugin_online == true                 │
│ - GE slots free  │     │ - ge_slots_available > 0                │
└────────┬─────────┘     └─────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│  OrderManager    │
│ create_buy_order │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│ Write to Firestore: /accounts/{id}/orders/{orderId}             │
│                                                                  │
│ {                                                                │
│   "order_id": "ord_abc123...",                                  │
│   "action": "buy",                                              │
│   "item_id": 4151,                                              │
│   "item_name": "Abyssal whip",                                  │
│   "quantity": 10,                                               │
│   "price": 2500000,                                             │
│   "status": "pending",        ◄── Initial status                │
│   "source": "ppo",            ◄── Identifies PPO as creator     │
│   "gold_exchanged": 0,                                          │
│   "tax_paid": 0,                                                │
│   "created_at": "2025-01-15T10:30:00Z"                         │
│ }                                                                │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 Sell Order Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SELL ORDER CREATION                         │
└─────────────────────────────────────────────────────────────────────┘

PPO Model Decision
       │
       ▼
┌──────────────────┐
│ InferenceBridge  │
│submit_sell_order │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│ CRITICAL VALIDATION: Portfolio Ownership Check                   │
│                                                                  │
│ portfolio_qty = ppo_portfolio.get_portfolio_quantity(item_id)   │
│                                                                  │
│ if portfolio_qty < quantity:                                    │
│     LOG WARNING: "Cannot sell: portfolio has {portfolio_qty},   │
│                   trying to sell {quantity}"                    │
│     return None  ◄── REJECTED                                   │
└────────┬─────────────────────────────────────────────────────────┘
         │ Passed
         ▼
┌──────────────────────────────────────────────────────────────────┐
│ PHYSICAL AVAILABILITY CHECK                                      │
│                                                                  │
│ inventory_qty = get_inventory_quantity(item_id)                 │
│ bank_qty = get_bank_quantity(item_id)                           │
│ total_available = inventory_qty + bank_qty                      │
│                                                                  │
│ if total_available < quantity:                                  │
│     LOG WARNING: "Physical availability insufficient"           │
│     return None  ◄── REJECTED                                   │
└────────┬─────────────────────────────────────────────────────────┘
         │ Passed
         ▼
┌──────────────────┐
│  OrderManager    │
│create_sell_order │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│ Write to Firestore: /accounts/{id}/orders/{orderId}             │
│                                                                  │
│ {                                                                │
│   "order_id": "ord_xyz789...",                                  │
│   "action": "sell",                                             │
│   "item_id": 4151,                                              │
│   "item_name": "Abyssal whip",                                  │
│   "quantity": 5,                                                │
│   "price": 2600000,                                             │
│   "status": "pending",                                          │
│   "source": "ppo",                                              │
│   ...                                                           │
│ }                                                                │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Plugin Order Processing

### 3.1 Order Listener

The plugin continuously listens for new orders:

```java
// FirebaseOrderListener.java
public void startListening() {
    ordersCollection.addSnapshotListener((snapshots, error) -> {
        for (DocumentChange change : snapshots.getDocumentChanges()) {
            if (change.getType() == ADDED) {
                FirebaseOrder order = parseOrder(change.getDocument());

                // Only process pending orders
                if (STATUS_PENDING.equals(order.getStatus())) {
                    // Mark as received
                    markOrderReceived(order.getOrderId());

                    // Queue for execution
                    queueManager.addOrder(order);
                }
            }
        }
    });
}
```

### 3.2 Order Status Transitions

```
┌─────────────────────────────────────────────────────────────────────┐
│                      ORDER STATUS LIFECYCLE                         │
└─────────────────────────────────────────────────────────────────────┘

   PPO Creates                Plugin Receives           Plugin Places in GE
       │                            │                          │
       ▼                            ▼                          ▼
  ┌─────────┐               ┌───────────┐               ┌──────────┐
  │ PENDING │──────────────►│ RECEIVED  │──────────────►│  PLACED  │
  └─────────┘               └───────────┘               └────┬─────┘
       │                          │                          │
       │                          │                          ▼
       │                          │                    ┌──────────┐
       │                          │                    │ PARTIAL  │◄──┐
       │                          │                    └────┬─────┘   │
       │                          │                         │         │
       │                          │                         │ More    │
       │                          │                         │ fills   │
       │                          │                         └─────────┘
       │                          │                         │
       │                          │                         │ Fully filled
       │                          │                         ▼
       │                          │                   ┌───────────┐
       │                          │                   │ COMPLETED │
       │                          │                   └───────────┘
       │                          │
       │                          │ Error
       │                          ▼
       │                    ┌──────────┐
       │                    │  FAILED  │
       │                    └──────────┘
       │
       │ User/System cancels
       ▼
  ┌───────────┐
  │ CANCELLED │
  └───────────┘


Status Updates Written to Firestore:
────────────────────────────────────
PENDING → RECEIVED:
  - updated_at: now

RECEIVED → PLACED:
  - ge_slot: 1-8
  - updated_at: now

PLACED → PARTIAL:
  - filled_quantity: X
  - updated_at: now

PLACED/PARTIAL → COMPLETED:
  - filled_quantity: total
  - gold_exchanged: net GP change
  - tax_paid: 1% of gross (sells only)
  - completed_at: now
```

### 3.3 GE Queue Manager

The queue manager handles order prioritization and slot allocation:

```java
// GEQueueManager.java
public class GEQueueManager {
    private PriorityQueue<GEOrder> orderQueue;
    private Map<Integer, GEOrder> activeSlots;  // slot -> order

    public void processQueue() {
        // Find available slots
        List<Integer> freeSlots = findFreeSlots();

        // Process queued orders
        while (!orderQueue.isEmpty() && !freeSlots.isEmpty()) {
            GEOrder order = orderQueue.poll();
            int slot = freeSlots.remove(0);

            // Place in GE
            boolean success = placeOrderInSlot(order, slot);

            if (success) {
                activeSlots.put(slot, order);
                orderListener.markOrderPlaced(order.getOrderId(), slot);
            } else {
                orderListener.markOrderFailed(order.getOrderId(), "Failed to place");
            }
        }
    }

    // Priority: older orders first, then by confidence
    private int compareOrders(GEOrder a, GEOrder b) {
        int timeCompare = a.getCreatedAt().compareTo(b.getCreatedAt());
        if (timeCompare != 0) return timeCompare;

        // Higher confidence = higher priority
        return Double.compare(
            b.getMetadata().getConfidence(),
            a.getMetadata().getConfidence()
        );
    }
}
```

---

## 4. Trade Execution

### 4.1 GE Interaction State Machine

The plugin uses a state machine to interact with the GE interface:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GE AUTOMATION STATE MACHINE                      │
└─────────────────────────────────────────────────────────────────────┘

                          ┌─────────────┐
                          │    IDLE     │◄─────────────────────────┐
                          └──────┬──────┘                          │
                                 │ Order queued                    │
                                 ▼                                 │
                          ┌─────────────┐                          │
                          │ WALKING_TO  │                          │
                          │     GE      │                          │
                          └──────┬──────┘                          │
                                 │ Arrived                         │
                                 ▼                                 │
                          ┌─────────────┐                          │
                          │ OPENING_GE  │                          │
                          └──────┬──────┘                          │
                                 │ GE Open                         │
                                 ▼                                 │
              ┌──────────────────┴──────────────────┐              │
              │                                     │              │
              ▼                                     ▼              │
       ┌─────────────┐                       ┌─────────────┐       │
       │  BUY_FLOW   │                       │ SELL_FLOW   │       │
       └──────┬──────┘                       └──────┬──────┘       │
              │                                     │              │
              ▼                                     ▼              │
       ┌─────────────┐                       ┌─────────────┐       │
       │SELECT_SLOT  │                       │SELECT_SLOT  │       │
       └──────┬──────┘                       └──────┬──────┘       │
              │                                     │              │
              ▼                                     ▼              │
       ┌─────────────┐                       ┌─────────────┐       │
       │SEARCH_ITEM  │                       │SELECT_ITEM  │       │
       │  (type name)│                       │(from inv)   │       │
       └──────┬──────┘                       └──────┬──────┘       │
              │                                     │              │
              ▼                                     ▼              │
       ┌─────────────┐                       ┌─────────────┐       │
       │ SET_PRICE   │                       │ SET_PRICE   │       │
       └──────┬──────┘                       └──────┬──────┘       │
              │                                     │              │
              ▼                                     ▼              │
       ┌─────────────┐                       ┌─────────────┐       │
       │SET_QUANTITY │                       │SET_QUANTITY │       │
       └──────┬──────┘                       └──────┬──────┘       │
              │                                     │              │
              ▼                                     ▼              │
       ┌─────────────┐                       ┌─────────────┐       │
       │  CONFIRM    │                       │  CONFIRM    │       │
       └──────┬──────┘                       └──────┬──────┘       │
              │                                     │              │
              └──────────────────┬──────────────────┘              │
                                 │                                 │
                                 ▼                                 │
                          ┌─────────────┐                          │
                          │   WAITING   │                          │
                          │  FOR_FILL   │                          │
                          └──────┬──────┘                          │
                                 │ Filled                          │
                                 ▼                                 │
                          ┌─────────────┐                          │
                          │ COLLECTING  │                          │
                          └──────┬──────┘                          │
                                 │                                 │
                                 └─────────────────────────────────┘
```

### 4.2 Trade Completion

When an offer fills, the plugin:

```java
// FirebaseTradeReporter.java
private void handleOfferFilled(GEEvent event, String action) {
    GEOrder order = event.getOrder();

    int quantity = order.getQuantity();
    int pricePerItem = order.getPricePerItem();
    int totalCost = quantity * pricePerItem;
    int taxPaid = 0;

    // Calculate tax for sells (1%)
    if (ACTION_SELL.equals(action)) {
        taxPaid = (int) Math.ceil(totalCost * GE_TAX_RATE);
    }

    // Calculate gold exchanged
    // Positive = gold received (sells)
    // Negative = gold spent (buys)
    int goldExchanged = ACTION_SELL.equals(action)
            ? totalCost - taxPaid   // Net after tax
            : -totalCost;           // Spent

    // 1. Update order status
    orderListener.markOrderCompletedWithDetails(
        order.getOrderId(),
        quantity,
        goldExchanged,
        taxPaid
    );

    // 2. Update portfolio
    if (ACTION_BUY.equals(action)) {
        portfolioManager.addToPortfolio(
            order.getItemId(),
            order.getItemName(),
            quantity,
            pricePerItem,
            order.getOrderId()
        );
    } else {
        portfolioManager.removeFromPortfolio(
            order.getItemId(),
            quantity,
            pricePerItem,
            order.getOrderId(),
            taxPaid
        );
    }

    // 3. Record trade for history
    recordTrade(order, action, goldExchanged, taxPaid);
}
```

---

## 5. Portfolio Updates

### 5.1 Buy Order Completes → Add to Portfolio

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PORTFOLIO UPDATE: BUY COMPLETE                   │
└─────────────────────────────────────────────────────────────────────┘

Order Completed: BUY 10x Abyssal Whip @ 2,500,000 GP each
Total Cost: 25,000,000 GP

                    BEFORE                          AFTER
              ┌────────────────┐              ┌────────────────┐
              │ Portfolio Doc  │              │ Portfolio Doc  │
              │ item_id: 4151  │              │ item_id: 4151  │
              ├────────────────┤              ├────────────────┤
Portfolio     │ quantity: 5    │    ───►      │ quantity: 15   │  (+10)
              │ avg_cost: 2.4M │              │ avg_cost: 2.47M│  (weighted)
              │ invested: 12M  │              │ invested: 37M  │  (+25M)
              │ location: inv  │              │ location: inv  │
              └────────────────┘              └────────────────┘

Weighted Average Cost Calculation:
──────────────────────────────────
existing_qty = 5
existing_avg = 2,400,000
existing_invested = 12,000,000

new_qty = 5 + 10 = 15
new_invested = 12,000,000 + 25,000,000 = 37,000,000
new_avg = 37,000,000 / 15 = 2,466,667

Trade Record Added:
{
  "order_id": "ord_abc123",
  "action": "buy",
  "quantity": 10,
  "price": 2500000,
  "timestamp": "2025-01-15T10:35:00Z"
}
```

### 5.2 Sell Order Completes → Remove from Portfolio

```
┌─────────────────────────────────────────────────────────────────────┐
│                   PORTFOLIO UPDATE: SELL COMPLETE                   │
└─────────────────────────────────────────────────────────────────────┘

Order Completed: SELL 5x Abyssal Whip @ 2,600,000 GP each
Gross: 13,000,000 GP
Tax (1%): 130,000 GP
Net Received: 12,870,000 GP

                    BEFORE                          AFTER
              ┌────────────────┐              ┌────────────────┐
              │ Portfolio Doc  │              │ Portfolio Doc  │
              │ item_id: 4151  │              │ item_id: 4151  │
              ├────────────────┤              ├────────────────┤
Portfolio     │ quantity: 15   │    ───►      │ quantity: 10   │  (-5)
              │ avg_cost: 2.47M│              │ avg_cost: 2.47M│  (unchanged)
              │ invested: 37M  │              │ invested: 24.7M│  (proportional)
              └────────────────┘              └────────────────┘

Cost Basis Reduction:
─────────────────────
cost_basis_sold = total_invested * (qty_sold / qty_before)
cost_basis_sold = 37,000,000 * (5 / 15) = 12,333,333

new_invested = 37,000,000 - 12,333,333 = 24,666,667

Profit Calculation:
───────────────────
Proceeds (net of tax): 12,870,000 GP
Cost Basis: 12,333,333 GP
Profit: 536,667 GP (+4.4%)

Trade Record Added:
{
  "order_id": "ord_xyz789",
  "action": "sell",
  "quantity": 5,
  "price": 2600000,
  "tax_paid": 130000,
  "timestamp": "2025-01-15T11:00:00Z"
}
```

### 5.3 Position Fully Closed

When all units of an item are sold:

```
Portfolio quantity reaches 0
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ portfolioManager.removeFromPortfolio() detects new_qty <= 0        │
│                                                                     │
│ Action: DELETE document from /portfolio/{itemId}                   │
│                                                                     │
│ This item is no longer in the PPO portfolio.                       │
│ Future sell orders for this item will be REJECTED.                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. Continuous Monitoring

### 6.1 PPO Side Monitoring

```python
# InferenceBridge monitors:
class InferenceBridge:
    def monitor_loop(self):
        while self.running:
            # Check for order status updates
            self._check_order_updates()

            # Refresh portfolio state
            self.portfolio_tracker.ppo_portfolio.refresh_cache()

            # Check plugin health
            if not self.portfolio_tracker.is_plugin_online():
                log.warning("Plugin offline - pausing orders")

            # Trigger callbacks for completed orders
            self._notify_order_callbacks()

            time.sleep(1)
```

### 6.2 Plugin Side Heartbeat

```java
// Plugin sends heartbeat every 30 seconds
public void sendHeartbeat() {
    Map<String, Object> updates = new HashMap<>();
    updates.put("heartbeat", Instant.now().toString());
    updates.put("plugin_online", true);
    updates.put("ge_slots_available", countFreeSlots());
    updates.put("queue_size", queueManager.getQueueSize());

    firebaseManager.updateDocument(accountDoc, updates);
}
```

---

## 7. Error Handling

### 7.1 Order Failures

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ERROR SCENARIOS                             │
└─────────────────────────────────────────────────────────────────────┘

Scenario: Not enough GP for buy order
───────────────────────────────────────
Plugin detects: player.getGold() < order.getTotalCost()
Action: markOrderFailed(orderId, "Insufficient gold")
Status: FAILED
PPO receives callback, logs error, skips this trade

Scenario: Item not in inventory for sell
─────────────────────────────────────────
Plugin detects: !inventory.contains(itemId)
Action: Check bank, if not there → markOrderFailed
Status: FAILED

Scenario: All GE slots occupied
────────────────────────────────
Plugin detects: no free slots
Action: Order stays in queue (status: RECEIVED)
When slot frees: automatically process queued order

Scenario: GE interface closes unexpectedly
───────────────────────────────────────────
Plugin detects: GE widget no longer visible
Action: Re-navigate to GE, resume from current state
Order status: unchanged (still PLACED or RECEIVED)

Scenario: Network disconnection
────────────────────────────────
Plugin: Firebase SDK handles retry automatically
PPO: Detects stale heartbeat → pauses new orders
Recovery: Automatic when connection restored
```

### 7.2 Recovery Flow

```
Plugin Restart Recovery:
────────────────────────

1. Load all orders with status != COMPLETED|FAILED|CANCELLED
2. Check current GE slot states
3. For each active order:
   - If in GE slot: continue monitoring
   - If not placed: re-queue for execution
4. Verify portfolio matches inventory + bank
5. Resume normal operation
```

---

## 8. Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                    COMPLETE DATA FLOW CYCLE                         │
└─────────────────────────────────────────────────────────────────────┘

Time ─────────────────────────────────────────────────────────────────►

1. INFERENCE (Python)
   │
   ├─ Fetch market data from OSRS Wiki API
   ├─ Load current portfolio from Firestore
   ├─ Run PPO model inference
   └─ Decision: BUY 10x Abyssal Whip @ 2.5M
         │
         ▼
2. ORDER CREATION (Python → Firestore)
   │
   ├─ Validate: enough gold, plugin online
   ├─ Write order to /orders/{orderId}
   └─ Status: PENDING
         │
         ▼
3. ORDER RECEIVED (Firestore → Java)
   │
   ├─ Listener detects new order
   ├─ Update status: RECEIVED
   └─ Add to queue
         │
         ▼
4. ORDER PLACED (Java → OSRS)
   │
   ├─ Find free GE slot
   ├─ Execute buy in GE interface
   ├─ Update status: PLACED, ge_slot: 3
   └─ Wait for fill
         │
         ▼
5. ORDER FILLED (OSRS → Java)
   │
   ├─ GE notifies: offer complete
   ├─ Collect items
   └─ Calculate: gold_exchanged = -25M
         │
         ▼
6. PORTFOLIO UPDATE (Java → Firestore)
   │
   ├─ Update /orders: COMPLETED, gold_exchanged, tax_paid
   ├─ Update /portfolio: +10 qty, new avg_cost
   ├─ Record /trades: trade history entry
   └─ Sync /inventory: new items added
         │
         ▼
7. PPO OBSERVES (Firestore → Python)
   │
   ├─ Order status callback: COMPLETED
   ├─ Portfolio cache refreshed
   └─ Ready for next inference cycle

CYCLE COMPLETE ─────────────────────────────────────────────────────►
```

---

## 9. Key Constraints & Rules

### 9.1 Portfolio Ownership Rule

```
╔═══════════════════════════════════════════════════════════════════╗
║  PPO CAN ONLY SELL ITEMS THAT EXIST IN THE PPO PORTFOLIO          ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  The portfolio tracks items acquired through PPO buy orders.      ║
║  This prevents PPO from selling the player's personal items.      ║
║                                                                   ║
║  Before any sell order:                                           ║
║  1. Check portfolio.quantity >= sell_quantity                     ║
║  2. If not, REJECT the sell order immediately                     ║
║  3. Also verify physical availability (inventory + bank)          ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
```

### 9.2 Tax Calculation

```
╔═══════════════════════════════════════════════════════════════════╗
║  GE TAX: 1% ON ALL SELL ORDERS                                    ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  Sell 10 items @ 1,000,000 GP each                               ║
║  ─────────────────────────────────                               ║
║  Gross proceeds:    10,000,000 GP                                ║
║  Tax (1%):            100,000 GP                                 ║
║  Net received:       9,900,000 GP                                ║
║                                                                   ║
║  tax_paid = ceil(gross * 0.01)                                   ║
║  gold_exchanged = gross - tax_paid                               ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
```

### 9.3 Source Tracking

```
╔═══════════════════════════════════════════════════════════════════╗
║  ORDER SOURCE FIELD                                               ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  source: "ppo"    → Created by PPOFlipperOpus inference          ║
║  source: "manual" → Created manually by user or plugin           ║
║                                                                   ║
║  This allows:                                                     ║
║  - Filtering trades by origin                                     ║
║  - Analyzing PPO performance separately                           ║
║  - Debugging which system created an order                        ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
```

---

## 10. Performance Considerations

| Metric | Target | Implementation |
|--------|--------|----------------|
| Order latency | < 500ms | Firebase real-time listeners |
| Portfolio cache TTL | 5 seconds | Prevents excessive reads |
| Heartbeat interval | 30 seconds | Balances freshness vs. load |
| Inference cycle | 1-5 seconds | Depends on model complexity |
| GE slot check | 1 second | Polling in game tick |

---

## 11. Future Improvements

1. **Partial Fill Handling**: Currently waits for full fill; could adjust price on stale orders
2. **Multi-Account**: Support running PPO for multiple characters
3. **Price Adjustment**: Auto-adjust prices if order not filling
4. **Order Expiry**: Cancel orders that haven't filled after X minutes
5. **Risk Limits**: Max position size, max portfolio exposure per item
6. **Real-time P&L**: Track unrealized P&L as prices change
