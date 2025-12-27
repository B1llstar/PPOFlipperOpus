# Firebase Integration Proposal: GE Auto Plugin ↔ PPO Flipper

## Overview

This proposal outlines a Firebase-based infrastructure to enable real-time communication between:
- **GE Auto Plugin** (RuneLite) - Executes trades in-game
- **PPO Flipper Inference** (Python) - AI decision-making engine

## Architecture

```
┌─────────────────────────┐         ┌──────────────────────────┐
│   RuneLite Client       │         │   PPO Inference Server   │
│   (GE Auto Plugin)      │         │   (Python)               │
│                         │         │                          │
│   - Executes trades     │◄───────►│   - AI decision making   │
│   - Reports status      │         │   - Market analysis      │
│   - Receives orders     │         │   - Strategy planning    │
└───────────┬─────────────┘         └────────────┬─────────────┘
            │                                    │
            │         Firebase Cloud             │
            │    ┌──────────────────────┐       │
            └───►│   Firestore DB       │◄──────┘
                 │   - Real-time sync   │
                 │   - Orders queue     │
                 │   - Trade history    │
                 │   - Portfolio state  │
                 └──────────────────────┘
                 ┌──────────────────────┐
                 │   Firebase Hosting   │
                 │   - Dashboard UI     │
                 │   - Monitoring       │
                 └──────────────────────┘
```

## Why Firebase?

### Advantages
1. **Real-time synchronization** - Firestore provides instant updates to both clients
2. **Offline support** - Plugin can queue operations when disconnected
3. **Scalability** - Handles multiple accounts/bots easily
4. **Security** - Built-in authentication and security rules
5. **Low latency** - Global CDN with edge locations
6. **Free tier** - Sufficient for small-scale operations
7. **SDK support** - Native Java (RuneLite) and Python support

### Alternative Considered
- **MongoDB Atlas + WebSocket**: More control, but requires managing infrastructure
- **Redis + REST API**: Fast but no built-in real-time sync
- **Direct HTTP polling**: Simple but inefficient and slower

## Firestore Data Model

### Collections Structure

#### 1. `/accounts/{accountId}`
Account-level configuration and state
```json
{
  "username": "player123",
  "status": "active|paused|stopped",
  "current_gold": 15000000,
  "ge_slots_available": 6,
  "last_heartbeat": "2025-12-27T10:30:00Z",
  "settings": {
    "max_risk_per_flip": 500000,
    "min_margin_percent": 5.0,
    "tax_aware": true
  }
}
```

#### 2. `/accounts/{accountId}/orders/{orderId}`
Trading orders (commands from AI to plugin)
```json
{
  "order_id": "ord_123456",
  "item_id": 2,
  "item_name": "Cannonball",
  "action": "buy|sell",
  "quantity": 5000,
  "price": 245,
  "status": "pending|placed|completed|cancelled|failed",
  "created_at": "2025-12-27T10:30:00Z",
  "updated_at": "2025-12-27T10:30:15Z",
  "ge_slot": 3,
  "error_message": null,
  "metadata": {
    "strategy": "momentum_flip",
    "expected_profit": 25000,
    "confidence_score": 0.87
  }
}
```

#### 3. `/accounts/{accountId}/trades/{tradeId}`
Completed trade history (reported by plugin)
```json
{
  "trade_id": "trade_789",
  "order_id": "ord_123456",
  "item_id": 2,
  "action": "buy|sell",
  "quantity": 5000,
  "price": 245,
  "total_cost": 1225000,
  "tax_paid": 12250,
  "completed_at": "2025-12-27T10:35:00Z",
  "time_to_complete_seconds": 180
}
```

#### 4. `/accounts/{accountId}/portfolio`
Current holdings snapshot
```json
{
  "updated_at": "2025-12-27T10:30:00Z",
  "gold": 15000000,
  "items": {
    "2": {
      "item_id": 2,
      "item_name": "Cannonball",
      "quantity": 5000,
      "avg_buy_price": 245,
      "current_value": 1250000
    }
  },
  "total_value": 16250000,
  "unrealized_profit": 25000
}
```

#### 5. `/market_data/items/{itemId}`
Market data cache (shared across accounts)
```json
{
  "item_id": 2,
  "item_name": "Cannonball",
  "current_buy_price": 245,
  "current_sell_price": 250,
  "spread": 5,
  "spread_percent": 2.04,
  "volume_24h": 2500000,
  "price_trend": "rising|falling|stable",
  "updated_at": "2025-12-27T10:30:00Z",
  "volatility_score": 0.15
}
```

#### 6. `/inference_queue/{requestId}`
AI inference requests (plugin → AI)
```json
{
  "request_id": "req_456",
  "account_id": "acc_123",
  "request_type": "get_recommendations|evaluate_portfolio",
  "payload": {
    "current_gold": 15000000,
    "current_portfolio": {...},
    "available_slots": 6
  },
  "status": "pending|processing|completed",
  "created_at": "2025-12-27T10:30:00Z",
  "response": null
}
```

## Data Flow

### 1. AI → Plugin (Trade Orders)
```
PPO Inference generates recommendation
    ↓
Create document in /accounts/{id}/orders
    ↓
Firestore real-time listener triggers in plugin
    ↓
Plugin receives order and executes trade
    ↓
Plugin updates order status to "placed"
    ↓
On completion, plugin updates order to "completed"
```

### 2. Plugin → AI (Trade Results)
```
Trade completes in-game
    ↓
Plugin writes to /accounts/{id}/trades
    ↓
Plugin updates /accounts/{id}/portfolio
    ↓
AI listener receives update
    ↓
AI adjusts strategy based on results
```

### 3. Market Data Sync
```
PPO collects price data from API
    ↓
Update /market_data/items/{itemId}
    ↓
Plugin subscribes to relevant items
    ↓
Plugin receives real-time price updates
```

## Security Rules

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    
    // Accounts - authenticated users can only access their own
    match /accounts/{accountId} {
      allow read, write: if request.auth != null 
                         && request.auth.uid == accountId;
      
      // Orders - writable by AI server, readable by plugin
      match /orders/{orderId} {
        allow read: if request.auth != null 
                    && request.auth.uid == accountId;
        allow write: if request.auth != null 
                     && (request.auth.uid == accountId 
                         || request.auth.token.role == 'inference_server');
      }
      
      // Trades - writable by plugin
      match /trades/{tradeId} {
        allow read, write: if request.auth != null 
                           && request.auth.uid == accountId;
      }
      
      // Portfolio - writable by plugin, readable by AI
      match /portfolio {
        allow read: if request.auth != null;
        allow write: if request.auth != null 
                     && request.auth.uid == accountId;
      }
    }
    
    // Market data - readable by all authenticated users
    match /market_data/items/{itemId} {
      allow read: if request.auth != null;
      allow write: if request.auth.token.role == 'inference_server';
    }
    
    // Inference queue
    match /inference_queue/{requestId} {
      allow read, write: if request.auth != null;
    }
  }
}
```

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)
- [ ] Set up Firebase project
- [ ] Configure Firestore database
- [ ] Implement security rules
- [ ] Create service accounts for authentication
- [ ] Set up Python Firebase Admin SDK
- [ ] Set up Java Firebase SDK in RuneLite plugin

### Phase 2: Plugin Integration (Week 2)
- [ ] Add Firebase dependencies to GE Auto plugin
- [ ] Implement order listener in plugin
- [ ] Implement trade reporting from plugin
- [ ] Add portfolio sync
- [ ] Add heartbeat mechanism
- [ ] Test order execution flow

### Phase 3: Inference Integration (Week 3)
- [ ] Create Firebase wrapper module in Python
- [ ] Implement order creation from PPO inference
- [ ] Add trade result listener
- [ ] Integrate with existing PPO training pipeline
- [ ] Add market data publishing

### Phase 4: Dashboard (Week 4)
- [ ] Create React dashboard hosted on Firebase Hosting
- [ ] Real-time portfolio view
- [ ] Trade history visualization
- [ ] Performance metrics
- [ ] Manual override controls

## Code Structure

### Python Side (PPO Flipper)
```
ppoflipperopus/
├── firebase/
│   ├── __init__.py
│   ├── firebase_client.py      # Main Firebase wrapper
│   ├── order_manager.py        # Create/manage orders
│   ├── trade_monitor.py        # Listen to trade results
│   ├── market_sync.py          # Publish market data
│   └── auth.py                 # Authentication helpers
├── inference/
│   ├── firebase_inference.py   # Inference with Firebase I/O
│   └── ...
└── dashboard/                   # Web dashboard
    ├── public/
    └── src/
```

### Java Side (GE Auto Plugin)
```
geautoplugin/
├── firebase/
│   ├── FirebaseManager.java
│   ├── OrderListener.java
│   ├── TradeReporter.java
│   └── PortfolioSync.java
├── ...
```

## Cost Estimation (Firebase Free Tier)

### Firestore Limits (Free)
- **Document reads**: 50,000/day (sufficient for ~1 read/sec)
- **Document writes**: 20,000/day (sufficient for ~0.2 writes/sec)
- **Document deletes**: 20,000/day
- **Storage**: 1 GB

### Expected Usage (Single Account)
- Orders created: ~100/day
- Trade updates: ~300/day
- Portfolio updates: ~1000/day
- Market data updates: ~200 items × 12/hour = 2400/day
- **Total writes**: ~3,800/day ✅ Within free tier

- Order reads: ~100/day
- Trade history reads: ~500/day
- Portfolio reads: ~2000/day
- Market data reads: ~200 items × 12/hour = 2400/day
- **Total reads**: ~5,000/day ✅ Within free tier

### Scaling (Paid Tier)
If exceeding free tier:
- **Additional reads**: $0.06 per 100,000
- **Additional writes**: $0.18 per 100,000
- **Storage**: $0.18/GB/month

Estimated cost for 10 accounts: ~$5-10/month

## Monitoring & Alerts

### Firebase Performance Monitoring
- Track order execution latency
- Monitor connection reliability
- Alert on failed orders

### Custom Metrics
- Average time from order → completion
- Success rate per item
- Profit per flip
- Portfolio value over time

## Risk Mitigation

### 1. Connection Loss
- Plugin queues orders locally
- Retry mechanism with exponential backoff
- Mark orders as "stale" after timeout

### 2. Order Conflicts
- Use transaction for GE slot allocation
- Implement order priority queue
- Cancel conflicting orders automatically

### 3. Data Consistency
- Validate portfolio state on reconnect
- Reconcile differences between plugin and Firestore
- Manual override capability in dashboard

### 4. Rate Limiting
- Respect Jagex rate limits (GE offers)
- Throttle order creation from AI
- Batch market data updates

## Alternative: Firebase Functions

For more complex logic, consider Firebase Cloud Functions:

```javascript
// Auto-cancel stale orders
exports.cancelStaleOrders = functions.pubsub
  .schedule('every 5 minutes')
  .onRun(async (context) => {
    const staleTime = Date.now() - 10 * 60 * 1000; // 10 min
    const staleOrders = await db.collection('accounts')
      .where('status', '==', 'pending')
      .where('created_at', '<', staleTime)
      .get();
    
    // Cancel orders...
  });
```

## Success Metrics

### Technical
- Order latency < 500ms (Firebase → Plugin)
- 99.9% order delivery success rate
- < 1% data sync conflicts
- Zero unauthorized access attempts

### Business
- 50%+ reduction in manual intervention
- Support for 5+ concurrent accounts
- Real-time portfolio visibility
- 10x faster decision-to-execution time

## Next Steps

1. **Review & Approve** this proposal
2. **Set up Firebase project** (30 min)
3. **Prototype order flow** (1 day)
4. **Test with single account** (1 week)
5. **Full rollout** (2 weeks)

---

**Estimated Total Implementation Time**: 3-4 weeks
**Upfront Cost**: $0 (free tier sufficient for testing)
**Ongoing Cost**: $0-10/month depending on scale
