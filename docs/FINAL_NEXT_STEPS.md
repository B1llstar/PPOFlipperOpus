# Final Next Steps - Firebase PPO Integration

## Current Status

The Firebase-based PPO inference system is now functional with:

- **Python Side**: `inference/run_firebase_inference.py` - Mock decision maker trading 21 magic runes
- **Java Side**: Firebase integration classes in `geauto/firebase/` package
- **Firestore**: Initialized with 1M gold budget at `accounts/default_account/portfolio/current`

## Immediate Testing Steps

### 1. Run the Python Inference Server
```bash
cd /Users/b1llstar/IdeaProjects/PPOFlipperOpus
python inference/run_firebase_inference.py
```

Expected output:
```
State synced: 1,000,000 gold, 0 items, 0 active orders, plugin_online=True
```

### 2. Compile the Java Plugin
```bash
cd /Users/b1llstar/IdeaProjects/RuneLite__Star/runelite_star_new
mvn compile -pl runelite-client -am
```

### 3. Run RuneLite with GE Auto Plugin
Launch RuneLite with the GE Auto plugin enabled. The plugin should:
- Connect to Firebase using the service account
- Listen for pending orders
- Update order status as trades execute

## Order Flow Verification

1. **Python creates order** → Firestore `accounts/{id}/orders/{order_id}` with `status: "pending"`
2. **Java plugin picks it up** → Updates status to `"received"`
3. **Plugin places on GE** → Updates status to `"placed"` with `ge_slot`
4. **Trade fills** → Updates to `"partial"` or `"completed"`
5. **Python sees completion** → Logs trade, updates portfolio

## Files Modified This Session

| File | Changes |
|------|---------|
| `inference/run_firebase_inference.py` | Created - trades 21 runes with mock prices |
| `firebase/inference_bridge.py` | Fixed `get_portfolio_summary()` → `get_state_summary()` |
| `FirebaseOrderListener.java` | Fixed to use `submitBuyOrder()`/`submitSellOrder()` |
| `FirebaseConfig.java` | Added `STATUS_RECEIVED` constant |
| `FirebaseCancelHandler.java` | Fixed `Optional<GEOrder>` handling |
| `pyproject.toml` | Added firebase-admin, google-cloud-firestore deps |

## Known Issues to Address

### Python Side
1. **Deprecation warning**: `query.where()` positional args - use `filter` keyword instead
2. **Python version**: 3.9.6 is past EOL - consider upgrading to 3.10+
3. **Real market data**: `GrandExchangeClient` code is commented out - uncomment when ready

### Java Side
1. **Service account path**: Verify `ppoflipperopus-firebase-adminsdk-*.json` is in resources
2. **GE cancel**: `cancelPlacedOrderStub()` is not implemented - manual cancel only
3. **Compile verification**: Need to verify full compile succeeds

## Runes Being Traded

| ID | Name | Mock Price Range |
|----|------|------------------|
| 554 | Fire rune | 4-5 gp |
| 555 | Water rune | 4-5 gp |
| 556 | Air rune | 4-5 gp |
| 557 | Earth rune | 4-5 gp |
| 558 | Mind rune | 3-4 gp |
| 559 | Body rune | 3-4 gp |
| 560 | Death rune | 180-200 gp |
| 561 | Nature rune | 180-200 gp |
| 562 | Chaos rune | 60-70 gp |
| 563 | Law rune | 150-170 gp |
| 564 | Cosmic rune | 120-140 gp |
| 565 | Blood rune | 350-400 gp |
| 566 | Soul rune | 300-350 gp |
| 9075 | Astral rune | 140-160 gp |
| 21880 | Wrath rune | 400-450 gp |
| 4694-4699 | Combo runes | 500-600 gp |

## Mock Decision Behavior

- **30% chance** to make a trade each decision cycle
- **50/50 split** between buy and sell (when holdings exist)
- **2% margin**: Buys at 98% of low price, sells at 102% of high price
- **Order size**: 100-1000 runes per order
- **Max spend**: 10-20% of gold per buy order

## After Testing Works

1. **Enable real market data**: Uncomment `GrandExchangeClient` in `_get_market_data()`
2. **Train actual PPO model**: Replace mock decision maker with trained agent
3. **Add more items**: Expand beyond runes to full item catalog
4. **Implement cancel**: Complete `cancelPlacedOrderStub()` in Java
5. **Add monitoring**: Dashboard for real-time trade monitoring

## Quick Commands

```bash
# Initialize/reset portfolio to 1M gold
cd /Users/b1llstar/IdeaProjects/PPOFlipperOpus
python -c "
from firebase_admin import credentials, firestore
import firebase_admin
from datetime import datetime, timezone

cred = credentials.Certificate('ppoflipperopus-firebase-adminsdk-fbsvc-25f1af05a0.json')
try: firebase_admin.get_app()
except: firebase_admin.initialize_app(cred)

db = firestore.client()
db.collection('accounts').document('default_account').collection('portfolio').document('current').set({
    'gold': 1000000, 'items': {}, 'total_value': 1000000,
    'updated_at': datetime.now(timezone.utc).isoformat()
})
print('Reset to 1M gold')
"

# Check current portfolio
python -c "
from firebase_admin import credentials, firestore
import firebase_admin

cred = credentials.Certificate('ppoflipperopus-firebase-adminsdk-fbsvc-25f1af05a0.json')
try: firebase_admin.get_app()
except: firebase_admin.initialize_app(cred)

db = firestore.client()
doc = db.collection('accounts').document('default_account').collection('portfolio').document('current').get()
print(doc.to_dict())
"

# View pending orders
python -c "
from firebase_admin import credentials, firestore
import firebase_admin

cred = credentials.Certificate('ppoflipperopus-firebase-adminsdk-fbsvc-25f1af05a0.json')
try: firebase_admin.get_app()
except: firebase_admin.initialize_app(cred)

db = firestore.client()
orders = db.collection('accounts').document('default_account').collection('orders').where('status', '==', 'pending').stream()
for o in orders: print(o.id, o.to_dict())
"
```
