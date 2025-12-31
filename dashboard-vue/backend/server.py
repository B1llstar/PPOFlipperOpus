#!/usr/bin/env python3
"""
Dashboard Backend API Server

Provides REST API endpoints for the Vue dashboard to fetch trading data from Firebase.
Uses the existing Firebase Admin SDK credentials.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime, timezone
import firebase_admin
from firebase_admin import credentials, firestore

from config.firebase_config import SERVICE_ACCOUNT_PATH, DEFAULT_ACCOUNT_ID

app = Flask(__name__)
CORS(app)  # Enable CORS for Vue frontend

# Initialize Firebase
db = None

def get_db():
    global db
    if db is None:
        if not firebase_admin._apps:
            cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
            firebase_admin.initialize_app(cred)
        db = firestore.client()
    return db


@app.route('/api/status')
def get_status():
    """Get overall system status."""
    try:
        db = get_db()
        account_ref = db.collection("accounts").document(DEFAULT_ACCOUNT_ID)
        account = account_ref.get()

        if account.exists:
            data = account.to_dict()

            # Check multiple possible fields for plugin online status
            plugin_online = (
                data.get("plugin_online", False) or
                data.get("pluginOnline", False) or
                data.get("online", False)
            )

            # Check heartbeat timestamp to determine if plugin is actually online
            last_heartbeat = data.get("last_heartbeat") or data.get("lastHeartbeat") or data.get("plugin_heartbeat")
            if last_heartbeat:
                try:
                    from datetime import datetime, timezone, timedelta
                    if isinstance(last_heartbeat, str):
                        # Parse ISO format
                        hb_time = datetime.fromisoformat(last_heartbeat.replace('Z', '+00:00'))
                    else:
                        # Assume it's a Firestore timestamp
                        hb_time = last_heartbeat
                    now = datetime.now(timezone.utc)
                    # Consider plugin online if heartbeat within last 2 minutes
                    if hasattr(hb_time, 'tzinfo') and (now - hb_time).total_seconds() < 120:
                        plugin_online = True
                except:
                    pass

            return jsonify({
                "success": True,
                "account_id": DEFAULT_ACCOUNT_ID,
                "gold": data.get("gold", 0),
                "current_gold": data.get("current_gold", 0),
                "portfolio_value": data.get("portfolio_value", 0),
                "total_profit": data.get("total_profit", 0),
                "inference_online": data.get("inference_online", False),
                "plugin_online": plugin_online,
                "last_heartbeat": str(last_heartbeat) if last_heartbeat else None,
                "status": data.get("status", "unknown")
            })
        else:
            return jsonify({"success": False, "error": "Account not found"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/debug')
def debug_data():
    """Debug endpoint to see raw Firebase data."""
    try:
        db = get_db()
        account_ref = db.collection("accounts").document(DEFAULT_ACCOUNT_ID)
        account = account_ref.get()

        result = {
            "account_exists": account.exists,
            "account_data": account.to_dict() if account.exists else None,
            "collections": {}
        }

        if account.exists:
            # Check subcollections
            for subcol in ["orders", "holdings", "trades", "inventory", "portfolio"]:
                try:
                    docs = list(account_ref.collection(subcol).limit(5).stream())
                    result["collections"][subcol] = {
                        "count": len(docs),
                        "sample": [d.to_dict() for d in docs[:2]]
                    }
                except:
                    result["collections"][subcol] = {"error": "failed to read"}

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/api/holdings')
def get_holdings():
    """Get all current holdings."""
    try:
        db = get_db()
        holdings_ref = db.collection("accounts").document(DEFAULT_ACCOUNT_ID).collection("holdings")
        holdings = []

        for doc in holdings_ref.stream():
            holding = doc.to_dict()
            holding["id"] = doc.id
            holdings.append(holding)

        return jsonify({
            "success": True,
            "holdings": holdings,
            "count": len(holdings)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/orders')
def get_orders():
    """Get all orders (active and recent)."""
    try:
        db = get_db()
        orders_ref = db.collection("accounts").document(DEFAULT_ACCOUNT_ID).collection("orders")
        orders = []

        for doc in orders_ref.stream():
            order = doc.to_dict()
            order["id"] = doc.id
            orders.append(order)

        # Sort by created_at descending
        orders.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return jsonify({
            "success": True,
            "orders": orders,
            "count": len(orders)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/orders/active')
def get_active_orders():
    """Get only active/pending orders."""
    try:
        db = get_db()
        orders_ref = db.collection("accounts").document(DEFAULT_ACCOUNT_ID).collection("orders")

        # Query for non-completed orders
        active_statuses = ["pending", "received", "placed", "active"]
        orders = []

        for doc in orders_ref.stream():
            order = doc.to_dict()
            if order.get("status") in active_statuses:
                order["id"] = doc.id
                orders.append(order)

        return jsonify({
            "success": True,
            "orders": orders,
            "count": len(orders)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/trades')
def get_trades():
    """Get trade history."""
    try:
        db = get_db()
        trades_ref = db.collection("accounts").document(DEFAULT_ACCOUNT_ID).collection("trades")
        trades = []

        for doc in trades_ref.stream():
            trade = doc.to_dict()
            trade["id"] = doc.id
            trades.append(trade)

        # Sort by timestamp descending
        trades.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        return jsonify({
            "success": True,
            "trades": trades[:100],  # Limit to last 100
            "count": len(trades)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/inventory')
def get_inventory():
    """Get current inventory from plugin."""
    try:
        db = get_db()
        inventory_ref = db.collection("accounts").document(DEFAULT_ACCOUNT_ID).collection("inventory").document("current")
        inventory = inventory_ref.get()

        if inventory.exists:
            data = inventory.to_dict()
            return jsonify({
                "success": True,
                "inventory": data
            })
        else:
            return jsonify({"success": True, "inventory": {}})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/portfolio')
def get_portfolio():
    """Get portfolio summary."""
    try:
        db = get_db()

        # Get account data
        account_ref = db.collection("accounts").document(DEFAULT_ACCOUNT_ID)
        account = account_ref.get()
        account_data = account.to_dict() if account.exists else {}

        # Get holdings
        holdings_ref = account_ref.collection("holdings")
        holdings = []
        total_holdings_value = 0

        for doc in holdings_ref.stream():
            holding = doc.to_dict()
            holding["id"] = doc.id
            value = holding.get("quantity", 0) * holding.get("avg_price", 0)
            holding["value"] = value
            total_holdings_value += value
            holdings.append(holding)

        # Get active orders
        orders_ref = account_ref.collection("orders")
        active_orders = 0
        pending_value = 0

        for doc in orders_ref.stream():
            order = doc.to_dict()
            if order.get("status") in ["pending", "received", "placed", "active"]:
                active_orders += 1
                pending_value += order.get("quantity", 0) * order.get("price", 0)

        gold = account_data.get("gold", 0) or account_data.get("current_gold", 0)

        return jsonify({
            "success": True,
            "portfolio": {
                "gold": gold,
                "holdings_value": total_holdings_value,
                "total_value": gold + total_holdings_value,
                "holdings_count": len(holdings),
                "active_orders": active_orders,
                "pending_value": pending_value,
                "total_profit": account_data.get("total_profit", 0),
                "holdings": holdings
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/stats')
def get_stats():
    """Get trading statistics."""
    try:
        db = get_db()

        # Get all trades for stats
        trades_ref = db.collection("accounts").document(DEFAULT_ACCOUNT_ID).collection("trades")

        total_trades = 0
        total_buys = 0
        total_sells = 0
        total_profit = 0
        total_volume = 0

        for doc in trades_ref.stream():
            trade = doc.to_dict()
            total_trades += 1

            if trade.get("type") == "buy":
                total_buys += 1
            elif trade.get("type") == "sell":
                total_sells += 1

            total_profit += trade.get("profit", 0)
            total_volume += trade.get("quantity", 0) * trade.get("price", 0)

        return jsonify({
            "success": True,
            "stats": {
                "total_trades": total_trades,
                "total_buys": total_buys,
                "total_sells": total_sells,
                "total_profit": total_profit,
                "total_volume": total_volume
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })


# =========================================================================
# Position Management Endpoints
# =========================================================================

@app.route('/api/positions')
def get_positions():
    """Get all active positions (PPO-acquired tradeable items)."""
    try:
        db = get_db()
        positions_ref = db.collection("accounts").document(DEFAULT_ACCOUNT_ID).collection("positions").document("active")
        positions_doc = positions_ref.get()

        if positions_doc.exists:
            data = positions_doc.to_dict()
            items = data.get("items", {})

            # Convert to list format
            positions = []
            for item_id, pos_data in items.items():
                pos_data["id"] = item_id
                positions.append(pos_data)

            # Sort by total_invested descending
            positions.sort(key=lambda x: x.get("total_invested", 0), reverse=True)

            return jsonify({
                "success": True,
                "positions": positions,
                "count": len(positions),
                "updated_at": data.get("updated_at")
            })
        else:
            return jsonify({
                "success": True,
                "positions": [],
                "count": 0
            })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/positions/<item_id>/lock', methods=['POST'])
def lock_position(item_id):
    """Lock a position to prevent PPO from selling it."""
    try:
        db = get_db()
        positions_ref = db.collection("accounts").document(DEFAULT_ACCOUNT_ID).collection("positions").document("active")
        positions_doc = positions_ref.get()

        if not positions_doc.exists:
            return jsonify({"success": False, "error": "No positions found"})

        data = positions_doc.to_dict()
        items = data.get("items", {})

        if item_id not in items:
            return jsonify({"success": False, "error": f"Position {item_id} not found"})

        items[item_id]["locked"] = True
        items[item_id]["last_updated"] = datetime.now(timezone.utc).isoformat()

        positions_ref.update({
            "items": items,
            "updated_at": datetime.now(timezone.utc).isoformat()
        })

        return jsonify({
            "success": True,
            "message": f"Position {item_id} locked"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/positions/<item_id>/unlock', methods=['POST'])
def unlock_position(item_id):
    """Unlock a position to allow PPO to sell it."""
    try:
        db = get_db()
        positions_ref = db.collection("accounts").document(DEFAULT_ACCOUNT_ID).collection("positions").document("active")
        positions_doc = positions_ref.get()

        if not positions_doc.exists:
            return jsonify({"success": False, "error": "No positions found"})

        data = positions_doc.to_dict()
        items = data.get("items", {})

        if item_id not in items:
            return jsonify({"success": False, "error": f"Position {item_id} not found"})

        items[item_id]["locked"] = False
        items[item_id]["last_updated"] = datetime.now(timezone.utc).isoformat()

        positions_ref.update({
            "items": items,
            "updated_at": datetime.now(timezone.utc).isoformat()
        })

        return jsonify({
            "success": True,
            "message": f"Position {item_id} unlocked"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/positions/<item_id>', methods=['DELETE'])
def remove_position(item_id):
    """Remove a position from the active positions list."""
    try:
        db = get_db()
        positions_ref = db.collection("accounts").document(DEFAULT_ACCOUNT_ID).collection("positions").document("active")
        positions_doc = positions_ref.get()

        if not positions_doc.exists:
            return jsonify({"success": False, "error": "No positions found"})

        data = positions_doc.to_dict()
        items = data.get("items", {})

        if item_id not in items:
            return jsonify({"success": False, "error": f"Position {item_id} not found"})

        item_name = items[item_id].get("item_name", f"Item {item_id}")
        del items[item_id]

        positions_ref.set({
            "items": items,
            "updated_at": datetime.now(timezone.utc).isoformat()
        })

        return jsonify({
            "success": True,
            "message": f"Position {item_name} removed"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/bank')
def get_bank():
    """Get bank contents from last sync."""
    try:
        db = get_db()
        bank_ref = db.collection("accounts").document(DEFAULT_ACCOUNT_ID).collection("bank").document("current")
        bank_doc = bank_ref.get()

        if bank_doc.exists:
            data = bank_doc.to_dict()
            items = data.get("items", {})

            # Convert to list
            bank_items = []
            for item_id, item_data in items.items():
                item_data["id"] = item_id
                bank_items.append(item_data)

            # Sort by total_value descending
            bank_items.sort(key=lambda x: x.get("total_value", 0), reverse=True)

            return jsonify({
                "success": True,
                "bank": bank_items,
                "total_value": data.get("total_value", 0),
                "item_count": data.get("item_count", len(bank_items)),
                "updated_at": data.get("updated_at")
            })
        else:
            return jsonify({
                "success": True,
                "bank": [],
                "total_value": 0,
                "item_count": 0
            })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/ge-slots')
def get_ge_slots():
    """Get GE slot states from last sync."""
    try:
        db = get_db()
        ge_ref = db.collection("accounts").document(DEFAULT_ACCOUNT_ID).collection("ge_slots").document("current")
        ge_doc = ge_ref.get()

        if ge_doc.exists:
            data = ge_doc.to_dict()
            return jsonify({
                "success": True,
                "slots": data.get("slots", {}),
                "slots_available": data.get("slots_available", 0),
                "buy_slots_used": data.get("buy_slots_used", 0),
                "sell_slots_used": data.get("sell_slots_used", 0),
                "updated_at": data.get("updated_at")
            })
        else:
            return jsonify({
                "success": True,
                "slots": {},
                "slots_available": 8
            })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


if __name__ == '__main__':
    print("=" * 60)
    print("PPO Flipper Dashboard API Server")
    print("=" * 60)
    print(f"Account: {DEFAULT_ACCOUNT_ID}")
    print(f"Firebase: {SERVICE_ACCOUNT_PATH}")
    print("=" * 60)
    print("\nStarting server on http://localhost:5001")
    print("Press Ctrl+C to stop\n")

    app.run(host='0.0.0.0', port=5001, debug=True)
