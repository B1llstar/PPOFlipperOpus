#!/usr/bin/env python3
"""
Wipe Firebase Database Clean

This script removes all trading data from Firebase Firestore:
- Orders (pending, active, completed, cancelled)
- Holdings/Inventory
- Portfolio state
- Trade history
- Heartbeats and status

Use this for a fresh start or to reset after testing.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timezone
import firebase_admin
from firebase_admin import credentials, firestore

from config.firebase_config import SERVICE_ACCOUNT_PATH, DEFAULT_ACCOUNT_ID


def get_firestore_client():
    """Initialize and return Firestore client."""
    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
        firebase_admin.initialize_app(cred)
    return firestore.client()


def delete_collection(db, collection_path: str, batch_size: int = 100) -> int:
    """Delete all documents in a collection. Returns count of deleted docs."""
    collection_ref = db.collection(collection_path)
    deleted = 0

    while True:
        docs = collection_ref.limit(batch_size).stream()
        docs_list = list(docs)

        if not docs_list:
            break

        batch = db.batch()
        for doc in docs_list:
            batch.delete(doc.reference)
            deleted += 1
        batch.commit()

        print(f"  Deleted {deleted} documents from {collection_path}...")

    return deleted


def delete_subcollections(db, doc_ref, batch_size: int = 100) -> int:
    """Delete all subcollections of a document."""
    deleted = 0

    # Known subcollections
    subcollections = ['orders', 'holdings', 'trades', 'history']

    for subcol_name in subcollections:
        subcol_ref = doc_ref.collection(subcol_name)
        docs = list(subcol_ref.limit(1).stream())

        if docs:
            count = delete_collection(db, f"{doc_ref.path}/{subcol_name}", batch_size)
            deleted += count
            if count > 0:
                print(f"  Cleared {subcol_name}: {count} documents")

    return deleted


def wipe_account_data(db, account_id: str) -> dict:
    """Wipe all data for a specific account."""
    print(f"\nWiping data for account: {account_id}")
    print("-" * 50)

    stats = {
        "orders": 0,
        "holdings": 0,
        "trades": 0,
        "other": 0
    }

    account_ref = db.collection("accounts").document(account_id)

    # Check if account exists
    account_doc = account_ref.get()
    if not account_doc.exists:
        print(f"  Account '{account_id}' does not exist in database")
        return stats

    # Delete orders subcollection
    orders_ref = account_ref.collection("orders")
    orders = list(orders_ref.stream())
    if orders:
        batch = db.batch()
        for order in orders:
            batch.delete(order.reference)
            stats["orders"] += 1
        batch.commit()
        print(f"  Deleted {stats['orders']} orders")

    # Delete holdings subcollection
    holdings_ref = account_ref.collection("holdings")
    holdings = list(holdings_ref.stream())
    if holdings:
        batch = db.batch()
        for holding in holdings:
            batch.delete(holding.reference)
            stats["holdings"] += 1
        batch.commit()
        print(f"  Deleted {stats['holdings']} holdings")

    # Delete trades subcollection
    trades_ref = account_ref.collection("trades")
    trades = list(trades_ref.stream())
    if trades:
        batch = db.batch()
        for trade in trades:
            batch.delete(trade.reference)
            stats["trades"] += 1
        batch.commit()
        print(f"  Deleted {stats['trades']} trades")

    # Reset the account document to clean state
    account_ref.set({
        "account_id": account_id,
        "gold": 0,
        "portfolio_value": 0,
        "total_profit": 0,
        "holdings_count": 0,
        "active_orders": 0,
        "status": "reset",
        "last_reset": datetime.now(timezone.utc).isoformat(),
        "reset_reason": "Manual database wipe",
        "inference_online": False,
        "plugin_online": False
    })
    print(f"  Reset account document to clean state")

    return stats


def wipe_global_collections(db) -> dict:
    """Wipe global collections that aren't account-specific."""
    print("\nWiping global collections...")
    print("-" * 50)

    stats = {}

    # Collections to wipe
    global_collections = [
        "pending_orders",
        "active_orders",
        "completed_orders",
        "cancelled_orders",
        "trade_history",
        "market_data_cache"
    ]

    for collection_name in global_collections:
        try:
            count = delete_collection(db, collection_name)
            stats[collection_name] = count
            if count > 0:
                print(f"  {collection_name}: {count} documents deleted")
        except Exception as e:
            print(f"  {collection_name}: Error - {e}")
            stats[collection_name] = 0

    return stats


def list_all_accounts(db) -> list:
    """List all account IDs in the database."""
    accounts_ref = db.collection("accounts")
    accounts = list(accounts_ref.stream())
    return [acc.id for acc in accounts]


def main():
    print("=" * 60)
    print("Firebase Database Wipe Tool")
    print("=" * 60)
    print(f"\nService Account: {SERVICE_ACCOUNT_PATH}")
    print(f"Default Account: {DEFAULT_ACCOUNT_ID}")

    # Initialize Firestore
    print("\nConnecting to Firestore...")
    try:
        db = get_firestore_client()
        print("Connected successfully!")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return 1

    # List all accounts
    accounts = list_all_accounts(db)
    print(f"\nFound {len(accounts)} account(s) in database:")
    for acc in accounts:
        print(f"  - {acc}")

    # Confirmation
    print("\n" + "!" * 60)
    print("WARNING: This will DELETE ALL trading data!")
    print("  - All orders (pending, active, completed)")
    print("  - All holdings/inventory")
    print("  - All trade history")
    print("  - All portfolio state")
    print("!" * 60)

    confirm = input("\nType 'WIPE' to confirm: ")
    if confirm != "WIPE":
        print("Aborted.")
        return 0

    # Wipe each account
    total_stats = {
        "orders": 0,
        "holdings": 0,
        "trades": 0,
        "other": 0
    }

    for account_id in accounts:
        stats = wipe_account_data(db, account_id)
        for key in total_stats:
            total_stats[key] += stats.get(key, 0)

    # Also wipe the default account if not in list
    if DEFAULT_ACCOUNT_ID not in accounts:
        stats = wipe_account_data(db, DEFAULT_ACCOUNT_ID)
        for key in total_stats:
            total_stats[key] += stats.get(key, 0)

    # Wipe global collections
    global_stats = wipe_global_collections(db)

    # Summary
    print("\n" + "=" * 60)
    print("WIPE COMPLETE")
    print("=" * 60)
    print(f"  Orders deleted: {total_stats['orders']}")
    print(f"  Holdings deleted: {total_stats['holdings']}")
    print(f"  Trades deleted: {total_stats['trades']}")
    print(f"  Global collections: {sum(global_stats.values())} documents")
    print("\nDatabase is now clean. Ready for fresh start!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
