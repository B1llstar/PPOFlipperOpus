#!/usr/bin/env python3
"""
Sync Orders Script

Use this when you need to manually sync the Firebase order state with
the actual in-game GE state. This should only be needed if the plugin
fails to update Firebase properly.

Usage:
    python scripts/sync_orders.py           # Just list active orders
    python scripts/sync_orders.py --all     # Force complete ALL active orders
    python scripts/sync_orders.py --id XXX  # Complete a specific order by ID
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import firebase_admin
from firebase_admin import credentials, firestore

from config.firebase_config import SERVICE_ACCOUNT_PATH, DEFAULT_ACCOUNT_ID
from firebase.firebase_client import FirebaseClient
from firebase.order_manager import OrderManager


def get_order_manager():
    """Initialize and return OrderManager."""
    client = FirebaseClient()
    if not client.initialize(SERVICE_ACCOUNT_PATH, DEFAULT_ACCOUNT_ID):
        print("Failed to initialize Firebase")
        sys.exit(1)
    return OrderManager(client)


def list_active_orders(om: OrderManager):
    """List all active orders."""
    orders = om.get_active_orders()

    if not orders:
        print("\nNo active orders in Firebase")
        return []

    print(f"\nActive Orders in Firebase ({len(orders)}):")
    print("-" * 80)

    for order in orders:
        order_id = order.get("order_id", "?")
        item_name = order.get("item_name", "Unknown")[:25]
        action = order.get("action", "?")
        qty = order.get("quantity", 0)
        price = order.get("price", 0)
        status = order.get("status", "?")
        created = order.get("created_at", "?")[:19] if order.get("created_at") else "?"

        print(f"  {order_id[:16]:<16}  {action.upper():4}  {qty:>6}x {item_name:<25} @ {price:>8}  [{status}]")

    print("-" * 80)
    return orders


def complete_order(om: OrderManager, order_id: str):
    """Complete a specific order."""
    if om.complete_order(order_id, metadata={"manual_sync": True}):
        print(f"Completed order: {order_id}")
        return True
    else:
        print(f"Failed to complete order: {order_id}")
        return False


def force_complete_all(om: OrderManager):
    """Force complete all active orders."""
    orders = om.get_active_orders()

    if not orders:
        print("No active orders to complete")
        return 0

    print(f"\n*** WARNING: This will mark ALL {len(orders)} active orders as completed ***")
    print("Only do this if you're sure all orders have been collected in-game.")
    confirm = input("\nType 'yes' to confirm: ")

    if confirm.lower() != 'yes':
        print("Cancelled")
        return 0

    count = om.force_complete_all_active(reason="Manual sync via script")
    print(f"\nCompleted {count} orders")
    return count


def main():
    parser = argparse.ArgumentParser(description="Sync Firebase order state")
    parser.add_argument("--all", action="store_true", help="Force complete ALL active orders")
    parser.add_argument("--id", type=str, help="Complete a specific order by ID")
    args = parser.parse_args()

    print("=" * 60)
    print("Order Sync Tool")
    print("=" * 60)
    print(f"Account: {DEFAULT_ACCOUNT_ID}")

    om = get_order_manager()

    if args.id:
        complete_order(om, args.id)
    elif args.all:
        list_active_orders(om)
        force_complete_all(om)
        print("\nAfter sync:")
        list_active_orders(om)
    else:
        orders = list_active_orders(om)
        if orders:
            print("\nTo complete all orders: python scripts/sync_orders.py --all")
            print("To complete one order:  python scripts/sync_orders.py --id <order_id>")


if __name__ == "__main__":
    main()
