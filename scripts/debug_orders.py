#!/usr/bin/env python3
"""
Debug script to verify order creation and check pending orders in Firestore.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from firebase.firebase_client import FirebaseClient
from firebase.order_manager import OrderManager
from config.firebase_config import SERVICE_ACCOUNT_PATH, DEFAULT_ACCOUNT_ID


def main():
    print("=" * 60)
    print("Firebase Order Debug Script")
    print("=" * 60)

    # Initialize Firebase
    print(f"\n1. Initializing Firebase...")
    print(f"   Service account: {SERVICE_ACCOUNT_PATH}")
    print(f"   Account ID: {DEFAULT_ACCOUNT_ID}")

    client = FirebaseClient()
    if not client.initialize(SERVICE_ACCOUNT_PATH, DEFAULT_ACCOUNT_ID):
        print("   ERROR: Failed to initialize Firebase!")
        return 1

    print("   SUCCESS: Firebase initialized")

    # Check orders collection path
    orders_ref = client.get_orders_ref()
    print(f"\n2. Orders collection path: accounts/{DEFAULT_ACCOUNT_ID}/orders")

    # List all orders
    print("\n3. Fetching all orders...")
    try:
        docs = list(orders_ref.stream())
        print(f"   Found {len(docs)} total orders")

        for doc in docs:
            data = doc.to_dict()
            order_id = data.get("order_id", doc.id)
            status = data.get("status", "unknown")
            action = data.get("action", "?")
            item_name = data.get("item_name", "?")
            quantity = data.get("quantity", 0)
            price = data.get("price", 0)
            created_at = data.get("created_at", "?")

            print(f"\n   [{status.upper():10}] {order_id}")
            print(f"      {action} {quantity}x {item_name} @ {price:,} gp")
            print(f"      Created: {created_at}")

    except Exception as e:
        print(f"   ERROR fetching orders: {e}")
        return 1

    # Check specifically for pending orders
    print("\n4. Checking for pending orders...")
    try:
        pending = list(orders_ref.where("status", "==", "pending").stream())
        print(f"   Found {len(pending)} pending orders")

        for doc in pending:
            data = doc.to_dict()
            print(f"   - {doc.id}: {data.get('action')} {data.get('quantity')}x {data.get('item_name')}")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Create a test order
    print("\n5. Creating a test order...")
    order_mgr = OrderManager(client=client, track_positions=False)

    order_id = order_mgr.create_buy_order(
        item_id=2,  # Cannonball
        item_name="Cannonball",
        quantity=100,
        price=150,
        confidence=0.9,
        strategy="debug_test"
    )

    if order_id:
        print(f"   SUCCESS: Created order {order_id}")

        # Verify it was written
        print("\n6. Verifying order in Firestore...")
        doc = orders_ref.document(order_id).get()
        if doc.exists:
            data = doc.to_dict()
            print(f"   Document path: accounts/{DEFAULT_ACCOUNT_ID}/orders/{order_id}")
            print(f"   Document data:")
            for key, value in data.items():
                print(f"      {key}: {value}")
        else:
            print(f"   ERROR: Order document not found!")
    else:
        print("   ERROR: Failed to create order")

    print("\n" + "=" * 60)
    print("Debug complete. If the test order was created successfully,")
    print("the Java plugin should pick it up if it's listening correctly.")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
