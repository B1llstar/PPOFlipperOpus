#!/usr/bin/env python3
"""Create a test buy order in Firestore."""

import sys
import uuid
from datetime import datetime

sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from google.cloud import firestore
from google.oauth2 import service_account
from config.firebase_config import SERVICE_ACCOUNT_PATH, PROJECT_ID, DEFAULT_ACCOUNT_ID


def create_test_order(
    item_name: str = "Fire rune",
    quantity: int = 100,
    price: int = 5,
    action: str = "buy",
    account_id: str = DEFAULT_ACCOUNT_ID
):
    """Create a test order in Firestore."""
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_PATH)
    db = firestore.Client(project=PROJECT_ID, credentials=credentials)

    order_id = f"test_{uuid.uuid4().hex[:12]}"
    now = datetime.utcnow().isoformat() + "Z"

    order_data = {
        "order_id": order_id,
        "action": action,
        "item_name": item_name,
        "item_id": -1,
        "quantity": quantity,
        "price": price,
        "status": "pending",
        "created_at": now,
        "updated_at": now,
        "filled_quantity": 0,
        "metadata": {"source": "test_script"}
    }

    orders_ref = db.collection("accounts").document(account_id).collection("orders")
    orders_ref.document(order_id).set(order_data)

    print(f"Created test order: {order_id}")
    print(f"  Action: {action}")
    print(f"  Item: {item_name}")
    print(f"  Quantity: {quantity}")
    print(f"  Price: {price}")
    print(f"  Status: pending")

    return order_id


if __name__ == "__main__":
    # Default: buy 100 fire runes at 5gp each
    item = sys.argv[1] if len(sys.argv) > 1 else "Fire rune"
    qty = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    prc = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    act = sys.argv[4] if len(sys.argv) > 4 else "buy"

    create_test_order(item, qty, prc, act)
