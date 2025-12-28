#!/usr/bin/env python3
"""Wipe all orders from Firestore."""

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from google.cloud import firestore
from google.oauth2 import service_account
from config.firebase_config import SERVICE_ACCOUNT_PATH, PROJECT_ID, DEFAULT_ACCOUNT_ID

def wipe_orders(account_id: str = DEFAULT_ACCOUNT_ID):
    """Delete all orders for an account."""
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_PATH)
    db = firestore.Client(project=PROJECT_ID, credentials=credentials)

    orders_ref = db.collection("accounts").document(account_id).collection("orders")
    docs = orders_ref.stream()

    count = 0
    for doc in docs:
        print(f"Deleting order: {doc.id}")
        doc.reference.delete()
        count += 1

    print(f"\nDeleted {count} orders")

if __name__ == "__main__":
    account_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ACCOUNT_ID
    print(f"Wiping all orders for account: {account_id}")
    wipe_orders(account_id)
