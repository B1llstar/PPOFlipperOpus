#!/usr/bin/env python3
"""
Set Gold Balance in Firebase

Quick script to manually set the gold balance for an account.
Use this when the RuneLite plugin isn't properly syncing gold.

Usage:
    python scripts/set_gold.py 1000000     # Set to 1M gold
    python scripts/set_gold.py 500000      # Set to 500k gold
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


def set_gold(amount: int, account_id: str = None):
    """Set the gold balance for an account."""
    account_id = account_id or DEFAULT_ACCOUNT_ID

    print(f"Setting gold balance for account: {account_id}")
    print(f"Amount: {amount:,} gp")

    db = get_firestore_client()

    # Update account document
    account_ref = db.collection("accounts").document(account_id)
    account_ref.set({
        "gold": amount,
        "current_gold": amount,
        "last_gold_update": datetime.now(timezone.utc).isoformat(),
        "gold_update_source": "manual_script"
    }, merge=True)
    print(f"  Updated accounts/{account_id}")

    # Update portfolio document
    portfolio_ref = account_ref.collection("portfolio").document("current")
    portfolio_ref.set({
        "gold": amount,
        "last_updated": datetime.now(timezone.utc).isoformat()
    }, merge=True)
    print(f"  Updated accounts/{account_id}/portfolio/current")

    # Update inventory document
    inventory_ref = account_ref.collection("inventory").document("current")
    inventory_ref.set({
        "gold": amount,
        "last_updated": datetime.now(timezone.utc).isoformat()
    }, merge=True)
    print(f"  Updated accounts/{account_id}/inventory/current")

    print(f"\nGold balance set to {amount:,} gp across all documents!")
    print("The inference script should now see this balance.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/set_gold.py <amount> [account_id]")
        print("Example: python scripts/set_gold.py 1000000")
        return 1

    try:
        amount = int(sys.argv[1].replace(",", "").replace("_", ""))
    except ValueError:
        print(f"Invalid amount: {sys.argv[1]}")
        return 1

    account_id = sys.argv[2] if len(sys.argv) > 2 else None

    set_gold(amount, account_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
