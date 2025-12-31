#!/usr/bin/env python3
"""
Sell Inventory Script

Creates sell orders for all tradeable items currently in inventory.
This is useful when items have been collected from buy orders and need to be sold.

Usage:
    python scripts/sell_inventory.py [--account-id ACCOUNT_ID] [--dry-run]
"""

import sys
import os
import argparse
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.firebase_config import get_service_account_path
from firebase.inference_bridge import InferenceBridge

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Sell Inventory Script")
    parser.add_argument(
        "--account-id",
        default="b1llstar",
        help="Account ID to use (default: b1llstar)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be sold without creating orders"
    )
    args = parser.parse_args()

    try:
        # Initialize the inference bridge
        bridge = InferenceBridge(
            service_account_path=get_service_account_path(),
            account_id=args.account_id
        )
        bridge.start()

        # Get current inventory
        inventory = bridge.portfolio_tracker.get_inventory()
        if not inventory:
            logger.error("No inventory data available. Is the plugin syncing?")
            return 1

        items = inventory.get("items", {})
        if not items:
            logger.info("No items in inventory to sell.")
            return 0

        logger.info(f"\n{'='*60}")
        logger.info("INVENTORY ITEMS")
        logger.info(f"{'='*60}")

        for item_id_str, item_data in items.items():
            item_id = int(item_id_str)
            item_name = item_data.get("item_name", f"Item #{item_id}")
            quantity = item_data.get("quantity", 0)
            price_each = item_data.get("price_each", 0)
            total_value = quantity * price_each

            logger.info(f"  {item_name}: {quantity:,}x @ {price_each:,} = {total_value:,} GP")

        logger.info(f"{'='*60}\n")

        if args.dry_run:
            logger.info("DRY RUN - No orders will be created.")
            logger.info("Remove --dry-run to actually create sell orders.")
            return 0

        # Check available GE slots
        available_slots = bridge.get_available_slots()
        logger.info(f"Available GE slots: {available_slots}")

        if available_slots <= 0:
            logger.error("No GE slots available for sell orders!")
            return 1

        # Confirm with user
        print(f"\nAbout to create sell orders for {len(items)} items.")
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm != 'y':
            logger.info("Aborted by user.")
            return 0

        # Create sell orders for all inventory items
        logger.info("\nCreating sell orders...")
        order_ids = bridge.create_sell_orders_from_inventory()

        if order_ids:
            logger.info(f"\nCreated {len(order_ids)} sell orders:")
            for oid in order_ids:
                logger.info(f"  - {oid}")
        else:
            logger.warning("No sell orders were created.")

        return 0

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1

    finally:
        if 'bridge' in locals():
            bridge.shutdown()


if __name__ == "__main__":
    sys.exit(main())
