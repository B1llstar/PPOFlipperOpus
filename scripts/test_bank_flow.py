#!/usr/bin/env python3
"""
Bank Flow Test Script

Tests bank integration by:
1. Buy 10 Fire Rune @ 10 GP (to get items in inventory)
2. Deposit 10 Fire Rune to bank
3. Sell 10 Fire Rune @ 2 GP (should auto-withdraw from bank)

Usage:
    python scripts/test_bank_flow.py [--account-id ACCOUNT_ID] [--bank-only]

The --bank-only flag skips the buy step and assumes you already have
Fire Runes in the bank to test deposit/withdraw/sell flow.
"""

import sys
import os
import time
import argparse
import logging
from datetime import datetime, timezone

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from firebase.firebase_client import FirebaseClient
from firebase.order_manager import OrderManager
from firebase.portfolio_tracker import PortfolioTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Item IDs
FIRE_RUNE_ID = 554

# Test configuration
BUY_PRICE = 10
SELL_PRICE = 2
QUANTITY = 10

# Timeouts
ORDER_TIMEOUT_SECONDS = 180  # 3 minutes per order
COMMAND_TIMEOUT_SECONDS = 60  # 1 minute per command
POLL_INTERVAL_SECONDS = 2


class BankFlowTest:
    """Tests the bank integration flow."""

    def __init__(self, account_id: str):
        self.account_id = account_id
        self.client = FirebaseClient()
        self.order_manager = None
        self.portfolio_tracker = None

    def initialize(self) -> bool:
        """Initialize Firebase connection."""
        service_account_paths = [
            "ppoflipperopus-firebase-adminsdk-fbsvc-0506134b11.json",
            "Firestore/service_account/ppoflipperopus-firebase-adminsdk-fbsvc-0506134b11.json",
            "../ppoflipperopus-firebase-adminsdk-fbsvc-0506134b11.json",
        ]

        service_account_path = None
        for path in service_account_paths:
            if os.path.exists(path):
                service_account_path = path
                break

        if not service_account_path:
            logger.error("Could not find service account file")
            return False

        if not self.client.initialize(service_account_path, self.account_id):
            logger.error("Failed to initialize Firebase client")
            return False

        self.order_manager = OrderManager(self.client)
        self.portfolio_tracker = PortfolioTracker(self.client)

        logger.info(f"Initialized for account: {self.account_id}")
        return True

    def check_plugin_online(self) -> bool:
        """Check if the GEAuto plugin is online."""
        if self.portfolio_tracker.is_plugin_online(max_age_seconds=60):
            logger.info("Plugin is ONLINE")
            return True
        else:
            logger.warning("Plugin appears to be OFFLINE (no recent heartbeat)")
            return False

    def print_state(self):
        """Print current state."""
        logger.info("=" * 50)

        gold = self.portfolio_tracker.get_gold()
        logger.info(f"Gold: {gold:,}")

        # Inventory
        inventory = self.portfolio_tracker.get_inventory()
        if inventory:
            fire_rune_inv = 0
            items = inventory.get("items", {})
            if str(FIRE_RUNE_ID) in items:
                fire_rune_inv = items[str(FIRE_RUNE_ID)].get("quantity", 0)
            logger.info(f"Inventory Fire Runes: {fire_rune_inv}")
            logger.info(f"Free slots: {inventory.get('free_slots', 28)}")

        # Bank
        bank = self.portfolio_tracker.get_bank()
        if bank:
            fire_rune_bank = 0
            items = bank.get("items", {})
            if str(FIRE_RUNE_ID) in items:
                fire_rune_bank = items[str(FIRE_RUNE_ID)].get("quantity", 0)
            logger.info(f"Bank Fire Runes: {fire_rune_bank}")

        logger.info("=" * 50)

    def wait_for_order(self, order_id: str, description: str) -> bool:
        """Wait for an order to complete."""
        logger.info(f"Waiting for: {description}")
        start_time = time.time()
        last_status = None

        while time.time() - start_time < ORDER_TIMEOUT_SECONDS:
            order = self.order_manager.get_order(order_id)
            if not order:
                logger.error(f"Order not found!")
                return False

            status = order.get('status')
            if status != last_status:
                logger.info(f"  Status: {status}")
                last_status = status

            if status == 'completed':
                logger.info(f"  COMPLETED!")
                return True
            elif status in ('failed', 'cancelled'):
                logger.error(f"  {status.upper()}")
                return False

            time.sleep(POLL_INTERVAL_SECONDS)

        logger.error(f"  TIMEOUT")
        return False

    def wait_for_command(self, command_id: str, description: str) -> bool:
        """Wait for a command to complete."""
        logger.info(f"Waiting for command: {description}")
        start_time = time.time()
        last_status = None

        while time.time() - start_time < COMMAND_TIMEOUT_SECONDS:
            doc = self.client.get_commands_ref().document(command_id).get()
            if not doc.exists:
                logger.error(f"Command not found!")
                return False

            command = doc.to_dict()
            status = command.get('status')

            if status != last_status:
                logger.info(f"  Status: {status}")
                last_status = status

            if status == 'completed':
                logger.info(f"  COMPLETED!")
                return True
            elif status == 'failed':
                error = command.get('error_message', 'Unknown')
                logger.error(f"  FAILED: {error}")
                return False

            time.sleep(POLL_INTERVAL_SECONDS)

        logger.error(f"  TIMEOUT")
        return False

    def submit_deposit_command(self, item_id: int, item_name: str, quantity: int) -> str:
        """Submit a bank deposit command."""
        import uuid
        command_id = f"cmd_{uuid.uuid4().hex[:12]}"

        command_data = {
            "command_id": command_id,
            "command_type": "deposit",
            "status": "pending",
            "item_id": item_id,
            "item_name": item_name,
            "quantity": quantity,
            "priority": 5,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {"source": "test_bank_flow"}
        }

        try:
            self.client.get_commands_ref().document(command_id).set(command_data)
            logger.info(f"Submitted DEPOSIT command: {quantity}x {item_name}")
            return command_id
        except Exception as e:
            logger.error(f"Failed to submit deposit: {e}")
            return None

    def run(self, bank_only: bool = False) -> bool:
        """Run the bank flow test."""
        logger.info("=" * 60)
        logger.info("BANK FLOW TEST")
        logger.info("=" * 60)

        if not self.check_plugin_online():
            logger.warning("Proceeding anyway...")

        self.print_state()

        # Step 1: Buy Fire Runes (unless bank-only mode)
        if not bank_only:
            logger.info("")
            logger.info("STEP 1: Buy 10 Fire Rune @ 10 GP")
            logger.info("-" * 40)

            order_id = self.order_manager.create_buy_order(
                item_id=FIRE_RUNE_ID,
                item_name="Fire rune",
                quantity=QUANTITY,
                price=BUY_PRICE,
                metadata={"source": "test_bank_flow"}
            )

            if not order_id:
                logger.error("Failed to create buy order")
                return False

            if not self.wait_for_order(order_id, "Buy Fire Rune"):
                return False

            self.print_state()
            time.sleep(2)
        else:
            logger.info("(Skipping buy step - bank-only mode)")

        # Step 2: Deposit Fire Runes to bank
        logger.info("")
        logger.info("STEP 2: Deposit 10 Fire Rune to bank")
        logger.info("-" * 40)

        command_id = self.submit_deposit_command(FIRE_RUNE_ID, "Fire rune", QUANTITY)
        if not command_id:
            return False

        if not self.wait_for_command(command_id, "Deposit Fire Rune"):
            return False

        self.print_state()
        time.sleep(2)

        # Step 3: Sell Fire Runes (should auto-withdraw from bank)
        logger.info("")
        logger.info("STEP 3: Sell 10 Fire Rune @ 2 GP (auto-withdraw)")
        logger.info("-" * 40)

        order_id = self.order_manager.create_sell_order(
            item_id=FIRE_RUNE_ID,
            item_name="Fire rune",
            quantity=QUANTITY,
            price=SELL_PRICE,
            metadata={"source": "test_bank_flow"}
        )

        if not order_id:
            logger.error("Failed to create sell order")
            return False

        if not self.wait_for_order(order_id, "Sell Fire Rune (with auto-withdraw)"):
            return False

        # Final state
        logger.info("")
        logger.info("=" * 60)
        logger.info("TEST COMPLETED!")
        logger.info("=" * 60)
        self.print_state()

        return True

    def shutdown(self):
        """Clean up."""
        if self.client:
            self.client.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Bank Flow Test")
    parser.add_argument(
        "--account-id",
        default="test_account",
        help="Account ID (default: test_account)"
    )
    parser.add_argument(
        "--bank-only",
        action="store_true",
        help="Skip buy step, assume items already in bank"
    )
    args = parser.parse_args()

    test = BankFlowTest(args.account_id)

    try:
        if not test.initialize():
            sys.exit(1)

        success = test.run(bank_only=args.bank_only)
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("\nInterrupted")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)

    finally:
        test.shutdown()


if __name__ == "__main__":
    main()
