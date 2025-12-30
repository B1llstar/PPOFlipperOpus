#!/usr/bin/env python3
"""
Full Flow Test Script

This script tests the complete PPO -> Plugin integration by:
1. Buy 100 Fire Rune @ 10 GP each (auto-collect)
2. Sell 100 Fire Rune @ 2 GP each (auto-collect)
3. Buy 100 Air Rune @ 10 GP each (auto-collect)
4. Bank Deposit Air Rune x 100
5. Sell 100 Air Rune @ 2 GP each (requires withdraw from bank first)

All operations are initiated via Firestore and executed by the GEAuto plugin.
The script monitors order status and waits for completion before proceeding.

Usage:
    python scripts/test_full_flow.py [--account-id ACCOUNT_ID]
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
AIR_RUNE_ID = 556

# Test configuration
BUY_PRICE = 10
SELL_PRICE = 2
QUANTITY = 100

# Timeouts
ORDER_TIMEOUT_SECONDS = 300  # 5 minutes per order
POLL_INTERVAL_SECONDS = 2


class FullFlowTest:
    """Orchestrates the full test flow through Firestore."""

    def __init__(self, account_id: str):
        self.account_id = account_id
        self.client = FirebaseClient()
        self.order_manager = None
        self.portfolio_tracker = None

    def initialize(self) -> bool:
        """Initialize Firebase connection."""
        # Find service account file
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
        """Print current state for debugging."""
        logger.info("=" * 60)
        logger.info("CURRENT STATE")
        logger.info("=" * 60)

        # Portfolio
        gold = self.portfolio_tracker.get_gold()
        logger.info(f"Gold: {gold:,}")

        # GE Slots
        ge_slots = self.portfolio_tracker.get_ge_slots()
        if ge_slots:
            logger.info(f"GE Slots - Buy: {ge_slots.get('buy_slots_used', 0)}, "
                       f"Sell: {ge_slots.get('sell_slots_used', 0)}, "
                       f"Available: {ge_slots.get('slots_available', 0)}")

        # Inventory
        inventory = self.portfolio_tracker.get_inventory()
        if inventory:
            logger.info(f"Inventory - Items: {inventory.get('item_count', 0)}, "
                       f"Free slots: {inventory.get('free_slots', 28)}")

        # Bank (if available)
        bank = self.portfolio_tracker.get_bank()
        if bank and bank.get('scan_type') == 'full':
            logger.info(f"Bank - Items: {bank.get('item_count', 0)}, "
                       f"Tradeable: {bank.get('tradeable_count', 0)}")

        logger.info("=" * 60)

    def wait_for_order_completion(self, order_id: str, description: str) -> bool:
        """Wait for an order to complete or fail."""
        logger.info(f"Waiting for order: {description} (ID: {order_id[:12]}...)")

        start_time = time.time()
        last_status = None

        while time.time() - start_time < ORDER_TIMEOUT_SECONDS:
            order = self.order_manager.get_order(order_id)
            if not order:
                logger.error(f"Order {order_id} not found!")
                return False

            status = order.get('status')

            if status != last_status:
                logger.info(f"  Status: {status}")
                last_status = status

            if status == 'completed':
                logger.info(f"  ORDER COMPLETED!")
                return True
            elif status == 'failed':
                error = order.get('error_message', 'Unknown error')
                logger.error(f"  ORDER FAILED: {error}")
                return False
            elif status == 'cancelled':
                logger.warning(f"  ORDER CANCELLED")
                return False

            time.sleep(POLL_INTERVAL_SECONDS)

        logger.error(f"  ORDER TIMEOUT after {ORDER_TIMEOUT_SECONDS}s")
        return False

    def wait_for_command_completion(self, command_id: str, description: str) -> bool:
        """Wait for a command to complete or fail."""
        logger.info(f"Waiting for command: {description} (ID: {command_id[:12]}...)")

        start_time = time.time()
        last_status = None

        while time.time() - start_time < ORDER_TIMEOUT_SECONDS:
            doc = self.client.get_commands_ref().document(command_id).get()
            if not doc.exists:
                logger.error(f"Command {command_id} not found!")
                return False

            command = doc.to_dict()
            status = command.get('status')

            if status != last_status:
                logger.info(f"  Status: {status}")
                last_status = status

            if status == 'completed':
                logger.info(f"  COMMAND COMPLETED!")
                return True
            elif status == 'failed':
                error = command.get('error_message', 'Unknown error')
                logger.error(f"  COMMAND FAILED: {error}")
                return False

            time.sleep(POLL_INTERVAL_SECONDS)

        logger.error(f"  COMMAND TIMEOUT after {ORDER_TIMEOUT_SECONDS}s")
        return False

    def submit_buy_order(self, item_id: int, item_name: str, quantity: int, price: int) -> str:
        """Submit a buy order and return order ID."""
        logger.info(f"Submitting BUY order: {quantity}x {item_name} @ {price} GP")

        order_id = self.order_manager.create_buy_order(
            item_id=item_id,
            item_name=item_name,
            quantity=quantity,
            price=price,
            metadata={"source": "test_full_flow", "test_run": True}
        )

        if order_id:
            logger.info(f"  Created order: {order_id[:12]}...")
        else:
            logger.error("  Failed to create order!")

        return order_id

    def submit_sell_order(self, item_id: int, item_name: str, quantity: int, price: int) -> str:
        """Submit a sell order and return order ID."""
        logger.info(f"Submitting SELL order: {quantity}x {item_name} @ {price} GP")

        order_id = self.order_manager.create_sell_order(
            item_id=item_id,
            item_name=item_name,
            quantity=quantity,
            price=price,
            metadata={"source": "test_full_flow", "test_run": True}
        )

        if order_id:
            logger.info(f"  Created order: {order_id[:12]}...")
        else:
            logger.error("  Failed to create order!")

        return order_id

    def submit_deposit_command(self, item_id: int, item_name: str, quantity: int) -> str:
        """Submit a bank deposit command."""
        logger.info(f"Submitting DEPOSIT command: {quantity}x {item_name}")

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
            "metadata": {"source": "test_full_flow"}
        }

        try:
            self.client.get_commands_ref().document(command_id).set(command_data)
            logger.info(f"  Created command: {command_id[:12]}...")
            return command_id
        except Exception as e:
            logger.error(f"  Failed to create command: {e}")
            return None

    def submit_withdraw_command(self, item_id: int, item_name: str, quantity: int) -> str:
        """Submit a bank withdraw command."""
        logger.info(f"Submitting WITHDRAW command: {quantity}x {item_name}")

        import uuid
        command_id = f"cmd_{uuid.uuid4().hex[:12]}"

        command_data = {
            "command_id": command_id,
            "command_type": "withdraw",
            "status": "pending",
            "item_id": item_id,
            "item_name": item_name,
            "quantity": quantity,
            "as_note": False,
            "priority": 5,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {"source": "test_full_flow"}
        }

        try:
            self.client.get_commands_ref().document(command_id).set(command_data)
            logger.info(f"  Created command: {command_id[:12]}...")
            return command_id
        except Exception as e:
            logger.error(f"  Failed to create command: {e}")
            return None

    def run_step(self, step_num: int, description: str):
        """Print step header."""
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"STEP {step_num}: {description}")
        logger.info("=" * 60)

    def run(self) -> bool:
        """Run the full test flow."""
        logger.info("Starting Full Flow Test")
        logger.info("=" * 60)

        # Check plugin is online
        if not self.check_plugin_online():
            logger.warning("Proceeding anyway, but orders may not execute...")

        self.print_state()

        # ==========================================
        # STEP 1: Buy 100 Fire Rune @ 10 GP
        # ==========================================
        self.run_step(1, "Buy 100 Fire Rune @ 10 GP each")

        order_id = self.submit_buy_order(FIRE_RUNE_ID, "Fire rune", QUANTITY, BUY_PRICE)
        if not order_id:
            return False

        if not self.wait_for_order_completion(order_id, "Buy Fire Rune"):
            return False

        self.print_state()
        time.sleep(2)  # Brief pause between operations

        # ==========================================
        # STEP 2: Sell 100 Fire Rune @ 2 GP
        # ==========================================
        self.run_step(2, "Sell 100 Fire Rune @ 2 GP each")

        order_id = self.submit_sell_order(FIRE_RUNE_ID, "Fire rune", QUANTITY, SELL_PRICE)
        if not order_id:
            return False

        if not self.wait_for_order_completion(order_id, "Sell Fire Rune"):
            return False

        self.print_state()
        time.sleep(2)

        # ==========================================
        # STEP 3: Buy 100 Air Rune @ 10 GP
        # ==========================================
        self.run_step(3, "Buy 100 Air Rune @ 10 GP each")

        order_id = self.submit_buy_order(AIR_RUNE_ID, "Air rune", QUANTITY, BUY_PRICE)
        if not order_id:
            return False

        if not self.wait_for_order_completion(order_id, "Buy Air Rune"):
            return False

        self.print_state()
        time.sleep(2)

        # ==========================================
        # STEP 4: Bank Deposit Air Rune x 100
        # ==========================================
        self.run_step(4, "Bank Deposit 100 Air Rune")

        command_id = self.submit_deposit_command(AIR_RUNE_ID, "Air rune", QUANTITY)
        if not command_id:
            return False

        if not self.wait_for_command_completion(command_id, "Deposit Air Rune"):
            return False

        self.print_state()
        time.sleep(2)

        # ==========================================
        # STEP 5: Sell 100 Air Rune @ 2 GP (requires withdraw)
        # ==========================================
        self.run_step(5, "Sell 100 Air Rune @ 2 GP (from bank)")

        # First, check if Air Runes are in bank
        bank_qty = self.portfolio_tracker.get_bank_item_quantity(AIR_RUNE_ID)
        logger.info(f"Air Runes in bank: {bank_qty}")

        if bank_qty >= QUANTITY:
            # Need to withdraw first
            logger.info("Air Runes are in bank, submitting withdraw command first...")

            command_id = self.submit_withdraw_command(AIR_RUNE_ID, "Air rune", QUANTITY)
            if not command_id:
                return False

            if not self.wait_for_command_completion(command_id, "Withdraw Air Rune"):
                return False

            self.print_state()
            time.sleep(2)

        # Now sell
        order_id = self.submit_sell_order(AIR_RUNE_ID, "Air rune", QUANTITY, SELL_PRICE)
        if not order_id:
            return False

        if not self.wait_for_order_completion(order_id, "Sell Air Rune"):
            return False

        # ==========================================
        # FINAL STATE
        # ==========================================
        logger.info("")
        logger.info("=" * 60)
        logger.info("TEST COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        self.print_state()

        return True

    def shutdown(self):
        """Clean up resources."""
        if self.client:
            self.client.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Full Flow Test Script")
    parser.add_argument(
        "--account-id",
        default="test_account",
        help="Account ID to use (default: test_account)"
    )
    args = parser.parse_args()

    test = FullFlowTest(args.account_id)

    try:
        if not test.initialize():
            logger.error("Failed to initialize, exiting")
            sys.exit(1)

        success = test.run()
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        sys.exit(1)

    finally:
        test.shutdown()


if __name__ == "__main__":
    main()
