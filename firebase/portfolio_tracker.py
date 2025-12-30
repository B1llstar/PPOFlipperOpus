"""
Portfolio Tracker - Track account portfolio state from the GE Auto plugin.

This module provides:
- Real-time portfolio updates (gold, inventory)
- Account state monitoring
- Holdings queries
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any, Callable

from .firebase_client import FirebaseClient

logger = logging.getLogger(__name__)


class PortfolioTracker:
    """
    Tracks the account portfolio state reported by the GE Auto plugin.

    Provides:
    - Current gold balance
    - Inventory holdings
    - GE slot availability
    - Account status
    """

    def __init__(self, client: Optional[FirebaseClient] = None):
        """Initialize with Firebase client."""
        self.client = client or FirebaseClient()
        self._portfolio_listener = None
        self._account_listener = None
        self._portfolio_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._account_callbacks: List[Callable[[Dict[str, Any]], None]] = []

        # Cached state
        self._current_portfolio: Optional[Dict[str, Any]] = None
        self._current_account: Optional[Dict[str, Any]] = None

    # =========================================================================
    # Portfolio Listening
    # =========================================================================

    def start_portfolio_listener(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Start listening for portfolio updates from the plugin.

        Args:
            callback: Function called with portfolio data on updates
        """
        if callback:
            self._portfolio_callbacks.append(callback)

        if self._portfolio_listener is not None:
            logger.warning("Portfolio listener already running")
            return

        def on_snapshot(doc_snapshot, changes, read_time):
            for doc in doc_snapshot:
                if doc.exists:
                    portfolio_data = doc.to_dict()
                    self._handle_portfolio_update(portfolio_data)

        self._portfolio_listener = self.client.get_portfolio_ref().on_snapshot(on_snapshot)
        logger.info("Started portfolio listener")

    def stop_portfolio_listener(self):
        """Stop listening for portfolio updates."""
        if self._portfolio_listener:
            self._portfolio_listener.unsubscribe()
            self._portfolio_listener = None
            logger.info("Stopped portfolio listener")

    def _handle_portfolio_update(self, portfolio_data: Dict[str, Any]):
        """Handle a portfolio update from the plugin."""
        gold = portfolio_data.get("gold", 0)
        items = portfolio_data.get("items", {})
        total_value = portfolio_data.get("total_value", 0)

        logger.debug(f"Portfolio update: {gold:,} gold, {len(items)} items, {total_value:,} total value")

        self._current_portfolio = portfolio_data

        # Notify callbacks
        for cb in self._portfolio_callbacks:
            try:
                cb(portfolio_data)
            except Exception as e:
                logger.error(f"Error in portfolio callback: {e}")

    def add_portfolio_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add a callback for portfolio updates."""
        self._portfolio_callbacks.append(callback)

    def remove_portfolio_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Remove a portfolio callback."""
        if callback in self._portfolio_callbacks:
            self._portfolio_callbacks.remove(callback)

    # =========================================================================
    # Account Listening
    # =========================================================================

    def start_account_listener(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Start listening for account state updates.

        Args:
            callback: Function called with account data on updates
        """
        if callback:
            self._account_callbacks.append(callback)

        if self._account_listener is not None:
            logger.warning("Account listener already running")
            return

        def on_snapshot(doc_snapshot, changes, read_time):
            for doc in doc_snapshot:
                if doc.exists:
                    account_data = doc.to_dict()
                    self._handle_account_update(account_data)

        self._account_listener = self.client.get_account_ref().on_snapshot(on_snapshot)
        logger.info("Started account listener")

    def stop_account_listener(self):
        """Stop listening for account updates."""
        if self._account_listener:
            self._account_listener.unsubscribe()
            self._account_listener = None
            logger.info("Stopped account listener")

    def _handle_account_update(self, account_data: Dict[str, Any]):
        """Handle an account state update from the plugin."""
        status = account_data.get("status", "unknown")
        gold = account_data.get("current_gold", 0)
        slots = account_data.get("ge_slots_available", 0)

        logger.debug(f"Account update: status={status}, gold={gold:,}, slots={slots}")

        self._current_account = account_data

        # Notify callbacks
        for cb in self._account_callbacks:
            try:
                cb(account_data)
            except Exception as e:
                logger.error(f"Error in account callback: {e}")

    def add_account_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add a callback for account updates."""
        self._account_callbacks.append(callback)

    # =========================================================================
    # Portfolio Queries
    # =========================================================================

    def get_portfolio(self) -> Optional[Dict[str, Any]]:
        """Get the current portfolio state (from cache or Firestore)."""
        if self._current_portfolio is not None:
            return self._current_portfolio

        try:
            doc = self.client.get_portfolio_ref().get()
            if doc.exists:
                self._current_portfolio = doc.to_dict()
                return self._current_portfolio
        except Exception as e:
            logger.error(f"Failed to get portfolio: {e}")
        return None

    def get_gold(self) -> int:
        """Get current gold balance."""
        portfolio = self.get_portfolio()
        if portfolio:
            return portfolio.get("gold", 0)
        return 0

    def get_holdings(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current item holdings.

        Returns:
            Dictionary mapping item_id (as string) to item data:
            {
                "item_id": int,
                "item_name": str,
                "quantity": int,
                "price_each": int (optional),
                "total_value": int (optional)
            }
        """
        portfolio = self.get_portfolio()
        if portfolio:
            return portfolio.get("items", {})
        return {}

    def get_item_quantity(self, item_id: int) -> int:
        """Get quantity of a specific item in holdings."""
        holdings = self.get_holdings()
        item_data = holdings.get(str(item_id), {})
        return item_data.get("quantity", 0)

    def has_item(self, item_id: int, min_quantity: int = 1) -> bool:
        """Check if we have at least min_quantity of an item."""
        return self.get_item_quantity(item_id) >= min_quantity

    def get_total_value(self) -> int:
        """Get total portfolio value (gold + items)."""
        portfolio = self.get_portfolio()
        if portfolio:
            return portfolio.get("total_value", 0)
        return 0

    # =========================================================================
    # Account Queries
    # =========================================================================

    def get_account(self) -> Optional[Dict[str, Any]]:
        """Get the current account state (from cache or Firestore)."""
        if self._current_account is not None:
            return self._current_account

        try:
            doc = self.client.get_account_ref().get()
            if doc.exists:
                self._current_account = doc.to_dict()
                return self._current_account
        except Exception as e:
            logger.error(f"Failed to get account: {e}")
        return None

    def get_available_slots(self) -> int:
        """Get number of available GE slots."""
        account = self.get_account()
        if account:
            return account.get("ge_slots_available", 0)
        return 0

    def get_account_status(self) -> str:
        """Get account status (active, paused, stopped)."""
        account = self.get_account()
        if account:
            return account.get("status", "unknown")
        return "unknown"

    def is_account_active(self) -> bool:
        """Check if account is active."""
        return self.get_account_status() == "active"

    def get_last_heartbeat(self) -> Optional[str]:
        """Get the last heartbeat timestamp from plugin."""
        account = self.get_account()
        if account:
            return account.get("last_heartbeat")
        return None

    def is_plugin_online(self, max_age_seconds: int = 60) -> bool:
        """
        Check if the plugin is online (sent heartbeat recently).

        Args:
            max_age_seconds: Maximum age of heartbeat to consider online

        Returns:
            True if plugin sent heartbeat within max_age_seconds
        """
        heartbeat = self.get_last_heartbeat()
        if heartbeat is None:
            return False

        try:
            heartbeat_time = datetime.fromisoformat(heartbeat.replace('Z', '+00:00'))
            age = (datetime.now(timezone.utc) - heartbeat_time).total_seconds()
            return age < max_age_seconds
        except Exception as e:
            logger.error(f"Failed to parse heartbeat time: {e}")
            return False

    # =========================================================================
    # Bank Queries
    # =========================================================================

    def get_bank(self) -> Optional[Dict[str, Any]]:
        """
        Get the current bank state from Firestore.

        Returns:
            Bank document with structure:
            {
                "updated_at": str,
                "items": { "item_id": {...}, ... },
                "item_count": int,
                "total_value": int,
                "tradeable_count": int,
                "awaiting_sell": [item_ids],
                "scan_type": "full" | "cached"
            }
        """
        try:
            doc = self.client.get_bank_ref().get()
            if doc.exists:
                return doc.to_dict()
        except Exception as e:
            logger.error(f"Failed to get bank: {e}")
        return None

    def get_bank_holdings(self) -> Dict[str, Dict[str, Any]]:
        """Get items currently in bank."""
        bank = self.get_bank()
        if bank:
            return bank.get("items", {})
        return {}

    def get_bank_item_quantity(self, item_id: int) -> int:
        """Get quantity of a specific item in bank."""
        holdings = self.get_bank_holdings()
        item_data = holdings.get(str(item_id), {})
        return item_data.get("quantity", 0)

    def has_bank_item(self, item_id: int, min_quantity: int = 1) -> bool:
        """Check if we have at least min_quantity of an item in bank."""
        return self.get_bank_item_quantity(item_id) >= min_quantity

    def get_awaiting_sell(self) -> List[int]:
        """Get list of tradeable item IDs in bank awaiting sell."""
        bank = self.get_bank()
        if bank:
            return bank.get("awaiting_sell", [])
        return []

    def get_bank_value(self) -> int:
        """Get total bank value."""
        bank = self.get_bank()
        if bank:
            return bank.get("total_value", 0)
        return 0

    # =========================================================================
    # Inventory Queries
    # =========================================================================

    def get_inventory(self) -> Optional[Dict[str, Any]]:
        """
        Get the current inventory state from Firestore.

        Returns:
            Inventory document with structure:
            {
                "updated_at": str,
                "items": { "item_id": {...}, ... },
                "gold": int,
                "item_count": int,
                "total_value": int,
                "free_slots": int
            }
        """
        try:
            doc = self.client.get_inventory_ref().get()
            if doc.exists:
                return doc.to_dict()
        except Exception as e:
            logger.error(f"Failed to get inventory: {e}")
        return None

    def get_inventory_items(self) -> Dict[str, Dict[str, Any]]:
        """Get items currently in inventory."""
        inventory = self.get_inventory()
        if inventory:
            return inventory.get("items", {})
        return {}

    def get_inventory_free_slots(self) -> int:
        """Get number of free inventory slots."""
        inventory = self.get_inventory()
        if inventory:
            return inventory.get("free_slots", 28)
        return 28

    # =========================================================================
    # GE Slots Queries
    # =========================================================================

    def get_ge_slots(self) -> Optional[Dict[str, Any]]:
        """
        Get the current GE slot states from Firestore.

        Returns:
            GE slots document with structure:
            {
                "updated_at": str,
                "slots": {
                    "1": { "order_id": str, "item_id": int, "type": "buy"|"sell", ... } | null,
                    ...
                },
                "buy_slots_used": int,
                "sell_slots_used": int,
                "slots_available": int
            }
        """
        try:
            doc = self.client.get_ge_slots_ref().get()
            if doc.exists:
                return doc.to_dict()
        except Exception as e:
            logger.error(f"Failed to get GE slots: {e}")
        return None

    def get_ge_slots_state(self) -> Dict[int, Optional[Dict[str, Any]]]:
        """
        Get GE slot states as a dictionary keyed by slot number (1-8).

        Returns:
            {1: None, 2: {"item_id": 554, "type": "buy", ...}, ...}
        """
        ge_data = self.get_ge_slots()
        if not ge_data:
            return {i: None for i in range(1, 9)}

        slots_data = ge_data.get("slots", {})
        result = {}
        for i in range(1, 9):
            slot_data = slots_data.get(str(i))
            result[i] = slot_data
        return result

    def get_ge_slots_available(self) -> int:
        """Get number of available GE slots."""
        ge_data = self.get_ge_slots()
        if ge_data:
            return ge_data.get("slots_available", 0)
        # Fallback to account document
        return self.get_available_slots()

    def get_buy_slots_used(self) -> int:
        """Get number of slots used for buy orders."""
        ge_data = self.get_ge_slots()
        if ge_data:
            return ge_data.get("buy_slots_used", 0)
        return 0

    def get_sell_slots_used(self) -> int:
        """Get number of slots used for sell orders."""
        ge_data = self.get_ge_slots()
        if ge_data:
            return ge_data.get("sell_slots_used", 0)
        return 0

    def get_empty_slots(self) -> List[int]:
        """Get list of empty slot numbers."""
        slots = self.get_ge_slots_state()
        return [i for i in range(1, 9) if slots.get(i) is None]

    def get_active_buy_slots(self) -> List[Dict[str, Any]]:
        """Get list of active buy orders with slot info."""
        slots = self.get_ge_slots_state()
        return [
            {"slot": i, **slot}
            for i, slot in slots.items()
            if slot is not None and slot.get("type") == "buy"
        ]

    def get_active_sell_slots(self) -> List[Dict[str, Any]]:
        """Get list of active sell orders with slot info."""
        slots = self.get_ge_slots_state()
        return [
            {"slot": i, **slot}
            for i, slot in slots.items()
            if slot is not None and slot.get("type") == "sell"
        ]

    # =========================================================================
    # Combined Holdings
    # =========================================================================

    def get_all_holdings(self) -> Dict[int, Dict[str, Any]]:
        """
        Get combined holdings from both inventory and bank.

        Returns:
            Dictionary mapping item_id (as int) to combined holdings:
            {
                item_id: {
                    "item_name": str,
                    "inventory_qty": int,
                    "bank_qty": int,
                    "total_qty": int,
                    "price_each": int,
                    "total_value": int
                }
            }
        """
        combined = {}

        # Add inventory items
        for item_id_str, item_data in self.get_inventory_items().items():
            item_id = int(item_id_str)
            combined[item_id] = {
                "item_id": item_id,
                "item_name": item_data.get("item_name", "Unknown"),
                "inventory_qty": item_data.get("quantity", 0),
                "bank_qty": 0,
                "total_qty": item_data.get("quantity", 0),
                "price_each": item_data.get("price_each", 0),
                "total_value": item_data.get("total_value", 0)
            }

        # Add/merge bank items
        for item_id_str, item_data in self.get_bank_holdings().items():
            item_id = int(item_id_str)
            bank_qty = item_data.get("quantity", 0)
            price = item_data.get("price_each", 0)

            if item_id in combined:
                combined[item_id]["bank_qty"] = bank_qty
                combined[item_id]["total_qty"] += bank_qty
                combined[item_id]["total_value"] += bank_qty * price
            else:
                combined[item_id] = {
                    "item_id": item_id,
                    "item_name": item_data.get("item_name", "Unknown"),
                    "inventory_qty": 0,
                    "bank_qty": bank_qty,
                    "total_qty": bank_qty,
                    "price_each": price,
                    "total_value": bank_qty * price
                }

        return combined

    def get_total_item_quantity(self, item_id: int) -> int:
        """Get total quantity of an item across inventory and bank."""
        return (
            self.get_item_quantity(item_id) +  # portfolio/inventory
            self.get_bank_item_quantity(item_id)
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def refresh(self):
        """Force refresh of cached portfolio and account data."""
        self._current_portfolio = None
        self._current_account = None
        self.get_portfolio()
        self.get_account()

    def stop_all_listeners(self):
        """Stop all listeners."""
        self.stop_portfolio_listener()
        self.stop_account_listener()

    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current state for PPO inference.

        Returns:
            Dictionary with key state information
        """
        portfolio = self.get_portfolio() or {}
        account = self.get_account() or {}

        return {
            "gold": portfolio.get("gold", 0),
            "total_value": portfolio.get("total_value", 0),
            "item_count": len(portfolio.get("items", {})),
            "holdings": portfolio.get("items", {}),
            "ge_slots_available": account.get("ge_slots_available", 0),
            "account_status": account.get("status", "unknown"),
            "plugin_online": self.is_plugin_online(),
            "last_heartbeat": account.get("last_heartbeat"),
            "updated_at": portfolio.get("updated_at")
        }
