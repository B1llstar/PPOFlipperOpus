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
