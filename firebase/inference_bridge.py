"""
Inference Bridge - Main orchestrator connecting PPO inference to Firebase and the plugin.

This is the central coordinator that:
- Manages all Firebase components (orders, trades, portfolio)
- Provides a unified interface for PPO inference
- Handles state synchronization
- Manages order lifecycle
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any, Callable

from .firebase_client import FirebaseClient
from .order_manager import OrderManager, OrderStatus, OrderAction
from .trade_monitor import TradeMonitor
from .portfolio_tracker import PortfolioTracker

logger = logging.getLogger(__name__)


class InferenceBridge:
    """
    Main orchestrator for PPO inference ↔ Firebase ↔ Plugin communication.

    Provides:
    - Unified interface for submitting orders
    - State management and synchronization
    - Event callbacks for trade completions
    - Portfolio and market data access
    """

    def __init__(
        self,
        service_account_path: str,
        account_id: str,
        on_trade_completed: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_order_status_changed: Optional[Callable[[str, str, Dict], None]] = None
    ):
        """
        Initialize the inference bridge.

        Args:
            service_account_path: Path to Firebase service account JSON
            account_id: Account identifier
            on_trade_completed: Callback when a trade completes
            on_order_status_changed: Callback when order status changes
        """
        self.account_id = account_id
        self._on_trade_completed = on_trade_completed
        self._on_order_status_changed = on_order_status_changed

        # Initialize Firebase
        self.client = FirebaseClient()
        if not self.client.initialize(service_account_path, account_id):
            raise RuntimeError("Failed to initialize Firebase")

        # Initialize components
        self.order_manager = OrderManager(self.client)
        self.trade_monitor = TradeMonitor(self.client)
        self.portfolio_tracker = PortfolioTracker(self.client)

        # State tracking
        self._running = False
        self._pending_order_ids: List[str] = []

        logger.info(f"InferenceBridge initialized for account: {account_id}")

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self):
        """Start all listeners and initialize state."""
        if self._running:
            logger.warning("InferenceBridge already running")
            return

        # Start listeners
        self.order_manager.start_status_listener(self._handle_order_status_change)
        self.trade_monitor.start_listening(self._handle_trade_completed)
        self.portfolio_tracker.start_portfolio_listener()
        self.portfolio_tracker.start_account_listener()

        # Initial state sync
        self.sync_state()

        self._running = True
        logger.info("InferenceBridge started")

    def stop(self):
        """Stop all listeners and cleanup."""
        if not self._running:
            return

        self.order_manager.stop_status_listener()
        self.trade_monitor.stop_listening()
        self.portfolio_tracker.stop_all_listeners()

        self._running = False
        logger.info("InferenceBridge stopped")

    def shutdown(self):
        """Full shutdown including Firebase connection."""
        self.stop()
        self.client.shutdown()
        logger.info("InferenceBridge shutdown complete")

    # =========================================================================
    # Order Submission
    # =========================================================================

    def submit_buy_order(
        self,
        item_id: int,
        item_name: str,
        quantity: int,
        price: int,
        confidence: float = 0.0,
        strategy: str = "ppo",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Submit a buy order to be executed by the plugin.

        Args:
            item_id: Item ID to buy
            item_name: Item name
            quantity: Quantity to buy
            price: Price per item
            confidence: PPO confidence score
            strategy: Strategy identifier
            metadata: Additional metadata

        Returns:
            Order ID if successful, None otherwise
        """
        # Validate
        if not self._validate_order(item_id, quantity, price, "buy"):
            return None

        order_id = self.order_manager.create_buy_order(
            item_id=item_id,
            item_name=item_name,
            quantity=quantity,
            price=price,
            confidence=confidence,
            strategy=strategy,
            metadata=metadata
        )

        if order_id:
            self._pending_order_ids.append(order_id)
            logger.info(f"Submitted buy order: {order_id}")

        return order_id

    def submit_sell_order(
        self,
        item_id: int,
        item_name: str,
        quantity: int,
        price: int,
        confidence: float = 0.0,
        strategy: str = "ppo",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Submit a sell order to be executed by the plugin.

        Args:
            item_id: Item ID to sell
            item_name: Item name
            quantity: Quantity to sell
            price: Price per item
            confidence: PPO confidence score
            strategy: Strategy identifier
            metadata: Additional metadata

        Returns:
            Order ID if successful, None otherwise
        """
        # Validate
        if not self._validate_order(item_id, quantity, price, "sell"):
            return None

        # Check if we have the item
        holdings = self.portfolio_tracker.get_item_quantity(item_id)
        if holdings < quantity:
            logger.warning(f"Insufficient holdings for sell: have {holdings}, need {quantity}")
            return None

        order_id = self.order_manager.create_sell_order(
            item_id=item_id,
            item_name=item_name,
            quantity=quantity,
            price=price,
            confidence=confidence,
            strategy=strategy,
            metadata=metadata
        )

        if order_id:
            self._pending_order_ids.append(order_id)
            logger.info(f"Submitted sell order: {order_id}")

        return order_id

    def _validate_order(self, item_id: int, quantity: int, price: int, action: str) -> bool:
        """Validate order parameters."""
        if item_id <= 0:
            logger.warning(f"Invalid item_id: {item_id}")
            return False

        if quantity <= 0:
            logger.warning(f"Invalid quantity: {quantity}")
            return False

        if price <= 0:
            logger.warning(f"Invalid price: {price}")
            return False

        # Check if plugin is online
        if not self.portfolio_tracker.is_plugin_online(max_age_seconds=120):
            logger.warning("Plugin appears to be offline")
            # Don't block, just warn

        # Check available GE slots for buys
        if action == "buy":
            available_slots = self.portfolio_tracker.get_available_slots()
            active_orders = len(self.order_manager.get_active_orders())
            if active_orders >= 8:
                logger.warning(f"Too many active orders: {active_orders}")
                return False

        return True

    # =========================================================================
    # Order Management
    # =========================================================================

    def cancel_order(self, order_id: str, reason: str = "Cancelled by inference") -> bool:
        """Cancel an order."""
        success = self.order_manager.cancel_order(order_id, reason)
        if success and order_id in self._pending_order_ids:
            self._pending_order_ids.remove(order_id)
        return success

    def cancel_all_pending(self, reason: str = "Bulk cancellation") -> int:
        """Cancel all pending orders."""
        count = self.order_manager.cancel_all_pending(reason)
        self._pending_order_ids.clear()
        return count

    def get_pending_orders(self) -> List[Dict[str, Any]]:
        """Get all pending orders."""
        return self.order_manager.get_pending_orders()

    def get_active_orders(self) -> List[Dict[str, Any]]:
        """Get all active orders (pending, placed, partial)."""
        return self.order_manager.get_active_orders()

    def get_active_order_count(self) -> int:
        """Get count of active orders."""
        return len(self.get_active_orders())

    # =========================================================================
    # State Access
    # =========================================================================

    def get_gold(self) -> int:
        """Get current gold balance."""
        return self.portfolio_tracker.get_gold()

    def get_holdings(self) -> Dict[str, Dict[str, Any]]:
        """Get current item holdings."""
        return self.portfolio_tracker.get_holdings()

    def get_item_quantity(self, item_id: int) -> int:
        """Get quantity of a specific item."""
        return self.portfolio_tracker.get_item_quantity(item_id)

    def get_available_slots(self) -> int:
        """Get number of available GE slots."""
        total_slots = 8
        active_orders = self.get_active_order_count()
        return max(0, total_slots - active_orders)

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get complete portfolio summary."""
        return self.portfolio_tracker.get_state_summary()

    def is_plugin_online(self) -> bool:
        """Check if plugin is online."""
        return self.portfolio_tracker.is_plugin_online()

    # =========================================================================
    # Trade History
    # =========================================================================

    def get_recent_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent completed trades."""
        return self.trade_monitor.get_recent_trades(limit)

    def get_pnl(self, item_id: Optional[int] = None) -> Dict[str, Any]:
        """Get profit/loss statistics."""
        return self.trade_monitor.calculate_pnl(item_id=item_id)

    def get_daily_pnl(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get daily P&L for the last N days."""
        return self.trade_monitor.calculate_daily_pnl(days)

    def get_item_performance(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get performance by item."""
        return self.trade_monitor.get_item_performance(limit)

    # =========================================================================
    # Item Lookups
    # =========================================================================

    def get_item_by_id(self, item_id: int) -> Optional[Dict[str, Any]]:
        """Get item data by ID."""
        return self.client.get_item_by_id(item_id)

    def get_item_id_by_name(self, item_name: str) -> Optional[int]:
        """Get item ID by name."""
        return self.client.get_item_id_by_name(item_name)

    def get_item_by_name(self, item_name: str) -> Optional[Dict[str, Any]]:
        """Get item data by name."""
        return self.client.get_item_by_name(item_name)

    # =========================================================================
    # State Synchronization
    # =========================================================================

    def sync_state(self):
        """Force synchronization of all state from Firestore."""
        logger.info("Synchronizing state from Firestore...")

        # Refresh portfolio tracker
        self.portfolio_tracker.refresh()

        # Load active orders
        active_orders = self.order_manager.get_active_orders()
        self._pending_order_ids = [o["order_id"] for o in active_orders]

        portfolio = self.portfolio_tracker.get_state_summary()
        logger.info(f"State synced: {portfolio['gold']:,} gold, "
                   f"{portfolio['item_count']} items, "
                   f"{len(self._pending_order_ids)} active orders, "
                   f"plugin_online={portfolio['plugin_online']}")

    # =========================================================================
    # Callbacks
    # =========================================================================

    def _handle_trade_completed(self, trade_data: Dict[str, Any]):
        """Handle trade completion from plugin."""
        order_id = trade_data.get("order_id")

        # Remove from pending if present
        if order_id and order_id in self._pending_order_ids:
            self._pending_order_ids.remove(order_id)

        # Mark the order as completed in Firestore
        if order_id:
            filled_quantity = trade_data.get("quantity")
            total_cost = trade_data.get("total_cost")
            self.order_manager.complete_order(
                order_id=order_id,
                filled_quantity=filled_quantity,
                total_cost=total_cost,
                metadata={"trade_id": trade_data.get("trade_id")}
            )

        # Notify external callback
        if self._on_trade_completed:
            try:
                self._on_trade_completed(trade_data)
            except Exception as e:
                logger.error(f"Error in trade completed callback: {e}")

    def _handle_order_status_change(self, order_id: str, status: str, order_data: Dict):
        """Handle order status change from plugin."""
        # Update local tracking
        if status in [OrderStatus.COMPLETED.value, OrderStatus.CANCELLED.value, OrderStatus.FAILED.value]:
            if order_id in self._pending_order_ids:
                self._pending_order_ids.remove(order_id)

        # Notify external callback
        if self._on_order_status_changed:
            try:
                self._on_order_status_changed(order_id, status, order_data)
            except Exception as e:
                logger.error(f"Error in order status callback: {e}")

    # =========================================================================
    # Utility
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get overall bridge status."""
        portfolio = self.portfolio_tracker.get_state_summary()
        pnl = self.trade_monitor.calculate_pnl()

        return {
            "running": self._running,
            "account_id": self.account_id,
            "plugin_online": portfolio["plugin_online"],
            "gold": portfolio["gold"],
            "total_value": portfolio["total_value"],
            "holdings_count": portfolio["item_count"],
            "active_orders": len(self._pending_order_ids),
            "available_slots": self.get_available_slots(),
            "net_profit": pnl["net_profit"],
            "total_trades": pnl["buy_count"] + pnl["sell_count"]
        }
