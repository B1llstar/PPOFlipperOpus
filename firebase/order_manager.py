"""
Order Manager - Create and manage trading orders sent to the GE Auto plugin.

This module handles the PPO → Plugin direction of communication:
- Create buy/sell orders
- Track order status
- Cancel orders
- Listen for status updates
"""

import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, List, Any, Callable

from google.cloud.firestore_v1 import DocumentSnapshot

from .firebase_client import FirebaseClient

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status values (must match Java/Plugin side)."""
    PENDING = "pending"
    RECEIVED = "received"  # Plugin has received the order
    PLACED = "placed"      # Order placed in GE slot
    PARTIAL = "partial"    # Partially filled
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class OrderAction(Enum):
    """Order action types."""
    BUY = "buy"
    SELL = "sell"


class OrderManager:
    """
    Manages trading orders in Firestore.

    Creates orders that the GE Auto plugin will pick up and execute.
    Listens for status updates from the plugin.
    """

    def __init__(self, client: Optional[FirebaseClient] = None):
        """Initialize with Firebase client."""
        self.client = client or FirebaseClient()
        self._status_listener = None
        self._pending_orders: Dict[str, Dict[str, Any]] = {}
        self._status_callbacks: List[Callable[[str, str, Dict], None]] = []

    def _generate_order_id(self) -> str:
        """Generate a unique order ID."""
        return f"ord_{uuid.uuid4().hex[:12]}"

    def _now_iso(self) -> str:
        """Get current UTC timestamp in ISO format."""
        return datetime.now(timezone.utc).isoformat()

    # =========================================================================
    # Order Creation
    # =========================================================================

    def create_buy_order(
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
        Create a buy order for the plugin to execute.

        Args:
            item_id: The item ID to buy
            item_name: The item name
            quantity: Number of items to buy
            price: Price per item
            confidence: PPO confidence score (0-1)
            strategy: Strategy identifier
            metadata: Additional metadata

        Returns:
            Order ID if successful, None otherwise
        """
        return self._create_order(
            OrderAction.BUY, item_id, item_name, quantity, price,
            confidence, strategy, metadata
        )

    def create_sell_order(
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
        Create a sell order for the plugin to execute.

        Args:
            item_id: The item ID to sell
            item_name: The item name
            quantity: Number of items to sell
            price: Price per item
            confidence: PPO confidence score (0-1)
            strategy: Strategy identifier
            metadata: Additional metadata

        Returns:
            Order ID if successful, None otherwise
        """
        return self._create_order(
            OrderAction.SELL, item_id, item_name, quantity, price,
            confidence, strategy, metadata
        )

    def _create_order(
        self,
        action: OrderAction,
        item_id: int,
        item_name: str,
        quantity: int,
        price: int,
        confidence: float,
        strategy: str,
        metadata: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Internal method to create an order."""
        try:
            order_id = self._generate_order_id()
            now = self._now_iso()

            order_data = {
                "order_id": order_id,
                "item_id": item_id,
                "item_name": item_name,
                "action": action.value,
                "quantity": quantity,
                "price": price,
                "status": OrderStatus.PENDING.value,
                "created_at": now,
                "updated_at": now,
                "ge_slot": None,
                "filled_quantity": 0,
                "error_message": None,
                "metadata": {
                    "confidence": confidence,
                    "strategy": strategy,
                    **(metadata or {})
                }
            }

            # Write to Firestore
            doc_ref = self.client.get_orders_ref().document(order_id)
            doc_ref.set(order_data)

            # Track locally
            self._pending_orders[order_id] = order_data

            logger.info(f"Created {action.value} order: {order_id} - {quantity}x {item_name} @ {price}")
            return order_id

        except Exception as e:
            logger.error(f"Failed to create order: {e}")
            return None

    # =========================================================================
    # Order Queries
    # =========================================================================

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get an order by ID."""
        try:
            doc = self.client.get_orders_ref().document(order_id).get()
            if doc.exists:
                return doc.to_dict()
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
        return None

    def get_pending_orders(self) -> List[Dict[str, Any]]:
        """Get all pending orders."""
        try:
            docs = (self.client.get_orders_ref()
                    .where("status", "==", OrderStatus.PENDING.value)
                    .stream())
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            logger.error(f"Failed to get pending orders: {e}")
            return []

    def get_active_orders(self) -> List[Dict[str, Any]]:
        """Get all active orders (pending, received, placed, partial)."""
        try:
            active_statuses = [
                OrderStatus.PENDING.value,
                OrderStatus.RECEIVED.value,
                OrderStatus.PLACED.value,
                OrderStatus.PARTIAL.value
            ]
            docs = (self.client.get_orders_ref()
                    .where("status", "in", active_statuses)
                    .stream())
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            logger.error(f"Failed to get active orders: {e}")
            return []

    def get_recent_orders(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get most recent orders."""
        try:
            docs = (self.client.get_orders_ref()
                    .order_by("created_at", direction="DESCENDING")
                    .limit(limit)
                    .stream())
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            logger.error(f"Failed to get recent orders: {e}")
            return []

    def get_orders_by_item(self, item_id: int, limit: int = 20) -> List[Dict[str, Any]]:
        """Get orders for a specific item."""
        try:
            docs = (self.client.get_orders_ref()
                    .where("item_id", "==", item_id)
                    .order_by("created_at", direction="DESCENDING")
                    .limit(limit)
                    .stream())
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            logger.error(f"Failed to get orders for item {item_id}: {e}")
            return []

    # =========================================================================
    # Order Cancellation
    # =========================================================================

    def cancel_order(self, order_id: str, reason: str = "Cancelled by inference") -> bool:
        """
        Request cancellation of an order.

        Note: If order is already placed on GE, the plugin needs to handle
        the actual cancellation. This just updates the status to request cancellation.

        Args:
            order_id: The order ID to cancel
            reason: Reason for cancellation

        Returns:
            True if cancellation request was recorded
        """
        try:
            doc_ref = self.client.get_orders_ref().document(order_id)
            doc = doc_ref.get()

            if not doc.exists:
                logger.warning(f"Order {order_id} not found for cancellation")
                return False

            order_data = doc.to_dict()
            current_status = order_data.get("status")

            # Can only cancel pending/received orders directly
            # Placed orders need plugin to cancel
            if current_status == OrderStatus.PENDING.value:
                doc_ref.update({
                    "status": OrderStatus.CANCELLED.value,
                    "error_message": reason,
                    "updated_at": self._now_iso()
                })
                logger.info(f"Cancelled pending order: {order_id}")
                return True

            elif current_status in [OrderStatus.RECEIVED.value, OrderStatus.PLACED.value, OrderStatus.PARTIAL.value]:
                # Request plugin to cancel
                doc_ref.update({
                    "cancel_requested": True,
                    "cancel_reason": reason,
                    "updated_at": self._now_iso()
                })
                logger.info(f"Requested cancellation for order: {order_id}")
                return True

            else:
                logger.warning(f"Cannot cancel order {order_id} with status {current_status}")
                return False

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def cancel_all_pending(self, reason: str = "Bulk cancellation") -> int:
        """Cancel all pending orders. Returns count of cancelled orders."""
        pending = self.get_pending_orders()
        count = 0
        for order in pending:
            if self.cancel_order(order["order_id"], reason):
                count += 1
        logger.info(f"Cancelled {count} pending orders")
        return count

    # =========================================================================
    # Status Listening
    # =========================================================================

    def start_status_listener(self, callback: Optional[Callable[[str, str, Dict], None]] = None):
        """
        Start listening for order status updates from the plugin.

        Args:
            callback: Function called with (order_id, new_status, order_data)
        """
        if callback:
            self._status_callbacks.append(callback)

        if self._status_listener is not None:
            logger.warning("Status listener already running")
            return

        def on_snapshot(doc_snapshot, changes, read_time):
            for change in changes:
                if change.type.name in ['ADDED', 'MODIFIED']:
                    order_data = change.document.to_dict()
                    order_id = order_data.get("order_id")
                    status = order_data.get("status")

                    # Update local tracking
                    if order_id in self._pending_orders:
                        old_status = self._pending_orders[order_id].get("status")
                        if old_status != status:
                            logger.info(f"Order {order_id} status: {old_status} → {status}")

                    self._pending_orders[order_id] = order_data

                    # Notify callbacks
                    for cb in self._status_callbacks:
                        try:
                            cb(order_id, status, order_data)
                        except Exception as e:
                            logger.error(f"Error in status callback: {e}")

        # Listen to all orders for this account
        self._status_listener = self.client.get_orders_ref().on_snapshot(on_snapshot)
        logger.info("Started order status listener")

    def stop_status_listener(self):
        """Stop listening for status updates."""
        if self._status_listener:
            self._status_listener.unsubscribe()
            self._status_listener = None
            logger.info("Stopped order status listener")

    def add_status_callback(self, callback: Callable[[str, str, Dict], None]):
        """Add a callback for status updates."""
        self._status_callbacks.append(callback)

    def remove_status_callback(self, callback: Callable[[str, str, Dict], None]):
        """Remove a status callback."""
        if callback in self._status_callbacks:
            self._status_callbacks.remove(callback)

    # =========================================================================
    # Utilities
    # =========================================================================

    def get_order_count_by_status(self) -> Dict[str, int]:
        """Get count of orders by status."""
        counts = {status.value: 0 for status in OrderStatus}
        try:
            docs = self.client.get_orders_ref().stream()
            for doc in docs:
                data = doc.to_dict()
                status = data.get("status", "unknown")
                counts[status] = counts.get(status, 0) + 1
        except Exception as e:
            logger.error(f"Failed to get order counts: {e}")
        return counts

    def cleanup_old_orders(self, days: int = 7) -> int:
        """
        Delete orders older than specified days.

        Args:
            days: Delete orders older than this many days

        Returns:
            Number of orders deleted
        """
        from datetime import timedelta

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        count = 0

        try:
            docs = (self.client.get_orders_ref()
                    .where("created_at", "<", cutoff)
                    .stream())

            for doc in docs:
                doc.reference.delete()
                count += 1

            logger.info(f"Deleted {count} orders older than {days} days")
        except Exception as e:
            logger.error(f"Failed to cleanup old orders: {e}")

        return count
