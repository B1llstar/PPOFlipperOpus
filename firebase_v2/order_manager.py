"""
Order Manager V2 - Create and monitor orders.

Handles order creation, status monitoring, and lifecycle management.
"""

import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional, List, Dict, Any

from google.cloud import firestore

from .firebase_client import FirebaseClientV2
from .types import Order, OrderAction, OrderStatus

logger = logging.getLogger(__name__)


class OrderManagerV2:
    """
    Manages order lifecycle in Firestore.

    Responsibilities:
    - Create new buy/sell orders
    - Monitor order status changes
    - Query orders by status
    - Cancel/delete orders
    """

    def __init__(self, client: Optional[FirebaseClientV2] = None):
        """
        Initialize the order manager.

        Args:
            client: FirebaseClientV2 instance (uses singleton if not provided)
        """
        self._client = client or FirebaseClientV2.get_instance()
        self._lock = threading.RLock()

        # Cached orders
        self._orders: Dict[str, Order] = {}

        # Listener handle
        self._order_listener = None

        # Callbacks
        self._on_order_completed: Optional[Callable[[Order], None]] = None
        self._on_order_failed: Optional[Callable[[Order], None]] = None
        self._on_order_cancelled: Optional[Callable[[Order], None]] = None
        self._on_order_placed: Optional[Callable[[Order], None]] = None

        self._running = False

    def start(
        self,
        on_order_completed: Optional[Callable[[Order], None]] = None,
        on_order_failed: Optional[Callable[[Order], None]] = None,
        on_order_cancelled: Optional[Callable[[Order], None]] = None,
        on_order_placed: Optional[Callable[[Order], None]] = None
    ):
        """
        Start listening to order changes.

        Args:
            on_order_completed: Callback when an order completes
            on_order_failed: Callback when an order fails
            on_order_cancelled: Callback when an order is cancelled
            on_order_placed: Callback when an order is placed in GE
        """
        if self._running:
            logger.warning("OrderManager already running")
            return

        self._on_order_completed = on_order_completed
        self._on_order_failed = on_order_failed
        self._on_order_cancelled = on_order_cancelled
        self._on_order_placed = on_order_placed

        self._start_order_listener()
        self._running = True
        logger.info("OrderManager started")

    def stop(self):
        """Stop listening to order changes."""
        if not self._running:
            return

        if self._order_listener:
            self._order_listener.unsubscribe()
            self._order_listener = None

        self._running = False
        logger.info("OrderManager stopped")

    def _start_order_listener(self):
        """Start listening to all orders."""
        def on_snapshot(doc_snapshot, changes, read_time):
            try:
                for change in changes:
                    doc = change.document
                    order_id = doc.id
                    data = doc.to_dict()

                    if change.type.name == 'REMOVED':
                        with self._lock:
                            if order_id in self._orders:
                                del self._orders[order_id]
                        continue

                    # Parse order
                    order = Order.from_dict(data)

                    # Get previous state
                    with self._lock:
                        prev_order = self._orders.get(order_id)
                        self._orders[order_id] = order

                    # Detect status changes and fire callbacks
                    if prev_order:
                        self._handle_status_change(prev_order, order)
                    elif change.type.name == 'ADDED':
                        logger.debug(f"Order tracked: {order_id} ({order.status.value})")

            except Exception as e:
                logger.error(f"Error processing order snapshot: {e}")

        self._order_listener = self._client.orders_ref.on_snapshot(on_snapshot)

    def _handle_status_change(self, prev: Order, current: Order):
        """Handle order status change and fire appropriate callback."""
        if prev.status == current.status:
            return

        logger.info(
            f"Order {current.order_id} status: {prev.status.value} -> {current.status.value}"
        )

        if current.status == OrderStatus.COMPLETED:
            if self._on_order_completed:
                self._on_order_completed(current)

        elif current.status == OrderStatus.FAILED:
            if self._on_order_failed:
                self._on_order_failed(current)

        elif current.status == OrderStatus.CANCELLED:
            if self._on_order_cancelled:
                self._on_order_cancelled(current)

        elif current.status == OrderStatus.PLACED and prev.status in [
            OrderStatus.PENDING, OrderStatus.RECEIVED
        ]:
            if self._on_order_placed:
                self._on_order_placed(current)

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
        strategy: str = "ppo_v2"
    ) -> Optional[str]:
        """
        Create a new buy order.

        Args:
            item_id: The item ID to buy
            item_name: The item name
            quantity: Quantity to buy
            price: Price per item
            confidence: Model confidence (0-1)
            strategy: Strategy identifier

        Returns:
            Order ID if successful, None if failed
        """
        return self._create_order(
            action=OrderAction.BUY,
            item_id=item_id,
            item_name=item_name,
            quantity=quantity,
            price=price,
            confidence=confidence,
            strategy=strategy
        )

    def create_sell_order(
        self,
        item_id: int,
        item_name: str,
        quantity: int,
        price: int,
        confidence: float = 0.0,
        strategy: str = "ppo_v2"
    ) -> Optional[str]:
        """
        Create a new sell order.

        Args:
            item_id: The item ID to sell
            item_name: The item name
            quantity: Quantity to sell
            price: Price per item
            confidence: Model confidence (0-1)
            strategy: Strategy identifier

        Returns:
            Order ID if successful, None if failed
        """
        return self._create_order(
            action=OrderAction.SELL,
            item_id=item_id,
            item_name=item_name,
            quantity=quantity,
            price=price,
            confidence=confidence,
            strategy=strategy
        )

    def _create_order(
        self,
        action: OrderAction,
        item_id: int,
        item_name: str,
        quantity: int,
        price: int,
        confidence: float,
        strategy: str
    ) -> Optional[str]:
        """Create an order in Firestore."""
        try:
            order_id = f"ord_{uuid.uuid4().hex[:12]}"

            order_data = {
                'order_id': order_id,
                'action': action.value,
                'item_id': item_id,
                'item_name': item_name,
                'quantity': quantity,
                'price': price,
                'status': OrderStatus.PENDING.value,
                'ge_slot': None,
                'filled_quantity': 0,
                'total_cost': 0,
                'created_at': firestore.SERVER_TIMESTAMP,
                'received_at': None,
                'placed_at': None,
                'completed_at': None,
                'error': None,
                'retry_count': 0,
                'confidence': confidence,
                'strategy': strategy
            }

            # Write to Firestore
            self._client.orders_ref.document(order_id).set(order_data)

            logger.info(
                f"Created {action.value} order: {order_id} for "
                f"{quantity}x {item_name} @ {price}gp"
            )

            return order_id

        except Exception as e:
            logger.error(f"Failed to create order: {e}")
            return None

    # =========================================================================
    # Order Queries
    # =========================================================================

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get an order by ID (from cache or Firestore)."""
        with self._lock:
            if order_id in self._orders:
                return self._orders[order_id]

        # Fetch from Firestore
        try:
            doc = self._client.orders_ref.document(order_id).get()
            if doc.exists:
                return Order.from_dict(doc.to_dict())
            return None
        except Exception as e:
            logger.error(f"Error fetching order {order_id}: {e}")
            return None

    def get_orders_by_status(self, *statuses: OrderStatus) -> List[Order]:
        """Get all orders with given status(es)."""
        with self._lock:
            return [
                order for order in self._orders.values()
                if order.status in statuses
            ]

    def get_pending_orders(self) -> List[Order]:
        """Get all pending orders (not yet in GE)."""
        return self.get_orders_by_status(
            OrderStatus.PENDING,
            OrderStatus.RECEIVED
        )

    def get_active_orders(self) -> List[Order]:
        """Get all active orders (in GE)."""
        return self.get_orders_by_status(
            OrderStatus.PLACED,
            OrderStatus.PARTIAL
        )

    def get_completed_orders(self) -> List[Order]:
        """Get all completed orders."""
        return self.get_orders_by_status(OrderStatus.COMPLETED)

    def get_all_orders(self) -> List[Order]:
        """Get all tracked orders."""
        with self._lock:
            return list(self._orders.values())

    def get_active_order_count(self) -> int:
        """Get count of active orders (using GE slots)."""
        return len(self.get_active_orders())

    def get_pending_order_count(self) -> int:
        """Get count of pending orders."""
        return len(self.get_pending_orders())

    # =========================================================================
    # Order Actions
    # =========================================================================

    def cancel_order(self, order_id: str, reason: str = "Cancelled by user") -> bool:
        """
        Cancel an order.

        Can only cancel pending/received orders. Placed orders need GE cancellation.

        Args:
            order_id: The order ID to cancel
            reason: Cancellation reason

        Returns:
            True if cancelled successfully
        """
        try:
            order = self.get_order(order_id)
            if not order:
                logger.warning(f"Order not found: {order_id}")
                return False

            if order.status not in [OrderStatus.PENDING, OrderStatus.RECEIVED]:
                logger.warning(
                    f"Cannot cancel order {order_id} with status {order.status.value}"
                )
                return False

            self._client.orders_ref.document(order_id).update({
                'status': OrderStatus.CANCELLED.value,
                'error': reason,
                'completed_at': firestore.SERVER_TIMESTAMP
            })

            logger.info(f"Cancelled order {order_id}: {reason}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def cancel_all_pending(self, reason: str = "Bulk cancel") -> int:
        """
        Cancel all pending orders.

        Returns:
            Number of orders cancelled
        """
        pending = self.get_pending_orders()
        cancelled = 0

        for order in pending:
            if self.cancel_order(order.order_id, reason):
                cancelled += 1

        logger.info(f"Cancelled {cancelled} pending orders")
        return cancelled

    def delete_order(self, order_id: str) -> bool:
        """
        Delete an order document.

        Only deletes terminal orders (completed/failed/cancelled).

        Args:
            order_id: The order ID to delete

        Returns:
            True if deleted successfully
        """
        try:
            order = self.get_order(order_id)
            if order and not order.is_terminal:
                logger.warning(
                    f"Cannot delete non-terminal order {order_id} "
                    f"with status {order.status.value}"
                )
                return False

            self._client.orders_ref.document(order_id).delete()

            with self._lock:
                if order_id in self._orders:
                    del self._orders[order_id]

            logger.info(f"Deleted order {order_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete order {order_id}: {e}")
            return False

    def clear_completed_orders(self) -> int:
        """
        Delete all completed orders.

        Returns:
            Number of orders deleted
        """
        completed = self.get_completed_orders()
        deleted = 0

        for order in completed:
            if self.delete_order(order.order_id):
                deleted += 1

        logger.info(f"Deleted {deleted} completed orders")
        return deleted

    # =========================================================================
    # Refresh
    # =========================================================================

    def refresh_orders(self):
        """Manually refresh all orders from Firestore."""
        try:
            docs = self._client.orders_ref.get()

            with self._lock:
                self._orders.clear()
                for doc in docs:
                    order = Order.from_dict(doc.to_dict())
                    self._orders[order.order_id] = order

            logger.info(f"Refreshed {len(self._orders)} orders from Firestore")

        except Exception as e:
            logger.error(f"Error refreshing orders: {e}")
            raise
