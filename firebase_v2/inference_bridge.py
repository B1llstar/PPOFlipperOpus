"""
Inference Bridge V2 - High-level interface for PPO inference.

Provides a clean, simple interface for the inference runner to:
- Access account state (gold, holdings, GE slots)
- Submit buy/sell orders
- Monitor order completions
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List, Callable, Any

from .firebase_client import FirebaseClientV2
from .state_listener import StateListener
from .order_manager import OrderManagerV2
from .types import (
    Order, OrderStatus, Holding, GEState, Portfolio,
    InventoryState, BankState, AccountState
)

logger = logging.getLogger(__name__)


class InferenceBridgeV2:
    """
    High-level interface for PPO inference to interact with Firestore.

    This is the main entry point for the inference runner. It provides:
    - Initialization and connection management
    - State access (gold, holdings, GE slots, plugin status)
    - Order submission with validation
    - Order monitoring via callbacks

    Usage:
        bridge = InferenceBridgeV2()
        await bridge.initialize(service_account_path, account_id)

        # Access state
        gold = bridge.get_gold()
        holdings = bridge.get_holdings()

        # Submit orders
        order_id = bridge.submit_buy_order(item_id, item_name, qty, price)

        # Shutdown
        bridge.shutdown()
    """

    def __init__(self):
        """Initialize the inference bridge (not connected yet)."""
        self._client: Optional[FirebaseClientV2] = None
        self._state_listener: Optional[StateListener] = None
        self._order_manager: Optional[OrderManagerV2] = None
        self._initialized = False

        # Callbacks
        self._on_order_completed: Optional[Callable[[Order], None]] = None
        self._on_order_failed: Optional[Callable[[Order], None]] = None

    # =========================================================================
    # Initialization
    # =========================================================================

    def initialize(
        self,
        service_account_path: Optional[str] = None,
        account_id: str = "b1llstar",
        project_id: str = "ppoflipperopus",
        on_order_completed: Optional[Callable[[Order], None]] = None,
        on_order_failed: Optional[Callable[[Order], None]] = None
    ) -> bool:
        """
        Initialize the bridge and connect to Firestore.

        Args:
            service_account_path: Path to service account JSON (auto-detect if None)
            account_id: The account ID (player name)
            project_id: Firebase project ID
            on_order_completed: Callback when orders complete
            on_order_failed: Callback when orders fail

        Returns:
            True if initialization successful
        """
        if self._initialized:
            logger.warning("InferenceBridge already initialized")
            return True

        try:
            # Find service account if not provided
            if service_account_path is None:
                service_account_path = self._find_service_account()

            if not service_account_path:
                logger.error("Could not find service account file")
                return False

            # Initialize Firebase client
            self._client = FirebaseClientV2.get_instance()
            if not self._client.initialize(service_account_path, account_id, project_id):
                return False

            # Initialize state listener
            self._state_listener = StateListener(self._client)
            self._state_listener.start()

            # Initialize order manager
            self._order_manager = OrderManagerV2(self._client)
            self._on_order_completed = on_order_completed
            self._on_order_failed = on_order_failed
            self._order_manager.start(
                on_order_completed=self._handle_order_completed,
                on_order_failed=self._handle_order_failed
            )

            # Wait for initial state
            logger.info("Waiting for initial state sync...")
            if not self._state_listener.wait_for_state(timeout=15.0):
                logger.warning("Timeout waiting for initial state - continuing anyway")

            self._initialized = True
            logger.info(f"InferenceBridge V2 initialized for account: {account_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize InferenceBridge: {e}")
            return False

    def _find_service_account(self) -> Optional[str]:
        """Find the service account JSON file."""
        possible_paths = [
            Path(__file__).parent.parent / "ppoflipperopus-firebase-adminsdk-fbsvc-907a50dae7.json",
            Path(__file__).parent.parent / "config" / "ppoflipperopus-firebase-adminsdk-fbsvc-907a50dae7.json",
            Path.home() / ".config" / "ppoflipperopus" / "service_account.json",
        ]

        for path in possible_paths:
            if path.exists():
                logger.info(f"Found service account: {path}")
                return str(path)

        return None

    def _handle_order_completed(self, order: Order):
        """Internal handler for order completion."""
        logger.info(
            f"Order completed: {order.order_id} - "
            f"{order.action.value} {order.filled_quantity}x {order.item_name} "
            f"@ {order.price}gp (total: {order.total_cost}gp)"
        )
        if self._on_order_completed:
            self._on_order_completed(order)

    def _handle_order_failed(self, order: Order):
        """Internal handler for order failure."""
        logger.warning(
            f"Order failed: {order.order_id} - "
            f"{order.action.value} {order.item_name}: {order.error}"
        )
        if self._on_order_failed:
            self._on_order_failed(order)

    def shutdown(self):
        """Shutdown the bridge and close connections."""
        if self._order_manager:
            self._order_manager.stop()
            self._order_manager = None

        if self._state_listener:
            self._state_listener.stop()
            self._state_listener = None

        if self._client:
            self._client.shutdown()
            self._client = None

        self._initialized = False
        logger.info("InferenceBridge V2 shutdown complete")

    @property
    def is_initialized(self) -> bool:
        """Check if bridge is initialized."""
        return self._initialized

    # =========================================================================
    # State Access
    # =========================================================================

    def get_gold(self) -> int:
        """Get current gold amount."""
        if not self._state_listener:
            return 0
        return self._state_listener.gold

    def get_portfolio(self) -> Optional[Portfolio]:
        """Get portfolio summary."""
        if not self._state_listener:
            return None
        return self._state_listener.portfolio

    def get_inventory(self) -> Optional[InventoryState]:
        """Get inventory state."""
        if not self._state_listener:
            return None
        return self._state_listener.inventory

    def get_bank(self) -> Optional[BankState]:
        """Get bank state."""
        if not self._state_listener:
            return None
        return self._state_listener.bank

    def get_ge_state(self) -> Optional[GEState]:
        """Get GE slots state."""
        if not self._state_listener:
            return None
        return self._state_listener.ge_state

    def get_free_slots(self) -> int:
        """Get number of free GE slots."""
        if not self._state_listener:
            return 8
        return self._state_listener.free_slots

    def get_holdings(self) -> Dict[int, Holding]:
        """Get all holdings (inventory + bank combined)."""
        if not self._state_listener:
            return {}
        return self._state_listener.get_holdings()

    def get_inventory_quantity(self, item_id: int) -> int:
        """Get quantity of an item in inventory."""
        if not self._state_listener:
            return 0
        return self._state_listener.get_inventory_quantity(item_id)

    def get_bank_quantity(self, item_id: int) -> int:
        """Get quantity of an item in bank."""
        if not self._state_listener:
            return 0
        return self._state_listener.get_bank_quantity(item_id)

    def get_total_quantity(self, item_id: int) -> int:
        """Get total quantity of an item (inventory + bank)."""
        if not self._state_listener:
            return 0
        return self._state_listener.get_total_quantity(item_id)

    def is_plugin_online(self) -> bool:
        """Check if plugin is online (heartbeat within 2 min)."""
        if not self._state_listener:
            return False
        return self._state_listener.plugin_online

    def get_account_state(self) -> Optional[AccountState]:
        """Get complete account state snapshot."""
        if not self._state_listener:
            return None
        return self._state_listener.get_account_state()

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
        strategy: str = "ppo_v2"
    ) -> Optional[str]:
        """
        Submit a buy order.

        Validates:
        - Plugin is online
        - Free GE slots available
        - Sufficient gold

        Args:
            item_id: Item ID to buy
            item_name: Item name
            quantity: Quantity to buy
            price: Price per item
            confidence: Model confidence (0-1)
            strategy: Strategy identifier

        Returns:
            Order ID if successful, None if validation fails
        """
        if not self._order_manager:
            logger.error("Order manager not initialized")
            return None

        # Validation
        if not self.is_plugin_online():
            logger.warning("Cannot submit order: plugin offline")
            return None

        if self.get_free_slots() < 1:
            logger.warning("Cannot submit order: no free GE slots")
            return None

        total_cost = quantity * price
        gold = self.get_gold()
        if total_cost > gold:
            logger.warning(
                f"Cannot submit order: insufficient gold "
                f"(need {total_cost:,}, have {gold:,})"
            )
            return None

        # Validate quantity/price
        if quantity < 1:
            logger.warning("Cannot submit order: quantity must be >= 1")
            return None
        if price < 1:
            logger.warning("Cannot submit order: price must be >= 1")
            return None

        return self._order_manager.create_buy_order(
            item_id=item_id,
            item_name=item_name,
            quantity=quantity,
            price=price,
            confidence=confidence,
            strategy=strategy
        )

    def submit_sell_order(
        self,
        item_id: int,
        item_name: str,
        quantity: int,
        price: int,
        confidence: float = 0.0,
        strategy: str = "ppo_v2"
    ) -> Optional[str]:
        """
        Submit a sell order.

        Validates:
        - Plugin is online
        - Free GE slots available
        - Sufficient item quantity (inventory + bank)

        Args:
            item_id: Item ID to sell
            item_name: Item name
            quantity: Quantity to sell
            price: Price per item
            confidence: Model confidence (0-1)
            strategy: Strategy identifier

        Returns:
            Order ID if successful, None if validation fails
        """
        if not self._order_manager:
            logger.error("Order manager not initialized")
            return None

        # Validation
        if not self.is_plugin_online():
            logger.warning("Cannot submit order: plugin offline")
            return None

        if self.get_free_slots() < 1:
            logger.warning("Cannot submit order: no free GE slots")
            return None

        available = self.get_total_quantity(item_id)
        if available < quantity:
            logger.warning(
                f"Cannot submit order: insufficient items "
                f"(need {quantity:,}, have {available:,})"
            )
            return None

        # Validate quantity/price
        if quantity < 1:
            logger.warning("Cannot submit order: quantity must be >= 1")
            return None
        if price < 1:
            logger.warning("Cannot submit order: price must be >= 1")
            return None

        return self._order_manager.create_sell_order(
            item_id=item_id,
            item_name=item_name,
            quantity=quantity,
            price=price,
            confidence=confidence,
            strategy=strategy
        )

    # =========================================================================
    # Order Management
    # =========================================================================

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get an order by ID."""
        if not self._order_manager:
            return None
        return self._order_manager.get_order(order_id)

    def get_pending_orders(self) -> List[Order]:
        """Get all pending orders."""
        if not self._order_manager:
            return []
        return self._order_manager.get_pending_orders()

    def get_active_orders(self) -> List[Order]:
        """Get all active orders (in GE)."""
        if not self._order_manager:
            return []
        return self._order_manager.get_active_orders()

    def get_active_order_count(self) -> int:
        """Get count of active orders."""
        if not self._order_manager:
            return 0
        return self._order_manager.get_active_order_count()

    def cancel_order(self, order_id: str, reason: str = "Cancelled") -> bool:
        """Cancel a pending order."""
        if not self._order_manager:
            return False
        return self._order_manager.cancel_order(order_id, reason)

    def cancel_all_pending(self, reason: str = "Bulk cancel") -> int:
        """Cancel all pending orders."""
        if not self._order_manager:
            return 0
        return self._order_manager.cancel_all_pending(reason)

    # =========================================================================
    # Utility
    # =========================================================================

    def refresh_state(self):
        """Force refresh all state from Firestore."""
        if self._state_listener:
            self._state_listener.refresh_all()
        if self._order_manager:
            self._order_manager.refresh_orders()

    def get_status(self) -> Dict[str, Any]:
        """Get bridge status summary."""
        return {
            'initialized': self._initialized,
            'plugin_online': self.is_plugin_online(),
            'gold': self.get_gold(),
            'free_slots': self.get_free_slots(),
            'holdings_count': len(self.get_holdings()),
            'pending_orders': self._order_manager.get_pending_order_count() if self._order_manager else 0,
            'active_orders': self.get_active_order_count()
        }

    def print_status(self):
        """Print current status to logger."""
        status = self.get_status()
        logger.info("=" * 50)
        logger.info("InferenceBridge V2 Status")
        logger.info("=" * 50)
        for key, value in status.items():
            if key == 'gold':
                logger.info(f"  {key}: {value:,} gp")
            else:
                logger.info(f"  {key}: {value}")
        logger.info("=" * 50)
