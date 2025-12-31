"""
State Listener V2 - Real-time Firestore state updates.

Listens to portfolio, inventory, bank, and GE state documents
using Firestore's onSnapshot for real-time updates.
"""

import logging
import threading
from datetime import datetime, timezone
from typing import Callable, Optional, Dict, Any

from .firebase_client import FirebaseClientV2
from .types import (
    Portfolio, InventoryState, BankState, GEState, AccountState,
    Holding
)

logger = logging.getLogger(__name__)


class StateListener:
    """
    Real-time listener for account state from Firestore.

    Maintains cached state that updates automatically via Firestore
    snapshot listeners. Provides synchronous access to the latest state.
    """

    def __init__(self, client: Optional[FirebaseClientV2] = None):
        """
        Initialize the state listener.

        Args:
            client: FirebaseClientV2 instance (uses singleton if not provided)
        """
        self._client = client or FirebaseClientV2.get_instance()
        self._lock = threading.RLock()

        # Cached state
        self._portfolio: Optional[Portfolio] = None
        self._inventory: Optional[InventoryState] = None
        self._bank: Optional[BankState] = None
        self._ge_state: Optional[GEState] = None

        # Listener handles for cleanup
        self._portfolio_listener = None
        self._inventory_listener = None
        self._bank_listener = None
        self._ge_state_listener = None

        # Callbacks
        self._on_portfolio_change: Optional[Callable[[Portfolio], None]] = None
        self._on_inventory_change: Optional[Callable[[InventoryState], None]] = None
        self._on_bank_change: Optional[Callable[[BankState], None]] = None
        self._on_ge_state_change: Optional[Callable[[GEState], None]] = None

        self._running = False

    def start(
        self,
        on_portfolio_change: Optional[Callable[[Portfolio], None]] = None,
        on_inventory_change: Optional[Callable[[InventoryState], None]] = None,
        on_bank_change: Optional[Callable[[BankState], None]] = None,
        on_ge_state_change: Optional[Callable[[GEState], None]] = None
    ):
        """
        Start listening to all state documents.

        Args:
            on_portfolio_change: Callback when portfolio updates
            on_inventory_change: Callback when inventory updates
            on_bank_change: Callback when bank updates
            on_ge_state_change: Callback when GE state updates
        """
        if self._running:
            logger.warning("StateListener already running")
            return

        self._on_portfolio_change = on_portfolio_change
        self._on_inventory_change = on_inventory_change
        self._on_bank_change = on_bank_change
        self._on_ge_state_change = on_ge_state_change

        # Start listeners
        self._start_portfolio_listener()
        self._start_inventory_listener()
        self._start_bank_listener()
        self._start_ge_state_listener()

        self._running = True
        logger.info("StateListener started")

    def stop(self):
        """Stop all listeners."""
        if not self._running:
            return

        # Unsubscribe from all listeners
        if self._portfolio_listener:
            self._portfolio_listener.unsubscribe()
            self._portfolio_listener = None

        if self._inventory_listener:
            self._inventory_listener.unsubscribe()
            self._inventory_listener = None

        if self._bank_listener:
            self._bank_listener.unsubscribe()
            self._bank_listener = None

        if self._ge_state_listener:
            self._ge_state_listener.unsubscribe()
            self._ge_state_listener = None

        self._running = False
        logger.info("StateListener stopped")

    # =========================================================================
    # Listener Setup
    # =========================================================================

    def _start_portfolio_listener(self):
        """Start listening to portfolio document."""
        def on_snapshot(doc_snapshot, changes, read_time):
            try:
                for doc in doc_snapshot:
                    if doc.exists:
                        data = doc.to_dict()
                        with self._lock:
                            self._portfolio = Portfolio.from_dict(data)
                        logger.debug(f"Portfolio updated: gold={self._portfolio.gold}")
                        if self._on_portfolio_change:
                            self._on_portfolio_change(self._portfolio)
            except Exception as e:
                logger.error(f"Error processing portfolio snapshot: {e}")

        self._portfolio_listener = self._client.portfolio_ref.on_snapshot(on_snapshot)

    def _start_inventory_listener(self):
        """Start listening to inventory document."""
        def on_snapshot(doc_snapshot, changes, read_time):
            try:
                for doc in doc_snapshot:
                    if doc.exists:
                        data = doc.to_dict()
                        with self._lock:
                            self._inventory = InventoryState.from_dict(data)
                        logger.debug(f"Inventory updated: {len(self._inventory.items)} items")
                        if self._on_inventory_change:
                            self._on_inventory_change(self._inventory)
            except Exception as e:
                logger.error(f"Error processing inventory snapshot: {e}")

        self._inventory_listener = self._client.inventory_ref.on_snapshot(on_snapshot)

    def _start_bank_listener(self):
        """Start listening to bank document."""
        def on_snapshot(doc_snapshot, changes, read_time):
            try:
                for doc in doc_snapshot:
                    if doc.exists:
                        data = doc.to_dict()
                        with self._lock:
                            self._bank = BankState.from_dict(data)
                        logger.debug(f"Bank updated: {len(self._bank.items)} items")
                        if self._on_bank_change:
                            self._on_bank_change(self._bank)
            except Exception as e:
                logger.error(f"Error processing bank snapshot: {e}")

        self._bank_listener = self._client.bank_ref.on_snapshot(on_snapshot)

    def _start_ge_state_listener(self):
        """Start listening to ge_state document."""
        def on_snapshot(doc_snapshot, changes, read_time):
            try:
                for doc in doc_snapshot:
                    if doc.exists:
                        data = doc.to_dict()
                        with self._lock:
                            self._ge_state = GEState.from_dict(data)
                        logger.debug(f"GE state updated: {self._ge_state.free_slots} free slots")
                        if self._on_ge_state_change:
                            self._on_ge_state_change(self._ge_state)
            except Exception as e:
                logger.error(f"Error processing GE state snapshot: {e}")

        self._ge_state_listener = self._client.ge_state_ref.on_snapshot(on_snapshot)

    # =========================================================================
    # State Access (Thread-safe)
    # =========================================================================

    @property
    def portfolio(self) -> Optional[Portfolio]:
        """Get current portfolio state."""
        with self._lock:
            return self._portfolio

    @property
    def inventory(self) -> Optional[InventoryState]:
        """Get current inventory state."""
        with self._lock:
            return self._inventory

    @property
    def bank(self) -> Optional[BankState]:
        """Get current bank state."""
        with self._lock:
            return self._bank

    @property
    def ge_state(self) -> Optional[GEState]:
        """Get current GE state."""
        with self._lock:
            return self._ge_state

    @property
    def gold(self) -> int:
        """Get current gold amount."""
        with self._lock:
            if self._portfolio:
                return self._portfolio.gold
            return 0

    @property
    def free_slots(self) -> int:
        """Get number of free GE slots."""
        with self._lock:
            if self._ge_state:
                return self._ge_state.free_slots
            return 8  # Assume all free if unknown

    @property
    def plugin_online(self) -> bool:
        """Check if plugin is online (based on last heartbeat)."""
        with self._lock:
            if not self._portfolio:
                return False
            if not self._portfolio.plugin_online:
                return False
            # Also check timestamp isn't too old
            if self._portfolio.last_updated:
                now = datetime.now(timezone.utc)
                if hasattr(self._portfolio.last_updated, 'timestamp'):
                    # Firestore timestamp
                    last = datetime.fromtimestamp(
                        self._portfolio.last_updated.timestamp(),
                        tz=timezone.utc
                    )
                else:
                    last = self._portfolio.last_updated
                age = (now - last).total_seconds()
                return age < 120  # 2 minutes
            return self._portfolio.plugin_online

    def get_account_state(self) -> Optional[AccountState]:
        """Get complete account state snapshot."""
        with self._lock:
            if not all([self._portfolio, self._inventory, self._bank, self._ge_state]):
                return None
            return AccountState(
                account_id=self._client.account_id,
                portfolio=self._portfolio,
                inventory=self._inventory,
                bank=self._bank,
                ge_state=self._ge_state
            )

    def get_holdings(self) -> Dict[int, Holding]:
        """Get all holdings (inventory + bank combined)."""
        holdings: Dict[int, Holding] = {}

        with self._lock:
            # Add inventory items
            if self._inventory:
                for item_id_str, holding in self._inventory.items.items():
                    item_id = int(item_id_str)
                    holdings[item_id] = Holding(
                        item_id=item_id,
                        name=holding.name,
                        quantity=holding.quantity
                    )

            # Add/merge bank items
            if self._bank:
                for item_id_str, holding in self._bank.items.items():
                    item_id = int(item_id_str)
                    if item_id in holdings:
                        # Create new Holding with combined quantity
                        existing = holdings[item_id]
                        holdings[item_id] = Holding(
                            item_id=item_id,
                            name=existing.name or holding.name,
                            quantity=existing.quantity + holding.quantity
                        )
                    else:
                        holdings[item_id] = Holding(
                            item_id=item_id,
                            name=holding.name,
                            quantity=holding.quantity
                        )

        return holdings

    def get_inventory_quantity(self, item_id: int) -> int:
        """Get quantity of an item in inventory."""
        with self._lock:
            if not self._inventory:
                return 0
            holding = self._inventory.items.get(str(item_id))
            return holding.quantity if holding else 0

    def get_bank_quantity(self, item_id: int) -> int:
        """Get quantity of an item in bank."""
        with self._lock:
            if not self._bank:
                return 0
            holding = self._bank.items.get(str(item_id))
            return holding.quantity if holding else 0

    def get_total_quantity(self, item_id: int) -> int:
        """Get total quantity of an item (inventory + bank)."""
        return self.get_inventory_quantity(item_id) + self.get_bank_quantity(item_id)

    # =========================================================================
    # Manual Refresh
    # =========================================================================

    def refresh_all(self):
        """Manually refresh all state from Firestore."""
        try:
            # Portfolio
            doc = self._client.portfolio_ref.get()
            if doc.exists:
                with self._lock:
                    self._portfolio = Portfolio.from_dict(doc.to_dict())

            # Inventory
            doc = self._client.inventory_ref.get()
            if doc.exists:
                with self._lock:
                    self._inventory = InventoryState.from_dict(doc.to_dict())

            # Bank
            doc = self._client.bank_ref.get()
            if doc.exists:
                with self._lock:
                    self._bank = BankState.from_dict(doc.to_dict())

            # GE State
            doc = self._client.ge_state_ref.get()
            if doc.exists:
                with self._lock:
                    self._ge_state = GEState.from_dict(doc.to_dict())

            logger.info("State refreshed from Firestore")

        except Exception as e:
            logger.error(f"Error refreshing state: {e}")
            raise

    def wait_for_state(self, timeout: float = 10.0) -> bool:
        """
        Wait for initial state to be populated.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if state is available, False if timeout
        """
        import time
        start = time.time()

        while time.time() - start < timeout:
            with self._lock:
                if all([self._portfolio, self._inventory, self._bank, self._ge_state]):
                    return True
            time.sleep(0.1)

        return False
