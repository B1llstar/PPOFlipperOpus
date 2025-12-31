"""
Type definitions for Firebase V2.

These dataclasses match the Firestore document schemas defined in GE_AUTO_V2_PROPOSAL.md.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, List


class OrderAction(str, Enum):
    """Order action type."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    """Order status lifecycle."""
    PENDING = "pending"      # Created by inference, waiting for plugin
    RECEIVED = "received"    # Plugin has received the order
    PLACED = "placed"        # Order placed in GE slot
    PARTIAL = "partial"      # Partially filled
    COMPLETED = "completed"  # Fully filled and collected
    CANCELLED = "cancelled"  # Cancelled
    FAILED = "failed"        # Failed to execute


class GESlotStatus(str, Enum):
    """GE slot status."""
    EMPTY = "empty"
    ACTIVE = "active"
    COMPLETE = "complete"


@dataclass
class Item:
    """Item from the items collection."""
    id: int
    name: str
    members: bool = True
    limit: int = 0
    high_alch: int = 0
    low_alch: int = 0
    tradeable: bool = True
    stackable: bool = True


@dataclass
class Holding:
    """An item holding (in bank or inventory)."""
    item_id: int
    name: str
    quantity: int

    @property
    def total_value(self) -> int:
        """Calculate total value (needs price lookup)."""
        return 0  # Placeholder - needs market price


@dataclass
class InventoryState:
    """Current inventory state from Firestore."""
    items: Dict[str, Holding]  # item_id -> Holding
    empty_slots: int
    scanned_at: Optional[datetime] = None

    @classmethod
    def from_dict(cls, data: dict) -> 'InventoryState':
        """Create from Firestore document."""
        items = {}
        for item_id, item_data in data.get('items', {}).items():
            items[item_id] = Holding(
                item_id=int(item_id),
                name=item_data.get('name', ''),
                quantity=item_data.get('quantity', 0)
            )
        return cls(
            items=items,
            empty_slots=data.get('empty_slots', 28),
            scanned_at=data.get('scanned_at')
        )


@dataclass
class BankState:
    """Current bank state from Firestore."""
    items: Dict[str, Holding]  # item_id -> Holding
    total_items: int
    scanned_at: Optional[datetime] = None

    @classmethod
    def from_dict(cls, data: dict) -> 'BankState':
        """Create from Firestore document."""
        items = {}
        for item_id, item_data in data.get('items', {}).items():
            items[item_id] = Holding(
                item_id=int(item_id),
                name=item_data.get('name', ''),
                quantity=item_data.get('quantity', 0)
            )
        return cls(
            items=items,
            total_items=data.get('total_items', 0),
            scanned_at=data.get('scanned_at')
        )


@dataclass
class GESlot:
    """State of a single GE slot."""
    slot_number: int
    status: GESlotStatus
    type: Optional[str] = None  # "buy" or "sell"
    item_id: Optional[int] = None
    item_name: Optional[str] = None
    quantity: int = 0
    price: int = 0
    filled: int = 0
    order_id: Optional[str] = None  # Links to our order if we placed it

    @property
    def is_empty(self) -> bool:
        return self.status == GESlotStatus.EMPTY

    @property
    def is_active(self) -> bool:
        return self.status == GESlotStatus.ACTIVE

    @property
    def is_complete(self) -> bool:
        return self.status == GESlotStatus.COMPLETE

    @property
    def fill_percentage(self) -> float:
        if self.quantity == 0:
            return 0.0
        return (self.filled / self.quantity) * 100

    @classmethod
    def from_dict(cls, slot_number: int, data: dict) -> 'GESlot':
        """Create from Firestore slot data."""
        status_str = data.get('status', 'empty')
        try:
            status = GESlotStatus(status_str)
        except ValueError:
            status = GESlotStatus.EMPTY

        return cls(
            slot_number=slot_number,
            status=status,
            type=data.get('type'),
            item_id=data.get('item_id'),
            item_name=data.get('item_name'),
            quantity=data.get('quantity', 0),
            price=data.get('price', 0),
            filled=data.get('filled', 0),
            order_id=data.get('order_id')
        )


@dataclass
class GEState:
    """Current state of all 8 GE slots."""
    slots: Dict[int, GESlot]  # slot_number (1-8) -> GESlot
    free_slots: int
    synced_at: Optional[datetime] = None

    @property
    def active_slots(self) -> List[GESlot]:
        """Get all active (non-empty) slots."""
        return [s for s in self.slots.values() if not s.is_empty]

    @property
    def empty_slot_numbers(self) -> List[int]:
        """Get numbers of empty slots."""
        return [s.slot_number for s in self.slots.values() if s.is_empty]

    @property
    def buy_slots(self) -> List[GESlot]:
        """Get all buy order slots."""
        return [s for s in self.slots.values() if s.type == 'buy']

    @property
    def sell_slots(self) -> List[GESlot]:
        """Get all sell order slots."""
        return [s for s in self.slots.values() if s.type == 'sell']

    @classmethod
    def from_dict(cls, data: dict) -> 'GEState':
        """Create from Firestore document."""
        slots = {}
        slots_data = data.get('slots', {})
        for slot_num in range(1, 9):
            slot_key = str(slot_num)
            if slot_key in slots_data:
                slots[slot_num] = GESlot.from_dict(slot_num, slots_data[slot_key])
            else:
                slots[slot_num] = GESlot(slot_number=slot_num, status=GESlotStatus.EMPTY)

        return cls(
            slots=slots,
            free_slots=data.get('free_slots', 8),
            synced_at=data.get('synced_at')
        )


@dataclass
class Portfolio:
    """Portfolio summary from Firestore."""
    gold: int
    total_value: int
    holdings_count: int
    active_order_count: int
    last_updated: Optional[datetime] = None
    plugin_online: bool = False
    plugin_version: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> 'Portfolio':
        """Create from Firestore document."""
        return cls(
            gold=data.get('gold', 0),
            total_value=data.get('total_value', 0),
            holdings_count=data.get('holdings_count', 0),
            active_order_count=data.get('active_order_count', 0),
            last_updated=data.get('last_updated'),
            plugin_online=data.get('plugin_online', False),
            plugin_version=data.get('plugin_version', '')
        )


@dataclass
class Order:
    """An order from the orders subcollection."""
    order_id: str
    action: OrderAction
    item_id: int
    item_name: str
    quantity: int
    price: int
    status: OrderStatus
    ge_slot: Optional[int] = None
    filled_quantity: int = 0
    total_cost: int = 0
    created_at: Optional[datetime] = None
    received_at: Optional[datetime] = None
    placed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    retry_count: int = 0
    confidence: float = 0.0
    strategy: str = "ppo_v2"

    @property
    def is_buy(self) -> bool:
        return self.action == OrderAction.BUY

    @property
    def is_sell(self) -> bool:
        return self.action == OrderAction.SELL

    @property
    def is_pending(self) -> bool:
        return self.status in [OrderStatus.PENDING, OrderStatus.RECEIVED]

    @property
    def is_active(self) -> bool:
        return self.status in [OrderStatus.PLACED, OrderStatus.PARTIAL]

    @property
    def is_complete(self) -> bool:
        return self.status == OrderStatus.COMPLETED

    @property
    def is_terminal(self) -> bool:
        return self.status in [OrderStatus.COMPLETED, OrderStatus.CANCELLED, OrderStatus.FAILED]

    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity

    @property
    def fill_percentage(self) -> float:
        if self.quantity == 0:
            return 0.0
        return (self.filled_quantity / self.quantity) * 100

    @property
    def expected_cost(self) -> int:
        return self.quantity * self.price

    @classmethod
    def from_dict(cls, data: dict) -> 'Order':
        """Create from Firestore document."""
        action_str = data.get('action', 'buy')
        status_str = data.get('status', 'pending')

        try:
            action = OrderAction(action_str)
        except ValueError:
            action = OrderAction.BUY

        try:
            status = OrderStatus(status_str)
        except ValueError:
            status = OrderStatus.PENDING

        return cls(
            order_id=data.get('order_id', ''),
            action=action,
            item_id=data.get('item_id', 0),
            item_name=data.get('item_name', ''),
            quantity=data.get('quantity', 0),
            price=data.get('price', 0),
            status=status,
            ge_slot=data.get('ge_slot'),
            filled_quantity=data.get('filled_quantity', 0),
            total_cost=data.get('total_cost', 0),
            created_at=data.get('created_at'),
            received_at=data.get('received_at'),
            placed_at=data.get('placed_at'),
            completed_at=data.get('completed_at'),
            error=data.get('error'),
            retry_count=data.get('retry_count', 0),
            confidence=data.get('confidence', 0.0),
            strategy=data.get('strategy', 'ppo_v2')
        )

    def to_dict(self) -> dict:
        """Convert to Firestore document."""
        return {
            'order_id': self.order_id,
            'action': self.action.value,
            'item_id': self.item_id,
            'item_name': self.item_name,
            'quantity': self.quantity,
            'price': self.price,
            'status': self.status.value,
            'ge_slot': self.ge_slot,
            'filled_quantity': self.filled_quantity,
            'total_cost': self.total_cost,
            'created_at': self.created_at,
            'received_at': self.received_at,
            'placed_at': self.placed_at,
            'completed_at': self.completed_at,
            'error': self.error,
            'retry_count': self.retry_count,
            'confidence': self.confidence,
            'strategy': self.strategy
        }


@dataclass
class AccountState:
    """Complete account state aggregated from all documents."""
    account_id: str
    portfolio: Portfolio
    inventory: InventoryState
    bank: BankState
    ge_state: GEState

    @property
    def gold(self) -> int:
        return self.portfolio.gold

    @property
    def total_value(self) -> int:
        return self.portfolio.total_value

    @property
    def free_ge_slots(self) -> int:
        return self.ge_state.free_slots

    @property
    def plugin_online(self) -> bool:
        return self.portfolio.plugin_online

    def get_holding(self, item_id: int) -> Optional[Holding]:
        """Get combined holding from inventory + bank."""
        item_id_str = str(item_id)
        inv_holding = self.inventory.items.get(item_id_str)
        bank_holding = self.bank.items.get(item_id_str)

        if not inv_holding and not bank_holding:
            return None

        total_qty = 0
        name = ""

        if inv_holding:
            total_qty += inv_holding.quantity
            name = inv_holding.name
        if bank_holding:
            total_qty += bank_holding.quantity
            if not name:
                name = bank_holding.name

        return Holding(item_id=item_id, name=name, quantity=total_qty)

    def get_all_holdings(self) -> Dict[int, Holding]:
        """Get all holdings (inventory + bank combined)."""
        holdings: Dict[int, Holding] = {}

        # Add inventory items
        for item_id_str, holding in self.inventory.items.items():
            item_id = int(item_id_str)
            holdings[item_id] = Holding(
                item_id=item_id,
                name=holding.name,
                quantity=holding.quantity
            )

        # Add/merge bank items
        for item_id_str, holding in self.bank.items.items():
            item_id = int(item_id_str)
            if item_id in holdings:
                holdings[item_id].quantity += holding.quantity
            else:
                holdings[item_id] = Holding(
                    item_id=item_id,
                    name=holding.name,
                    quantity=holding.quantity
                )

        return holdings
