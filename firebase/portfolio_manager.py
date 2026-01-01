"""
Portfolio Manager - Manages the portfolio collection for PPO trading.

The portfolio tracks items acquired through PPO buy orders minus items sold.
This is the source of truth for what PPO "owns" and can sell.

Schema:
/accounts/{accountId}/portfolio/{itemId}
  - item_id: number
  - item_name: string
  - quantity: number (bought - sold)
  - avg_cost: number (weighted average purchase price)
  - total_invested: number (total GP spent acquiring)
  - location: string (inventory|bank|mixed)
  - updated_at: timestamp
  - trades: array [{ order_id, action, qty, price, timestamp }]
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any, Set
from dataclasses import dataclass

from .firebase_client import FirebaseClient

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.firebase_config import (
    SUBCOLLECTION_PORTFOLIO,
    FIELD_ITEM_ID, FIELD_ITEM_NAME, FIELD_QUANTITY,
    FIELD_AVG_COST, FIELD_TOTAL_INVESTED, FIELD_LOCATION,
    FIELD_UPDATED_AT, FIELD_CREATED_AT, FIELD_TRADES,
    FIELD_ORDER_ID, FIELD_ACTION, FIELD_PRICE, FIELD_TAX_PAID,
    LOCATION_INVENTORY, LOCATION_BANK, LOCATION_MIXED,
    ACTION_BUY, ACTION_SELL, GE_TAX_RATE
)

logger = logging.getLogger(__name__)


@dataclass
class PortfolioItem:
    """Represents a single item in the portfolio."""
    item_id: int
    item_name: str
    quantity: int
    avg_cost: float
    total_invested: int
    location: str
    updated_at: str
    trades: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.trades is None:
            self.trades = []

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PortfolioItem":
        """Create a PortfolioItem from a Firestore document."""
        return cls(
            item_id=data.get(FIELD_ITEM_ID, 0),
            item_name=data.get(FIELD_ITEM_NAME, "Unknown"),
            quantity=data.get(FIELD_QUANTITY, 0),
            avg_cost=data.get(FIELD_AVG_COST, 0.0),
            total_invested=data.get(FIELD_TOTAL_INVESTED, 0),
            location=data.get(FIELD_LOCATION, "unknown"),
            updated_at=data.get(FIELD_UPDATED_AT, ""),
            trades=data.get(FIELD_TRADES, [])
        )

    def get_unrealized_pnl(self, current_price: int) -> int:
        """Calculate unrealized P&L at current price."""
        return int((current_price - self.avg_cost) * self.quantity)

    def get_unrealized_pnl_percent(self, current_price: int) -> float:
        """Calculate unrealized P&L percentage."""
        if self.avg_cost <= 0:
            return 0.0
        return (current_price - self.avg_cost) / self.avg_cost * 100


@dataclass
class PortfolioDiscrepancy:
    """Represents a discrepancy between portfolio and actual holdings."""
    item_id: int
    item_name: str
    expected_quantity: int
    actual_quantity: int

    @property
    def difference(self) -> int:
        return self.actual_quantity - self.expected_quantity


class PortfolioManager:
    """
    Manages the portfolio collection in Firebase.

    Portfolio tracks items acquired through PPO buy orders minus items sold.
    Each portfolio document represents one item type and tracks:
    - Total quantity owned
    - Average cost basis
    - Location (inventory, bank, or mixed)
    - Trade history for that item
    """

    def __init__(self, client: Optional[FirebaseClient] = None):
        """Initialize with Firebase client."""
        self.client = client or FirebaseClient()
        self._cache: Dict[int, PortfolioItem] = {}
        self._cache_time: float = 0
        self._cache_ttl: float = 5.0  # 5 second cache

    def _get_portfolio_collection(self):
        """Get the portfolio collection reference."""
        return self.client.get_account_ref().collection(SUBCOLLECTION_PORTFOLIO)

    def _now_iso(self) -> str:
        """Get current UTC timestamp in ISO format."""
        return datetime.now(timezone.utc).isoformat()

    # =========================================================================
    # Portfolio Queries
    # =========================================================================

    def get_portfolio_quantity(self, item_id: int) -> int:
        """Get the quantity of an item in the portfolio. Returns 0 if not found."""
        self._refresh_cache_if_needed()
        item = self._cache.get(item_id)
        return item.quantity if item else 0

    def is_in_portfolio(self, item_id: int) -> bool:
        """Check if an item is in the portfolio."""
        return self.get_portfolio_quantity(item_id) > 0

    def can_sell(self, item_id: int, quantity: int) -> bool:
        """Check if we can sell a specific quantity of an item."""
        return self.get_portfolio_quantity(item_id) >= quantity

    def get_portfolio_item(self, item_id: int) -> Optional[PortfolioItem]:
        """Get a portfolio item by ID."""
        self._refresh_cache_if_needed()
        return self._cache.get(item_id)

    def get_all_portfolio_items(self) -> Dict[int, PortfolioItem]:
        """Get all portfolio items."""
        self._refresh_cache_if_needed()
        return dict(self._cache)

    def get_portfolio_item_ids(self) -> Set[int]:
        """Get set of all item IDs in the portfolio."""
        self._refresh_cache_if_needed()
        return set(self._cache.keys())

    def get_total_portfolio_value(self, price_lookup: Dict[int, int]) -> int:
        """Get total portfolio value at given prices."""
        self._refresh_cache_if_needed()
        total = 0
        for item in self._cache.values():
            price = price_lookup.get(item.item_id, 0)
            total += price * item.quantity
        return total

    def get_total_invested(self) -> int:
        """Get total invested (cost basis) across all positions."""
        self._refresh_cache_if_needed()
        return sum(item.total_invested for item in self._cache.values())

    # =========================================================================
    # Portfolio Updates (typically done by plugin, but can be used for testing)
    # =========================================================================

    def add_to_portfolio(
        self,
        item_id: int,
        item_name: str,
        quantity: int,
        price_per_item: int,
        order_id: str
    ) -> bool:
        """
        Add items to the portfolio when a buy order completes.

        Args:
            item_id: The item ID
            item_name: The item name
            quantity: Quantity bought
            price_per_item: Price paid per item
            order_id: The order that caused this addition

        Returns:
            True if successful
        """
        try:
            doc_ref = self._get_portfolio_collection().document(str(item_id))
            doc = doc_ref.get()

            now = self._now_iso()
            total_cost = quantity * price_per_item

            trade_record = {
                FIELD_ORDER_ID: order_id,
                FIELD_ACTION: ACTION_BUY,
                FIELD_QUANTITY: quantity,
                FIELD_PRICE: price_per_item,
                "timestamp": now
            }

            if doc.exists:
                # Update existing position
                data = doc.to_dict()
                existing_qty = data.get(FIELD_QUANTITY, 0)
                existing_avg_cost = data.get(FIELD_AVG_COST, 0.0)
                existing_invested = data.get(FIELD_TOTAL_INVESTED, 0)

                # Calculate new weighted average cost
                new_qty = existing_qty + quantity
                new_invested = existing_invested + total_cost
                new_avg_cost = new_invested / new_qty if new_qty > 0 else 0

                # Get existing trades list
                trades = list(data.get(FIELD_TRADES, []))
                trades.append(trade_record)

                doc_ref.update({
                    FIELD_QUANTITY: new_qty,
                    FIELD_AVG_COST: new_avg_cost,
                    FIELD_TOTAL_INVESTED: new_invested,
                    FIELD_UPDATED_AT: now,
                    FIELD_TRADES: trades
                })

                logger.info(f"Updated portfolio: {new_qty}x {item_name} (avg cost: {int(new_avg_cost)})")

            else:
                # Create new position
                doc_ref.set({
                    FIELD_ITEM_ID: item_id,
                    FIELD_ITEM_NAME: item_name,
                    FIELD_QUANTITY: quantity,
                    FIELD_AVG_COST: float(price_per_item),
                    FIELD_TOTAL_INVESTED: total_cost,
                    FIELD_LOCATION: LOCATION_INVENTORY,
                    FIELD_CREATED_AT: now,
                    FIELD_UPDATED_AT: now,
                    FIELD_TRADES: [trade_record]
                })

                logger.info(f"Added to portfolio: {quantity}x {item_name} @ {price_per_item}")

            # Update cache
            self._update_cache_item(item_id)
            return True

        except Exception as e:
            logger.error(f"Failed to add to portfolio: {quantity}x {item_name}: {e}")
            return False

    def remove_from_portfolio(
        self,
        item_id: int,
        quantity: int,
        sale_price: int,
        order_id: str,
        tax_paid: int = 0
    ) -> bool:
        """
        Remove items from the portfolio when a sell order completes.

        Args:
            item_id: The item ID
            quantity: Quantity sold
            sale_price: Price received per item
            order_id: The order that caused this removal
            tax_paid: Tax paid on the sale

        Returns:
            True if successful
        """
        try:
            doc_ref = self._get_portfolio_collection().document(str(item_id))
            doc = doc_ref.get()

            if not doc.exists:
                logger.warning(f"Cannot remove from portfolio: item {item_id} not found")
                return False

            data = doc.to_dict()
            item_name = data.get(FIELD_ITEM_NAME, "Unknown")
            existing_qty = data.get(FIELD_QUANTITY, 0)

            if quantity > existing_qty:
                logger.warning(f"Trying to sell {quantity} but only have {existing_qty} of item {item_id}")
                quantity = existing_qty

            now = self._now_iso()

            trade_record = {
                FIELD_ORDER_ID: order_id,
                FIELD_ACTION: ACTION_SELL,
                FIELD_QUANTITY: quantity,
                FIELD_PRICE: sale_price,
                FIELD_TAX_PAID: tax_paid,
                "timestamp": now
            }

            new_qty = existing_qty - quantity

            if new_qty <= 0:
                # Position fully closed - delete document
                doc_ref.delete()
                logger.info(f"Closed portfolio position: {item_name} (sold all {existing_qty})")
            else:
                # Update remaining position
                existing_invested = data.get(FIELD_TOTAL_INVESTED, 0)
                cost_basis_sold = int(existing_invested * quantity / existing_qty)
                new_invested = existing_invested - cost_basis_sold

                trades = list(data.get(FIELD_TRADES, []))
                trades.append(trade_record)

                doc_ref.update({
                    FIELD_QUANTITY: new_qty,
                    FIELD_TOTAL_INVESTED: new_invested,
                    FIELD_UPDATED_AT: now,
                    FIELD_TRADES: trades
                })

                logger.info(f"Reduced portfolio: {new_qty} remaining {item_name} (sold {quantity})")

            # Update cache
            self._update_cache_item(item_id)
            return True

        except Exception as e:
            logger.error(f"Failed to remove from portfolio: {quantity} of item {item_id}: {e}")
            return False

    def update_location(self, item_id: int, location: str) -> bool:
        """Update the location field for a portfolio item."""
        try:
            doc_ref = self._get_portfolio_collection().document(str(item_id))
            doc_ref.update({
                FIELD_LOCATION: location,
                FIELD_UPDATED_AT: self._now_iso()
            })
            logger.debug(f"Updated location for item {item_id}: {location}")
            return True
        except Exception as e:
            logger.error(f"Failed to update location for item {item_id}: {e}")
            return False

    # =========================================================================
    # Portfolio Sync (reconcile with actual inventory/bank)
    # =========================================================================

    def sync_locations(
        self,
        inventory_items: Dict[int, int],
        bank_items: Dict[int, int]
    ) -> None:
        """
        Sync portfolio locations based on current inventory and bank state.

        Args:
            inventory_items: Map of itemId to quantity in inventory
            bank_items: Map of itemId to quantity in bank
        """
        self._refresh_cache_if_needed()

        for item in self._cache.values():
            in_inventory = inventory_items.get(item.item_id, 0)
            in_bank = bank_items.get(item.item_id, 0)

            if in_inventory > 0 and in_bank > 0:
                new_location = LOCATION_MIXED
            elif in_inventory > 0:
                new_location = LOCATION_INVENTORY
            elif in_bank > 0:
                new_location = LOCATION_BANK
            else:
                new_location = "unknown"

            if new_location != item.location:
                self.update_location(item.item_id, new_location)

    def verify_portfolio(
        self,
        inventory_items: Dict[int, int],
        bank_items: Dict[int, int]
    ) -> List[PortfolioDiscrepancy]:
        """
        Verify portfolio quantities match actual holdings.

        Args:
            inventory_items: Map of itemId to quantity in inventory
            bank_items: Map of itemId to quantity in bank

        Returns:
            List of discrepancies found
        """
        self._refresh_cache_if_needed()

        discrepancies = []
        for item in self._cache.values():
            actual_qty = inventory_items.get(item.item_id, 0) + bank_items.get(item.item_id, 0)

            if actual_qty != item.quantity:
                discrepancies.append(PortfolioDiscrepancy(
                    item_id=item.item_id,
                    item_name=item.item_name,
                    expected_quantity=item.quantity,
                    actual_quantity=actual_qty
                ))

        return discrepancies

    # =========================================================================
    # Cache Management
    # =========================================================================

    def _refresh_cache_if_needed(self) -> None:
        """Refresh the cache if it's stale."""
        import time
        now = time.time()
        if now - self._cache_time > self._cache_ttl:
            self.refresh_cache()

    def refresh_cache(self) -> None:
        """Force refresh the entire cache from Firebase."""
        import time
        try:
            self._cache.clear()

            docs = self._get_portfolio_collection().stream()
            for doc in docs:
                data = doc.to_dict()
                if data:
                    item = PortfolioItem.from_dict(data)
                    self._cache[item.item_id] = item

            self._cache_time = time.time()
            logger.debug(f"Refreshed portfolio cache: {len(self._cache)} items")

        except Exception as e:
            logger.error(f"Failed to refresh portfolio cache: {e}")

    def _update_cache_item(self, item_id: int) -> None:
        """Update cache for a single item from Firebase."""
        try:
            doc = self._get_portfolio_collection().document(str(item_id)).get()

            if doc.exists:
                item = PortfolioItem.from_dict(doc.to_dict())
                self._cache[item_id] = item
            else:
                self._cache.pop(item_id, None)

        except Exception as e:
            logger.error(f"Failed to update cache for item {item_id}: {e}")

    # =========================================================================
    # Summary Methods
    # =========================================================================

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get a summary of the portfolio state."""
        self._refresh_cache_if_needed()

        items = list(self._cache.values())
        total_invested = sum(item.total_invested for item in items)
        total_items = sum(item.quantity for item in items)

        return {
            "item_count": len(items),
            "total_items": total_items,
            "total_invested": total_invested,
            "items": [
                {
                    "item_id": item.item_id,
                    "item_name": item.item_name,
                    "quantity": item.quantity,
                    "avg_cost": item.avg_cost,
                    "total_invested": item.total_invested,
                    "location": item.location
                }
                for item in items
            ]
        }
