"""
Position Tracker - Tracks positions acquired during PPO inference.

Only items in the active positions list are fair game for trading.
Pre-existing bank items are ignored unless manually marked as active.

Firestore structure:
    /accounts/{id}/positions/active
    {
        "items": {
            "2": {  # item_id as string key
                "item_id": 2,
                "item_name": "Cannonball",
                "quantity": 10000,           # Total qty PPO owns
                "avg_cost": 150,             # Average cost basis
                "total_invested": 1500000,   # Total GP invested
                "first_acquired": "...",     # When first bought
                "last_updated": "...",
                "source": "ppo",             # "ppo" or "manual"
                "locked": false              # If true, won't be sold
            }
        },
        "updated_at": "..."
    }
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from .firebase_client import FirebaseClient

logger = logging.getLogger(__name__)


class PositionTracker:
    """
    Tracks active positions that PPO inference can trade.

    Items not in active positions are off-limits for selling.
    """

    POSITIONS_DOC = "active"
    POSITIONS_COLLECTION = "positions"

    def __init__(self, client: Optional[FirebaseClient] = None, account_id: Optional[str] = None):
        """Initialize position tracker."""
        self.client = client or FirebaseClient()
        self.account_id = account_id or self.client.account_id
        self._cache: Optional[Dict[str, Any]] = None
        self._cache_time: float = 0
        self._cache_ttl: float = 5.0  # 5 second cache

    def _get_positions_ref(self):
        """Get reference to positions document."""
        return (self.client.db
                .collection("accounts")
                .document(self.account_id)
                .collection(self.POSITIONS_COLLECTION)
                .document(self.POSITIONS_DOC))

    def _now_iso(self) -> str:
        """Get current UTC timestamp."""
        return datetime.now(timezone.utc).isoformat()

    def _invalidate_cache(self):
        """Invalidate the cache."""
        self._cache = None
        self._cache_time = 0

    # =========================================================================
    # Read Operations
    # =========================================================================

    def get_active_positions(self, use_cache: bool = True) -> Dict[int, Dict[str, Any]]:
        """
        Get all active positions.

        Returns:
            Dict mapping item_id (int) to position data
        """
        import time

        if use_cache and self._cache is not None:
            if time.time() - self._cache_time < self._cache_ttl:
                return self._cache

        try:
            doc = self._get_positions_ref().get()
            if doc.exists:
                data = doc.to_dict()
                items = data.get("items", {})
                # Convert string keys to int
                result = {int(k): v for k, v in items.items()}
                self._cache = result
                self._cache_time = time.time()
                return result
            return {}
        except Exception as e:
            logger.error(f"Failed to get active positions: {e}")
            return {}

    def get_position(self, item_id: int) -> Optional[Dict[str, Any]]:
        """Get position for a specific item."""
        positions = self.get_active_positions()
        return positions.get(item_id)

    def has_position(self, item_id: int) -> bool:
        """Check if item is in active positions."""
        return item_id in self.get_active_positions()

    def get_position_quantity(self, item_id: int) -> int:
        """Get quantity of item in active positions."""
        pos = self.get_position(item_id)
        return pos.get("quantity", 0) if pos else 0

    def is_tradeable(self, item_id: int) -> bool:
        """
        Check if an item can be traded by PPO.

        Item is tradeable if:
        - It's in active positions AND
        - It's not locked
        """
        pos = self.get_position(item_id)
        if not pos:
            return False
        return not pos.get("locked", False)

    def get_tradeable_positions(self) -> Dict[int, Dict[str, Any]]:
        """Get all positions that can be traded (not locked)."""
        positions = self.get_active_positions()
        return {k: v for k, v in positions.items() if not v.get("locked", False)}

    def get_locked_positions(self) -> Dict[int, Dict[str, Any]]:
        """Get all locked positions."""
        positions = self.get_active_positions()
        return {k: v for k, v in positions.items() if v.get("locked", False)}

    # =========================================================================
    # Write Operations - Called when PPO trades
    # =========================================================================

    def add_position(
        self,
        item_id: int,
        item_name: str,
        quantity: int,
        cost_per_item: int,
        source: str = "ppo"
    ) -> bool:
        """
        Add or update a position after a buy.

        If position exists, updates quantity and recalculates avg cost.

        Args:
            item_id: Item ID
            item_name: Item name
            quantity: Quantity bought
            cost_per_item: Price paid per item
            source: "ppo" for inference buys, "manual" for user-added

        Returns:
            True if successful
        """
        try:
            now = self._now_iso()
            positions = self.get_active_positions(use_cache=False)

            existing = positions.get(item_id)

            if existing:
                # Update existing position with new avg cost
                old_qty = existing.get("quantity", 0)
                old_invested = existing.get("total_invested", 0)

                new_qty = old_qty + quantity
                new_invested = old_invested + (quantity * cost_per_item)
                new_avg_cost = new_invested // new_qty if new_qty > 0 else cost_per_item

                position_data = {
                    **existing,
                    "quantity": new_qty,
                    "avg_cost": new_avg_cost,
                    "total_invested": new_invested,
                    "last_updated": now
                }
            else:
                # New position
                position_data = {
                    "item_id": item_id,
                    "item_name": item_name,
                    "quantity": quantity,
                    "avg_cost": cost_per_item,
                    "total_invested": quantity * cost_per_item,
                    "first_acquired": now,
                    "last_updated": now,
                    "source": source,
                    "locked": False
                }

            # Update Firestore
            self._get_positions_ref().set({
                "items": {
                    **{str(k): v for k, v in positions.items()},
                    str(item_id): position_data
                },
                "updated_at": now
            }, merge=True)

            self._invalidate_cache()

            logger.info(f"Added position: {quantity}x {item_name} @ {cost_per_item} "
                       f"(total: {position_data['quantity']}, avg: {position_data['avg_cost']})")
            return True

        except Exception as e:
            logger.error(f"Failed to add position: {e}")
            return False

    def reduce_position(
        self,
        item_id: int,
        quantity: int,
        sale_price: int = 0
    ) -> bool:
        """
        Reduce a position after a sell.

        If quantity reaches 0, removes the position entirely.

        Args:
            item_id: Item ID
            quantity: Quantity sold
            sale_price: Price per item (for P&L tracking)

        Returns:
            True if successful
        """
        try:
            positions = self.get_active_positions(use_cache=False)
            existing = positions.get(item_id)

            if not existing:
                logger.warning(f"Cannot reduce position for item {item_id}: not in active positions")
                return False

            old_qty = existing.get("quantity", 0)
            new_qty = max(0, old_qty - quantity)

            now = self._now_iso()

            if new_qty == 0:
                # Remove position entirely
                del positions[item_id]
                logger.info(f"Removed position: item {item_id} (sold all {quantity})")
            else:
                # Reduce quantity, keep avg cost the same
                # Reduce total_invested proportionally
                old_invested = existing.get("total_invested", 0)
                reduction_ratio = quantity / old_qty if old_qty > 0 else 1
                new_invested = int(old_invested * (1 - reduction_ratio))

                positions[item_id] = {
                    **existing,
                    "quantity": new_qty,
                    "total_invested": new_invested,
                    "last_updated": now
                }
                logger.info(f"Reduced position: item {item_id} by {quantity} (remaining: {new_qty})")

            # Update Firestore
            self._get_positions_ref().set({
                "items": {str(k): v for k, v in positions.items()},
                "updated_at": now
            })

            self._invalidate_cache()
            return True

        except Exception as e:
            logger.error(f"Failed to reduce position: {e}")
            return False

    def remove_position(self, item_id: int) -> bool:
        """
        Remove a position entirely.

        Use this when manually removing an item from active positions.
        """
        try:
            positions = self.get_active_positions(use_cache=False)

            if item_id not in positions:
                logger.warning(f"Position {item_id} not found")
                return False

            item_name = positions[item_id].get("item_name", f"Item {item_id}")
            del positions[item_id]

            now = self._now_iso()
            self._get_positions_ref().set({
                "items": {str(k): v for k, v in positions.items()},
                "updated_at": now
            })

            self._invalidate_cache()
            logger.info(f"Removed position: {item_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to remove position: {e}")
            return False

    # =========================================================================
    # Lock/Unlock - For Vue dashboard
    # =========================================================================

    def lock_position(self, item_id: int) -> bool:
        """
        Lock a position so it won't be sold by PPO.

        Locked positions stay in active list but aren't tradeable.
        """
        try:
            positions = self.get_active_positions(use_cache=False)

            if item_id not in positions:
                logger.warning(f"Cannot lock position {item_id}: not found")
                return False

            positions[item_id]["locked"] = True
            positions[item_id]["last_updated"] = self._now_iso()

            self._get_positions_ref().set({
                "items": {str(k): v for k, v in positions.items()},
                "updated_at": self._now_iso()
            })

            self._invalidate_cache()
            logger.info(f"Locked position: {positions[item_id].get('item_name', item_id)}")
            return True

        except Exception as e:
            logger.error(f"Failed to lock position: {e}")
            return False

    def unlock_position(self, item_id: int) -> bool:
        """Unlock a position so PPO can trade it again."""
        try:
            positions = self.get_active_positions(use_cache=False)

            if item_id not in positions:
                logger.warning(f"Cannot unlock position {item_id}: not found")
                return False

            positions[item_id]["locked"] = False
            positions[item_id]["last_updated"] = self._now_iso()

            self._get_positions_ref().set({
                "items": {str(k): v for k, v in positions.items()},
                "updated_at": self._now_iso()
            })

            self._invalidate_cache()
            logger.info(f"Unlocked position: {positions[item_id].get('item_name', item_id)}")
            return True

        except Exception as e:
            logger.error(f"Failed to unlock position: {e}")
            return False

    # =========================================================================
    # Manual Position Management - For Vue dashboard
    # =========================================================================

    def add_manual_position(
        self,
        item_id: int,
        item_name: str,
        quantity: int,
        avg_cost: int = 0
    ) -> bool:
        """
        Manually add a bank item to active positions.

        Use this from Vue dashboard to mark pre-existing items as tradeable.
        """
        return self.add_position(
            item_id=item_id,
            item_name=item_name,
            quantity=quantity,
            cost_per_item=avg_cost,
            source="manual"
        )

    def sync_with_inventory(
        self,
        inventory_items: Dict[int, int],
        ge_items: Dict[int, int]
    ):
        """
        Sync positions with current inventory and GE state.

        Called after inventory sync to update position quantities.
        Items in inventory or GE that are in active positions get quantity updated.
        Items NOT in active positions are ignored.

        Args:
            inventory_items: Dict of item_id -> quantity in inventory
            ge_items: Dict of item_id -> quantity in active GE orders
        """
        try:
            positions = self.get_active_positions(use_cache=False)
            now = self._now_iso()
            updated = False

            for item_id, pos in positions.items():
                inv_qty = inventory_items.get(item_id, 0)
                ge_qty = ge_items.get(item_id, 0)
                actual_qty = inv_qty + ge_qty

                # Only update if we have the item somewhere
                # Don't reduce to 0 just because it's in bank (bank isn't checked here)
                if actual_qty > 0 and pos.get("quantity", 0) != actual_qty:
                    # Position quantity might need adjustment
                    # But be careful - item might be in bank
                    logger.debug(f"Position sync: {pos.get('item_name')} - "
                               f"tracked: {pos.get('quantity')}, in inv/ge: {actual_qty}")

            if updated:
                self._get_positions_ref().set({
                    "items": {str(k): v for k, v in positions.items()},
                    "updated_at": now
                })
                self._invalidate_cache()

        except Exception as e:
            logger.error(f"Failed to sync positions: {e}")

    # =========================================================================
    # Queries for Inference
    # =========================================================================

    def get_sellable_items(self, bank_items: Dict[int, int]) -> Dict[int, int]:
        """
        Get items from bank that can be sold (are in active positions).

        Args:
            bank_items: Dict of item_id -> quantity in bank

        Returns:
            Dict of item_id -> sellable quantity
        """
        positions = self.get_tradeable_positions()
        sellable = {}

        for item_id, bank_qty in bank_items.items():
            if item_id in positions:
                pos = positions[item_id]
                # Can only sell up to position quantity
                max_sellable = min(bank_qty, pos.get("quantity", 0))
                if max_sellable > 0:
                    sellable[item_id] = max_sellable

        return sellable

    def get_position_value(self, item_id: int, current_price: int) -> Dict[str, Any]:
        """
        Get position value and P&L for an item.

        Args:
            item_id: Item ID
            current_price: Current market price

        Returns:
            {
                "quantity": int,
                "avg_cost": int,
                "total_invested": int,
                "current_value": int,
                "unrealized_pnl": int,
                "unrealized_pnl_pct": float
            }
        """
        pos = self.get_position(item_id)
        if not pos:
            return {
                "quantity": 0,
                "avg_cost": 0,
                "total_invested": 0,
                "current_value": 0,
                "unrealized_pnl": 0,
                "unrealized_pnl_pct": 0.0
            }

        qty = pos.get("quantity", 0)
        avg_cost = pos.get("avg_cost", 0)
        invested = pos.get("total_invested", 0)
        current_value = qty * current_price
        pnl = current_value - invested
        pnl_pct = (pnl / invested * 100) if invested > 0 else 0.0

        return {
            "quantity": qty,
            "avg_cost": avg_cost,
            "total_invested": invested,
            "current_value": current_value,
            "unrealized_pnl": pnl,
            "unrealized_pnl_pct": pnl_pct
        }

    def print_positions(self):
        """Print all active positions to console."""
        positions = self.get_active_positions()

        print("\n" + "=" * 80)
        print("Active Positions (PPO Tradeable)")
        print("=" * 80)

        if not positions:
            print("  (no active positions)")
            print("=" * 80 + "\n")
            return

        print(f"{'Item Name':<30} {'Qty':>10} {'Avg Cost':>10} {'Invested':>12} {'Source':>8} {'Locked':>6}")
        print("-" * 80)

        total_invested = 0
        for item_id, pos in sorted(positions.items(), key=lambda x: x[1].get("total_invested", 0), reverse=True):
            name = pos.get("item_name", f"Item {item_id}")[:29]
            qty = pos.get("quantity", 0)
            avg = pos.get("avg_cost", 0)
            invested = pos.get("total_invested", 0)
            source = pos.get("source", "?")
            locked = "Yes" if pos.get("locked", False) else "No"

            total_invested += invested
            print(f"{name:<30} {qty:>10,} {avg:>10,} {invested:>12,} {source:>8} {locked:>6}")

        print("-" * 80)
        print(f"{'TOTAL':<30} {'':<10} {'':<10} {total_invested:>12,}")
        print("=" * 80 + "\n")
