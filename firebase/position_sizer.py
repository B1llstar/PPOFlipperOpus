"""
Position Sizer - Enforces spending limits per position.

Ensures no single item takes up more than a configurable % of the portfolio,
and provides methods to calculate safe order sizes.
"""

import logging
from typing import Dict, Any, Optional, Tuple

from .inventory_scanner import InventoryScanner

logger = logging.getLogger(__name__)


class PositionSizer:
    """
    Enforces position sizing limits based on portfolio %.

    Example:
        sizer = PositionSizer(scanner, max_position_pct=10.0)

        # Check if we can buy more of an item
        can_buy, max_qty, reason = sizer.check_buy_limit(item_id=2, price=150, quantity=1000)
        if can_buy:
            order_manager.create_buy_order(...)
        else:
            print(f"Blocked: {reason}")

        # Get recommended quantity that stays within limits
        safe_qty = sizer.get_safe_buy_quantity(item_id=2, price=150, desired_qty=1000)
    """

    def __init__(
        self,
        scanner: InventoryScanner,
        max_position_pct: float = 10.0,
        max_single_order_pct: float = 5.0,
        min_cash_reserve_pct: float = 10.0,
        max_total_exposure_pct: float = 80.0
    ):
        """
        Initialize position sizer.

        Args:
            scanner: InventoryScanner instance for reading portfolio state
            max_position_pct: Maximum % of portfolio in any single item (default 10%)
            max_single_order_pct: Maximum % of portfolio per single order (default 5%)
            min_cash_reserve_pct: Minimum % to keep as cash (default 10%)
            max_total_exposure_pct: Maximum % of portfolio in items (default 80%)
        """
        self.scanner = scanner
        self.max_position_pct = max_position_pct
        self.max_single_order_pct = max_single_order_pct
        self.min_cash_reserve_pct = min_cash_reserve_pct
        self.max_total_exposure_pct = max_total_exposure_pct

    def get_limits(self) -> Dict[str, Any]:
        """Get current limit settings."""
        return {
            "max_position_pct": self.max_position_pct,
            "max_single_order_pct": self.max_single_order_pct,
            "min_cash_reserve_pct": self.min_cash_reserve_pct,
            "max_total_exposure_pct": self.max_total_exposure_pct
        }

    def set_limits(
        self,
        max_position_pct: Optional[float] = None,
        max_single_order_pct: Optional[float] = None,
        min_cash_reserve_pct: Optional[float] = None,
        max_total_exposure_pct: Optional[float] = None
    ):
        """Update limit settings."""
        if max_position_pct is not None:
            self.max_position_pct = max_position_pct
        if max_single_order_pct is not None:
            self.max_single_order_pct = max_single_order_pct
        if min_cash_reserve_pct is not None:
            self.min_cash_reserve_pct = min_cash_reserve_pct
        if max_total_exposure_pct is not None:
            self.max_total_exposure_pct = max_total_exposure_pct

    def get_available_capital(self) -> Dict[str, Any]:
        """
        Get available capital for new positions.

        Returns:
            {
                "total_portfolio": int,
                "gold": int,
                "items_exposure": int,
                "items_exposure_pct": float,
                "cash_reserve_required": int,
                "available_for_orders": int,  # Gold available after reserve
                "can_add_exposure": bool,     # Below max total exposure
                "exposure_room": int          # How much more exposure allowed
            }
        """
        util = self.scanner.get_portfolio_utilization()

        total = util["total_portfolio_value"]
        gold = util["gold"]
        items_value = util["items_value"]
        orders_value = util["active_orders_value"]

        # Current exposure = items + active buy orders
        current_exposure = items_value + orders_value
        exposure_pct = (current_exposure / total * 100) if total > 0 else 0

        # Required cash reserve
        cash_reserve = int(total * self.min_cash_reserve_pct / 100)

        # Available gold after reserve
        available = max(0, gold - cash_reserve)

        # Check if we can add more exposure
        max_exposure = int(total * self.max_total_exposure_pct / 100)
        exposure_room = max(0, max_exposure - current_exposure)
        can_add = exposure_pct < self.max_total_exposure_pct

        return {
            "total_portfolio": total,
            "gold": gold,
            "items_exposure": current_exposure,
            "items_exposure_pct": exposure_pct,
            "cash_reserve_required": cash_reserve,
            "available_for_orders": available,
            "can_add_exposure": can_add,
            "exposure_room": exposure_room
        }

    def check_buy_limit(
        self,
        item_id: int,
        price: int,
        quantity: int
    ) -> Tuple[bool, int, str]:
        """
        Check if a buy order is within position limits.

        Args:
            item_id: Item to buy
            price: Price per item
            quantity: Desired quantity

        Returns:
            Tuple of (allowed, max_quantity, reason)
            - allowed: True if order can proceed
            - max_quantity: Maximum quantity allowed (may be less than requested)
            - reason: Explanation if blocked or reduced
        """
        order_value = price * quantity
        util = self.scanner.get_portfolio_utilization()
        capital = self.get_available_capital()

        total_portfolio = util["total_portfolio_value"]

        if total_portfolio == 0:
            return False, 0, "Portfolio value is zero"

        # Check 1: Do we have enough available gold?
        if order_value > capital["available_for_orders"]:
            max_qty = capital["available_for_orders"] // price if price > 0 else 0
            if max_qty == 0:
                return False, 0, f"Insufficient funds (available: {capital['available_for_orders']:,}, need: {order_value:,})"
            return True, max_qty, f"Reduced to {max_qty} due to available funds"

        # Check 2: Would this exceed max total exposure?
        if not capital["can_add_exposure"]:
            return False, 0, f"At max exposure ({self.max_total_exposure_pct}%)"

        if order_value > capital["exposure_room"]:
            max_qty = capital["exposure_room"] // price if price > 0 else 0
            if max_qty == 0:
                return False, 0, f"Exposure room exhausted (room: {capital['exposure_room']:,})"
            return True, max_qty, f"Reduced to {max_qty} to stay within exposure limit"

        # Check 3: Would this single order exceed max single order %?
        max_single_order_value = int(total_portfolio * self.max_single_order_pct / 100)
        if order_value > max_single_order_value:
            max_qty = max_single_order_value // price if price > 0 else 0
            if max_qty == 0:
                return False, 0, f"Order too large (max: {max_single_order_value:,})"
            return True, max_qty, f"Reduced to {max_qty} (max order: {self.max_single_order_pct}% of portfolio)"

        # Check 4: Would this item exceed max position %?
        item_util = self.scanner.get_item_utilization(item_id)
        current_value = item_util["value"]
        new_value = current_value + order_value
        new_pct = (new_value / total_portfolio * 100) if total_portfolio > 0 else 0

        if new_pct > self.max_position_pct:
            # Calculate how much we can add
            max_position_value = int(total_portfolio * self.max_position_pct / 100)
            remaining_room = max(0, max_position_value - current_value)
            max_qty = remaining_room // price if price > 0 else 0

            if max_qty == 0:
                return False, 0, f"Position limit reached ({item_util['item_name']}: {item_util['percent_of_portfolio']:.1f}% of portfolio)"

            return True, max_qty, f"Reduced to {max_qty} to stay within {self.max_position_pct}% position limit"

        # All checks passed
        return True, quantity, "OK"

    def get_safe_buy_quantity(
        self,
        item_id: int,
        price: int,
        desired_qty: int
    ) -> int:
        """
        Get the maximum safe quantity to buy within all limits.

        Args:
            item_id: Item to buy
            price: Price per item
            desired_qty: Desired quantity

        Returns:
            Maximum quantity that respects all limits (may be 0)
        """
        allowed, max_qty, reason = self.check_buy_limit(item_id, price, desired_qty)
        if not allowed:
            logger.info(f"Buy blocked for item {item_id}: {reason}")
            return 0
        return max_qty

    def get_position_summary(self, item_id: int) -> Dict[str, Any]:
        """
        Get detailed position summary for an item.

        Args:
            item_id: Item to check

        Returns:
            {
                "item_id": int,
                "item_name": str,
                "current_value": int,
                "current_pct": float,
                "max_allowed_pct": float,
                "max_allowed_value": int,
                "remaining_room": int,
                "at_limit": bool
            }
        """
        util = self.scanner.get_portfolio_utilization()
        item_util = self.scanner.get_item_utilization(item_id)

        total = util["total_portfolio_value"]
        max_value = int(total * self.max_position_pct / 100) if total > 0 else 0
        remaining = max(0, max_value - item_util["value"])

        return {
            "item_id": item_id,
            "item_name": item_util["item_name"],
            "current_value": item_util["value"],
            "current_pct": item_util["percent_of_portfolio"],
            "max_allowed_pct": self.max_position_pct,
            "max_allowed_value": max_value,
            "remaining_room": remaining,
            "at_limit": item_util["percent_of_portfolio"] >= self.max_position_pct
        }

    def get_all_positions_status(self) -> Dict[str, Any]:
        """
        Get status of all positions relative to limits.

        Returns:
            {
                "portfolio_total": int,
                "max_position_pct": float,
                "positions": [
                    {
                        "item_id": int,
                        "item_name": str,
                        "value": int,
                        "pct": float,
                        "at_limit": bool,
                        "over_limit": bool,
                        "room": int
                    }
                ],
                "over_limit_count": int,
                "at_limit_count": int
            }
        """
        util = self.scanner.get_portfolio_utilization()
        total = util["total_portfolio_value"]
        max_value = int(total * self.max_position_pct / 100) if total > 0 else 0

        positions = []
        over_count = 0
        at_count = 0

        for item in util["by_item"]:
            pct = item["percent"]
            at_limit = pct >= self.max_position_pct
            over_limit = pct > self.max_position_pct
            room = max(0, max_value - item["value"])

            if over_limit:
                over_count += 1
            elif at_limit:
                at_count += 1

            positions.append({
                "item_id": item["item_id"],
                "item_name": item["item_name"],
                "value": item["value"],
                "pct": pct,
                "at_limit": at_limit,
                "over_limit": over_limit,
                "room": room
            })

        return {
            "portfolio_total": total,
            "max_position_pct": self.max_position_pct,
            "positions": positions,
            "over_limit_count": over_count,
            "at_limit_count": at_count
        }

    def print_limits_status(self):
        """Print current limits and status to console."""
        capital = self.get_available_capital()
        status = self.get_all_positions_status()

        print("\n" + "=" * 70)
        print("Position Sizing Status")
        print("=" * 70)

        print(f"\nLimits:")
        print(f"  Max position per item:    {self.max_position_pct}%")
        print(f"  Max single order:         {self.max_single_order_pct}%")
        print(f"  Min cash reserve:         {self.min_cash_reserve_pct}%")
        print(f"  Max total exposure:       {self.max_total_exposure_pct}%")

        print(f"\nCapital Status:")
        print(f"  Total Portfolio:          {capital['total_portfolio']:,} GP")
        print(f"  Gold:                     {capital['gold']:,} GP")
        print(f"  Cash Reserve Required:    {capital['cash_reserve_required']:,} GP")
        print(f"  Available for Orders:     {capital['available_for_orders']:,} GP")
        print(f"  Current Exposure:         {capital['items_exposure']:,} GP ({capital['items_exposure_pct']:.1f}%)")
        print(f"  Exposure Room:            {capital['exposure_room']:,} GP")
        print(f"  Can Add Exposure:         {'Yes' if capital['can_add_exposure'] else 'No'}")

        if status["over_limit_count"] > 0 or status["at_limit_count"] > 0:
            print(f"\nPosition Warnings:")
            print(f"  Over Limit:               {status['over_limit_count']}")
            print(f"  At Limit:                 {status['at_limit_count']}")

            print(f"\n  Positions at/over limit:")
            for pos in status["positions"]:
                if pos["at_limit"] or pos["over_limit"]:
                    flag = "OVER" if pos["over_limit"] else "AT"
                    print(f"    [{flag}] {pos['item_name']}: {pos['pct']:.1f}% ({pos['value']:,} GP)")

        print("=" * 70 + "\n")
