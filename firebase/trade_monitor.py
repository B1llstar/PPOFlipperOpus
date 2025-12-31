"""
Trade Monitor - Listen for completed trades from the GE Auto plugin.

This module handles the Plugin → PPO direction of communication:
- Listen for completed trades
- Calculate profit/loss
- Track trade history
- Provide data for reward calculation
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Any, Callable

from .firebase_client import FirebaseClient

logger = logging.getLogger(__name__)


class TradeMonitor:
    """
    Monitors completed trades reported by the GE Auto plugin.

    Listens to the trades collection and provides:
    - Real-time trade notifications
    - P&L calculations
    - Trade history queries
    """

    # GE tax rate (1%)
    GE_TAX_RATE = 0.01

    def __init__(self, client: Optional[FirebaseClient] = None):
        """Initialize with Firebase client."""
        self.client = client or FirebaseClient()
        self._trade_listener = None
        self._trade_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._recent_trades: List[Dict[str, Any]] = []
        self._max_recent_trades = 100
        self._initial_snapshot_received = False  # Track if initial snapshot processed

    # =========================================================================
    # Trade Listening
    # =========================================================================

    def start_listening(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Start listening for new trades from the plugin.

        Args:
            callback: Function called with trade data when new trade arrives
        """
        if callback:
            self._trade_callbacks.append(callback)

        if self._trade_listener is not None:
            logger.warning("Trade listener already running")
            return

        def on_snapshot(doc_snapshot, changes, read_time):
            is_initial = not self._initial_snapshot_received

            for change in changes:
                if change.type.name == 'ADDED':
                    trade_data = change.document.to_dict()
                    # Only process new trades after initial snapshot
                    # Skip historical trades on startup to avoid re-processing
                    if not is_initial:
                        self._handle_new_trade(trade_data)

            # Mark initial snapshot as received
            if is_initial:
                self._initial_snapshot_received = True
                logger.info(f"Initial trade snapshot received: {len(changes)} trades (skipped)")

        self._trade_listener = self.client.get_trades_ref().on_snapshot(on_snapshot)
        logger.info("Started trade monitor listener")

    def stop_listening(self):
        """Stop listening for trades."""
        if self._trade_listener:
            self._trade_listener.unsubscribe()
            self._trade_listener = None
            self._initial_snapshot_received = False  # Reset for next startup
            logger.info("Stopped trade monitor listener")

    def _handle_new_trade(self, trade_data: Dict[str, Any]):
        """Handle a new trade from the plugin."""
        trade_id = trade_data.get("trade_id", "unknown")
        action = trade_data.get("action", "unknown")
        item_name = trade_data.get("item_name", "unknown")
        quantity = trade_data.get("quantity", 0)
        price = trade_data.get("price", 0)
        total_cost = trade_data.get("total_cost", quantity * price)

        logger.info(f"Trade completed: {trade_id} - {action} {quantity}x {item_name} @ {price} = {total_cost}")

        # Add to recent trades
        self._recent_trades.insert(0, trade_data)
        if len(self._recent_trades) > self._max_recent_trades:
            self._recent_trades = self._recent_trades[:self._max_recent_trades]

        # Notify callbacks
        for cb in self._trade_callbacks:
            try:
                cb(trade_data)
            except Exception as e:
                logger.error(f"Error in trade callback: {e}")

    def add_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add a callback for new trades."""
        self._trade_callbacks.append(callback)

    def remove_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Remove a trade callback."""
        if callback in self._trade_callbacks:
            self._trade_callbacks.remove(callback)

    # =========================================================================
    # Trade Queries
    # =========================================================================

    def get_recent_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get most recent trades."""
        try:
            docs = (self.client.get_trades_ref()
                    .order_by("completed_at", direction="DESCENDING")
                    .limit(limit)
                    .stream())
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            logger.error(f"Failed to get recent trades: {e}")
            return []

    def get_trades_for_item(self, item_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """Get trades for a specific item."""
        try:
            docs = (self.client.get_trades_ref()
                    .where("item_id", "==", item_id)
                    .order_by("completed_at", direction="DESCENDING")
                    .limit(limit)
                    .stream())
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            logger.error(f"Failed to get trades for item {item_id}: {e}")
            return []

    def get_trades_since(self, since: datetime) -> List[Dict[str, Any]]:
        """Get trades since a specific datetime."""
        try:
            since_iso = since.isoformat()
            docs = (self.client.get_trades_ref()
                    .where("completed_at", ">=", since_iso)
                    .order_by("completed_at", direction="DESCENDING")
                    .stream())
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            logger.error(f"Failed to get trades since {since}: {e}")
            return []

    def get_trade_by_order_id(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get trade associated with a specific order."""
        try:
            docs = (self.client.get_trades_ref()
                    .where("order_id", "==", order_id)
                    .limit(1)
                    .stream())
            for doc in docs:
                return doc.to_dict()
        except Exception as e:
            logger.error(f"Failed to get trade for order {order_id}: {e}")
        return None

    # =========================================================================
    # P&L Calculations
    # =========================================================================

    def calculate_pnl(
        self,
        item_id: Optional[int] = None,
        since: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Calculate profit/loss from trades.

        Args:
            item_id: Filter by item ID (optional)
            since: Filter by start date (optional)

        Returns:
            Dictionary with P&L statistics
        """
        try:
            query = self.client.get_trades_ref()

            if item_id is not None:
                query = query.where("item_id", "==", item_id)
            if since is not None:
                query = query.where("completed_at", ">=", since.isoformat())

            docs = query.stream()

            total_bought = 0
            total_sold = 0
            total_tax = 0
            buy_count = 0
            sell_count = 0
            buy_quantity = 0
            sell_quantity = 0

            for doc in docs:
                trade = doc.to_dict()
                action = trade.get("action", "")
                total_cost = trade.get("total_cost", 0)
                tax_paid = trade.get("tax_paid", 0)
                quantity = trade.get("quantity", 0)

                if action == "buy":
                    total_bought += total_cost
                    buy_count += 1
                    buy_quantity += quantity
                elif action == "sell":
                    total_sold += total_cost
                    total_tax += tax_paid
                    sell_count += 1
                    sell_quantity += quantity

            gross_profit = total_sold - total_bought
            net_profit = gross_profit - total_tax

            return {
                "total_bought": total_bought,
                "total_sold": total_sold,
                "total_tax": total_tax,
                "gross_profit": gross_profit,
                "net_profit": net_profit,
                "buy_count": buy_count,
                "sell_count": sell_count,
                "buy_quantity": buy_quantity,
                "sell_quantity": sell_quantity,
                "roi_percent": (net_profit / total_bought * 100) if total_bought > 0 else 0
            }

        except Exception as e:
            logger.error(f"Failed to calculate P&L: {e}")
            return {
                "total_bought": 0,
                "total_sold": 0,
                "total_tax": 0,
                "gross_profit": 0,
                "net_profit": 0,
                "buy_count": 0,
                "sell_count": 0,
                "buy_quantity": 0,
                "sell_quantity": 0,
                "roi_percent": 0
            }

    def calculate_daily_pnl(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Calculate P&L for each of the last N days.

        Args:
            days: Number of days to calculate

        Returns:
            List of daily P&L dictionaries
        """
        daily_pnl = []
        now = datetime.now(timezone.utc)

        for i in range(days):
            day_start = (now - timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)

            # Get trades for this day
            try:
                docs = (self.client.get_trades_ref()
                        .where("completed_at", ">=", day_start.isoformat())
                        .where("completed_at", "<", day_end.isoformat())
                        .stream())

                day_bought = 0
                day_sold = 0
                day_tax = 0
                trade_count = 0

                for doc in docs:
                    trade = doc.to_dict()
                    action = trade.get("action", "")
                    total_cost = trade.get("total_cost", 0)
                    tax_paid = trade.get("tax_paid", 0)

                    if action == "buy":
                        day_bought += total_cost
                    elif action == "sell":
                        day_sold += total_cost
                        day_tax += tax_paid
                    trade_count += 1

                daily_pnl.append({
                    "date": day_start.strftime("%Y-%m-%d"),
                    "total_bought": day_bought,
                    "total_sold": day_sold,
                    "total_tax": day_tax,
                    "gross_profit": day_sold - day_bought,
                    "net_profit": day_sold - day_bought - day_tax,
                    "trade_count": trade_count
                })

            except Exception as e:
                logger.error(f"Failed to calculate P&L for {day_start.date()}: {e}")
                daily_pnl.append({
                    "date": day_start.strftime("%Y-%m-%d"),
                    "error": str(e)
                })

        return daily_pnl

    def get_item_performance(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get performance statistics by item.

        Returns:
            List of items sorted by net profit
        """
        try:
            # Get all trades
            docs = self.client.get_trades_ref().stream()

            # Aggregate by item
            item_stats: Dict[int, Dict[str, Any]] = {}

            for doc in docs:
                trade = doc.to_dict()
                item_id = trade.get("item_id")
                item_name = trade.get("item_name", "Unknown")
                action = trade.get("action", "")
                total_cost = trade.get("total_cost", 0)
                tax_paid = trade.get("tax_paid", 0)
                quantity = trade.get("quantity", 0)

                if item_id not in item_stats:
                    item_stats[item_id] = {
                        "item_id": item_id,
                        "item_name": item_name,
                        "total_bought": 0,
                        "total_sold": 0,
                        "total_tax": 0,
                        "buy_count": 0,
                        "sell_count": 0,
                        "buy_quantity": 0,
                        "sell_quantity": 0
                    }

                if action == "buy":
                    item_stats[item_id]["total_bought"] += total_cost
                    item_stats[item_id]["buy_count"] += 1
                    item_stats[item_id]["buy_quantity"] += quantity
                elif action == "sell":
                    item_stats[item_id]["total_sold"] += total_cost
                    item_stats[item_id]["total_tax"] += tax_paid
                    item_stats[item_id]["sell_count"] += 1
                    item_stats[item_id]["sell_quantity"] += quantity

            # Calculate profit for each item
            for stats in item_stats.values():
                stats["gross_profit"] = stats["total_sold"] - stats["total_bought"]
                stats["net_profit"] = stats["gross_profit"] - stats["total_tax"]
                if stats["total_bought"] > 0:
                    stats["roi_percent"] = stats["net_profit"] / stats["total_bought"] * 100
                else:
                    stats["roi_percent"] = 0

            # Sort by net profit and return top items
            sorted_items = sorted(
                item_stats.values(),
                key=lambda x: x["net_profit"],
                reverse=True
            )

            return sorted_items[:limit]

        except Exception as e:
            logger.error(f"Failed to get item performance: {e}")
            return []

    # =========================================================================
    # Cached Recent Trades
    # =========================================================================

    def get_cached_recent_trades(self) -> List[Dict[str, Any]]:
        """Get trades from local cache (faster than Firestore query)."""
        return self._recent_trades.copy()

    def clear_cache(self):
        """Clear the local trade cache."""
        self._recent_trades.clear()
