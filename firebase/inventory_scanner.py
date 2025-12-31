"""
Inventory Scanner - Scans GE-relevant items and provides full inventory accounting.

This utility coordinates with the GEAuto plugin via Firestore to:
- Scan all tradeable items in the GE relevant to PPO inferencing
- Get complete bank inventory with quantities and names
- Retrieve current gold balance for the account
- Provide a consolidated view for the PPO inference system

Usage:
    from firebase.inventory_scanner import InventoryScanner

    scanner = InventoryScanner(service_account_path, account_id)

    # Get all tradeable items from Firestore items collection
    items = scanner.get_all_tradeable_items()

    # Get complete bank accounting
    bank = scanner.get_bank_accounting()

    # Get current gold
    gold = scanner.get_gold()

    # Get comprehensive state for inference
    state = scanner.get_inference_state()
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path

from .firebase_client import FirebaseClient

logger = logging.getLogger(__name__)


class InventoryScanner:
    """
    Scans and retrieves inventory/bank data from Firestore for PPO inference.

    Coordinates with the GEAuto RuneLite plugin which syncs:
    - Inventory state to /accounts/{id}/inventory/current
    - Bank state to /accounts/{id}/bank/current
    - GE slot state to /accounts/{id}/ge_slots/current
    - Account state (gold, heartbeat) to /accounts/{id}

    And provides access to:
    - /items collection: All item data (id, name, value, ge_limit, etc.)
    - /itemNames collection: Fast name→id lookup
    """

    def __init__(
        self,
        service_account_path: Optional[str] = None,
        account_id: str = "b1llstar"
    ):
        """
        Initialize the inventory scanner.

        Args:
            service_account_path: Path to Firebase service account JSON.
                                 If None, uses default from firebase_config.
            account_id: Account identifier (matches RuneLite player name).
        """
        self.account_id = account_id
        self.client = FirebaseClient()

        # Initialize Firebase if not already done
        if not self.client.initialized:
            if service_account_path is None:
                try:
                    from config.firebase_config import get_service_account_path
                    service_account_path = get_service_account_path()
                except ImportError:
                    # Try relative import as fallback
                    import sys
                    from pathlib import Path
                    sys.path.insert(0, str(Path(__file__).parent.parent))
                    from config.firebase_config import get_service_account_path
                    service_account_path = get_service_account_path()

            if not self.client.initialize(service_account_path, account_id):
                raise RuntimeError("Failed to initialize Firebase connection")

        # Cache for items collection (rarely changes)
        self._items_cache: Optional[Dict[int, Dict[str, Any]]] = None
        self._items_cache_time: float = 0
        self._items_cache_ttl: float = 3600  # 1 hour cache for items

        # Cache for name→id mapping
        self._name_to_id_cache: Optional[Dict[str, int]] = None

        logger.info(f"InventoryScanner initialized for account: {account_id}")

    # =========================================================================
    # Tradeable Items from Items Collection
    # =========================================================================

    def get_all_tradeable_items(
        self,
        min_value: int = 0,
        max_value: Optional[int] = None,
        members_only: Optional[bool] = None,
        force_refresh: bool = False
    ) -> Dict[int, Dict[str, Any]]:
        """
        Get all tradeable items from the Firestore items collection.

        These are items that exist in the GE and are relevant for trading.

        Args:
            min_value: Minimum item value (filters low value items)
            max_value: Maximum item value (filters high value items)
            members_only: If True, only members items; if False, only F2P; if None, all
            force_refresh: Bypass cache and fetch fresh data

        Returns:
            Dictionary mapping item_id (int) to item data:
            {
                item_id: {
                    "id": int,
                    "name": str,
                    "value": int,          # Base value
                    "highalch": int,       # High alch value
                    "lowalch": int,        # Low alch value
                    "ge_limit": int,       # GE buy limit
                    "members": bool,       # Members only item
                    "examine": str,        # Examine text
                    "icon": str,           # Icon filename
                    "updated_at": str      # Last update timestamp
                }
            }
        """
        import time
        current_time = time.time()

        # Check cache
        if (not force_refresh and
            self._items_cache is not None and
            current_time - self._items_cache_time < self._items_cache_ttl):
            items = self._items_cache
        else:
            # Fetch from Firestore
            items = self._fetch_all_items()
            self._items_cache = items
            self._items_cache_time = current_time

        # Apply filters
        filtered = {}
        for item_id, item_data in items.items():
            # Value filter
            item_value = item_data.get("value", 0)
            if item_value < min_value:
                continue
            if max_value is not None and item_value > max_value:
                continue

            # Members filter
            if members_only is not None:
                is_members = item_data.get("members", False)
                if members_only and not is_members:
                    continue
                if not members_only and is_members:
                    continue

            filtered[item_id] = item_data

        logger.info(f"Found {len(filtered)} tradeable items (filtered from {len(items)} total)")
        return filtered

    def _fetch_all_items(self) -> Dict[int, Dict[str, Any]]:
        """Fetch all items from Firestore items collection."""
        items = {}
        try:
            docs = self.client.collection(FirebaseClient.COLLECTION_ITEMS).stream()
            for doc in docs:
                try:
                    item_id = int(doc.id)
                    item_data = doc.to_dict()
                    if item_data:
                        item_data["id"] = item_id
                        items[item_id] = item_data
                except (ValueError, TypeError):
                    # Skip documents with non-integer IDs
                    continue

            logger.info(f"Fetched {len(items)} items from Firestore")
        except Exception as e:
            logger.error(f"Failed to fetch items: {e}")

        return items

    def get_item_by_id(self, item_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific item by ID."""
        return self.client.get_item_by_id(item_id)

    def get_item_id_by_name(self, item_name: str) -> Optional[int]:
        """Get item ID by name (uses itemNames collection for fast lookup)."""
        return self.client.get_item_id_by_name(item_name)

    def get_item_by_name(self, item_name: str) -> Optional[Dict[str, Any]]:
        """Get item data by name."""
        return self.client.get_item_by_name(item_name)

    def build_name_to_id_map(self, force_refresh: bool = False) -> Dict[str, int]:
        """
        Build a mapping of item names to IDs for fast lookup.

        Args:
            force_refresh: Bypass cache

        Returns:
            Dictionary mapping item name to item ID
        """
        if not force_refresh and self._name_to_id_cache is not None:
            return self._name_to_id_cache

        name_to_id = {}
        try:
            docs = self.client.collection(FirebaseClient.COLLECTION_ITEM_NAMES).stream()
            for doc in docs:
                data = doc.to_dict()
                if data and "id" in data:
                    # Document ID is the sanitized name
                    name = doc.id.replace("_", "/")  # Reverse sanitization
                    name_to_id[name] = data["id"]

            self._name_to_id_cache = name_to_id
            logger.info(f"Built name→ID map with {len(name_to_id)} entries")
        except Exception as e:
            logger.error(f"Failed to build name→ID map: {e}")

        return name_to_id

    # =========================================================================
    # Bank Accounting
    # =========================================================================

    def get_bank_accounting(self) -> Dict[str, Any]:
        """
        Get complete accounting of all items in the bank.

        Returns:
            {
                "updated_at": str,           # Last sync timestamp
                "total_value": int,          # Total GP value of all items
                "item_count": int,           # Number of unique items
                "total_quantity": int,       # Sum of all quantities
                "tradeable_count": int,      # Items flagged as tradeable
                "awaiting_sell": [int],      # Item IDs ready to sell
                "items": {
                    "item_id": {
                        "item_id": int,
                        "item_name": str,
                        "quantity": int,
                        "price_each": int,
                        "total_value": int,
                        "is_tradeable": bool
                    }
                }
            }
        """
        try:
            doc = self.client.get_bank_ref().get()
            if doc.exists:
                bank_data = doc.to_dict()

                # Calculate total quantity
                items = bank_data.get("items", {})
                total_qty = sum(
                    item.get("quantity", 0)
                    for item in items.values()
                )
                bank_data["total_quantity"] = total_qty

                logger.info(f"Bank accounting: {len(items)} items, {total_qty:,} total quantity, "
                           f"{bank_data.get('total_value', 0):,} GP value")
                return bank_data
            else:
                logger.warning("Bank document not found - plugin may not have synced yet")
                return self._empty_bank_response()
        except Exception as e:
            logger.error(f"Failed to get bank accounting: {e}")
            return self._empty_bank_response()

    def _empty_bank_response(self) -> Dict[str, Any]:
        """Return empty bank response structure."""
        return {
            "updated_at": None,
            "total_value": 0,
            "item_count": 0,
            "total_quantity": 0,
            "tradeable_count": 0,
            "awaiting_sell": [],
            "items": {}
        }

    def get_bank_items_by_value(
        self,
        min_value: int = 0,
        descending: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get bank items sorted by total value.

        Args:
            min_value: Minimum total value to include
            descending: Sort high to low if True

        Returns:
            List of item dicts sorted by total_value
        """
        bank = self.get_bank_accounting()
        items = list(bank.get("items", {}).values())

        # Filter by minimum value
        items = [i for i in items if i.get("total_value", 0) >= min_value]

        # Sort by value
        items.sort(key=lambda x: x.get("total_value", 0), reverse=descending)

        return items

    def get_bank_item_quantity(self, item_id: int) -> int:
        """Get quantity of a specific item in bank."""
        bank = self.get_bank_accounting()
        item_data = bank.get("items", {}).get(str(item_id), {})
        return item_data.get("quantity", 0)

    def get_tradeable_bank_items(self) -> Dict[int, Dict[str, Any]]:
        """
        Get only tradeable items from bank (those flagged for potential sale).

        Returns:
            Dictionary mapping item_id (int) to item data
        """
        bank = self.get_bank_accounting()
        tradeable = {}

        awaiting_sell = set(bank.get("awaiting_sell", []))

        for item_id_str, item_data in bank.get("items", {}).items():
            item_id = int(item_id_str)
            if item_data.get("is_tradeable", False) or item_id in awaiting_sell:
                tradeable[item_id] = item_data

        return tradeable

    # =========================================================================
    # Inventory Accounting
    # =========================================================================

    def get_inventory_accounting(self) -> Dict[str, Any]:
        """
        Get complete accounting of player inventory (28 slots).

        Returns:
            {
                "updated_at": str,
                "gold": int,                 # Coins in inventory
                "item_count": int,           # Number of unique items (not counting coins)
                "total_item_value": int,     # Value of items (not coins)
                "total_value": int,          # Items + coins
                "free_slots": int,           # Empty slots
                "items": {
                    "item_id": {
                        "item_id": int,
                        "item_name": str,
                        "quantity": int,
                        "price_each": int,
                        "total_value": int,
                        "is_tradeable": bool
                    }
                }
            }
        """
        try:
            doc = self.client.get_inventory_ref().get()
            if doc.exists:
                inv_data = doc.to_dict()
                logger.info(f"Inventory: {inv_data.get('gold', 0):,} gold, "
                           f"{len(inv_data.get('items', {}))} items, "
                           f"{inv_data.get('free_slots', 0)} free slots")
                return inv_data
            else:
                logger.warning("Inventory document not found - plugin may not have synced yet")
                return self._empty_inventory_response()
        except Exception as e:
            logger.error(f"Failed to get inventory: {e}")
            return self._empty_inventory_response()

    def _empty_inventory_response(self) -> Dict[str, Any]:
        """Return empty inventory response structure."""
        return {
            "updated_at": None,
            "gold": 0,
            "item_count": 0,
            "total_item_value": 0,
            "total_value": 0,
            "free_slots": 28,
            "items": {}
        }

    # =========================================================================
    # Gold/Account Status
    # =========================================================================

    def get_gold(self) -> int:
        """
        Get current gold balance for the account.

        Checks multiple sources:
        1. Account document current_gold field
        2. Inventory gold field
        3. Portfolio gold field

        Returns:
            Current gold balance in GP
        """
        # Try account document first (most authoritative)
        try:
            doc = self.client.get_account_ref().get()
            if doc.exists:
                data = doc.to_dict()
                # Check various field names the plugin might use
                for field in ["current_gold", "gold"]:
                    gold = data.get(field)
                    if gold is not None and gold > 0:
                        return gold
        except Exception as e:
            logger.debug(f"Could not get gold from account: {e}")

        # Try inventory
        try:
            doc = self.client.get_inventory_ref().get()
            if doc.exists:
                data = doc.to_dict()
                gold = data.get("gold", 0)
                if gold > 0:
                    return gold
        except Exception as e:
            logger.debug(f"Could not get gold from inventory: {e}")

        # Try portfolio
        try:
            doc = self.client.get_portfolio_ref().get()
            if doc.exists:
                data = doc.to_dict()
                gold = data.get("gold", 0)
                if gold > 0:
                    return gold
        except Exception as e:
            logger.debug(f"Could not get gold from portfolio: {e}")

        logger.warning("Could not find gold balance in any document")
        return 0

    def get_account_status(self) -> Dict[str, Any]:
        """
        Get current account status and metadata.

        Returns:
            {
                "account_id": str,
                "status": str,               # "active", "paused", "offline"
                "plugin_version": str,
                "last_heartbeat": str,       # ISO timestamp
                "is_online": bool,           # True if heartbeat recent
                "ge_slots_available": int,
                "queue_size": int,
                "settings": {...},
                "capabilities": {...}
            }
        """
        try:
            doc = self.client.get_account_ref().get()
            if doc.exists:
                data = doc.to_dict()

                # Determine online status from heartbeat
                is_online = False
                heartbeat = data.get("last_heartbeat")
                if heartbeat:
                    try:
                        hb_time = datetime.fromisoformat(heartbeat.replace('Z', '+00:00'))
                        age = (datetime.now(timezone.utc) - hb_time).total_seconds()
                        is_online = age < 120  # Consider online if heartbeat within 2 minutes
                    except:
                        pass

                return {
                    "account_id": data.get("account_id", self.account_id),
                    "status": data.get("status", "unknown"),
                    "plugin_version": data.get("plugin_version", "unknown"),
                    "last_heartbeat": heartbeat,
                    "is_online": is_online,
                    "ge_slots_available": data.get("ge_slots_available", 0),
                    "queue_size": data.get("queue_size", 0),
                    "settings": data.get("settings", {}),
                    "capabilities": data.get("capabilities", {})
                }
            else:
                logger.warning("Account document not found")
                return self._empty_account_status()
        except Exception as e:
            logger.error(f"Failed to get account status: {e}")
            return self._empty_account_status()

    def _empty_account_status(self) -> Dict[str, Any]:
        """Return empty account status structure."""
        return {
            "account_id": self.account_id,
            "status": "offline",
            "plugin_version": "unknown",
            "last_heartbeat": None,
            "is_online": False,
            "ge_slots_available": 0,
            "queue_size": 0,
            "settings": {},
            "capabilities": {}
        }

    def is_plugin_online(self, max_age_seconds: int = 120) -> bool:
        """Check if the plugin is online (sent heartbeat recently)."""
        status = self.get_account_status()
        return status.get("is_online", False)

    # =========================================================================
    # GE Slots State
    # =========================================================================

    def get_ge_slots(self) -> Dict[str, Any]:
        """
        Get current GE slot states (8 slots).

        Returns:
            {
                "updated_at": str,
                "buy_slots_used": int,
                "sell_slots_used": int,
                "slots_available": int,
                "slots": {
                    "1": {
                        "order_id": str,
                        "item_id": int,
                        "item_name": str,
                        "type": "buy" | "sell",
                        "status": "active" | "completed" | "cancelled" | "empty",
                        "quantity": int,
                        "filled_quantity": int,
                        "price": int
                    } | None,
                    ...
                }
            }
        """
        try:
            doc = self.client.get_ge_slots_ref().get()
            if doc.exists:
                return doc.to_dict()
            else:
                logger.warning("GE slots document not found")
                return self._empty_ge_slots()
        except Exception as e:
            logger.error(f"Failed to get GE slots: {e}")
            return self._empty_ge_slots()

    def _empty_ge_slots(self) -> Dict[str, Any]:
        """Return empty GE slots structure."""
        return {
            "updated_at": None,
            "buy_slots_used": 0,
            "sell_slots_used": 0,
            "slots_available": 8,
            "slots": {str(i): None for i in range(1, 9)}
        }

    def get_available_slot_count(self) -> int:
        """Get number of available GE slots."""
        ge = self.get_ge_slots()
        return ge.get("slots_available", 0)

    # =========================================================================
    # Comprehensive Inference State
    # =========================================================================

    def get_inference_state(self) -> Dict[str, Any]:
        """
        Get comprehensive state for PPO inference.

        This is the main method for getting everything the inference
        system needs to make decisions.

        Returns:
            {
                "timestamp": str,
                "account": {
                    "account_id": str,
                    "is_online": bool,
                    "gold": int,
                    "ge_slots_available": int
                },
                "inventory": {
                    "gold": int,
                    "items": {...},
                    "free_slots": int
                },
                "bank": {
                    "total_value": int,
                    "item_count": int,
                    "tradeable_items": {...}   # Only tradeable items
                },
                "ge_slots": {
                    "slots_available": int,
                    "buy_slots_used": int,
                    "sell_slots_used": int,
                    "active_orders": [...]     # List of active slot orders
                },
                "holdings": {
                    "combined": {...}          # Merged inventory + bank holdings
                }
            }
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # Get account status
        account_status = self.get_account_status()
        gold = self.get_gold()

        # Get inventory
        inventory = self.get_inventory_accounting()

        # Get bank
        bank = self.get_bank_accounting()
        tradeable_bank = self.get_tradeable_bank_items()

        # Get GE slots
        ge_slots = self.get_ge_slots()

        # Build active orders list
        active_orders = []
        for slot_num, slot_data in ge_slots.get("slots", {}).items():
            if slot_data is not None and slot_data.get("status") == "active":
                slot_data["slot_number"] = int(slot_num)
                active_orders.append(slot_data)

        # Combine holdings
        combined_holdings = self._merge_holdings(inventory, bank)

        return {
            "timestamp": timestamp,
            "account": {
                "account_id": account_status["account_id"],
                "is_online": account_status["is_online"],
                "gold": gold,
                "ge_slots_available": account_status["ge_slots_available"]
            },
            "inventory": {
                "gold": inventory.get("gold", 0),
                "items": inventory.get("items", {}),
                "free_slots": inventory.get("free_slots", 28),
                "item_count": len(inventory.get("items", {}))
            },
            "bank": {
                "total_value": bank.get("total_value", 0),
                "item_count": bank.get("item_count", 0),
                "tradeable_items": tradeable_bank,
                "tradeable_count": len(tradeable_bank)
            },
            "ge_slots": {
                "slots_available": ge_slots.get("slots_available", 0),
                "buy_slots_used": ge_slots.get("buy_slots_used", 0),
                "sell_slots_used": ge_slots.get("sell_slots_used", 0),
                "active_orders": active_orders
            },
            "holdings": {
                "combined": combined_holdings
            }
        }

    def _merge_holdings(
        self,
        inventory: Dict[str, Any],
        bank: Dict[str, Any]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Merge inventory and bank holdings into a single view.

        Returns:
            Dictionary mapping item_id (int) to combined holdings:
            {
                item_id: {
                    "item_id": int,
                    "item_name": str,
                    "inventory_qty": int,
                    "bank_qty": int,
                    "total_qty": int,
                    "price_each": int,
                    "total_value": int
                }
            }
        """
        combined = {}

        # Add inventory items
        for item_id_str, item_data in inventory.get("items", {}).items():
            try:
                item_id = int(item_id_str)
            except (ValueError, TypeError):
                continue

            combined[item_id] = {
                "item_id": item_id,
                "item_name": item_data.get("item_name", "Unknown"),
                "inventory_qty": item_data.get("quantity", 0),
                "bank_qty": 0,
                "total_qty": item_data.get("quantity", 0),
                "price_each": item_data.get("price_each", 0),
                "total_value": item_data.get("total_value", 0)
            }

        # Add/merge bank items
        for item_id_str, item_data in bank.get("items", {}).items():
            try:
                item_id = int(item_id_str)
            except (ValueError, TypeError):
                continue

            bank_qty = item_data.get("quantity", 0)
            price = item_data.get("price_each", 0)

            if item_id in combined:
                combined[item_id]["bank_qty"] = bank_qty
                combined[item_id]["total_qty"] += bank_qty
                combined[item_id]["total_value"] += bank_qty * price
            else:
                combined[item_id] = {
                    "item_id": item_id,
                    "item_name": item_data.get("item_name", "Unknown"),
                    "inventory_qty": 0,
                    "bank_qty": bank_qty,
                    "total_qty": bank_qty,
                    "price_each": price,
                    "total_value": bank_qty * price
                }

        return combined

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_items_by_ids(self, item_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Get item data for a list of item IDs.

        Args:
            item_ids: List of item IDs to fetch

        Returns:
            Dictionary mapping item_id to item data
        """
        result = {}
        for item_id in item_ids:
            item_data = self.get_item_by_id(item_id)
            if item_data:
                result[item_id] = item_data
        return result

    def get_item_quantity_total(self, item_id: int) -> int:
        """
        Get total quantity of an item across inventory and bank.

        Args:
            item_id: Item ID to check

        Returns:
            Total quantity held
        """
        inventory = self.get_inventory_accounting()
        bank = self.get_bank_accounting()

        inv_qty = inventory.get("items", {}).get(str(item_id), {}).get("quantity", 0)
        bank_qty = bank.get("items", {}).get(str(item_id), {}).get("quantity", 0)

        return inv_qty + bank_qty

    def get_tradeable_items_for_inference(
        self,
        item_ids: Optional[List[int]] = None,
        min_ge_limit: int = 100
    ) -> Dict[int, Dict[str, Any]]:
        """
        Get tradeable items suitable for PPO inference.

        Filters items to those that:
        - Have GE limit >= min_ge_limit
        - Are tradeable on the GE
        - Optionally, only specific item IDs

        Args:
            item_ids: Optional list of specific item IDs to filter to
            min_ge_limit: Minimum GE buy limit

        Returns:
            Dictionary mapping item_id to item data with additional fields:
            - "ge_limit": int
            - "current_holding": int (total across inv+bank)
        """
        all_items = self.get_all_tradeable_items()
        inference_state = self.get_inference_state()
        combined_holdings = inference_state["holdings"]["combined"]

        result = {}
        for item_id, item_data in all_items.items():
            # Filter by specific IDs if provided
            if item_ids is not None and item_id not in item_ids:
                continue

            # Filter by GE limit
            ge_limit = item_data.get("ge_limit", 0)
            if ge_limit < min_ge_limit:
                continue

            # Add current holding info
            holding = combined_holdings.get(item_id, {})
            item_data["current_holding"] = holding.get("total_qty", 0)
            item_data["holding_value"] = holding.get("total_value", 0)

            result[item_id] = item_data

        return result

    # =========================================================================
    # Portfolio Utilization
    # =========================================================================

    def get_portfolio_utilization(self) -> Dict[str, Any]:
        """
        Get portfolio utilization showing % allocation per item.

        Returns:
            {
                "total_portfolio_value": int,    # Gold + all item values
                "gold": int,
                "gold_percent": float,           # % of portfolio in gold
                "items_value": int,              # Total value of items
                "items_percent": float,          # % of portfolio in items
                "by_item": [                     # Sorted by value descending
                    {
                        "item_id": int,
                        "item_name": str,
                        "quantity": int,
                        "value": int,
                        "percent": float,        # % of total portfolio
                        "percent_of_items": float  # % of items-only portion
                    }
                ],
                "active_orders_value": int,      # Value tied up in active GE orders
                "active_orders_percent": float
            }
        """
        gold = self.get_gold()
        inventory = self.get_inventory_accounting()
        bank = self.get_bank_accounting()
        ge_slots = self.get_ge_slots()

        # Merge holdings
        combined = self._merge_holdings(inventory, bank)

        # Calculate total items value
        items_value = sum(item.get("total_value", 0) for item in combined.values())

        # Calculate active orders value (money tied up in GE)
        active_orders_value = 0
        for slot_num, slot_data in ge_slots.get("slots", {}).items():
            if slot_data and slot_data.get("status") == "active":
                qty = slot_data.get("quantity", 0)
                filled = slot_data.get("filled_quantity", 0)
                price = slot_data.get("price", 0)
                remaining = qty - filled
                if slot_data.get("type") == "buy":
                    # Buy order: money is tied up
                    active_orders_value += remaining * price

        # Total portfolio = gold + items + active buy orders
        total_portfolio = gold + items_value + active_orders_value

        # Build per-item breakdown
        by_item = []
        for item_id, item_data in combined.items():
            item_value = item_data.get("total_value", 0)
            if item_value > 0:
                by_item.append({
                    "item_id": item_id,
                    "item_name": item_data.get("item_name", "Unknown"),
                    "quantity": item_data.get("total_qty", 0),
                    "value": item_value,
                    "percent": (item_value / total_portfolio * 100) if total_portfolio > 0 else 0,
                    "percent_of_items": (item_value / items_value * 100) if items_value > 0 else 0
                })

        # Sort by value descending
        by_item.sort(key=lambda x: x["value"], reverse=True)

        return {
            "total_portfolio_value": total_portfolio,
            "gold": gold,
            "gold_percent": (gold / total_portfolio * 100) if total_portfolio > 0 else 0,
            "items_value": items_value,
            "items_percent": (items_value / total_portfolio * 100) if total_portfolio > 0 else 0,
            "by_item": by_item,
            "active_orders_value": active_orders_value,
            "active_orders_percent": (active_orders_value / total_portfolio * 100) if total_portfolio > 0 else 0
        }

    def get_item_utilization(self, item_id: int) -> Dict[str, Any]:
        """
        Get utilization info for a specific item.

        Args:
            item_id: Item ID to check

        Returns:
            {
                "item_id": int,
                "item_name": str,
                "quantity": int,
                "value": int,
                "percent_of_portfolio": float,
                "percent_of_items": float,
                "in_inventory": int,
                "in_bank": int,
                "in_ge_orders": int  # quantity in active GE orders
            }
        """
        utilization = self.get_portfolio_utilization()
        inventory = self.get_inventory_accounting()
        bank = self.get_bank_accounting()
        ge_slots = self.get_ge_slots()

        # Find item in by_item list
        item_util = None
        for item in utilization["by_item"]:
            if item["item_id"] == item_id:
                item_util = item
                break

        if not item_util:
            return {
                "item_id": item_id,
                "item_name": "Unknown",
                "quantity": 0,
                "value": 0,
                "percent_of_portfolio": 0,
                "percent_of_items": 0,
                "in_inventory": 0,
                "in_bank": 0,
                "in_ge_orders": 0
            }

        # Get breakdown by location
        inv_qty = inventory.get("items", {}).get(str(item_id), {}).get("quantity", 0)
        bank_qty = bank.get("items", {}).get(str(item_id), {}).get("quantity", 0)

        # Check GE orders
        ge_qty = 0
        for slot_num, slot_data in ge_slots.get("slots", {}).items():
            if slot_data and slot_data.get("item_id") == item_id:
                if slot_data.get("status") == "active":
                    ge_qty += slot_data.get("quantity", 0) - slot_data.get("filled_quantity", 0)

        return {
            "item_id": item_id,
            "item_name": item_util["item_name"],
            "quantity": item_util["quantity"],
            "value": item_util["value"],
            "percent_of_portfolio": item_util["percent"],
            "percent_of_items": item_util["percent_of_items"],
            "in_inventory": inv_qty,
            "in_bank": bank_qty,
            "in_ge_orders": ge_qty
        }

    def print_portfolio_utilization(self, top_n: int = 20):
        """Print portfolio utilization breakdown to console."""
        util = self.get_portfolio_utilization()

        print("\n" + "=" * 70)
        print("Portfolio Utilization")
        print("=" * 70)
        print(f"Total Portfolio Value: {util['total_portfolio_value']:,} GP")
        print("-" * 70)
        print(f"{'Component':<25} {'Value':>15} {'Percent':>10}")
        print("-" * 70)
        print(f"{'Gold':<25} {util['gold']:>15,} {util['gold_percent']:>9.1f}%")
        print(f"{'Items':<25} {util['items_value']:>15,} {util['items_percent']:>9.1f}%")
        print(f"{'Active GE Orders':<25} {util['active_orders_value']:>15,} {util['active_orders_percent']:>9.1f}%")
        print("-" * 70)

        if util["by_item"]:
            print(f"\nTop {min(top_n, len(util['by_item']))} Items by Value:")
            print(f"{'Item Name':<30} {'Qty':>10} {'Value':>12} {'% Port':>8} {'% Items':>8}")
            print("-" * 70)

            for item in util["by_item"][:top_n]:
                name = item["item_name"][:29]
                print(f"{name:<30} {item['quantity']:>10,} {item['value']:>12,} "
                      f"{item['percent']:>7.1f}% {item['percent_of_items']:>7.1f}%")

            if len(util["by_item"]) > top_n:
                remaining = len(util["by_item"]) - top_n
                remaining_value = sum(i["value"] for i in util["by_item"][top_n:])
                remaining_pct = sum(i["percent"] for i in util["by_item"][top_n:])
                print(f"{'... and ' + str(remaining) + ' more items':<30} {'':<10} "
                      f"{remaining_value:>12,} {remaining_pct:>7.1f}%")

        print("=" * 70 + "\n")

    def print_summary(self):
        """Print a summary of current inventory state to console."""
        state = self.get_inference_state()

        print("\n" + "=" * 60)
        print("Inventory Scanner Summary")
        print("=" * 60)
        print(f"Account: {state['account']['account_id']}")
        print(f"Online: {state['account']['is_online']}")
        print(f"Gold: {state['account']['gold']:,}")
        print(f"GE Slots Available: {state['account']['ge_slots_available']}")
        print("-" * 60)
        print(f"Inventory Items: {state['inventory']['item_count']}")
        print(f"Inventory Free Slots: {state['inventory']['free_slots']}")
        print(f"Bank Items: {state['bank']['item_count']}")
        print(f"Bank Value: {state['bank']['total_value']:,}")
        print(f"Tradeable Items in Bank: {state['bank']['tradeable_count']}")
        print("-" * 60)
        print(f"Active Orders: {len(state['ge_slots']['active_orders'])}")
        print(f"Buy Slots Used: {state['ge_slots']['buy_slots_used']}")
        print(f"Sell Slots Used: {state['ge_slots']['sell_slots_used']}")
        print("=" * 60 + "\n")

    def shutdown(self):
        """Clean up resources."""
        # Clear caches
        self._items_cache = None
        self._name_to_id_cache = None
        logger.info("InventoryScanner shutdown")


# =============================================================================
# Standalone Usage
# =============================================================================

def main():
    """Main entry point for standalone usage."""
    import argparse
    import sys

    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    parser = argparse.ArgumentParser(description="Inventory Scanner Utility")
    parser.add_argument("--account", "-a", default="b1llstar", help="Account ID")
    parser.add_argument("--summary", "-s", action="store_true", help="Print summary")
    parser.add_argument("--items", "-i", action="store_true", help="List all tradeable items")
    parser.add_argument("--bank", "-b", action="store_true", help="Show bank contents")
    parser.add_argument("--gold", "-g", action="store_true", help="Show gold balance")
    parser.add_argument("--state", "-S", action="store_true", help="Show full inference state (JSON)")

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        from config.firebase_config import get_service_account_path
        scanner = InventoryScanner(
            service_account_path=get_service_account_path(),
            account_id=args.account
        )

        if args.summary or (not args.items and not args.bank and not args.gold and not args.state):
            scanner.print_summary()

        if args.items:
            items = scanner.get_all_tradeable_items()
            print(f"\nTradeable Items ({len(items)} total):")
            for item_id, item_data in list(items.items())[:20]:  # First 20
                print(f"  {item_id}: {item_data.get('name', 'Unknown')} "
                      f"(GE limit: {item_data.get('ge_limit', 'N/A')})")
            if len(items) > 20:
                print(f"  ... and {len(items) - 20} more")

        if args.bank:
            bank = scanner.get_bank_accounting()
            print(f"\nBank Contents ({bank.get('item_count', 0)} items):")
            items = scanner.get_bank_items_by_value(min_value=1000)
            for item in items[:20]:
                print(f"  {item.get('item_name', 'Unknown')}: "
                      f"{item.get('quantity', 0):,}x @ {item.get('price_each', 0):,} = "
                      f"{item.get('total_value', 0):,}")

        if args.gold:
            gold = scanner.get_gold()
            print(f"\nGold Balance: {gold:,} GP")

        if args.state:
            import json
            state = scanner.get_inference_state()
            print("\nFull Inference State:")
            print(json.dumps(state, indent=2, default=str))

        scanner.shutdown()

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
