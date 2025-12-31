"""
Firebase Configuration V2 for GE Auto V2 Architecture.

This configuration matches the V2 Firestore schema defined in GE_AUTO_V2_PROPOSAL.md.

Key differences from V1:
- Simplified collection structure (4 docs + 1 subcollection vs 7+ subcollections)
- Single documents for portfolio, inventory, bank, ge_state
- Orders as the only subcollection
"""

import os
from pathlib import Path

# ============================================================================
# Firebase Connection
# ============================================================================

PROJECT_ID = "ppoflipperopus"


def get_service_account_path() -> str:
    """Find the service account JSON file."""
    possible_paths = [
        Path(__file__).parent.parent / "ppoflipperopus-firebase-adminsdk-fbsvc-907a50dae7.json",
        Path(__file__).parent / "ppoflipperopus-firebase-adminsdk-fbsvc-907a50dae7.json",
        Path(os.environ.get("FIREBASE_SERVICE_ACCOUNT", "")),
        Path.home() / ".config" / "ppoflipperopus" / "service_account.json",
    ]

    for path in possible_paths:
        if path.exists():
            return str(path)

    return str(possible_paths[0])


SERVICE_ACCOUNT_PATH = get_service_account_path()

# Default account ID (player name, lowercased, spaces -> underscores)
DEFAULT_ACCOUNT_ID = os.environ.get("PPO_ACCOUNT_ID", "b1llstar")


# ============================================================================
# Collection Structure (V2)
# ============================================================================

# Top-level collections
COLLECTION_ITEMS = "items"            # Static item database
COLLECTION_ITEM_NAMES = "itemNames"   # name -> id lookup
COLLECTION_ACCOUNTS = "accounts"      # Per-account data

# Account subdocuments (NOT subcollections - single documents)
# Path: accounts/{accountId}/{docName}/current
DOC_PORTFOLIO = "portfolio"           # Summary: gold, total_value, plugin_online
DOC_INVENTORY = "inventory"           # Current inventory contents
DOC_BANK = "bank"                     # Current bank contents
DOC_GE_STATE = "ge_state"             # All 8 GE slot states

# Account subcollections
SUBCOLLECTION_ORDERS = "orders"       # Buy/sell orders


# ============================================================================
# Order Schema
# ============================================================================

# Order actions
ACTION_BUY = "buy"
ACTION_SELL = "sell"

# Order status lifecycle
STATUS_PENDING = "pending"      # Created by inference, waiting for plugin
STATUS_RECEIVED = "received"    # Plugin has received the order
STATUS_PLACED = "placed"        # Order placed in GE slot
STATUS_PARTIAL = "partial"      # Partially filled
STATUS_COMPLETED = "completed"  # Fully filled and collected
STATUS_CANCELLED = "cancelled"  # Cancelled
STATUS_FAILED = "failed"        # Failed to execute

# Active statuses (order is being processed)
ACTIVE_STATUSES = [STATUS_PENDING, STATUS_RECEIVED, STATUS_PLACED, STATUS_PARTIAL]

# Terminal statuses (order is done)
TERMINAL_STATUSES = [STATUS_COMPLETED, STATUS_CANCELLED, STATUS_FAILED]


# ============================================================================
# GE Slot Schema
# ============================================================================

SLOT_EMPTY = "empty"
SLOT_ACTIVE = "active"
SLOT_COMPLETE = "complete"


# ============================================================================
# Timing Configuration
# ============================================================================

# Inference decision interval (seconds)
DECISION_INTERVAL = 5

# Plugin heartbeat interval (seconds) - how often plugin updates portfolio
HEARTBEAT_INTERVAL = 30

# Maximum age of heartbeat to consider plugin online (seconds)
PLUGIN_ONLINE_THRESHOLD = 120

# Market data cache TTL (seconds)
MARKET_CACHE_TTL = 60


# ============================================================================
# Trading Limits
# ============================================================================

# Maximum concurrent orders (GE has 8 slots)
MAX_ACTIVE_ORDERS = 8

# Minimum confidence threshold to execute a trade
MIN_CONFIDENCE_THRESHOLD = 0.05

# Maximum value for a single order (gp)
MAX_ORDER_VALUE = 10_000_000  # 10M

# Minimum value for a single order (gp)
MIN_ORDER_VALUE = 10_000  # 10K

# Maximum quantity per order
MAX_ORDER_QUANTITY = 100_000


# ============================================================================
# Error Handling
# ============================================================================

# Maximum consecutive errors before pausing
MAX_CONSECUTIVE_ERRORS = 5

# Pause duration after max errors (seconds)
ERROR_PAUSE_DURATION = 60


# ============================================================================
# Feature Flags
# ============================================================================

# Enable trade execution (False for dry-run)
EXECUTE_TRADES = True

# Enable verbose logging
VERBOSE_LOGGING = False


# ============================================================================
# Helper Functions
# ============================================================================

def get_account_path(account_id: str) -> str:
    """Get the Firestore path to an account document."""
    return f"{COLLECTION_ACCOUNTS}/{account_id}"


def get_portfolio_path(account_id: str) -> str:
    """Get the Firestore path to portfolio document."""
    return f"{COLLECTION_ACCOUNTS}/{account_id}/{DOC_PORTFOLIO}/current"


def get_inventory_path(account_id: str) -> str:
    """Get the Firestore path to inventory document."""
    return f"{COLLECTION_ACCOUNTS}/{account_id}/{DOC_INVENTORY}/current"


def get_bank_path(account_id: str) -> str:
    """Get the Firestore path to bank document."""
    return f"{COLLECTION_ACCOUNTS}/{account_id}/{DOC_BANK}/current"


def get_ge_state_path(account_id: str) -> str:
    """Get the Firestore path to GE state document."""
    return f"{COLLECTION_ACCOUNTS}/{account_id}/{DOC_GE_STATE}/current"


def get_orders_path(account_id: str) -> str:
    """Get the Firestore path to orders subcollection."""
    return f"{COLLECTION_ACCOUNTS}/{account_id}/{SUBCOLLECTION_ORDERS}"


# ============================================================================
# Config Dictionary
# ============================================================================

CONFIG_V2 = {
    # Connection
    "project_id": PROJECT_ID,
    "service_account_path": SERVICE_ACCOUNT_PATH,
    "default_account_id": DEFAULT_ACCOUNT_ID,

    # Collections
    "collection_items": COLLECTION_ITEMS,
    "collection_item_names": COLLECTION_ITEM_NAMES,
    "collection_accounts": COLLECTION_ACCOUNTS,

    # Documents
    "doc_portfolio": DOC_PORTFOLIO,
    "doc_inventory": DOC_INVENTORY,
    "doc_bank": DOC_BANK,
    "doc_ge_state": DOC_GE_STATE,
    "subcollection_orders": SUBCOLLECTION_ORDERS,

    # Timing
    "decision_interval": DECISION_INTERVAL,
    "heartbeat_interval": HEARTBEAT_INTERVAL,
    "plugin_online_threshold": PLUGIN_ONLINE_THRESHOLD,
    "market_cache_ttl": MARKET_CACHE_TTL,

    # Limits
    "max_active_orders": MAX_ACTIVE_ORDERS,
    "min_confidence_threshold": MIN_CONFIDENCE_THRESHOLD,
    "max_order_value": MAX_ORDER_VALUE,
    "min_order_value": MIN_ORDER_VALUE,
    "max_order_quantity": MAX_ORDER_QUANTITY,

    # Error handling
    "max_consecutive_errors": MAX_CONSECUTIVE_ERRORS,
    "error_pause_duration": ERROR_PAUSE_DURATION,

    # Features
    "execute_trades": EXECUTE_TRADES,
    "verbose_logging": VERBOSE_LOGGING,
}


def print_config():
    """Print current configuration."""
    print("=" * 60)
    print("Firebase Configuration V2")
    print("=" * 60)
    for key, value in CONFIG_V2.items():
        print(f"  {key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
