"""
Firebase Configuration for PPO Inference.

This module contains all configuration settings for Firebase integration
with the PPO inference server.

Schema Overview:
/accounts/{accountId}          - Parent collection (one per character)
  /orders/{orderId}            - Buy/sell orders (status tracking)
  /portfolio/{itemId}          - Accumulated items from buy orders minus sells
  /inventory/{itemId}          - Current inventory items with portfolio flag
  /bank/{itemId}               - Current bank items with portfolio flag
  /trades/{tradeId}            - Completed trade records
  /actions/{actionId}          - Action audit log
  /ge_slots/current            - GE slot states
  /commands/{commandId}        - Commands from PPO to plugin

/items/{itemId}                - Item metadata (IDs, prices)
/itemNames/{normalizedName}    - Item name to ID mapping
"""

import os
from pathlib import Path

# ============================================================================
# Firebase Connection
# ============================================================================

# Project ID
PROJECT_ID = "ppoflipperopus"

# Service account path - check multiple locations
def get_service_account_path() -> str:
    """Find the service account JSON file."""
    possible_paths = [
        # Root of PPOFlipperOpus (new service account)
        Path(__file__).parent.parent / "ppoflipperopus-firebase-adminsdk-fbsvc-907a50dae7.json",
        # Config directory
        Path(__file__).parent / "ppoflipperopus-firebase-adminsdk-fbsvc-907a50dae7.json",
        # Environment variable
        Path(os.environ.get("FIREBASE_SERVICE_ACCOUNT", "")),
        # User home
        Path.home() / ".config" / "ppoflipperopus" / "service_account.json",
    ]

    for path in possible_paths:
        if path.exists():
            return str(path)

    # Default to root location
    return str(Path(__file__).parent.parent / "ppoflipperopus-firebase-adminsdk-fbsvc-907a50dae7.json")


SERVICE_ACCOUNT_PATH = get_service_account_path()

# Default account ID (can be overridden at runtime)
# IMPORTANT: This must match the player name in RuneLite (lowercased, spaces replaced with underscores)
# The Java plugin uses client.getLocalPlayer().getName().toLowerCase().replace(" ", "_")
DEFAULT_ACCOUNT_ID = os.environ.get("PPO_ACCOUNT_ID", "b1llstar")

# ============================================================================
# Collection and Field Constants (matching Java FirebaseConfig.java)
# ============================================================================

# Top-Level Collection Names
COLLECTION_ITEMS = "items"
COLLECTION_ITEM_NAMES = "itemNames"
COLLECTION_ACCOUNTS = "accounts"
COLLECTION_MARKET_DATA = "market_data"

# Account Subcollection Names
SUBCOLLECTION_ORDERS = "orders"
SUBCOLLECTION_TRADES = "trades"
SUBCOLLECTION_PORTFOLIO = "portfolio"
SUBCOLLECTION_INVENTORY = "inventory"
SUBCOLLECTION_BANK = "bank"
SUBCOLLECTION_ACTIONS = "actions"
SUBCOLLECTION_GE_SLOTS = "ge_slots"
SUBCOLLECTION_COMMANDS = "commands"

# Document Names
DOC_CURRENT = "current"
DOC_SNAPSHOT = "snapshot"
DOC_SUMMARY = "summary"

# Command Types (PPO -> Plugin)
CMD_WITHDRAW = "withdraw"
CMD_DEPOSIT = "deposit"
CMD_DEPOSIT_ALL = "deposit_all"
CMD_OPEN_BANK = "open_bank"
CMD_CLOSE_BANK = "close_bank"
CMD_OPEN_GE = "open_ge"
CMD_CLOSE_GE = "close_ge"
CMD_SYNC_PORTFOLIO = "sync_portfolio"
CMD_SYNC_ORDERS = "sync_orders"

# Command Status
CMD_STATUS_PENDING = "pending"
CMD_STATUS_RECEIVED = "received"
CMD_STATUS_EXECUTING = "executing"
CMD_STATUS_COMPLETED = "completed"
CMD_STATUS_FAILED = "failed"

# Order Status Values (matching Java side)
STATUS_PENDING = "pending"
STATUS_RECEIVED = "received"
STATUS_PLACED = "placed"
STATUS_PARTIAL = "partial"
STATUS_COMPLETED = "completed"
STATUS_CANCELLED = "cancelled"
STATUS_FAILED = "failed"

# Action Types
ACTION_BUY = "buy"
ACTION_SELL = "sell"
ACTION_WITHDRAW = "withdraw"
ACTION_DEPOSIT = "deposit"
ACTION_CANCEL = "cancel"
ACTION_COLLECT = "collect"

# Order Source (who created the order)
SOURCE_PPO = "ppo"
SOURCE_MANUAL = "manual"

# Order Field Names
FIELD_ORDER_ID = "order_id"
FIELD_ITEM_ID = "item_id"
FIELD_ITEM_NAME = "item_name"
FIELD_ACTION = "action"
FIELD_QUANTITY = "quantity"
FIELD_PRICE = "price"
FIELD_STATUS = "status"
FIELD_CREATED_AT = "created_at"
FIELD_UPDATED_AT = "updated_at"
FIELD_COMPLETED_AT = "completed_at"
FIELD_GE_SLOT = "ge_slot"
FIELD_ERROR_MESSAGE = "error_message"
FIELD_FILLED_QUANTITY = "filled_quantity"
FIELD_METADATA = "metadata"
FIELD_SOURCE = "source"
FIELD_GOLD_EXCHANGED = "gold_exchanged"
FIELD_TAX_PAID = "tax_paid"

# Portfolio Field Names
FIELD_AVG_COST = "avg_cost"
FIELD_TOTAL_INVESTED = "total_invested"
FIELD_LOCATION = "location"
FIELD_TRADES = "trades"

# Location Values (where portfolio items are stored)
LOCATION_INVENTORY = "inventory"
LOCATION_BANK = "bank"
LOCATION_MIXED = "mixed"

# Inventory/Bank Field Names
FIELD_SLOT = "slot"
FIELD_TAB = "tab"
FIELD_IS_PORTFOLIO_ITEM = "is_portfolio_item"
FIELD_NOTED = "noted"

# Common Field Names
FIELD_GOLD = "gold"
FIELD_ITEMS = "items"
FIELD_TOTAL_VALUE = "total_value"
FIELD_TOTAL_COST = "total_cost"

# GE Tax Rate
GE_TAX_RATE = 0.01  # 1%

# ============================================================================
# Timing Configuration
# ============================================================================

# How often to run inference decisions (seconds)
DECISION_INTERVAL = 5

# How often to send heartbeat to Firestore (seconds)
HEARTBEAT_INTERVAL = 30

# How often to sync portfolio state (seconds)
PORTFOLIO_SYNC_INTERVAL = 60

# Maximum age of plugin heartbeat to consider online (seconds)
PLUGIN_ONLINE_THRESHOLD = 120

# Order timeout - cancel if not placed within this time (seconds)
ORDER_TIMEOUT = 600  # 10 minutes

# Stale order threshold - warn if order pending too long (seconds)
STALE_ORDER_THRESHOLD = 300  # 5 minutes

# ============================================================================
# Trading Limits
# ============================================================================

# Maximum number of concurrent active orders
MAX_ACTIVE_ORDERS = 8

# Maximum pending orders before blocking new submissions
MAX_PENDING_ORDERS = 8

# Maximum number of outstanding positions (different items held)
MAX_OUTSTANDING_POSITIONS = 25

# Minimum confidence threshold to execute a trade
MIN_CONFIDENCE_THRESHOLD = 0.05  # Very low for early-stage model testing

# Maximum value for a single order (gp)
MAX_ORDER_VALUE = 10_000_000  # 10M

# Minimum value for a single order (gp)
MIN_ORDER_VALUE = 10_000  # 10K

# Maximum quantity per order (for safety)
MAX_ORDER_QUANTITY = 100_000

# ============================================================================
# Retry and Error Handling
# ============================================================================

# Maximum retries for failed orders
MAX_ORDER_RETRIES = 3

# Delay between retries (seconds)
RETRY_DELAY = 5

# Maximum consecutive errors before pausing inference
MAX_CONSECUTIVE_ERRORS = 5

# Pause duration after max errors (seconds)
ERROR_PAUSE_DURATION = 60

# ============================================================================
# Logging
# ============================================================================

# Log level for Firebase operations
FIREBASE_LOG_LEVEL = "INFO"

# Log file path
LOG_FILE = "firebase_inference.log"

# ============================================================================
# Feature Flags
# ============================================================================

# Enable trade execution (set False for dry-run mode)
EXECUTE_TRADES = True

# Enable automatic order cancellation for stale orders
AUTO_CANCEL_STALE = True

# Enable portfolio sync on startup
SYNC_ON_STARTUP = True

# Enable verbose logging of all Firebase operations
VERBOSE_FIREBASE_LOGGING = False

# ============================================================================
# Combined Config Dictionary
# ============================================================================

FIREBASE_CONFIG = {
    # Connection
    "project_id": PROJECT_ID,
    "service_account_path": SERVICE_ACCOUNT_PATH,
    "default_account_id": DEFAULT_ACCOUNT_ID,

    # Timing
    "decision_interval": DECISION_INTERVAL,
    "heartbeat_interval": HEARTBEAT_INTERVAL,
    "portfolio_sync_interval": PORTFOLIO_SYNC_INTERVAL,
    "plugin_online_threshold": PLUGIN_ONLINE_THRESHOLD,
    "order_timeout": ORDER_TIMEOUT,
    "stale_order_threshold": STALE_ORDER_THRESHOLD,

    # Trading Limits
    "max_active_orders": MAX_ACTIVE_ORDERS,
    "max_pending_orders": MAX_PENDING_ORDERS,
    "max_outstanding_positions": MAX_OUTSTANDING_POSITIONS,
    "min_confidence_threshold": MIN_CONFIDENCE_THRESHOLD,
    "max_order_value": MAX_ORDER_VALUE,
    "min_order_value": MIN_ORDER_VALUE,
    "max_order_quantity": MAX_ORDER_QUANTITY,

    # Retry and Error Handling
    "max_order_retries": MAX_ORDER_RETRIES,
    "retry_delay": RETRY_DELAY,
    "max_consecutive_errors": MAX_CONSECUTIVE_ERRORS,
    "error_pause_duration": ERROR_PAUSE_DURATION,

    # Feature Flags
    "execute_trades": EXECUTE_TRADES,
    "auto_cancel_stale": AUTO_CANCEL_STALE,
    "sync_on_startup": SYNC_ON_STARTUP,
    "verbose_firebase_logging": VERBOSE_FIREBASE_LOGGING,
}


def get_config() -> dict:
    """Get the Firebase configuration dictionary."""
    return FIREBASE_CONFIG.copy()


def print_config():
    """Print current configuration for debugging."""
    print("=" * 60)
    print("Firebase Configuration")
    print("=" * 60)
    for key, value in FIREBASE_CONFIG.items():
        print(f"  {key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
