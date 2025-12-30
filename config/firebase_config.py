"""
Firebase Configuration for PPO Inference.

This module contains all configuration settings for Firebase integration
with the PPO inference server.
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
        # Root of PPOFlipperOpus
        Path(__file__).parent.parent / "ppoflipperopus-firebase-adminsdk-fbsvc-0506134b11.json",
        # Config directory
        Path(__file__).parent / "ppoflipperopus-firebase-adminsdk-fbsvc-0506134b11.json",
        # Environment variable
        Path(os.environ.get("FIREBASE_SERVICE_ACCOUNT", "")),
        # User home
        Path.home() / ".config" / "ppoflipperopus" / "service_account.json",
    ]

    for path in possible_paths:
        if path.exists():
            return str(path)

    # Default to root location
    return str(Path(__file__).parent.parent / "ppoflipperopus-firebase-adminsdk-fbsvc-0506134b11.json")


SERVICE_ACCOUNT_PATH = get_service_account_path()

# Default account ID (can be overridden at runtime)
# IMPORTANT: This must match the player name in RuneLite (lowercased, spaces replaced with underscores)
# The Java plugin uses client.getLocalPlayer().getName().toLowerCase().replace(" ", "_")
DEFAULT_ACCOUNT_ID = os.environ.get("PPO_ACCOUNT_ID", "b1llstar")

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
MIN_CONFIDENCE_THRESHOLD = 0.7

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
