"""
Firebase-based PPO Inference Runner.

This module replaces the WebSocket-based inference system with Firebase Firestore.
It provides real-time bidirectional communication between PPO inference and the
GE Auto plugin.

Architecture:
    PPO Inference → Firebase → GE Auto Plugin
    PPO Inference ← Firebase ← GE Auto Plugin
"""

import asyncio
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("firebase_inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("run_firebase_inference")

# Add parent directory to path for imports
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

# Import Firebase components
try:
    from firebase.inference_bridge import InferenceBridge
    from config.firebase_config import (
        SERVICE_ACCOUNT_PATH,
        DEFAULT_ACCOUNT_ID,
        DECISION_INTERVAL,
        MIN_CONFIDENCE_THRESHOLD,
        MAX_ACTIVE_ORDERS,
        get_config
    )
    logger.info("Successfully imported Firebase components")
except ImportError as e:
    logger.error(f"Error importing Firebase components: {str(e)}")
    raise

# Import PPO components
try:
    from ge_env import GrandExchangeEnv
    from ppo_agent import PPOAgent
    from shared_knowledge import SharedKnowledgeRepository
    from volume_analysis import VolumeAnalyzer, create_volume_analyzer
    from ge_rest_client import GrandExchangeClient
    logger.info("Successfully imported PPO components")
except ImportError as e:
    logger.warning(f"PPO components not available: {str(e)}")
    # Will use mock agent if real components not available


class FirebaseInferenceRunner:
    """
    Main runner for Firebase-based PPO inference.

    This class orchestrates:
    - Connection to Firebase
    - PPO agent decision making
    - Order submission and tracking
    - Portfolio state management
    """

    def __init__(
        self,
        service_account_path: str = SERVICE_ACCOUNT_PATH,
        account_id: str = DEFAULT_ACCOUNT_ID,
        decision_interval: float = DECISION_INTERVAL,
        min_confidence: float = MIN_CONFIDENCE_THRESHOLD
    ):
        """
        Initialize the Firebase inference runner.

        Args:
            service_account_path: Path to Firebase service account JSON
            account_id: Account identifier
            decision_interval: Seconds between inference decisions
            min_confidence: Minimum confidence to execute trades
        """
        self.service_account_path = service_account_path
        self.account_id = account_id
        self.decision_interval = decision_interval
        self.min_confidence = min_confidence

        # Components
        self.bridge: Optional[InferenceBridge] = None
        self.agent: Optional[PPOAgent] = None
        self.volume_analyzer: Optional[VolumeAnalyzer] = None

        # State
        self._running = False
        self._paused = False
        self._consecutive_errors = 0
        self._max_consecutive_errors = 5

        # Statistics
        self.stats = {
            "decisions_made": 0,
            "orders_submitted": 0,
            "trades_completed": 0,
            "total_profit": 0,
            "start_time": None,
            "last_decision_time": None
        }

        # Item data - Only trade runes
        self.tradeable_items: Dict[int, str] = {}  # item_id -> name
        self.id_to_name_map: Dict[str, str] = {}
        self.name_to_id_map: Dict[str, str] = {}
        self.buy_limits: Dict[str, int] = {}

        # Magic runes we trade
        self.RUNES = {
            554: "Fire rune",
            555: "Water rune",
            556: "Air rune",
            557: "Earth rune",
            558: "Mind rune",
            559: "Body rune",
            560: "Death rune",
            561: "Nature rune",
            562: "Chaos rune",
            563: "Law rune",
            564: "Cosmic rune",
            565: "Blood rune",
            566: "Soul rune",
            9075: "Astral rune",
            21880: "Wrath rune",
            4694: "Steam rune",
            4695: "Mist rune",
            4696: "Dust rune",
            4697: "Smoke rune",
            4698: "Mud rune",
            4699: "Lava rune",
        }

    # =========================================================================
    # Initialization
    # =========================================================================

    def initialize(self) -> bool:
        """Initialize all components."""
        logger.info("Initializing Firebase inference runner...")

        try:
            # Initialize Firebase bridge
            self.bridge = InferenceBridge(
                service_account_path=self.service_account_path,
                account_id=self.account_id,
                on_trade_completed=self._on_trade_completed,
                on_order_status_changed=self._on_order_status_changed
            )

            # Load item data
            self._load_item_data()

            # Initialize PPO agent (if available)
            self._initialize_agent()

            logger.info("Firebase inference runner initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            return False

    def _load_item_data(self):
        """Load rune data for trading."""
        logger.info("Loading rune data for trading...")

        # Use only the predefined runes
        self.tradeable_items = self.RUNES.copy()

        for item_id, item_name in self.RUNES.items():
            self.id_to_name_map[str(item_id)] = item_name
            self.name_to_id_map[item_name] = str(item_id)
            # Runes have high buy limits
            self.buy_limits[item_name] = 25000

        logger.info(f"Loaded {len(self.tradeable_items)} runes for trading:")
        for item_id, name in sorted(self.tradeable_items.items()):
            logger.info(f"  {item_id}: {name}")

    def _initialize_agent(self):
        """Initialize the PPO agent."""
        logger.info("Initializing PPO agent...")

        try:
            # Try to load a trained agent
            # For now, we'll create a placeholder that can be replaced
            # with actual agent loading logic

            # Check for trained model
            import os
            model_path = os.path.join(
                os.path.dirname(__file__),
                "..", "models", "ppo_agent.pt"
            )

            if os.path.exists(model_path):
                logger.info(f"Loading trained model from {model_path}")
                # TODO: Load actual trained model
                # self.agent = PPOAgent.load(model_path)
            else:
                logger.warning("No trained model found, using mock decision maker")

        except Exception as e:
            logger.warning(f"Could not initialize PPO agent: {e}")

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self):
        """Start the inference runner."""
        if self._running:
            logger.warning("Inference runner already running")
            return

        logger.info("Starting Firebase inference runner...")

        # Start the bridge
        self.bridge.start()

        self._running = True
        self.stats["start_time"] = datetime.now(timezone.utc).isoformat()

        logger.info("Firebase inference runner started")

    def stop(self):
        """Stop the inference runner."""
        if not self._running:
            return

        logger.info("Stopping Firebase inference runner...")

        self._running = False

        if self.bridge:
            self.bridge.stop()

        logger.info("Firebase inference runner stopped")

    def shutdown(self):
        """Full shutdown."""
        self.stop()

        if self.bridge:
            self.bridge.shutdown()

        logger.info("Firebase inference runner shutdown complete")

    def pause(self):
        """Pause inference decisions (orders still tracked)."""
        self._paused = True
        logger.info("Inference paused")

    def resume(self):
        """Resume inference decisions."""
        self._paused = False
        logger.info("Inference resumed")

    # =========================================================================
    # Main Loop
    # =========================================================================

    async def run(self):
        """Main inference loop."""
        logger.info("Starting main inference loop...")

        while self._running:
            try:
                # Check if paused
                if self._paused:
                    await asyncio.sleep(1)
                    continue

                # Check if plugin is online
                if not self.bridge.is_plugin_online():
                    logger.debug("Plugin offline, waiting...")
                    await asyncio.sleep(5)
                    continue

                # Make inference decision
                await self._make_decision()

                # Reset error counter on success
                self._consecutive_errors = 0

                # Wait for next decision interval
                await asyncio.sleep(self.decision_interval)

            except Exception as e:
                logger.error(f"Error in inference loop: {e}")
                self._consecutive_errors += 1

                if self._consecutive_errors >= self._max_consecutive_errors:
                    logger.error(f"Too many consecutive errors ({self._consecutive_errors}), pausing...")
                    self._paused = True
                    await asyncio.sleep(60)
                    self._paused = False
                    self._consecutive_errors = 0
                else:
                    await asyncio.sleep(5)

        logger.info("Main inference loop ended")

    async def _make_decision(self):
        """Make a single inference decision."""
        self.stats["decisions_made"] += 1
        self.stats["last_decision_time"] = datetime.now(timezone.utc).isoformat()

        # Get current state
        portfolio = self.bridge.get_portfolio_summary()
        gold = portfolio.get("gold", 0)
        holdings = portfolio.get("holdings", {})

        # Get active orders
        active_orders = self.bridge.get_active_orders()
        active_count = len(active_orders)

        # Check if we have room for more orders
        if active_count >= MAX_ACTIVE_ORDERS:
            logger.debug(f"Max active orders reached ({active_count})")
            return

        # Get market data for decision making
        market_data = await self._get_market_data()

        # Make PPO decision
        decision = await self._get_ppo_decision(
            gold=gold,
            holdings=holdings,
            active_orders=active_orders,
            market_data=market_data
        )

        if decision is None:
            return

        action = decision.get("action")
        confidence = decision.get("confidence", 0)

        # Check confidence threshold
        if confidence < self.min_confidence:
            logger.debug(f"Decision confidence {confidence:.2f} below threshold {self.min_confidence}")
            return

        # Execute decision
        if action == "buy":
            await self._execute_buy(decision)
        elif action == "sell":
            await self._execute_sell(decision)
        elif action == "hold":
            logger.debug("Decision: HOLD")

    async def _get_market_data(self) -> Dict[str, Any]:
        """Get current market data for decision making."""
        # TODO: Uncomment this when using real market data from OSRS Wiki API
        # try:
        #     client = GrandExchangeClient()
        #     latest = client.get_latest()
        #     data_5m = client.get_5m()
        #
        #     return {
        #         "latest": latest,
        #         "5m": data_5m,
        #         "timestamp": datetime.now(timezone.utc).isoformat()
        #     }
        # except Exception as e:
        #     logger.warning(f"Could not get market data: {e}")
        #     return {}

        # Mock prices for runes (approximate GE prices) - for testing
        rune_prices = {
            "554": {"low": 4, "high": 5},      # Fire rune
            "555": {"low": 4, "high": 5},      # Water rune
            "556": {"low": 4, "high": 5},      # Air rune
            "557": {"low": 4, "high": 5},      # Earth rune
            "558": {"low": 3, "high": 4},      # Mind rune
            "559": {"low": 3, "high": 4},      # Body rune
            "560": {"low": 180, "high": 200},  # Death rune
            "561": {"low": 180, "high": 200},  # Nature rune
            "562": {"low": 60, "high": 70},    # Chaos rune
            "563": {"low": 150, "high": 170},  # Law rune
            "564": {"low": 120, "high": 140},  # Cosmic rune
            "565": {"low": 350, "high": 400},  # Blood rune
            "566": {"low": 300, "high": 350},  # Soul rune
            "9075": {"low": 140, "high": 160}, # Astral rune
            "21880": {"low": 400, "high": 450},# Wrath rune
            "4694": {"low": 500, "high": 600}, # Steam rune
            "4695": {"low": 500, "high": 600}, # Mist rune
            "4696": {"low": 500, "high": 600}, # Dust rune
            "4697": {"low": 500, "high": 600}, # Smoke rune
            "4698": {"low": 500, "high": 600}, # Mud rune
            "4699": {"low": 500, "high": 600}, # Lava rune
        }

        return {
            "latest": rune_prices,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def _get_ppo_decision(
        self,
        gold: int,
        holdings: Dict[str, Any],
        active_orders: List[Dict[str, Any]],
        market_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Get a decision from the PPO agent.

        Returns:
            Decision dict with action, item_id, quantity, price, confidence
            or None if no action should be taken
        """
        if self.agent is not None:
            # Use actual PPO agent
            # TODO: Implement actual PPO decision making
            pass

        import random

        # Get items we're already trading (have orders for)
        active_item_ids = set()
        for order in active_orders:
            active_item_ids.add(order.get("item_id"))

        # Get rune holdings (filter holdings to only runes we trade)
        rune_holdings = {}
        for item_id_str, item_data in holdings.items():
            item_id = int(item_id_str) if item_id_str.isdigit() else None
            if item_id and item_id in self.RUNES:
                rune_holdings[item_id] = item_data

        # Decision: 30% chance to act (buy or sell)
        if random.random() > 0.3:
            return None

        latest = market_data.get("latest", {})

        # If we have runes, 50% chance to sell one
        if rune_holdings and random.random() > 0.5:
            # Sell a rune we own
            item_id = random.choice(list(rune_holdings.keys()))
            item_data = rune_holdings[item_id]
            item_name = self.RUNES[item_id]

            # Skip if we already have an order for this item
            if item_id in active_item_ids:
                return None

            quantity = item_data.get("quantity", 0)
            if quantity <= 0:
                return None

            # Sell up to 1000 at a time
            sell_quantity = min(quantity, random.randint(100, 1000))

            # Get current high price from market data
            item_prices = latest.get(str(item_id), {})
            price = item_prices.get("high", 5)  # Default 5gp for runes

            # Add small margin for profit
            sell_price = int(price * 1.02)  # 2% above market

            return {
                "action": "sell",
                "item_id": item_id,
                "item_name": item_name,
                "quantity": sell_quantity,
                "price": sell_price,
                "confidence": random.uniform(0.75, 0.95)
            }
        else:
            # Buy a random rune
            # Pick a rune we don't have an active order for
            available_runes = [
                (item_id, name) for item_id, name in self.RUNES.items()
                if item_id not in active_item_ids
            ]

            if not available_runes:
                return None

            item_id, item_name = random.choice(available_runes)

            # Get current low price from market data
            item_prices = latest.get(str(item_id), {})
            price = item_prices.get("low", 5)  # Default 5gp for runes

            # Bid slightly below market for better fills
            buy_price = max(1, int(price * 0.98))  # 2% below market

            # Calculate affordable quantity (use 10-20% of gold per order)
            max_spend = int(gold * random.uniform(0.1, 0.2))
            max_quantity = max_spend // max(buy_price, 1)

            if max_quantity <= 0:
                return None

            # Buy between 100-1000 runes
            buy_quantity = min(max_quantity, random.randint(100, 1000))

            if buy_quantity < 10:  # Don't bother with tiny orders
                return None

            return {
                "action": "buy",
                "item_id": item_id,
                "item_name": item_name,
                "quantity": buy_quantity,
                "price": buy_price,
                "confidence": random.uniform(0.75, 0.95)
            }

    async def _execute_buy(self, decision: Dict[str, Any]):
        """Execute a buy order."""
        item_id = decision["item_id"]
        item_name = decision["item_name"]
        quantity = decision["quantity"]
        price = decision["price"]
        confidence = decision["confidence"]

        logger.info(f"Executing BUY: {quantity}x {item_name} @ {price} (confidence: {confidence:.2f})")

        order_id = self.bridge.submit_buy_order(
            item_id=item_id,
            item_name=item_name,
            quantity=quantity,
            price=price,
            confidence=confidence,
            strategy="ppo_firebase",
            metadata={"decision_time": datetime.now(timezone.utc).isoformat()}
        )

        if order_id:
            self.stats["orders_submitted"] += 1
            logger.info(f"Buy order submitted: {order_id}")
        else:
            logger.warning("Failed to submit buy order")

    async def _execute_sell(self, decision: Dict[str, Any]):
        """Execute a sell order."""
        item_id = decision["item_id"]
        item_name = decision["item_name"]
        quantity = decision["quantity"]
        price = decision["price"]
        confidence = decision["confidence"]

        logger.info(f"Executing SELL: {quantity}x {item_name} @ {price} (confidence: {confidence:.2f})")

        order_id = self.bridge.submit_sell_order(
            item_id=item_id,
            item_name=item_name,
            quantity=quantity,
            price=price,
            confidence=confidence,
            strategy="ppo_firebase",
            metadata={"decision_time": datetime.now(timezone.utc).isoformat()}
        )

        if order_id:
            self.stats["orders_submitted"] += 1
            logger.info(f"Sell order submitted: {order_id}")
        else:
            logger.warning("Failed to submit sell order")

    # =========================================================================
    # Callbacks
    # =========================================================================

    def _on_trade_completed(self, trade_data: Dict[str, Any]):
        """Called when a trade is completed by the plugin."""
        self.stats["trades_completed"] += 1

        action = trade_data.get("action", "unknown")
        item_name = trade_data.get("item_name", "unknown")
        quantity = trade_data.get("quantity", 0)
        price = trade_data.get("price", 0)
        total_cost = trade_data.get("total_cost", quantity * price)

        logger.info(f"Trade completed: {action} {quantity}x {item_name} @ {price} = {total_cost}")

        # Track profit (rough estimate - actual P&L calculated from trades collection)
        if action == "sell":
            self.stats["total_profit"] += total_cost
        elif action == "buy":
            self.stats["total_profit"] -= total_cost

    def _on_order_status_changed(self, order_id: str, status: str, order_data: Dict):
        """Called when an order status changes."""
        logger.info(f"Order {order_id} status changed to: {status}")

    # =========================================================================
    # Status and Statistics
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get current runner status."""
        bridge_status = self.bridge.get_status() if self.bridge else {}

        return {
            "running": self._running,
            "paused": self._paused,
            "account_id": self.account_id,
            "consecutive_errors": self._consecutive_errors,
            **self.stats,
            **bridge_status
        }

    def print_status(self):
        """Print current status to console."""
        status = self.get_status()

        print("\n" + "=" * 60)
        print("Firebase Inference Status")
        print("=" * 60)
        print(f"  Running: {status.get('running')}")
        print(f"  Paused: {status.get('paused')}")
        print(f"  Account: {status.get('account_id')}")
        print(f"  Plugin Online: {status.get('plugin_online')}")
        print(f"  Gold: {status.get('gold', 0):,}")
        print(f"  Holdings: {status.get('holdings_count', 0)} items")
        print(f"  Active Orders: {status.get('active_orders', 0)}")
        print(f"  Available Slots: {status.get('available_slots', 0)}")
        print("-" * 60)
        print(f"  Decisions Made: {status.get('decisions_made', 0)}")
        print(f"  Orders Submitted: {status.get('orders_submitted', 0)}")
        print(f"  Trades Completed: {status.get('trades_completed', 0)}")
        print(f"  Net Profit: {status.get('net_profit', 0):,}")
        print("=" * 60 + "\n")


# =============================================================================
# Main Entry Point
# =============================================================================

async def main():
    """Main entry point for Firebase inference."""
    logger.info("=" * 60)
    logger.info("Firebase-based PPO Inference Runner")
    logger.info("=" * 60)

    # Print configuration
    config = get_config()
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    # Create runner
    runner = FirebaseInferenceRunner()

    # Handle shutdown signals
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal")
        runner.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize
    if not runner.initialize():
        logger.error("Failed to initialize, exiting")
        return

    # Start
    runner.start()

    try:
        # Run main loop
        await runner.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Cleanup
        runner.shutdown()
        runner.print_status()


if __name__ == "__main__":
    asyncio.run(main())
