"""
Inference Runner V2 - Simplified PPO inference using Firebase V2.

This module provides a clean, streamlined inference loop using the V2 Firebase
architecture. It focuses on simplicity and clarity while maintaining full
compatibility with the trained PPO model.
"""

import asyncio
import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set

import numpy as np
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("inference_v2.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("inference_v2")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import V2 Firebase components
from firebase_v2 import InferenceBridgeV2
from firebase_v2.types import Order, Holding


class InferenceRunnerV2:
    """
    Simplified PPO inference runner using Firebase V2 architecture.

    Key simplifications from V1:
    - Uses InferenceBridgeV2 for all Firebase operations
    - Single responsibility: make decisions and submit orders
    - Cleaner state access via typed dataclasses
    - No complex retry logic (handled by plugin)
    """

    # Configuration
    DECISION_INTERVAL = 5.0  # Seconds between decisions
    MIN_CONFIDENCE = 0.05   # Minimum confidence to execute
    MAX_ACTIVE_ORDERS = 8   # Maximum concurrent orders
    MARKET_CACHE_TTL = 60   # Market data cache TTL in seconds

    def __init__(
        self,
        account_id: str = "b1llstar",
        decision_interval: float = DECISION_INTERVAL,
        min_confidence: float = MIN_CONFIDENCE,
        dry_run: bool = False
    ):
        """
        Initialize the inference runner.

        Args:
            account_id: Account ID (player name)
            decision_interval: Seconds between inference decisions
            min_confidence: Minimum confidence to execute trades
            dry_run: If True, don't submit orders (just log)
        """
        self.account_id = account_id
        self.decision_interval = decision_interval
        self.min_confidence = min_confidence
        self.dry_run = dry_run

        # Firebase bridge
        self.bridge: Optional[InferenceBridgeV2] = None

        # PPO Model components
        self.feature_extractor = None
        self.actor = None
        self.critic = None
        self.device = None

        # Model metadata
        self.item_list: List[int] = []
        self.model_config: Dict[str, Any] = {}
        self.price_ranges: Dict[int, Tuple[float, float]] = {}

        # Item data
        self.items: Dict[int, Dict[str, Any]] = {}  # item_id -> {name, ge_limit}

        # Market data cache
        self._market_cache: Optional[Dict[str, Any]] = None
        self._market_cache_time: float = 0
        self._ge_client = None

        # Runtime state
        self._running = False
        self._consecutive_errors = 0

        # Statistics
        self.stats = {
            "decisions": 0,
            "orders_submitted": 0,
            "buys": 0,
            "sells": 0,
            "holds": 0,
            "start_time": None
        }

    # =========================================================================
    # Initialization
    # =========================================================================

    def initialize(self) -> bool:
        """Initialize all components."""
        logger.info("Initializing Inference Runner V2...")

        try:
            # Initialize Firebase bridge
            self.bridge = InferenceBridgeV2()
            if not self.bridge.initialize(
                account_id=self.account_id,
                on_order_completed=self._on_order_completed,
                on_order_failed=self._on_order_failed
            ):
                logger.error("Failed to initialize Firebase bridge")
                return False

            # Load PPO model
            if not self._load_model():
                logger.error("Failed to load PPO model")
                return False

            # Load item data
            self._load_items()

            logger.info("Inference Runner V2 initialized successfully")
            self.bridge.print_status()
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_model(self) -> bool:
        """Load the trained PPO model."""
        model_path = Path(__file__).parent.parent / "model" / "shared_model_final.pt"

        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return False

        try:
            # Determine device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("Using Apple Silicon MPS")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU")

            # Load checkpoint
            logger.info(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)

            # Extract metadata
            self.item_list = checkpoint.get('item_list', [])
            self.model_config = checkpoint.get('config', {})
            self.price_ranges = checkpoint.get('price_ranges', {})

            logger.info(f"Model trained on {len(self.item_list)} items")

            # Import and create network components
            from ppo_agent import FeatureExtractor, ActorHead, CriticHead

            hidden_size = self.model_config.get('hidden_size', 256)
            num_layers = self.model_config.get('num_layers', 3)
            obs_dim = self.model_config.get('obs_dim', 1407)
            n_items = self.model_config.get('n_items', 200)
            price_bins = self.model_config.get('price_bins', 21)
            quantity_bins = self.model_config.get('quantity_bins', 11)
            wait_steps_bins = self.model_config.get('wait_steps_bins', 5)

            self.feature_extractor = FeatureExtractor(
                input_dim=obs_dim,
                hidden_size=hidden_size,
                num_layers=num_layers
            ).to(self.device)

            self.actor = ActorHead(
                feature_dim=hidden_size,
                n_items=n_items,
                price_bins=price_bins,
                quantity_bins=quantity_bins,
                wait_steps_bins=wait_steps_bins
            ).to(self.device)

            self.critic = CriticHead(feature_dim=hidden_size).to(self.device)

            # Load weights
            self.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])

            # Eval mode
            self.feature_extractor.eval()
            self.actor.eval()
            self.critic.eval()

            logger.info("PPO model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_items(self):
        """Load item data for tradeable items."""
        items_path = Path(__file__).parent.parent / "mongo_data" / "items.json"

        try:
            with open(items_path, 'r') as f:
                items_data = json.load(f)

            # Build lookup
            all_items = {}
            for item in items_data:
                item_id = item.get('id')
                if item_id:
                    all_items[item_id] = {
                        'name': item.get('name', f'Item_{item_id}'),
                        'ge_limit': item.get('ge_limit', 5000) or 5000
                    }

            # Filter to model items
            for item_id in self.item_list:
                if item_id in all_items:
                    self.items[item_id] = all_items[item_id]
                else:
                    self.items[item_id] = {
                        'name': f'Item_{item_id}',
                        'ge_limit': 5000
                    }

            logger.info(f"Loaded {len(self.items)} tradeable items")

        except Exception as e:
            logger.error(f"Failed to load items: {e}")

    # =========================================================================
    # Callbacks
    # =========================================================================

    def _on_order_completed(self, order: Order):
        """Handle order completion."""
        action = "BOUGHT" if order.is_buy else "SOLD"
        logger.info(
            f"Order completed: {action} {order.filled_quantity}x {order.item_name} "
            f"@ {order.price}gp (total: {order.total_cost:,}gp)"
        )

    def _on_order_failed(self, order: Order):
        """Handle order failure."""
        logger.warning(f"Order failed: {order.item_name} - {order.error}")

    # =========================================================================
    # Main Loop
    # =========================================================================

    async def run(self):
        """Main inference loop."""
        if not self.bridge:
            logger.error("Bridge not initialized")
            return

        self._running = True
        self.stats["start_time"] = datetime.now(timezone.utc).isoformat()

        logger.info(f"Starting inference loop (interval={self.decision_interval}s)")

        # Set up signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self.stop)

        while self._running:
            try:
                await self._decision_cycle()
                self._consecutive_errors = 0
            except Exception as e:
                self._consecutive_errors += 1
                logger.error(f"Decision error ({self._consecutive_errors}): {e}")

                if self._consecutive_errors >= 5:
                    logger.error("Too many errors, pausing for 60s")
                    await asyncio.sleep(60)
                    self._consecutive_errors = 0

            await asyncio.sleep(self.decision_interval)

        logger.info("Inference loop stopped")

    def stop(self):
        """Stop the inference loop."""
        logger.info("Stopping inference...")
        self._running = False

    def shutdown(self):
        """Shutdown all components."""
        self.stop()
        if self.bridge:
            self.bridge.shutdown()
        logger.info("Inference Runner V2 shutdown complete")

    # =========================================================================
    # Decision Making
    # =========================================================================

    async def _decision_cycle(self):
        """Single decision cycle."""
        self.stats["decisions"] += 1

        # Check plugin online
        if not self.bridge.is_plugin_online():
            logger.debug("Plugin offline, skipping")
            return

        # Check free slots
        free_slots = self.bridge.get_free_slots()
        active_orders = self.bridge.get_active_order_count()

        if free_slots < 1:
            logger.debug(f"No free slots ({active_orders} active orders)")
            return

        # Get state
        gold = self.bridge.get_gold()
        holdings = self.bridge.get_holdings()

        # Get market data
        market_data = await self._get_market_data()

        # Get PPO decision
        decision = await self._get_ppo_decision(gold, holdings, market_data)

        if not decision:
            return

        action = decision.get("action", "hold")
        confidence = decision.get("confidence", 0)

        # Check confidence threshold
        if confidence < self.min_confidence:
            logger.debug(f"Low confidence ({confidence:.2f}), skipping")
            return

        # Execute
        if action == "buy":
            await self._execute_buy(decision)
        elif action == "sell":
            await self._execute_sell(decision)
        else:
            self.stats["holds"] += 1
            logger.debug(f"HOLD (confidence: {confidence:.2f})")

    async def _get_market_data(self) -> Dict[str, Any]:
        """Get current market data with caching."""
        now = time.time()

        if self._market_cache and (now - self._market_cache_time) < self.MARKET_CACHE_TTL:
            return self._market_cache

        latest_prices = {}
        volume_data = {}

        try:
            # Try API first
            from api.ge_rest_client import GrandExchangeClient

            if not self._ge_client:
                self._ge_client = GrandExchangeClient(
                    contact_email="ppoflipperopus@example.com",
                    project_name="PPOFlipperOpus-V2"
                )

            # Latest prices
            latest_raw = self._ge_client.get_latest()
            for item_id, data in latest_raw.items():
                latest_prices[str(item_id)] = {
                    'high': data.get('high', 0),
                    'low': data.get('low', 0)
                }

            # 5-minute volume
            data_5m = self._ge_client.get_5m()
            for item_id, data in data_5m.items():
                volume_data[str(item_id)] = {
                    'high_volume': data.get('highPriceVolume', 0),
                    'low_volume': data.get('lowPriceVolume', 0)
                }

            logger.debug(f"Fetched {len(latest_prices)} prices from API")

        except Exception as e:
            logger.warning(f"API fetch failed: {e}, using local data")
            latest_prices, volume_data = self._load_local_market_data()

        self._market_cache = {"latest": latest_prices, "volume": volume_data}
        self._market_cache_time = now

        return self._market_cache

    def _load_local_market_data(self) -> Tuple[Dict, Dict]:
        """Load market data from local files."""
        latest_prices = {}
        volume_data = {}

        base_dir = Path(__file__).parent.parent

        try:
            # Latest prices
            latest_path = base_dir / "mongo_data" / "latest_prices.json"
            if latest_path.exists():
                with open(latest_path, 'r') as f:
                    for item in json.load(f):
                        latest_prices[str(item.get('item_id'))] = {
                            'high': item.get('high_price', 0),
                            'low': item.get('low_price', 0)
                        }

            # Volume data
            prices_1h_path = base_dir / "mongo_data" / "prices_1h.json"
            if prices_1h_path.exists():
                with open(prices_1h_path, 'r') as f:
                    for item in json.load(f):
                        volume_data[str(item.get('item_id'))] = {
                            'high_volume': item.get('high_price_volume', 0),
                            'low_volume': item.get('low_price_volume', 0)
                        }

        except Exception as e:
            logger.error(f"Failed to load local market data: {e}")

        return latest_prices, volume_data

    async def _get_ppo_decision(
        self,
        gold: int,
        holdings: Dict[int, Holding],
        market_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get decision from PPO model."""
        if not self.feature_extractor or not self.actor:
            return None

        try:
            # Build observation
            obs = self._build_observation(gold, holdings, market_data)
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.feature_extractor(obs_tensor)
                logits = self.actor(features)

                # Get probabilities
                action_probs = torch.softmax(logits["action_type"], dim=-1)
                item_probs = torch.softmax(logits["item"], dim=-1)
                price_probs = torch.softmax(logits["price"], dim=-1)
                quantity_probs = torch.softmax(logits["quantity"], dim=-1)

                # Sample actions
                action_type = torch.distributions.Categorical(action_probs).sample().item()
                item_idx = torch.distributions.Categorical(item_probs).sample().item()
                price_bin = torch.distributions.Categorical(price_probs).sample().item()
                quantity_bin = torch.distributions.Categorical(quantity_probs).sample().item()

                # Confidence
                action_conf = action_probs[0, action_type].item()
                item_conf = item_probs[0, item_idx].item()
                confidence = (action_conf + item_conf) / 2

            # Decode action
            actions = ['hold', 'buy', 'sell']
            action = actions[action_type]

            if action == 'hold':
                return {"action": "hold", "confidence": confidence}

            # Get item
            if item_idx >= len(self.item_list):
                return {"action": "hold", "confidence": 0}

            item_id = self.item_list[item_idx]
            item_data = self.items.get(item_id, {})
            item_name = item_data.get('name', f'Item_{item_id}')
            ge_limit = item_data.get('ge_limit', 5000)

            # Get market prices
            latest = market_data.get("latest", {})
            price_data = latest.get(str(item_id), {})
            market_high = price_data.get('high', 0)
            market_low = price_data.get('low', 0)

            if market_high == 0 or market_low == 0:
                return {"action": "hold", "confidence": 0}

            # Calculate price and quantity from bins
            price_frac = price_bin / 20.0  # 0 to 1
            quantity_frac = quantity_bin / 10.0  # 0 to 1

            # For sells, can only sell what we have
            if action == 'sell':
                holding = holdings.get(item_id)
                if not holding or holding.quantity < 1:
                    return {"action": "hold", "confidence": confidence}

                # Price: 98-108% of high
                price = int(market_high * (0.98 + 0.1 * price_frac))
                # Quantity: 10-100% of holdings
                max_qty = holding.quantity
                quantity = max(1, int(max_qty * (0.1 + 0.9 * quantity_frac)))
                quantity = min(quantity, max_qty)

            else:  # buy
                # Price: 95-105% of low
                price = int(market_low * (0.95 + 0.1 * price_frac))
                # Quantity based on gold and limits
                max_from_gold = gold // max(price, 1)
                max_qty = min(ge_limit, max_from_gold)
                quantity = max(1, int(max_qty * quantity_frac * 0.5))
                quantity = min(quantity, max_from_gold, ge_limit)

            if quantity < 1 or price < 1:
                return {"action": "hold", "confidence": 0}

            return {
                "action": action,
                "item_id": item_id,
                "item_name": item_name,
                "quantity": quantity,
                "price": price,
                "confidence": confidence
            }

        except Exception as e:
            logger.error(f"PPO decision error: {e}")
            return None

    def _build_observation(
        self,
        gold: int,
        holdings: Dict[int, Holding],
        market_data: Dict[str, Any]
    ) -> np.ndarray:
        """Build observation vector for PPO model."""
        obs_dim = self.model_config.get('obs_dim', 1407)
        n_items = len(self.item_list)

        obs = np.zeros(obs_dim, dtype=np.float32)
        latest = market_data.get("latest", {})
        volume = market_data.get("volume", {})

        # Per-item features (7 per item)
        idx = 0
        for item_id in self.item_list:
            item_id_str = str(item_id)

            price_data = latest.get(item_id_str, {})
            high = price_data.get('high', 0)
            low = price_data.get('low', 0)

            vol_data = volume.get(item_id_str, {})
            high_vol = vol_data.get('high_volume', 0)
            low_vol = vol_data.get('low_volume', 0)

            # Normalize
            norm_high = np.log1p(high) / 15.0 if high > 0 else 0
            norm_low = np.log1p(low) / 15.0 if low > 0 else 0
            norm_high_vol = np.log1p(high_vol) / 15.0 if high_vol > 0 else 0
            norm_low_vol = np.log1p(low_vol) / 15.0 if low_vol > 0 else 0
            spread = (high - low) / max(low, 1) if low > 0 else 0
            spread = min(spread, 1.0)

            # Position
            holding = holdings.get(item_id)
            qty = holding.quantity if holding else 0
            value = qty * high if high > 0 else 0
            norm_qty = np.log1p(qty) / 15.0
            norm_value = np.log1p(value) / 20.0

            if idx + 7 <= obs_dim:
                obs[idx:idx+7] = [norm_high, norm_low, norm_high_vol, norm_low_vol, spread, norm_qty, norm_value]
            idx += 7

        # Portfolio features
        portfolio_start = n_items * 7
        if portfolio_start + 3 <= obs_dim:
            obs[portfolio_start] = np.log1p(gold) / 25.0

            total_value = gold
            for item_id, holding in holdings.items():
                price_data = latest.get(str(item_id), {})
                price = price_data.get('high', 0)
                total_value += holding.quantity * price

            obs[portfolio_start + 1] = np.log1p(total_value) / 25.0
            obs[portfolio_start + 2] = 0  # PnL placeholder

        # Time features
        time_start = portfolio_start + 3
        if time_start + 4 <= obs_dim:
            now = datetime.now()
            hour = now.hour
            day = now.weekday()
            obs[time_start] = np.sin(2 * np.pi * hour / 24)
            obs[time_start + 1] = np.cos(2 * np.pi * hour / 24)
            obs[time_start + 2] = np.sin(2 * np.pi * day / 7)
            obs[time_start + 3] = np.cos(2 * np.pi * day / 7)

        return obs

    # =========================================================================
    # Order Execution
    # =========================================================================

    async def _execute_buy(self, decision: Dict[str, Any]):
        """Execute a buy order."""
        item_id = decision["item_id"]
        item_name = decision["item_name"]
        quantity = decision["quantity"]
        price = decision["price"]
        confidence = decision["confidence"]

        total_cost = quantity * price
        gold = self.bridge.get_gold()

        if total_cost > gold:
            # Adjust quantity
            quantity = gold // price
            if quantity < 1:
                logger.debug(f"Cannot afford {item_name}")
                return

        logger.info(
            f"BUY: {quantity}x {item_name} @ {price}gp "
            f"(total: {quantity * price:,}gp, confidence: {confidence:.2f})"
        )

        if self.dry_run:
            logger.info("[DRY RUN] Order not submitted")
            return

        order_id = self.bridge.submit_buy_order(
            item_id=item_id,
            item_name=item_name,
            quantity=quantity,
            price=price,
            confidence=confidence,
            strategy="ppo_v2"
        )

        if order_id:
            self.stats["orders_submitted"] += 1
            self.stats["buys"] += 1
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

        # Verify we have the items
        available = self.bridge.get_total_quantity(item_id)
        if available < quantity:
            quantity = available
            if quantity < 1:
                logger.debug(f"No {item_name} to sell")
                return

        logger.info(
            f"SELL: {quantity}x {item_name} @ {price}gp "
            f"(total: {quantity * price:,}gp, confidence: {confidence:.2f})"
        )

        if self.dry_run:
            logger.info("[DRY RUN] Order not submitted")
            return

        order_id = self.bridge.submit_sell_order(
            item_id=item_id,
            item_name=item_name,
            quantity=quantity,
            price=price,
            confidence=confidence,
            strategy="ppo_v2"
        )

        if order_id:
            self.stats["orders_submitted"] += 1
            self.stats["sells"] += 1
            logger.info(f"Sell order submitted: {order_id}")
        else:
            logger.warning("Failed to submit sell order")

    # =========================================================================
    # Status
    # =========================================================================

    def print_stats(self):
        """Print current statistics."""
        logger.info("=" * 50)
        logger.info("Inference Runner V2 Statistics")
        logger.info("=" * 50)
        for key, value in self.stats.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 50)


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="PPO Inference Runner V2")
    parser.add_argument("--account", default="b1llstar", help="Account ID")
    parser.add_argument("--interval", type=float, default=5.0, help="Decision interval")
    parser.add_argument("--confidence", type=float, default=0.05, help="Min confidence")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    args = parser.parse_args()

    runner = InferenceRunnerV2(
        account_id=args.account,
        decision_interval=args.interval,
        min_confidence=args.confidence,
        dry_run=args.dry_run
    )

    if not runner.initialize():
        logger.error("Failed to initialize")
        return 1

    try:
        await runner.run()
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        runner.print_stats()
        runner.shutdown()

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
