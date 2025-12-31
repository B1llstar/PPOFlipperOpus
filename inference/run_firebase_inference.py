"""
Firebase-based PPO Inference Runner.

This module replaces the WebSocket-based inference system with Firebase Firestore.
It provides real-time bidirectional communication between PPO inference and the
GE Auto plugin.

Architecture:
    PPO Inference → Firebase → GE Auto Plugin
    PPO Inference ← Firebase ← GE Auto Plugin

Now uses trained PPO model (shared_model_final.pt) for actual decision making
with the top 200 traded items from training data.
"""

import asyncio
import logging
import signal
import sys
import time
import os
import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

import torch
import numpy as np

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
        MAX_OUTSTANDING_POSITIONS,
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

        # Item data - loaded from trained model
        self.tradeable_items: Dict[int, str] = {}  # item_id -> name
        self.id_to_name_map: Dict[str, str] = {}
        self.name_to_id_map: Dict[str, str] = {}
        self.buy_limits: Dict[str, int] = {}

        # Model data - loaded from checkpoint
        self.item_list: List[int] = []  # List of item IDs from training
        self.price_ranges: Dict[int, Tuple[float, float]] = {}
        self.model_buy_limits: Dict[int, int] = {}
        self.model_config: Dict[str, Any] = {}

        # PPO model components
        self.feature_extractor = None
        self.actor = None
        self.critic = None
        self.device = None

        # Market data cache (to avoid hitting API too frequently)
        self._market_data_cache: Optional[Dict[str, Any]] = None
        self._market_data_cache_time: float = 0
        self._market_data_cache_ttl: float = 60.0  # Cache for 60 seconds
        self._ge_client = None  # Reusable API client

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

            # Initialize PPO agent first (loads model and item_list)
            if not self._initialize_agent():
                logger.error("Failed to initialize PPO agent")
                return False

            # Load item data (uses item_list from model)
            self._load_item_data()

            logger.info("Firebase inference runner initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_item_data(self):
        """Load item data from mongo_data/items.json for the trained items."""
        logger.info("Loading item data for trading...")

        # Get the base directory (parent of inference/)
        base_dir = Path(__file__).parent.parent

        # Load item mapping from mongo_data/items.json
        items_path = base_dir / "mongo_data" / "items.json"
        try:
            with open(items_path, 'r') as f:
                items_data = json.load(f)

            # Build ID to name mapping
            all_items_map = {}
            for item in items_data:
                item_id = item.get('id')
                item_name = item.get('name')
                ge_limit = item.get('ge_limit', 5000)
                if item_id and item_name:
                    all_items_map[item_id] = {
                        'name': item_name,
                        'ge_limit': ge_limit if ge_limit else 5000
                    }

            logger.info(f"Loaded {len(all_items_map)} items from items.json")

        except FileNotFoundError:
            logger.error(f"Items file not found: {items_path}")
            all_items_map = {}
        except Exception as e:
            logger.error(f"Error loading items: {e}")
            all_items_map = {}

        # Filter to only the items from training (self.item_list is loaded from model)
        for item_id in self.item_list:
            if item_id in all_items_map:
                item_info = all_items_map[item_id]
                item_name = item_info['name']
                ge_limit = item_info['ge_limit']

                self.tradeable_items[item_id] = item_name
                self.id_to_name_map[str(item_id)] = item_name
                self.name_to_id_map[item_name] = str(item_id)
                self.buy_limits[item_name] = ge_limit
            else:
                # Use item ID as name if not found in mapping
                logger.warning(f"Item ID {item_id} not found in items.json")
                self.tradeable_items[item_id] = f"Item_{item_id}"
                self.id_to_name_map[str(item_id)] = f"Item_{item_id}"
                self.name_to_id_map[f"Item_{item_id}"] = str(item_id)
                self.buy_limits[f"Item_{item_id}"] = 5000

        logger.info(f"Loaded {len(self.tradeable_items)} items for trading (from model training)")
        logger.info(f"First 10 items: {list(self.tradeable_items.items())[:10]}")

    def _initialize_agent(self):
        """Initialize the PPO agent by loading the trained model."""
        logger.info("Initializing PPO agent...")

        # Get the base directory
        base_dir = Path(__file__).parent.parent
        model_path = base_dir / "model" / "shared_model_final.pt"

        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False

        try:
            # Determine the best device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("Using Apple Silicon MPS")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU")

            # Load the checkpoint
            logger.info(f"Loading trained model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)

            # Extract model metadata
            self.item_list = checkpoint.get('item_list', [])
            self.price_ranges = checkpoint.get('price_ranges', {})
            self.model_buy_limits = checkpoint.get('buy_limits', {})
            self.model_config = checkpoint.get('config', {})

            logger.info(f"Model trained on {len(self.item_list)} items")
            logger.info(f"Model config: hidden_size={self.model_config.get('hidden_size')}, "
                       f"num_layers={self.model_config.get('num_layers')}, "
                       f"obs_dim={self.model_config.get('obs_dim')}")

            # Import network components from ppo_agent
            from ppo_agent import FeatureExtractor, ActorHead, CriticHead

            # Create network components with the same architecture as training
            hidden_size = self.model_config.get('hidden_size', 1024)
            num_layers = self.model_config.get('num_layers', 4)
            n_items = self.model_config.get('n_items', len(self.item_list))
            # obs_dim = n_items * 13 (per-item) + 6 (portfolio) + 4 (time)
            default_obs_dim = n_items * 13 + 6 + 4
            obs_dim = self.model_config.get('obs_dim', default_obs_dim)
            price_bins = self.model_config.get('price_bins', 20)
            quantity_bins = self.model_config.get('quantity_bins', 10)
            wait_steps_bins = self.model_config.get('wait_steps_bins', 10)

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

            # Load the trained weights
            self.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])

            # Set to evaluation mode
            self.feature_extractor.eval()
            self.actor.eval()
            self.critic.eval()

            logger.info(f"Successfully loaded trained PPO model")
            logger.info(f"Model has been updated {checkpoint.get('n_updates', 0)} times, "
                       f"trained on {checkpoint.get('total_timesteps', 0)} timesteps")

            return True

        except Exception as e:
            logger.error(f"Failed to load PPO model: {e}")
            import traceback
            traceback.print_exc()
            return False

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

        # Get current state - use inventory for positions (what we can sell)
        inventory = self.bridge.portfolio_tracker.get_inventory()
        inventory_items = inventory.get("items", {}) if inventory else {}

        # Get gold from inventory first (most accurate), then fallback to other sources
        gold = 0
        if inventory:
            gold = inventory.get("gold", 0)
        if gold == 0:
            gold = self.bridge.get_gold()

        # Use inventory items as holdings (these are what the bot can actually trade)
        holdings = {}
        for item_id_str, item_data in inventory_items.items():
            holdings[item_id_str] = {
                'quantity': item_data.get('quantity', 0),
                'price': item_data.get('price_each', 0)
            }

        logger.info(f"State: gold={gold:,}, inventory_items={len(holdings)}, items={list(holdings.keys())[:5]}")

        # Get active orders from Firebase
        active_orders = self.bridge.get_active_orders()
        firebase_order_count = len(active_orders)

        # Try to get actual GE slot usage from the plugin (more accurate than our tracking)
        plugin_slots_used = self.bridge.portfolio_tracker.get_used_slots()

        # Use the plugin's slot count if available, otherwise fall back to Firebase order count
        if plugin_slots_used is not None and plugin_slots_used >= 0:
            active_count = plugin_slots_used
            if firebase_order_count != plugin_slots_used:
                logger.debug(f"Slot count: plugin={plugin_slots_used}, firebase={firebase_order_count}")
        else:
            active_count = firebase_order_count

        # Check if we have room for more orders
        if active_count >= MAX_ACTIVE_ORDERS:
            logger.info(f"Slots full ({active_count}/{MAX_ACTIVE_ORDERS}) - waiting for orders to complete")
            return

        # Check outstanding positions limit (number of different items held)
        positions_count = len([h for h in holdings.values()
                              if isinstance(h, dict) and h.get('quantity', 0) > 0])
        if positions_count >= MAX_OUTSTANDING_POSITIONS:
            logger.info(f"Max positions reached ({positions_count}/{MAX_OUTSTANDING_POSITIONS}) - waiting to sell")
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
            logger.warning(f"[THRESHOLD-BLOCK] Decision BLOCKED: {action} confidence {confidence:.2f} < threshold {self.min_confidence} - NO ORDER CREATED")
            return

        logger.info(f"[THRESHOLD-PASS] Decision ACCEPTED: {action} confidence {confidence:.2f} >= threshold {self.min_confidence} - CREATING ORDER")

        # Execute decision
        if action == "buy":
            await self._execute_buy(decision)
        elif action == "sell":
            await self._execute_sell(decision)
        elif action == "hold":
            logger.info(f"Decision: HOLD (confidence: {confidence:.2f})")

    async def _get_market_data(self) -> Dict[str, Any]:
        """Get current market data for decision making.

        Fetches real-time data from OSRS Wiki API with caching (60s TTL),
        with fallback to local mongo_data files.
        """
        # Check cache first
        current_time = time.time()
        if (self._market_data_cache is not None and
            current_time - self._market_data_cache_time < self._market_data_cache_ttl):
            logger.debug("Using cached market data")
            return self._market_data_cache

        latest_prices = {}
        volume_data = {}

        # Try to fetch real-time data from OSRS Wiki API
        try:
            from api.ge_rest_client import GrandExchangeClient

            # Reuse client if available, otherwise create new one
            if self._ge_client is None:
                self._ge_client = GrandExchangeClient(
                    contact_email="ppoflipperopus@example.com",
                    project_name="PPOFlipperOpus-Firebase"
                )

            # Fetch latest prices (instant buy/sell)
            latest_raw = self._ge_client.get_latest()
            for item_id, data in latest_raw.items():
                latest_prices[str(item_id)] = {
                    'high': data.get('high', 0),
                    'low': data.get('low', 0),
                    'high_time': data.get('highTime', 0),
                    'low_time': data.get('lowTime', 0)
                }
            logger.info(f"Fetched {len(latest_prices)} latest prices from API")

            # Fetch 5-minute averaged data (includes volume)
            data_5m = self._ge_client.get_5m()
            for item_id, data in data_5m.items():
                volume_data[str(item_id)] = {
                    'avg_high_price': data.get('avgHighPrice', 0),
                    'avg_low_price': data.get('avgLowPrice', 0),
                    'high_volume': data.get('highPriceVolume', 0),
                    'low_volume': data.get('lowPriceVolume', 0)
                }
            logger.info(f"Fetched {len(volume_data)} 5m volume entries from API")

        except Exception as e:
            logger.warning(f"Could not fetch real-time data from API: {e}")
            logger.info("Falling back to local mongo_data files...")

            # Fallback to local mongo_data files
            base_dir = Path(__file__).parent.parent
            latest_path = base_dir / "mongo_data" / "latest_prices.json"
            prices_1h_path = base_dir / "mongo_data" / "prices_1h.json"

            try:
                # Load latest prices
                if latest_path.exists():
                    with open(latest_path, 'r') as f:
                        prices_list = json.load(f)
                        for item in prices_list:
                            item_id = str(item.get('item_id'))
                            latest_prices[item_id] = {
                                'high': item.get('high_price', 0),
                                'low': item.get('low_price', 0),
                                'high_time': item.get('high_time', 0),
                                'low_time': item.get('low_time', 0)
                            }
                    logger.debug(f"Loaded {len(latest_prices)} latest prices from local file")

                # Load 1h volume data
                if prices_1h_path.exists():
                    with open(prices_1h_path, 'r') as f:
                        volume_list = json.load(f)
                        for item in volume_list:
                            item_id = str(item.get('item_id'))
                            volume_data[item_id] = {
                                'avg_high_price': item.get('avg_high_price', 0),
                                'avg_low_price': item.get('avg_low_price', 0),
                                'high_volume': item.get('high_price_volume', 0),
                                'low_volume': item.get('low_price_volume', 0)
                            }
                    logger.debug(f"Loaded {len(volume_data)} volume entries from local file")

            except Exception as file_error:
                logger.error(f"Could not load local market data: {file_error}")

        # Cache the result
        self._market_data_cache = {
            "latest": latest_prices,
            "volume": volume_data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        self._market_data_cache_time = current_time

        return self._market_data_cache

    def _build_observation(
        self,
        gold: int,
        holdings: Dict[str, Any],
        active_orders: List[Dict[str, Any]],
        market_data: Dict[str, Any]
    ) -> np.ndarray:
        """
        Build observation vector MATCHING THE TRAINING ENVIRONMENT FORMAT EXACTLY.

        Training environment (ge_environment.py) uses:
        Per item: 13 features
            - Current market: high, low, spread_pct, high_vol (log), low_vol (log), vol_ratio = 6
            - Rolling stats: sma_6, sma_24, vol_sma (log), volatility = 4
            - Position: quantity (log), avg_cost (normalized), unrealized_pnl_pct = 3
        Portfolio: 6 features
            - log_cash/20, log_total_value/20, realized_pnl_ratio, unrealized_pnl_ratio,
              position_count_ratio, concentration
        Time: 4 cyclical features
            - sin/cos hour, sin/cos day

        Total: n_items * 13 + 6 + 4 = 200 * 13 + 10 = 2610
        """
        n_items = len(self.item_list)
        per_item_features = 13
        portfolio_features = 6
        time_features = 4

        # Calculate expected obs_dim to match training
        expected_obs_dim = n_items * per_item_features + portfolio_features + time_features
        obs_dim = self.model_config.get('obs_dim', expected_obs_dim)

        # Initialize observation array
        obs = np.zeros(obs_dim, dtype=np.float32)

        latest = market_data.get("latest", {})
        volume = market_data.get("volume", {})

        # Track portfolio stats
        total_value = gold
        unrealized_pnl = 0.0
        n_positions = 0
        position_values = []

        # Build per-item features (13 per item)
        idx = 0
        for item_id in self.item_list:
            item_id_str = str(item_id)

            # Get price data
            price_data = latest.get(item_id_str, {})
            high_price = float(price_data.get('high', 0))
            low_price = float(price_data.get('low', 0))
            mid_price = (high_price + low_price) / 2 if (high_price + low_price) > 0 else 1.0
            price_scale = mid_price if mid_price > 0 else 1.0

            # Get volume data
            vol_data = volume.get(item_id_str, {})
            high_vol = float(vol_data.get('high_volume', 0))
            low_vol = float(vol_data.get('low_volume', 0))

            # Get position for this item
            holding = holdings.get(item_id_str, {})
            position_qty = holding.get('quantity', 0) if isinstance(holding, dict) else 0
            position_avg_cost = holding.get('avg_cost', holding.get('price', 0)) if isinstance(holding, dict) else 0
            position_value = position_qty * low_price

            # Track portfolio stats
            if position_qty > 0:
                n_positions += 1
                total_value += position_value
                position_values.append(position_value)
                if position_avg_cost > 0:
                    unrealized_pnl += position_value - (position_qty * position_avg_cost)

            # === 13 FEATURES PER ITEM (matching training exactly) ===

            # Feature 1-2: Normalized high/low prices
            obs[idx] = high_price / price_scale if price_scale > 0 else 0
            obs[idx + 1] = low_price / price_scale if price_scale > 0 else 0

            # Feature 3: Spread percentage
            spread_pct = (high_price - low_price) / price_scale if price_scale > 0 else 0
            obs[idx + 2] = np.clip(spread_pct, 0, 1)

            # Feature 4-5: Log volumes
            obs[idx + 3] = np.log1p(high_vol)
            obs[idx + 4] = np.log1p(low_vol)

            # Feature 6: Volume ratio
            vol_ratio = high_vol / max(low_vol, 1)
            obs[idx + 5] = np.clip(vol_ratio, 0.1, 10)

            # Features 7-10: Rolling statistics
            # Since we don't have historical data in inference, use reasonable defaults
            # sma_6 and sma_24 normalized by price_scale - assume current price is the average
            obs[idx + 6] = 1.0  # sma_6 / price_scale ≈ 1.0 (current price)
            obs[idx + 7] = 1.0  # sma_24 / price_scale ≈ 1.0 (current price)
            # vol_sma (log) - use current total volume as estimate
            obs[idx + 8] = np.log1p((high_vol + low_vol) / 2)
            # volatility - assume low volatility (0.02 = 2%)
            obs[idx + 9] = 0.02

            # Features 11-13: Position features
            obs[idx + 10] = np.log1p(position_qty)  # Log quantity

            if position_qty > 0 and position_avg_cost > 0:
                obs[idx + 11] = position_avg_cost / price_scale  # Normalized avg cost
                unrealized_pnl_pct = (low_price - position_avg_cost) / position_avg_cost
                obs[idx + 12] = np.clip(unrealized_pnl_pct, -0.5, 0.5)  # Unrealized P&L %
            else:
                obs[idx + 11] = 0.0
                obs[idx + 12] = 0.0

            idx += per_item_features

        # === PORTFOLIO FEATURES (6) ===
        portfolio_start = n_items * per_item_features

        # 1. Normalized log cash
        obs[portfolio_start] = np.log1p(gold) / 20.0

        # 2. Normalized log total value
        obs[portfolio_start + 1] = np.log1p(total_value) / 20.0

        # 3. Realized P&L ratio (we don't track this, use 0)
        obs[portfolio_start + 2] = 0.0

        # 4. Unrealized P&L ratio
        initial_cash = 1_000_000  # Assume same as training
        obs[portfolio_start + 3] = unrealized_pnl / initial_cash

        # 5. Position count ratio
        obs[portfolio_start + 4] = n_positions / n_items

        # 6. Concentration (Herfindahl index)
        if position_values:
            total_pos_value = sum(position_values)
            if total_pos_value > 0:
                shares = [v / total_pos_value for v in position_values]
                concentration = sum(s * s for s in shares)
            else:
                concentration = 0
        else:
            concentration = 0
        obs[portfolio_start + 5] = concentration

        # === TIME FEATURES (4) ===
        time_start = portfolio_start + portfolio_features
        now = datetime.now()
        hour = now.hour
        day = now.weekday()

        obs[time_start] = np.sin(2 * np.pi * hour / 24)
        obs[time_start + 1] = np.cos(2 * np.pi * hour / 24)
        obs[time_start + 2] = np.sin(2 * np.pi * day / 7)
        obs[time_start + 3] = np.cos(2 * np.pi * day / 7)

        return obs

    async def _get_ppo_decision(
        self,
        gold: int,
        holdings: Dict[str, Any],
        active_orders: List[Dict[str, Any]],
        market_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Get a decision from the trained PPO model.

        Returns:
            Decision dict with action, item_id, quantity, price, confidence
            or None if no action should be taken
        """
        # Check if model is loaded
        if self.feature_extractor is None or self.actor is None:
            logger.warning("Model not loaded, cannot make decision")
            return None

        # Get items we're already trading
        active_item_ids = set()
        for order in active_orders:
            active_item_ids.add(order.get("item_id"))

        try:
            # Build observation
            obs = self._build_observation(gold, holdings, active_orders, market_data)

            # Convert to tensor
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)

            # Get model output
            with torch.no_grad():
                features = self.feature_extractor(obs_tensor)
                logits = self.actor(features)
                value = self.critic(features)

                # Get probabilities for action type
                action_type_probs = torch.softmax(logits["action_type"], dim=-1)
                item_probs = torch.softmax(logits["item"], dim=-1)
                price_probs = torch.softmax(logits["price"], dim=-1)
                quantity_probs = torch.softmax(logits["quantity"], dim=-1)

                # Stochastic sampling from probability distributions
                # This allows exploration across different items instead of always picking the same one
                action_type_dist = torch.distributions.Categorical(action_type_probs)
                item_dist = torch.distributions.Categorical(item_probs)
                price_dist = torch.distributions.Categorical(price_probs)
                quantity_dist = torch.distributions.Categorical(quantity_probs)

                action_type = action_type_dist.sample().item()
                item_idx = item_dist.sample().item()
                price_bin = price_dist.sample().item()
                quantity_bin = quantity_dist.sample().item()

                # Calculate confidence from the probability of chosen action
                action_type_conf = action_type_probs[0, action_type].item()
                item_conf = item_probs[0, item_idx].item()
                confidence = (action_type_conf + item_conf) / 2

            # Decode action type: 0=hold, 1=buy, 2=sell
            action_types = ['hold', 'buy', 'sell']
            action = action_types[action_type]

            if action == 'hold':
                logger.info(f"Model decision: HOLD (confidence: {confidence:.2f})")
                return {"action": "hold", "confidence": confidence}

            # For SELL actions, we can ONLY sell items we actually hold
            # Build list of items we have in holdings
            held_item_ids = set()
            for item_id_str, holding in holdings.items():
                if isinstance(holding, dict) and holding.get('quantity', 0) > 0:
                    try:
                        held_item_ids.add(int(item_id_str))
                    except (ValueError, TypeError):
                        pass

            if action == 'sell':
                if not held_item_ids:
                    logger.info("SELL requested but no holdings available - switching to HOLD")
                    return {"action": "hold", "confidence": confidence}

                # For sells, only consider items we actually hold
                # Mask the item probabilities to only include held items
                held_indices = [i for i, item_id in enumerate(self.item_list) if item_id in held_item_ids]

                if not held_indices:
                    logger.info("SELL requested but none of our holdings are in tradeable items - switching to HOLD")
                    return {"action": "hold", "confidence": confidence}

                # Sample only from held items
                held_probs = item_probs[0, held_indices]
                held_probs = held_probs / held_probs.sum()  # Renormalize
                held_dist = torch.distributions.Categorical(held_probs)
                held_sample_idx = held_dist.sample().item()
                item_idx = held_indices[held_sample_idx]

            # Get item from index
            if item_idx >= len(self.item_list):
                item_idx = len(self.item_list) - 1
            item_id = self.item_list[item_idx]
            item_name = self.tradeable_items.get(item_id, f"Item_{item_id}")

            # Skip if we already have an active order for this item
            if item_id in active_item_ids:
                logger.debug(f"Skipping {item_name} - already have active order")
                return None

            # Get price range for this item
            price_range = self.price_ranges.get(item_id, (100, 10000))
            min_price, max_price = price_range

            # Get current market price
            latest = market_data.get("latest", {})
            price_data = latest.get(str(item_id), {})
            market_high = price_data.get('high', max_price)
            market_low = price_data.get('low', min_price)

            if market_high == 0:
                market_high = max_price
            if market_low == 0:
                market_low = min_price

            # Decode price from bin
            price_bins = self.model_config.get('price_bins', 21)
            price_frac = price_bin / max(price_bins - 1, 1)

            if action == 'buy':
                # For buying, use the low end of market
                price = int(market_low * (0.95 + 0.1 * price_frac))  # 95% to 105% of low
            else:
                # For selling, use the high end of market
                price = int(market_high * (0.98 + 0.1 * price_frac))  # 98% to 108% of high

            price = max(1, price)

            # Decode quantity from bin
            quantity_bins = self.model_config.get('quantity_bins', 11)
            buy_limit = self.model_buy_limits.get(item_id, 5000)
            quantity_frac = quantity_bin / max(quantity_bins - 1, 1)

            # Calculate total portfolio value for position sizing
            total_portfolio = gold
            for h_id, h_data in holdings.items():
                if isinstance(h_data, dict):
                    h_qty = h_data.get('quantity', 0)
                    h_price = h_data.get('price', 0)
                    total_portfolio += h_qty * h_price

            # 5% max position size per item
            MAX_POSITION_PCT = 0.05
            max_position_value = int(total_portfolio * MAX_POSITION_PCT)

            # Calculate quantity based on available gold, buy limit, AND 5% portfolio cap
            if action == 'buy':
                # Check current position value for this item
                current_holding = holdings.get(str(item_id), {})
                current_qty = current_holding.get('quantity', 0) if isinstance(current_holding, dict) else 0
                current_value = current_qty * price

                # How much more can we buy before hitting 5% cap?
                remaining_allocation = max(0, max_position_value - current_value)
                max_from_cap = remaining_allocation // max(price, 1)

                if max_from_cap <= 0:
                    logger.info(f"Skipping BUY {item_name} - already at 5% cap ({current_value:,}/{max_position_value:,})")
                    return None

                max_affordable = gold // max(price, 1)
                max_quantity = min(buy_limit, max_affordable, max_from_cap)
                # Use quantity_frac directly - model learned this distribution
                quantity = max(1, int(max_quantity * quantity_frac))

                logger.debug(f"Position sizing: portfolio={total_portfolio:,}, 5%cap={max_position_value:,}, "
                           f"current={current_value:,}, max_from_cap={max_from_cap}, qty={quantity}")
            else:
                # For selling, check holdings - ONLY sell what we actually have
                holding = holdings.get(str(item_id), {})
                available_qty = holding.get('quantity', 0) if isinstance(holding, dict) else 0

                if available_qty <= 0:
                    logger.info(f"Skipping SELL {item_name} - no holdings (qty: {available_qty})")
                    return None

                # Use quantity_frac directly - model learned this distribution
                quantity = max(1, int(available_qty * quantity_frac))
                # Never sell more than we have
                quantity = min(quantity, available_qty)

            if quantity <= 0:
                logger.debug(f"Skipping {action} {item_name} - quantity is 0")
                return None

            logger.info(f"Model decision: {action.upper()} {quantity}x {item_name} @ {price} "
                       f"(confidence: {confidence:.2f}, value: {value.item():.4f})")

            return {
                "action": action,
                "item_id": item_id,
                "item_name": item_name,
                "quantity": quantity,
                "price": price,
                "confidence": confidence,
                "value_estimate": value.item()
            }

        except Exception as e:
            logger.error(f"Error in PPO decision: {e}")
            import traceback
            traceback.print_exc()
            return None

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
        print(f"  Positions: {status.get('holdings_count', 0)}/{MAX_OUTSTANDING_POSITIONS}")
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
