#!/usr/bin/env python3
"""
Grand Exchange Trading Environment for PPO Training

A Gymnasium-compatible environment simulating the OSRS Grand Exchange
for training reinforcement learning agents to flip items profitably.

Key Features:
- Multi-item portfolio management
- Realistic market constraints (volume, GE limits, slots)
- Rich observation space with market indicators
- Continuous action space for nuanced trading decisions
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import IntEnum
import sqlite3
import logging
from pathlib import Path

logger = logging.getLogger("GEEnvironment")


class ActionType(IntEnum):
    """Discrete action types for trading."""
    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass
class Position:
    """Represents a position in an item."""
    item_id: int
    quantity: int = 0
    avg_cost: float = 0.0

    @property
    def value(self) -> float:
        return self.quantity * self.avg_cost

    def add(self, quantity: int, price: float):
        """Add to position with new average cost."""
        if quantity <= 0:
            return
        total_cost = self.quantity * self.avg_cost + quantity * price
        self.quantity += quantity
        if self.quantity > 0:
            self.avg_cost = total_cost / self.quantity

    def remove(self, quantity: int) -> int:
        """Remove from position, returns actual quantity removed."""
        actual = min(quantity, self.quantity)
        self.quantity -= actual
        if self.quantity == 0:
            self.avg_cost = 0.0
        return actual


@dataclass
class MarketState:
    """Current state of an item in the market."""
    item_id: int
    high_price: float  # Instant buy price
    low_price: float   # Instant sell price
    high_volume: int   # Buy volume
    low_volume: int    # Sell volume
    timestamp: int

    @property
    def spread(self) -> float:
        return self.high_price - self.low_price

    @property
    def spread_pct(self) -> float:
        mid = (self.high_price + self.low_price) / 2
        return self.spread / mid if mid > 0 else 0

    @property
    def total_volume(self) -> int:
        return self.high_volume + self.low_volume

    @property
    def mid_price(self) -> float:
        return (self.high_price + self.low_price) / 2


@dataclass
class TradeResult:
    """Result of a trade execution."""
    success: bool
    item_id: int
    action: ActionType
    quantity: int
    price: float
    cost: float  # Total cost including any fees
    pnl: float = 0.0  # Realized P&L for sells
    message: str = ""


class GrandExchangeEnv(gym.Env):
    """
    Grand Exchange Trading Environment

    Observation Space:
    - Market features for each tradeable item (prices, volumes, spreads, trends)
    - Portfolio state (positions, cash, unrealized P&L)
    - Time features (hour of day, day of week)
    - Historical context (rolling windows of price/volume)

    Action Space:
    - Hybrid: discrete action type + continuous parameters
    - For each item: (action_type, quantity_fraction, price_offset)

    Rewards:
    - Realized profit from completed trades
    - Unrealized P&L changes (discounted)
    - Penalties for excessive trading, holding costs
    """

    metadata = {"render_modes": ["human", "ansi"]}

    # Constants
    GE_TAX_RATE = 0.01  # 1% GE tax on sells
    GE_SLOTS = 8  # Number of GE slots available
    MAX_ITEMS_TRACKED = 100  # Top items to track
    LOOKBACK_PERIODS = 24  # Hours of history in observation

    def __init__(
        self,
        db_path: str,
        initial_cash: float = 10_000_000,  # 10M GP starting capital
        episode_length: int = 168,  # 1 week of hourly steps
        top_n_items: int = 50,  # Number of items to trade
        ge_limit_multiplier: float = 1.0,  # Scale GE limits (1.0 = realistic)
        include_volume_constraint: bool = True,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None
    ):
        super().__init__()

        self.db_path = db_path
        self.initial_cash = initial_cash
        self.episode_length = episode_length
        self.top_n_items = top_n_items
        self.ge_limit_multiplier = ge_limit_multiplier
        self.include_volume_constraint = include_volume_constraint
        self.render_mode = render_mode

        # Load market data
        self._load_market_data()

        # Define spaces
        self._define_spaces()

        # Initialize state
        self.reset(seed=seed)

        logger.info(f"GE Environment initialized with {len(self.tradeable_items)} items")

    def _load_market_data(self):
        """Load historical market data from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get item metadata
        cursor.execute("""
            SELECT id, name, ge_limit, highalch
            FROM items
            WHERE ge_limit IS NOT NULL AND ge_limit > 0
        """)
        self.item_metadata = {
            row[0]: {"name": row[1], "ge_limit": row[2], "highalch": row[3]}
            for row in cursor.fetchall()
        }

        # Get items with most trading data (top by volume)
        cursor.execute("""
            SELECT item_id, SUM(high_price_volume + low_price_volume) as total_vol
            FROM timeseries
            WHERE timestep = '1h'
            GROUP BY item_id
            ORDER BY total_vol DESC
            LIMIT ?
        """, (self.top_n_items,))

        self.tradeable_items = [row[0] for row in cursor.fetchall()]
        self.item_to_idx = {item_id: idx for idx, item_id in enumerate(self.tradeable_items)}

        # Load all timeseries data for these items
        placeholders = ",".join("?" * len(self.tradeable_items))
        cursor.execute(f"""
            SELECT item_id, timestamp, avg_high_price, high_price_volume,
                   avg_low_price, low_price_volume
            FROM timeseries
            WHERE timestep = '1h' AND item_id IN ({placeholders})
            ORDER BY timestamp
        """, self.tradeable_items)

        # Organize by item and timestamp
        self.market_history: Dict[int, List[MarketState]] = {
            item_id: [] for item_id in self.tradeable_items
        }

        for row in cursor.fetchall():
            item_id, timestamp, high_price, high_vol, low_price, low_vol = row
            if high_price and low_price:  # Skip null data
                self.market_history[item_id].append(MarketState(
                    item_id=item_id,
                    high_price=float(high_price),
                    low_price=float(low_price),
                    high_volume=int(high_vol or 0),
                    low_volume=int(low_vol or 0),
                    timestamp=timestamp
                ))

        # Get unique timestamps for episode sampling
        cursor.execute("""
            SELECT DISTINCT timestamp FROM timeseries
            WHERE timestep = '1h'
            ORDER BY timestamp
        """)
        self.all_timestamps = [row[0] for row in cursor.fetchall()]

        conn.close()

        # Filter items with enough data
        min_data_points = self.episode_length + self.LOOKBACK_PERIODS
        self.tradeable_items = [
            item_id for item_id in self.tradeable_items
            if len(self.market_history[item_id]) >= min_data_points
        ]
        self.item_to_idx = {item_id: idx for idx, item_id in enumerate(self.tradeable_items)}

        logger.info(f"Loaded {len(self.tradeable_items)} items with sufficient data")
        logger.info(f"Timeline: {len(self.all_timestamps)} hourly data points")

    def _define_spaces(self):
        """Define observation and action spaces."""
        n_items = len(self.tradeable_items)

        # === OBSERVATION SPACE ===
        # Per-item features (for each of top_n_items):
        # - Current: high_price, low_price, spread_pct, high_vol, low_vol, vol_ratio
        # - Rolling stats: price_sma_6h, price_sma_24h, vol_sma_6h, volatility_6h
        # - Position: quantity_held, avg_cost, unrealized_pnl_pct
        # Total per item: 13 features

        # Portfolio features:
        # - cash, total_value, realized_pnl, unrealized_pnl
        # - n_positions, portfolio_concentration
        # Total: 6 features

        # Time features:
        # - hour_sin, hour_cos, dow_sin, dow_cos
        # Total: 4 features

        per_item_features = 13
        portfolio_features = 6
        time_features = 4

        obs_dim = n_items * per_item_features + portfolio_features + time_features

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # === ACTION SPACE ===
        # Hybrid action space:
        # For each item: [action_type (0-2), quantity_fraction (0-1), price_aggressiveness (-1 to 1)]
        # action_type: 0=hold, 1=buy, 2=sell
        # quantity_fraction: fraction of affordable/sellable quantity
        # price_aggressiveness: -1=passive (better price), 0=mid, 1=aggressive (instant fill)

        # Using MultiDiscrete for action types + Box for continuous params
        # Simplified: Box action space with 3 values per item
        self.action_space = spaces.Box(
            low=np.array([[-1, 0, -1]] * n_items, dtype=np.float32).flatten(),
            high=np.array([[1, 1, 1]] * n_items, dtype=np.float32).flatten(),
            dtype=np.float32
        )

        self.n_items = n_items
        self.per_item_features = per_item_features
        self.portfolio_features = portfolio_features
        self.time_features = time_features

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Initialize portfolio
        self.cash = self.initial_cash
        self.positions: Dict[int, Position] = {
            item_id: Position(item_id=item_id)
            for item_id in self.tradeable_items
        }

        # Track P&L
        self.realized_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_volume_traded = 0.0

        # Sample random starting point
        max_start = len(self.all_timestamps) - self.episode_length - self.LOOKBACK_PERIODS
        if max_start <= 0:
            self.start_idx = self.LOOKBACK_PERIODS
        else:
            self.start_idx = self.np_random.integers(
                self.LOOKBACK_PERIODS,
                max_start
            )

        self.current_step = 0
        self.current_timestamp = self.all_timestamps[self.start_idx]

        # Build timestamp index for fast lookup
        self._build_timestamp_index()

        # Trade history for this episode
        self.trade_history: List[TradeResult] = []

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def _build_timestamp_index(self):
        """Build index from timestamp to market state for each item."""
        self.timestamp_to_state: Dict[int, Dict[int, MarketState]] = {}

        for item_id in self.tradeable_items:
            for state in self.market_history[item_id]:
                if state.timestamp not in self.timestamp_to_state:
                    self.timestamp_to_state[state.timestamp] = {}
                self.timestamp_to_state[state.timestamp][item_id] = state

    def _get_market_state(self, item_id: int, timestamp: int) -> Optional[MarketState]:
        """Get market state for item at timestamp."""
        if timestamp in self.timestamp_to_state:
            return self.timestamp_to_state[timestamp].get(item_id)
        return None

    def _get_observation(self) -> np.ndarray:
        """Build observation vector."""
        obs = []

        # Per-item features
        for item_id in self.tradeable_items:
            item_obs = self._get_item_features(item_id)
            obs.extend(item_obs)

        # Portfolio features
        portfolio_obs = self._get_portfolio_features()
        obs.extend(portfolio_obs)

        # Time features
        time_obs = self._get_time_features()
        obs.extend(time_obs)

        return np.array(obs, dtype=np.float32)

    def _get_item_features(self, item_id: int) -> List[float]:
        """Get features for a single item."""
        features = []

        # Current market state
        current_state = self._get_market_state(item_id, self.current_timestamp)

        if current_state is None:
            # No data - return zeros
            return [0.0] * self.per_item_features

        # Normalize prices by item's typical range
        price_scale = current_state.mid_price if current_state.mid_price > 0 else 1.0

        # Current market features (6)
        features.append(current_state.high_price / price_scale)  # Normalized high
        features.append(current_state.low_price / price_scale)   # Normalized low
        features.append(current_state.spread_pct)                # Spread %
        features.append(np.log1p(current_state.high_volume))     # Log buy volume
        features.append(np.log1p(current_state.low_volume))      # Log sell volume
        vol_ratio = (current_state.high_volume / max(current_state.low_volume, 1))
        features.append(np.clip(vol_ratio, 0.1, 10))             # Volume ratio

        # Rolling statistics (4)
        history = self._get_item_history(item_id, self.LOOKBACK_PERIODS)
        if len(history) >= 6:
            prices = [s.mid_price for s in history]
            volumes = [s.total_volume for s in history]

            # SMAs
            sma_6 = np.mean(prices[-6:])
            sma_24 = np.mean(prices[-24:]) if len(prices) >= 24 else sma_6
            features.append(sma_6 / price_scale)
            features.append(sma_24 / price_scale)

            # Volume SMA
            vol_sma = np.mean(volumes[-6:])
            features.append(np.log1p(vol_sma))

            # Volatility
            volatility = np.std(prices[-6:]) / price_scale if len(prices) >= 6 else 0
            features.append(volatility)
        else:
            features.extend([1.0, 1.0, 0.0, 0.0])

        # Position features (3)
        position = self.positions[item_id]
        features.append(np.log1p(position.quantity))  # Log quantity

        if position.quantity > 0:
            features.append(position.avg_cost / price_scale)  # Normalized avg cost
            unrealized_pnl_pct = (current_state.low_price - position.avg_cost) / position.avg_cost
            features.append(np.clip(unrealized_pnl_pct, -0.5, 0.5))  # Unrealized P&L %
        else:
            features.append(0.0)
            features.append(0.0)

        return features

    def _get_item_history(self, item_id: int, periods: int) -> List[MarketState]:
        """Get historical market states for item."""
        history = []
        current_idx = self.start_idx + self.current_step

        for i in range(periods):
            idx = current_idx - periods + i + 1
            if 0 <= idx < len(self.all_timestamps):
                ts = self.all_timestamps[idx]
                state = self._get_market_state(item_id, ts)
                if state:
                    history.append(state)

        return history

    def _get_portfolio_features(self) -> List[float]:
        """Get portfolio-level features."""
        total_value = self.cash
        unrealized_pnl = 0.0
        n_positions = 0
        position_values = []

        for item_id, position in self.positions.items():
            if position.quantity > 0:
                n_positions += 1
                state = self._get_market_state(item_id, self.current_timestamp)
                if state:
                    current_value = position.quantity * state.low_price
                    total_value += current_value
                    position_values.append(current_value)
                    unrealized_pnl += current_value - position.value

        # Concentration (Herfindahl index)
        if position_values:
            total_pos_value = sum(position_values)
            if total_pos_value > 0:
                shares = [v / total_pos_value for v in position_values]
                concentration = sum(s * s for s in shares)
            else:
                concentration = 0
        else:
            concentration = 0

        return [
            np.log1p(self.cash) / 20,              # Normalized log cash
            np.log1p(total_value) / 20,            # Normalized log total value
            self.realized_pnl / self.initial_cash, # Realized P&L ratio
            unrealized_pnl / self.initial_cash,    # Unrealized P&L ratio
            n_positions / len(self.tradeable_items), # Position count ratio
            concentration                           # Portfolio concentration
        ]

    def _get_time_features(self) -> List[float]:
        """Get cyclical time features."""
        # Convert timestamp to datetime components
        from datetime import datetime
        dt = datetime.fromtimestamp(self.current_timestamp)

        hour = dt.hour
        dow = dt.weekday()

        # Cyclical encoding
        return [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7)
        ]

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Reshape action to (n_items, 3)
        action = action.reshape(self.n_items, 3)

        # Execute trades
        step_pnl = 0.0
        step_trades = []

        for idx, item_id in enumerate(self.tradeable_items):
            item_action = action[idx]
            result = self._execute_trade(item_id, item_action)

            if result.success:
                step_trades.append(result)
                step_pnl += result.pnl
                self.total_trades += 1
                if result.pnl > 0:
                    self.winning_trades += 1
                self.total_volume_traded += result.cost

        self.trade_history.extend(step_trades)
        self.realized_pnl += step_pnl

        # Advance time
        self.current_step += 1
        if self.start_idx + self.current_step < len(self.all_timestamps):
            self.current_timestamp = self.all_timestamps[self.start_idx + self.current_step]

        # Calculate reward
        reward = self._calculate_reward(step_pnl, step_trades)

        # Check termination
        terminated = self.current_step >= self.episode_length
        truncated = self.cash < 0  # Bankrupt

        obs = self._get_observation()
        info = self._get_info()
        info["step_trades"] = len(step_trades)
        info["step_pnl"] = step_pnl

        return obs, reward, terminated, truncated, info

    def _execute_trade(self, item_id: int, action: np.ndarray) -> TradeResult:
        """Execute a trade for a single item."""
        action_signal, quantity_frac, price_aggression = action

        # Decode action type from continuous signal
        if action_signal < -0.33:
            action_type = ActionType.SELL
        elif action_signal > 0.33:
            action_type = ActionType.BUY
        else:
            action_type = ActionType.HOLD

        if action_type == ActionType.HOLD:
            return TradeResult(
                success=False, item_id=item_id, action=action_type,
                quantity=0, price=0, cost=0, message="Hold"
            )

        # Get current market state
        state = self._get_market_state(item_id, self.current_timestamp)
        if state is None:
            return TradeResult(
                success=False, item_id=item_id, action=action_type,
                quantity=0, price=0, cost=0, message="No market data"
            )

        # Determine price based on aggressiveness
        # -1 = passive (limit order, better price but may not fill)
        # 0 = mid price
        # +1 = aggressive (market order, instant fill at worse price)
        if action_type == ActionType.BUY:
            # Aggressive = pay high price (instant buy)
            # Passive = try to pay low price (may not fill)
            fill_prob = (price_aggression + 1) / 2  # 0 to 1
            price = state.low_price + (state.high_price - state.low_price) * fill_prob

            # Check if order fills (based on aggressiveness and volume)
            if self.np_random.random() > fill_prob * 0.8 + 0.2:
                return TradeResult(
                    success=False, item_id=item_id, action=action_type,
                    quantity=0, price=price, cost=0, message="Order not filled"
                )
        else:  # SELL
            # Aggressive = accept low price (instant sell)
            # Passive = try to get high price
            fill_prob = (price_aggression + 1) / 2
            price = state.high_price - (state.high_price - state.low_price) * fill_prob

            if self.np_random.random() > fill_prob * 0.8 + 0.2:
                return TradeResult(
                    success=False, item_id=item_id, action=action_type,
                    quantity=0, price=price, cost=0, message="Order not filled"
                )

        # Determine quantity
        quantity_frac = np.clip(quantity_frac, 0, 1)
        position = self.positions[item_id]

        if action_type == ActionType.BUY:
            # How many can we afford?
            max_affordable = int(self.cash / price) if price > 0 else 0

            # Apply GE limit if set
            ge_limit = self.item_metadata.get(item_id, {}).get("ge_limit", 10000)
            ge_limit = int(ge_limit * self.ge_limit_multiplier)
            max_quantity = min(max_affordable, ge_limit)

            # Volume constraint
            if self.include_volume_constraint:
                available_volume = state.low_volume  # Sellers available
                max_quantity = min(max_quantity, available_volume)

            quantity = int(max_quantity * quantity_frac)

            if quantity <= 0:
                return TradeResult(
                    success=False, item_id=item_id, action=action_type,
                    quantity=0, price=price, cost=0, message="Cannot afford"
                )

            # Execute buy
            cost = quantity * price
            self.cash -= cost
            position.add(quantity, price)

            return TradeResult(
                success=True, item_id=item_id, action=action_type,
                quantity=quantity, price=price, cost=cost, pnl=0,
                message=f"Bought {quantity} @ {price:.0f}"
            )

        else:  # SELL
            # How many do we have?
            max_sellable = position.quantity

            if max_sellable <= 0:
                return TradeResult(
                    success=False, item_id=item_id, action=action_type,
                    quantity=0, price=price, cost=0, message="No position"
                )

            # Volume constraint
            if self.include_volume_constraint:
                available_volume = state.high_volume  # Buyers available
                max_sellable = min(max_sellable, available_volume)

            quantity = int(max_sellable * quantity_frac)

            if quantity <= 0:
                return TradeResult(
                    success=False, item_id=item_id, action=action_type,
                    quantity=0, price=price, cost=0, message="Quantity too small"
                )

            # Calculate P&L
            avg_cost = position.avg_cost
            gross_proceeds = quantity * price
            tax = gross_proceeds * self.GE_TAX_RATE
            net_proceeds = gross_proceeds - tax
            pnl = net_proceeds - (quantity * avg_cost)

            # Execute sell
            position.remove(quantity)
            self.cash += net_proceeds

            return TradeResult(
                success=True, item_id=item_id, action=action_type,
                quantity=quantity, price=price, cost=gross_proceeds, pnl=pnl,
                message=f"Sold {quantity} @ {price:.0f}, P&L: {pnl:.0f}"
            )

    def _calculate_reward(self, step_pnl: float, trades: List[TradeResult]) -> float:
        """Calculate reward for the step."""
        reward = 0.0

        # Primary reward: realized P&L (normalized)
        reward += step_pnl / self.initial_cash * 100  # Scale up for learning

        # Secondary: unrealized P&L change (smaller weight)
        # This encourages building positions in good opportunities
        unrealized = 0.0
        for item_id, position in self.positions.items():
            if position.quantity > 0:
                state = self._get_market_state(item_id, self.current_timestamp)
                if state:
                    current_value = position.quantity * state.low_price
                    unrealized += current_value - position.value

        reward += unrealized / self.initial_cash * 10  # Lower weight

        # Small penalty for excessive trading (transaction costs already in P&L)
        if len(trades) > 5:
            reward -= 0.01 * (len(trades) - 5)

        return reward

    def _get_info(self) -> Dict[str, Any]:
        """Get episode info."""
        total_value = self.cash
        for item_id, position in self.positions.items():
            if position.quantity > 0:
                state = self._get_market_state(item_id, self.current_timestamp)
                if state:
                    total_value += position.quantity * state.low_price

        return {
            "cash": self.cash,
            "total_value": total_value,
            "realized_pnl": self.realized_pnl,
            "total_return": (total_value - self.initial_cash) / self.initial_cash,
            "total_trades": self.total_trades,
            "win_rate": self.winning_trades / max(self.total_trades, 1),
            "current_step": self.current_step,
            "episode_length": self.episode_length
        }

    def render(self):
        """Render the environment."""
        if self.render_mode == "human" or self.render_mode == "ansi":
            info = self._get_info()
            print(f"\n=== Step {self.current_step}/{self.episode_length} ===")
            print(f"Cash: {self.cash:,.0f} GP")
            print(f"Total Value: {info['total_value']:,.0f} GP")
            print(f"Return: {info['total_return']*100:.2f}%")
            print(f"Trades: {self.total_trades} (Win: {info['win_rate']*100:.1f}%)")

            # Show top positions
            positions = [
                (item_id, pos) for item_id, pos in self.positions.items()
                if pos.quantity > 0
            ]
            positions.sort(key=lambda x: x[1].value, reverse=True)

            if positions:
                print("\nTop Positions:")
                for item_id, pos in positions[:5]:
                    name = self.item_metadata.get(item_id, {}).get("name", f"Item {item_id}")
                    print(f"  {name}: {pos.quantity:,} @ {pos.avg_cost:.0f}")

    def close(self):
        """Clean up resources."""
        pass


def make_env(
    db_path: str,
    seed: int = 0,
    **kwargs
) -> GrandExchangeEnv:
    """Factory function to create environment."""
    return GrandExchangeEnv(db_path=db_path, seed=seed, **kwargs)
