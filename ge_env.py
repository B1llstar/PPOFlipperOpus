#!/usr/bin/env python3
"""
Grand Exchange Trading Environment

A Gymnasium-compatible environment simulating the OSRS Grand Exchange
for training reinforcement learning agents to flip items profitably.

This environment supports:
- Multi-item portfolio management
- Realistic market constraints (volume, GE limits, tax)
- Volume-based order validation
- Flexible observation and action spaces
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import IntEnum
import logging
import time

# Import volume analysis if available
try:
    from volume_analysis import get_volume_metrics_for_item
    VOLUME_ANALYSIS_AVAILABLE = True
except ImportError:
    VOLUME_ANALYSIS_AVAILABLE = False
    def get_volume_metrics_for_item(analyzer, item):
        return {}

logger = logging.getLogger("GEEnv")


@dataclass
class Position:
    """Represents a position in an item."""
    item: str
    quantity: int = 0
    avg_cost: float = 0.0
    buy_limit_remaining: int = 0
    buy_limit_reset_time: float = 0.0

    @property
    def value(self) -> float:
        return self.quantity * self.avg_cost

    def add(self, quantity: int, price: float, buy_limit: int):
        """Add to position with new average cost."""
        if quantity <= 0:
            return

        total_cost = self.quantity * self.avg_cost + quantity * price
        self.quantity += quantity
        if self.quantity > 0:
            self.avg_cost = total_cost / self.quantity

        # Track buy limit
        self.buy_limit_remaining = buy_limit - quantity
        self.buy_limit_reset_time = time.time() + 4 * 3600  # 4 hour reset

    def remove(self, quantity: int) -> Tuple[int, float]:
        """Remove from position, returns (actual_qty, avg_cost)."""
        actual = min(quantity, self.quantity)
        cost = self.avg_cost
        self.quantity -= actual
        if self.quantity == 0:
            self.avg_cost = 0.0
        return actual, cost


@dataclass
class Order:
    """Represents an active GE order."""
    order_id: int
    item: str
    order_type: str  # 'buy' or 'sell'
    price: float
    quantity: int
    quantity_filled: int = 0
    placed_at: float = field(default_factory=time.time)
    status: str = "pending"  # pending, partial, filled, cancelled


class GrandExchangeEnv(gym.Env):
    """
    Grand Exchange Trading Environment

    Observation Space:
    - Market features for each item (prices, volumes, spreads)
    - Portfolio state (positions, cash, P&L)
    - Volume metrics if analyzer provided

    Action Space:
    - Dict action with: type, item, price, quantity

    Rewards:
    - Realized profit from completed trades
    - Volume-adjusted rewards
    - Penalties for bad trades
    """

    metadata = {"render_modes": ["human", "ansi"]}

    # GE Constants
    GE_TAX_RATE = 0.01  # 1% tax on sells
    GE_SLOTS = 8
    BUY_LIMIT_RESET_HOURS = 4

    def __init__(
        self,
        items: Dict[str, Dict] = None,
        starting_gp: float = 10_000_000,
        max_steps: int = 1000,
        volume_analyzer: Any = None,
        random_seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        **kwargs  # Accept additional kwargs for flexibility
    ):
        super().__init__()

        self.items = items or {}
        self.starting_gp = starting_gp
        self.max_steps = max_steps
        self.volume_analyzer = volume_analyzer
        self.render_mode = render_mode

        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)

        # Extract item list
        self.item_list = list(self.items.keys())
        self.n_items = len(self.item_list)
        self.item_to_idx = {item: idx for idx, item in enumerate(self.item_list)}

        # Define spaces
        self._define_spaces()

        # Initialize state
        self.reset()

        logger.info(f"GE Environment initialized with {self.n_items} items, {starting_gp:,} GP")

    def _define_spaces(self):
        """Define observation and action spaces."""
        # Observation space: flattened features
        # Per item: price, spread, spread_pct, volume metrics (4), position qty, position value
        # Portfolio: cash, total_value, realized_pnl, n_positions
        # Total: n_items * 8 + 4

        per_item_features = 8
        portfolio_features = 4

        obs_dim = max(1, self.n_items * per_item_features + portfolio_features)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Action space is a dict
        # In practice, the agent produces an action dict
        # We'll validate it in step()
        self.action_space = spaces.Dict({
            "type": spaces.Discrete(3),  # 0=hold, 1=buy, 2=sell
            "item_idx": spaces.Discrete(max(1, self.n_items)),
            "price_offset": spaces.Box(low=-0.1, high=0.1, shape=(1,), dtype=np.float32),
            "quantity_frac": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)

        # Initialize portfolio
        self.gp = self.starting_gp
        self.positions: Dict[str, Position] = {}

        for item in self.item_list:
            buy_limit = self.items[item].get('buy_limit', 5000)
            self.positions[item] = Position(
                item=item,
                buy_limit_remaining=buy_limit
            )

        # Orders
        self.active_orders: List[Order] = []
        self.next_order_id = 1

        # P&L tracking
        self.realized_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_volume_traded = 0.0
        self.total_tax_paid = 0.0

        # Step counter
        self.current_step = 0

        # Simulated prices (start at base prices)
        self.prices = {}
        for item, data in self.items.items():
            base = data.get('base_price', (data.get('min_price', 100) + data.get('max_price', 200)) / 2)
            self.prices[item] = float(base)

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def _get_obs(self) -> Union[np.ndarray, Dict]:
        """Build observation."""
        obs_list = []

        # Per-item features
        for item in self.item_list:
            item_data = self.items.get(item, {})
            position = self.positions.get(item)
            price = self.prices.get(item, 0)

            min_price = item_data.get('min_price', price * 0.9)
            max_price = item_data.get('max_price', price * 1.1)
            spread = max_price - min_price
            spread_pct = spread / price if price > 0 else 0

            # Volume metrics
            vol_metrics = {}
            if self.volume_analyzer and VOLUME_ANALYSIS_AVAILABLE:
                vol_metrics = get_volume_metrics_for_item(self.volume_analyzer, item)

            obs_list.extend([
                price / 10000,  # Normalized price
                spread / 1000,  # Normalized spread
                spread_pct,
                vol_metrics.get('recent_volume', 0) / 100000,
                vol_metrics.get('volume_momentum', 0),
                vol_metrics.get('buy_sell_imbalance', 0),
                vol_metrics.get('market_activity', 0),
                position.quantity / 1000 if position else 0,
            ])

        # Portfolio features
        total_value = self.gp
        for item, pos in self.positions.items():
            if pos.quantity > 0:
                total_value += pos.quantity * self.prices.get(item, pos.avg_cost)

        n_positions = sum(1 for pos in self.positions.values() if pos.quantity > 0)

        obs_list.extend([
            self.gp / self.starting_gp,
            total_value / self.starting_gp,
            self.realized_pnl / self.starting_gp,
            n_positions / max(1, self.n_items),
        ])

        return np.array(obs_list, dtype=np.float32)

    def step(self, action: Dict) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self.current_step += 1

        # Parse action
        action_type = action.get('type', 'hold')
        item = action.get('item')
        price = action.get('price', 0)
        quantity = action.get('quantity', 0)

        # Default info
        info = {'msg': '', 'action': action_type, 'item': item}
        reward = 0.0

        # Validate action
        if action_type == 'hold' or item is None:
            info['msg'] = 'Holding'
        elif item not in self.items:
            info['msg'] = f'Invalid item: {item}'
            reward = -0.1
        else:
            # Volume validation
            if self.volume_analyzer and VOLUME_ANALYSIS_AVAILABLE:
                vol_metrics = get_volume_metrics_for_item(self.volume_analyzer, item)

                # Check volume thresholds
                if vol_metrics.get('recent_volume', 0) < 500:
                    info['msg'] = 'Volume too low for trading'
                    reward = -0.05
                elif vol_metrics.get('volume_momentum', 0) < -0.3:
                    info['msg'] = 'Declining volume - risky trade'
                    reward = -0.02
                elif action_type == 'buy' and vol_metrics.get('buy_sell_imbalance', 0) < -0.3:
                    info['msg'] = 'Heavy selling pressure - avoid buying'
                    reward = -0.02
                else:
                    # Execute trade
                    reward, trade_info = self._execute_trade(action_type, item, price, quantity)
                    info.update(trade_info)
            else:
                # No volume validation
                reward, trade_info = self._execute_trade(action_type, item, price, quantity)
                info.update(trade_info)

        # Simulate price changes
        self._simulate_price_movement()

        # Check termination
        terminated = self.current_step >= self.max_steps
        truncated = self.gp < 0  # Bankrupt

        obs = self._get_obs()
        info.update(self._get_info())

        return obs, reward, terminated, truncated, info

    def _execute_trade(self, action_type: str, item: str, price: float, quantity: int) -> Tuple[float, Dict]:
        """Execute a buy or sell trade."""
        info = {'msg': ''}
        reward = 0.0

        item_data = self.items[item]
        position = self.positions[item]
        current_price = self.prices.get(item, item_data.get('base_price', 100))

        if action_type == 'buy':
            # Validate buy
            max_affordable = int(self.gp / price) if price > 0 else 0
            buy_limit = item_data.get('buy_limit', 5000)
            max_quantity = min(max_affordable, buy_limit, quantity)

            if max_quantity <= 0:
                info['msg'] = 'Cannot afford or invalid quantity'
                return -0.1, info

            # Check if price is reasonable
            if price > current_price * 1.1:
                info['msg'] = 'Price too high above market'
                return -0.05, info

            # Execute buy
            cost = max_quantity * price
            self.gp -= cost
            position.add(max_quantity, price, buy_limit)
            self.total_trades += 1
            self.total_volume_traded += cost

            info['msg'] = f'Bought {max_quantity} {item} @ {price}'
            info['quantity'] = max_quantity
            info['cost'] = cost

            # Small reward for buying (actual profit realized on sell)
            reward = 0.01

        elif action_type == 'sell':
            # Validate sell
            if position.quantity <= 0:
                info['msg'] = 'No position to sell'
                return -0.1, info

            sell_quantity = min(quantity, position.quantity)

            if sell_quantity <= 0:
                info['msg'] = 'Invalid sell quantity'
                return -0.05, info

            # Check if price is reasonable
            if price < current_price * 0.9:
                info['msg'] = 'Price too low below market'
                return -0.05, info

            # Calculate P&L
            actual_qty, avg_cost = position.remove(sell_quantity)
            gross = actual_qty * price
            tax = gross * self.GE_TAX_RATE
            net = gross - tax
            pnl = net - (actual_qty * avg_cost)

            self.gp += net
            self.realized_pnl += pnl
            self.total_tax_paid += tax
            self.total_trades += 1
            self.total_volume_traded += gross

            if pnl > 0:
                self.winning_trades += 1

            info['msg'] = f'Sold {actual_qty} {item} @ {price}, P&L: {pnl:.0f}'
            info['quantity'] = actual_qty
            info['gross'] = gross
            info['tax'] = tax
            info['net'] = net
            info['pnl'] = pnl

            # Reward based on P&L
            reward = pnl / self.starting_gp * 100  # Scaled

        return reward, info

    def _simulate_price_movement(self):
        """Simulate random price movements."""
        for item in self.item_list:
            item_data = self.items[item]
            current = self.prices[item]
            min_price = item_data.get('min_price', current * 0.5)
            max_price = item_data.get('max_price', current * 2.0)

            # Random walk with mean reversion
            change = np.random.normal(0, 0.01) * current
            base = item_data.get('base_price', (min_price + max_price) / 2)
            reversion = 0.01 * (base - current)

            new_price = current + change + reversion
            self.prices[item] = np.clip(new_price, min_price, max_price)

    def _get_info(self) -> Dict[str, Any]:
        """Get episode info."""
        total_value = self.gp
        for item, pos in self.positions.items():
            if pos.quantity > 0:
                total_value += pos.quantity * self.prices.get(item, pos.avg_cost)

        return {
            'gp': self.gp,
            'total_value': total_value,
            'realized_pnl': self.realized_pnl,
            'total_return': (total_value - self.starting_gp) / self.starting_gp,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'current_step': self.current_step,
            'max_steps': self.max_steps,
            'total_tax_paid': self.total_tax_paid,
        }

    def render(self):
        """Render the environment."""
        if self.render_mode == "human" or self.render_mode == "ansi":
            info = self._get_info()
            print(f"\n=== Step {self.current_step}/{self.max_steps} ===")
            print(f"GP: {self.gp:,.0f}")
            print(f"Total Value: {info['total_value']:,.0f}")
            print(f"Return: {info['total_return']*100:.2f}%")
            print(f"Trades: {self.total_trades} (Win: {info['win_rate']*100:.1f}%)")
            print(f"Tax Paid: {self.total_tax_paid:,.0f}")

    def close(self):
        """Clean up resources."""
        pass


# Convenience function for creating env
def make_env(**kwargs) -> GrandExchangeEnv:
    """Factory function to create environment."""
    return GrandExchangeEnv(**kwargs)
