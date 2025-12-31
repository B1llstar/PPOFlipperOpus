#!/usr/bin/env python3
"""
PPO Agent for Grand Exchange Flipping

Hardware-accelerated PPO implementation using PyTorch with:
- Actor-Critic architecture with shared feature extraction
- Generalized Advantage Estimation (GAE)
- Clipped surrogate objective
- Entropy bonus for exploration
- Value function clipping
- Gradient clipping
- Mixed precision training support
- Rollout buffer for experience storage

Optimized for:
- Apple Silicon (MPS)
- NVIDIA GPUs (CUDA)
- CPU fallback
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import random
import time

logger = logging.getLogger("PPOAgent")


def get_device() -> torch.device:
    """Get best available device for training."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


class RolloutBuffer:
    """
    Buffer for storing rollout experiences.

    Stores transitions and computes advantages using GAE.
    """

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        n_items: int,
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.n_items = n_items
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Pre-allocate buffers on CPU (moved to device during training)
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.action_types = np.zeros(buffer_size, dtype=np.int64)
        self.action_items = np.zeros(buffer_size, dtype=np.int64)
        self.action_prices = np.zeros(buffer_size, dtype=np.int64)
        self.action_quantities = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)

        # Computed during finalization
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        self.pos = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action_type: int,
        action_item: int,
        action_price: int,
        action_quantity: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float
    ):
        """Add a transition to the buffer."""
        if self.pos >= self.buffer_size:
            logger.warning("Buffer full, overwriting old data")
            self.pos = 0

        self.observations[self.pos] = obs
        self.action_types[self.pos] = action_type
        self.action_items[self.pos] = action_item
        self.action_prices[self.pos] = action_price
        self.action_quantities[self.pos] = action_quantity
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob

        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True

    def compute_advantages(self, last_value: float, last_done: bool):
        """Compute GAE advantages and returns."""
        last_gae = 0
        n = self.pos if not self.full else self.buffer_size

        for t in reversed(range(n)):
            if t == n - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns[:n] = self.advantages[:n] + self.values[:n]

    def get_batches(self, batch_size: int):
        """Generate random mini-batches as tensors on device."""
        n = self.pos if not self.full else self.buffer_size
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]

            yield {
                "observations": torch.from_numpy(self.observations[batch_indices]).to(self.device),
                "action_types": torch.from_numpy(self.action_types[batch_indices]).to(self.device),
                "action_items": torch.from_numpy(self.action_items[batch_indices]).to(self.device),
                "action_prices": torch.from_numpy(self.action_prices[batch_indices]).to(self.device),
                "action_quantities": torch.from_numpy(self.action_quantities[batch_indices]).to(self.device),
                "old_log_probs": torch.from_numpy(self.log_probs[batch_indices]).to(self.device),
                "advantages": torch.from_numpy(self.advantages[batch_indices]).to(self.device),
                "returns": torch.from_numpy(self.returns[batch_indices]).to(self.device),
                "old_values": torch.from_numpy(self.values[batch_indices]).to(self.device)
            }

    def reset(self):
        """Reset buffer for new rollout."""
        self.pos = 0
        self.full = False


class FeatureExtractor(nn.Module):
    """Shared feature extraction network with layer normalization."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for i in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0 and i < num_layers - 1:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_size

        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_size

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ActorHead(nn.Module):
    """
    Actor network head for discrete actions.

    Outputs logits for:
    - Action type (hold, buy, sell)
    - Item selection
    - Price bin
    - Quantity bin
    - Wait steps bin
    """

    def __init__(
        self,
        feature_dim: int,
        n_items: int,
        price_bins: int = 21,
        quantity_bins: int = 11,
        wait_steps_bins: int = 5
    ):
        super().__init__()

        self.n_items = n_items
        self.price_bins = price_bins
        self.quantity_bins = quantity_bins
        self.wait_steps_bins = wait_steps_bins

        # Shared hidden layer for all action heads
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.ReLU()
        )

        # Action type head (hold, buy, sell)
        self.action_type_head = nn.Linear(feature_dim // 2, 3)

        # Item selection head
        self.item_head = nn.Linear(feature_dim // 2, n_items)

        # Price bin head
        self.price_head = nn.Linear(feature_dim // 2, price_bins)

        # Quantity bin head
        self.quantity_head = nn.Linear(feature_dim // 2, quantity_bins)

        # Wait steps head
        self.wait_head = nn.Linear(feature_dim // 2, wait_steps_bins)

        self._init_weights()

    def _init_weights(self):
        for module in [self.action_type_head, self.item_head,
                       self.price_head, self.quantity_head, self.wait_head]:
            nn.init.orthogonal_(module.weight, gain=0.01)
            nn.init.constant_(module.bias, 0)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns logits for each action component."""
        shared = self.shared(features)
        return {
            "action_type": self.action_type_head(shared),
            "item": self.item_head(shared),
            "price": self.price_head(shared),
            "quantity": self.quantity_head(shared),
            "wait_steps": self.wait_head(shared)
        }


class CriticHead(nn.Module):
    """Critic network head for value estimation."""

    def __init__(self, feature_dim: int):
        super().__init__()

        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.value_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1)
                nn.init.constant_(module.bias, 0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


class PPOAgent:
    """
    PPO Agent for Grand Exchange trading.

    Features:
    - Hardware acceleration (CUDA, MPS, CPU)
    - Discrete action space for trading decisions
    - Volume blacklist integration
    - Shared knowledge repository support
    - Risk tolerance per agent
    - Rollout buffer for efficient training
    - Learning rate scheduling
    - Gradient clipping
    """

    def __init__(
        self,
        item_list: List[str],
        price_ranges: Dict[str, Tuple[float, float]],
        buy_limits: Dict[str, int],
        device: Optional[torch.device] = None,
        hidden_size: int = 256,
        num_layers: int = 3,
        price_bins: int = 21,
        quantity_bins: int = 11,
        wait_steps_bins: int = 5,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        clip_value: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        buffer_size: int = 2048,
        volume_blacklist: Optional[set] = None,
        volume_analyzer: Any = None,
        shared_knowledge: Any = None,
        agent_id: int = 0
    ):
        self.item_list = item_list
        self.price_ranges = price_ranges
        self.buy_limits = buy_limits
        self.device = device or get_device()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.price_bins = price_bins
        self.quantity_bins = quantity_bins
        self.wait_steps_bins = wait_steps_bins

        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.clip_value = clip_value
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.buffer_size = buffer_size

        self.volume_blacklist = volume_blacklist or set()
        self.volume_analyzer = volume_analyzer
        self.shared_knowledge = shared_knowledge
        self.agent_id = agent_id

        # Risk tolerance varies per agent for diversity
        self.risk_tolerance = 0.3 + random.random() * 0.4  # 0.3 to 0.7

        # Calculate observation dimension (MUST MATCH ge_environment.py)
        # Per item: 13 features
        #   - Current market: high, low, spread_pct, high_vol, low_vol, vol_ratio = 6
        #   - Rolling stats: sma_6, sma_24, vol_sma, volatility = 4
        #   - Position: quantity, avg_cost, unrealized_pnl_pct = 3
        # Portfolio: 6 features (cash, value, realized_pnl, unrealized_pnl, position_count, concentration)
        # Time: 4 cyclical features (sin/cos hour, sin/cos day)
        self.n_items = len(item_list)
        per_item_features = 13
        portfolio_features = 6
        time_features = 4
        self.obs_dim = max(1, self.n_items * per_item_features + portfolio_features + time_features)

        # Create networks
        self.feature_extractor = FeatureExtractor(
            input_dim=self.obs_dim,
            hidden_size=hidden_size,
            num_layers=num_layers
        ).to(self.device)

        self.actor = ActorHead(
            feature_dim=hidden_size,
            n_items=max(1, self.n_items),
            price_bins=price_bins,
            quantity_bins=quantity_bins,
            wait_steps_bins=wait_steps_bins
        ).to(self.device)

        self.critic = CriticHead(feature_dim=hidden_size).to(self.device)

        # Collect all parameters
        self.all_parameters = (
            list(self.feature_extractor.parameters()) +
            list(self.actor.parameters()) +
            list(self.critic.parameters())
        )

        # Optimizer with weight decay
        self.optimizer = AdamW(
            self.all_parameters,
            lr=lr,
            weight_decay=0.0001,
            eps=1e-5
        )

        # Learning rate scheduler
        self.scheduler = None  # Set during training if needed

        # Rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=buffer_size,
            obs_dim=self.obs_dim,
            n_items=self.n_items,
            device=self.device,
            gamma=gamma,
            gae_lambda=gae_lambda
        )

        # Training stats
        self.n_updates = 0
        self.total_timesteps = 0
        self.training_start_time = None

        # Metrics tracking
        self.recent_losses = []
        self.recent_rewards = []

        logger.info(f"PPO Agent {agent_id} initialized on {self.device}")
        logger.info(f"  Items: {self.n_items}, Obs dim: {self.obs_dim}")
        logger.info(f"  Hidden: {hidden_size}, Layers: {num_layers}")
        logger.info(f"  Risk tolerance: {self.risk_tolerance:.3f}")
        logger.info(f"  Parameters: {sum(p.numel() for p in self.all_parameters):,}")

    def _obs_to_tensor(self, obs: Any) -> torch.Tensor:
        """Convert observation to tensor."""
        if isinstance(obs, dict):
            # Flatten dict observation
            flat = []
            for key in sorted(obs.keys()):
                val = obs[key]
                if isinstance(val, dict):
                    for k2 in sorted(val.keys()):
                        v = val[k2]
                        if isinstance(v, (list, np.ndarray)):
                            flat.extend(np.asarray(v).flatten())
                        elif isinstance(v, (int, float)):
                            flat.append(float(v))
                elif isinstance(val, (list, np.ndarray)):
                    flat.extend(np.asarray(val).flatten())
                elif isinstance(val, (int, float)):
                    flat.append(float(val))
            obs = np.array(flat, dtype=np.float32)

        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()

        # Ensure correct dimension
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # Pad or truncate to expected dimension
        if obs.shape[1] != self.obs_dim:
            if obs.shape[1] < self.obs_dim:
                padding = torch.zeros(obs.shape[0], self.obs_dim - obs.shape[1])
                obs = torch.cat([obs, padding], dim=1)
            else:
                obs = obs[:, :self.obs_dim]

        return obs.to(self.device)

    def sample_action(
        self,
        obs: Any,
        deterministic: bool = False
    ) -> Tuple[Dict[str, Any], Optional[float]]:
        """
        Sample action from policy.

        Returns:
            action_dict: Dict with 'type', 'item', 'price', 'quantity'
            log_prob: Log probability of the action (None if deterministic)
        """
        with torch.no_grad():
            obs_tensor = self._obs_to_tensor(obs)
            features = self.feature_extractor(obs_tensor)
            logits = self.actor(features)

            # Sample each component
            action_type_dist = Categorical(logits=logits["action_type"])
            item_dist = Categorical(logits=logits["item"])
            price_dist = Categorical(logits=logits["price"])
            quantity_dist = Categorical(logits=logits["quantity"])

            if deterministic:
                action_type = logits["action_type"].argmax(dim=-1)
                item_idx = logits["item"].argmax(dim=-1)
                price_bin = logits["price"].argmax(dim=-1)
                quantity_bin = logits["quantity"].argmax(dim=-1)
                log_prob = None
            else:
                action_type = action_type_dist.sample()
                item_idx = item_dist.sample()
                price_bin = price_dist.sample()
                quantity_bin = quantity_dist.sample()

                log_prob = (
                    action_type_dist.log_prob(action_type) +
                    item_dist.log_prob(item_idx) +
                    price_dist.log_prob(price_bin) +
                    quantity_dist.log_prob(quantity_bin)
                ).item()

            # Convert to action dict
            action_type_val = action_type.item()
            item_idx_val = item_idx.item()
            price_bin_val = price_bin.item()
            quantity_bin_val = quantity_bin.item()

            # Ensure item_idx is in bounds
            item_idx_val = min(item_idx_val, len(self.item_list) - 1)
            item = self.item_list[item_idx_val]

            # Check volume blacklist
            if item in self.volume_blacklist:
                # Force hold action for blacklisted items
                action_type_val = 0

            # Decode action type
            action_types = ['hold', 'buy', 'sell']
            action_type_str = action_types[action_type_val]

            # Decode price from bin
            price_range = self.price_ranges.get(item, (100, 200))
            min_price, max_price = price_range
            price_frac = price_bin_val / max(self.price_bins - 1, 1)
            price = int(min_price + (max_price - min_price) * price_frac)

            # Decode quantity from bin
            buy_limit = self.buy_limits.get(item, 1000)
            quantity_frac = quantity_bin_val / max(self.quantity_bins - 1, 1)
            quantity = max(1, int(buy_limit * quantity_frac * self.risk_tolerance))

            action_dict = {
                'type': action_type_str,
                'item': item,
                'price': price,
                'quantity': quantity,
                '_action_type_idx': action_type_val,
                '_item_idx': item_idx_val,
                '_price_bin': price_bin_val,
                '_quantity_bin': quantity_bin_val
            }

            return action_dict, log_prob

    def get_value(self, obs: Any) -> float:
        """Get value estimate for observation."""
        with torch.no_grad():
            obs_tensor = self._obs_to_tensor(obs)
            features = self.feature_extractor(obs_tensor)
            value = self.critic(features)
            return value.item()

    def store_transition(
        self,
        obs: Any,
        action: Dict,
        reward: float,
        done: bool,
        value: float,
        log_prob: float
    ):
        """Store a transition in the buffer."""
        obs_array = self._obs_to_numpy(obs)

        self.buffer.add(
            obs=obs_array,
            action_type=action.get('_action_type_idx', 0),
            action_item=action.get('_item_idx', 0),
            action_price=action.get('_price_bin', 0),
            action_quantity=action.get('_quantity_bin', 0),
            reward=reward,
            done=done,
            value=value,
            log_prob=log_prob
        )
        self.total_timesteps += 1

    def _obs_to_numpy(self, obs: Any) -> np.ndarray:
        """Convert observation to numpy array."""
        if isinstance(obs, dict):
            flat = []
            for key in sorted(obs.keys()):
                val = obs[key]
                if isinstance(val, dict):
                    for k2 in sorted(val.keys()):
                        v = val[k2]
                        if isinstance(v, (list, np.ndarray)):
                            flat.extend(np.asarray(v).flatten())
                        elif isinstance(v, (int, float)):
                            flat.append(float(v))
                elif isinstance(val, (list, np.ndarray)):
                    flat.extend(np.asarray(val).flatten())
                elif isinstance(val, (int, float)):
                    flat.append(float(val))
            obs = np.array(flat, dtype=np.float32)

        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()

        obs = np.asarray(obs, dtype=np.float32).flatten()

        # Pad or truncate
        if len(obs) < self.obs_dim:
            obs = np.concatenate([obs, np.zeros(self.obs_dim - len(obs))])
        elif len(obs) > self.obs_dim:
            obs = obs[:self.obs_dim]

        return obs

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        action_types: torch.Tensor,
        action_items: torch.Tensor,
        action_prices: torch.Tensor,
        action_quantities: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and entropy for given actions.

        Returns:
            log_prob: Log probability of actions
            entropy: Policy entropy
            value: State values
        """
        features = self.feature_extractor(obs)
        logits = self.actor(features)
        value = self.critic(features)

        # Create distributions
        action_type_dist = Categorical(logits=logits["action_type"])
        item_dist = Categorical(logits=logits["item"])
        price_dist = Categorical(logits=logits["price"])
        quantity_dist = Categorical(logits=logits["quantity"])

        # Log probs
        log_prob = (
            action_type_dist.log_prob(action_types) +
            item_dist.log_prob(action_items) +
            price_dist.log_prob(action_prices) +
            quantity_dist.log_prob(action_quantities)
        )

        # Entropy
        entropy = (
            action_type_dist.entropy() +
            item_dist.entropy() +
            price_dist.entropy() +
            quantity_dist.entropy()
        )

        return log_prob, entropy, value.squeeze(-1)

    def update(self, last_obs: Any, last_done: bool, n_epochs: int = 10, batch_size: int = 64) -> Dict[str, float]:
        """
        Perform PPO update using collected rollout.

        Returns dict of training metrics.
        """
        # Compute final value for GAE
        with torch.no_grad():
            last_value = self.get_value(last_obs)

        # Compute advantages
        self.buffer.compute_advantages(last_value, last_done)

        # Normalize advantages
        n = self.buffer.pos if not self.buffer.full else self.buffer.buffer_size
        adv = self.buffer.advantages[:n]
        self.buffer.advantages[:n] = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Training metrics
        all_losses = []
        all_pg_losses = []
        all_value_losses = []
        all_entropy_losses = []
        all_clip_fractions = []

        # Multiple epochs over the data
        for epoch in range(n_epochs):
            for batch in self.buffer.get_batches(batch_size):
                loss, metrics = self._update_batch(batch)
                all_losses.append(loss)
                all_pg_losses.append(metrics["pg_loss"])
                all_value_losses.append(metrics["value_loss"])
                all_entropy_losses.append(metrics["entropy_loss"])
                all_clip_fractions.append(metrics["clip_fraction"])

        self.n_updates += 1
        self.buffer.reset()

        result = {
            "loss": np.mean(all_losses),
            "pg_loss": np.mean(all_pg_losses),
            "value_loss": np.mean(all_value_losses),
            "entropy_loss": np.mean(all_entropy_losses),
            "clip_fraction": np.mean(all_clip_fractions),
            "n_updates": self.n_updates,
            "total_timesteps": self.total_timesteps
        }

        self.recent_losses.append(result["loss"])
        if len(self.recent_losses) > 100:
            self.recent_losses.pop(0)

        return result

    def _update_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, Dict]:
        """Update network on a single batch."""
        obs = batch["observations"]
        action_types = batch["action_types"]
        action_items = batch["action_items"]
        action_prices = batch["action_prices"]
        action_quantities = batch["action_quantities"]
        old_log_probs = batch["old_log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        old_values = batch["old_values"]

        # Forward pass
        log_probs, entropy, values = self.evaluate_actions(
            obs, action_types, action_items, action_prices, action_quantities
        )

        # Policy loss (clipped surrogate objective)
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        pg_loss = -torch.min(surr1, surr2).mean()

        # Value loss (clipped)
        if self.clip_value > 0:
            values_clipped = old_values + torch.clamp(
                values - old_values,
                -self.clip_value,
                self.clip_value
            )
            value_loss1 = F.mse_loss(values, returns)
            value_loss2 = F.mse_loss(values_clipped, returns)
            value_loss = torch.max(value_loss1, value_loss2)
        else:
            value_loss = F.mse_loss(values, returns)

        # Entropy loss (for exploration)
        entropy_loss = -entropy.mean()

        # Total loss
        loss = pg_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.all_parameters, self.max_grad_norm)
        self.optimizer.step()

        # Step scheduler if present
        if self.scheduler is not None:
            self.scheduler.step()

        # Metrics
        with torch.no_grad():
            clip_fraction = (torch.abs(ratio - 1) > self.clip_epsilon).float().mean().item()

        metrics = {
            "pg_loss": pg_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "clip_fraction": clip_fraction
        }

        return loss.item(), metrics

    def compute_gradients(self, last_obs: Any, last_done: bool, n_epochs: int = 10, batch_size: int = 64) -> Dict[str, float]:
        """
        Compute gradients from collected rollout WITHOUT applying optimizer step.
        For use in distributed training where gradients are aggregated externally.

        FIXED: Now properly averages gradients across batches instead of accumulating.

        Returns dict of training metrics (same as update()).
        """
        # Compute final value for GAE
        with torch.no_grad():
            last_value = self.get_value(last_obs)

        # Compute advantages
        self.buffer.compute_advantages(last_value, last_done)

        # Normalize advantages
        n = self.buffer.pos if not self.buffer.full else self.buffer.buffer_size
        adv = self.buffer.advantages[:n]
        self.buffer.advantages[:n] = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Training metrics
        all_losses = []
        all_pg_losses = []
        all_value_losses = []
        all_entropy_losses = []
        all_clip_fractions = []

        # Accumulate gradients properly by averaging across all batches
        accumulated_grads = {}
        num_batches = 0

        # Multiple epochs over the data
        for epoch in range(n_epochs):
            for batch in self.buffer.get_batches(batch_size):
                # Zero gradients for this batch
                self.optimizer.zero_grad()

                # Get observations
                obs = batch["observations"]
                action_types = batch["action_types"]
                action_items = batch["action_items"]
                action_prices = batch["action_prices"]
                action_quantities = batch["action_quantities"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                old_values = batch["old_values"]

                # Forward pass
                log_probs, entropy, values = self.evaluate_actions(
                    obs, action_types, action_items, action_prices, action_quantities
                )

                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                pg_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                if self.clip_value > 0:
                    values_clipped = old_values + torch.clamp(
                        values - old_values,
                        -self.clip_value,
                        self.clip_value
                    )
                    value_loss1 = F.mse_loss(values, returns)
                    value_loss2 = F.mse_loss(values_clipped, returns)
                    value_loss = torch.max(value_loss1, value_loss2)
                else:
                    value_loss = F.mse_loss(values, returns)

                # Entropy loss (for exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = pg_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Backward pass for this batch
                loss.backward()

                # Accumulate gradients (we'll average later)
                for name, param in self.feature_extractor.named_parameters():
                    if param.grad is not None:
                        key = f'feature_extractor.{name}'
                        if key not in accumulated_grads:
                            accumulated_grads[key] = param.grad.clone()
                        else:
                            accumulated_grads[key] += param.grad

                for name, param in self.actor.named_parameters():
                    if param.grad is not None:
                        key = f'actor.{name}'
                        if key not in accumulated_grads:
                            accumulated_grads[key] = param.grad.clone()
                        else:
                            accumulated_grads[key] += param.grad

                for name, param in self.critic.named_parameters():
                    if param.grad is not None:
                        key = f'critic.{name}'
                        if key not in accumulated_grads:
                            accumulated_grads[key] = param.grad.clone()
                        else:
                            accumulated_grads[key] += param.grad

                num_batches += 1

                # Metrics
                with torch.no_grad():
                    clip_fraction = (torch.abs(ratio - 1) > self.clip_epsilon).float().mean().item()

                all_losses.append(loss.item())
                all_pg_losses.append(pg_loss.item())
                all_value_losses.append(value_loss.item())
                all_entropy_losses.append(entropy_loss.item())
                all_clip_fractions.append(clip_fraction)

        # Average the accumulated gradients and set them on the parameters
        self.optimizer.zero_grad()
        for name, param in self.feature_extractor.named_parameters():
            key = f'feature_extractor.{name}'
            if key in accumulated_grads:
                param.grad = accumulated_grads[key] / num_batches

        for name, param in self.actor.named_parameters():
            key = f'actor.{name}'
            if key in accumulated_grads:
                param.grad = accumulated_grads[key] / num_batches

        for name, param in self.critic.named_parameters():
            key = f'critic.{name}'
            if key in accumulated_grads:
                param.grad = accumulated_grads[key] / num_batches

        # Clip averaged gradients
        nn.utils.clip_grad_norm_(self.all_parameters, self.max_grad_norm)

        self.n_updates += 1
        self.buffer.reset()

        result = {
            "loss": np.mean(all_losses),
            "pg_loss": np.mean(all_pg_losses),
            "value_loss": np.mean(all_value_losses),
            "entropy_loss": np.mean(all_entropy_losses),
            "clip_fraction": np.mean(all_clip_fractions),
            "n_updates": self.n_updates,
            "total_timesteps": self.total_timesteps
        }

        self.recent_losses.append(result["loss"])
        if len(self.recent_losses) > 100:
            self.recent_losses.pop(0)

        return result

    def save(self, path: str):
        """Save agent to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "feature_extractor": self.feature_extractor.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "n_updates": self.n_updates,
            "total_timesteps": self.total_timesteps,
            "item_list": self.item_list,
            "price_ranges": self.price_ranges,
            "buy_limits": self.buy_limits,
            "config": {
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "price_bins": self.price_bins,
                "quantity_bins": self.quantity_bins,
                "wait_steps_bins": self.wait_steps_bins,
                "lr": self.lr,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_epsilon": self.clip_epsilon,
                "entropy_coef": self.entropy_coef,
                "value_coef": self.value_coef,
                "risk_tolerance": self.risk_tolerance,
                "obs_dim": self.obs_dim,
                "n_items": self.n_items
            }
        }, path)

        logger.info(f"Agent {self.agent_id} saved to {path}")

    def load(self, path: str):
        """Load agent from file."""
        checkpoint = torch.load(path, map_location=self.device)

        self.feature_extractor.load_state_dict(checkpoint["feature_extractor"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.n_updates = checkpoint.get("n_updates", 0)
        self.total_timesteps = checkpoint.get("total_timesteps", 0)

        if "config" in checkpoint:
            config = checkpoint["config"]
            self.risk_tolerance = config.get("risk_tolerance", self.risk_tolerance)

        logger.info(f"Agent loaded from {path}")

    def set_training_mode(self, training: bool):
        """Set network to training or eval mode."""
        if training:
            self.feature_extractor.train()
            self.actor.train()
            self.critic.train()
        else:
            self.feature_extractor.eval()
            self.actor.eval()
            self.critic.eval()

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "agent_id": self.agent_id,
            "n_updates": self.n_updates,
            "total_timesteps": self.total_timesteps,
            "risk_tolerance": self.risk_tolerance,
            "n_items": self.n_items,
            "obs_dim": self.obs_dim,
            "avg_recent_loss": np.mean(self.recent_losses) if self.recent_losses else 0,
            "blacklisted_items": len(self.volume_blacklist)
        }
