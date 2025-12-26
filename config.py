#!/usr/bin/env python3
"""
Configuration for PPO Flipper

Contains environment and PPO hyperparameters used across training and inference.
"""

import torch

# Environment configuration
ENV_KWARGS = {
    # Starting capital
    "starting_gp": 10_000_000,  # 10M GP

    # Episode settings
    "max_steps": 1000,  # Maximum steps per episode
    "step_interval": 60,  # Seconds per step (simulated)

    # Trading constraints
    "ge_tax_rate": 0.01,  # 1% GE tax on sells
    "ge_slots": 8,  # Number of GE slots

    # Volume filtering
    "min_volume_threshold": 1000,  # Minimum volume to consider trading
    "volume_momentum_threshold": -0.2,  # Reject items with declining volume below this

    # Price constraints
    "max_spread_pct": 0.15,  # Maximum spread percentage to consider
    "min_spread_pct": 0.01,  # Minimum spread to be profitable after tax

    # Risk management
    "max_position_pct": 0.25,  # Max % of portfolio in single item
    "max_order_pct": 0.10,  # Max % of GP per single order

    # Strict mode for item selection (only high volume items)
    "strict_mode": False,
    "high_vol_items_path": "config/high_vol_items.txt",

    # Random subset selection (for diversity in training)
    "random_item_subset": False,
    "subset_fraction": 0.7,
}

# PPO hyperparameters
PPO_KWARGS = {
    # Network architecture
    "hidden_size": 256,
    "num_layers": 3,

    # Action space discretization
    "price_bins": 21,  # Number of price offset bins
    "quantity_bins": 11,  # Number of quantity bins
    "wait_steps_bins": 5,  # Number of wait time bins

    # PPO algorithm parameters
    "lr": 3e-4,
    "gamma": 0.99,  # Discount factor
    "gae_lambda": 0.95,  # GAE lambda
    "clip_epsilon": 0.2,  # PPO clip parameter
    "entropy_coef": 0.01,  # Entropy bonus coefficient
    "value_coef": 0.5,  # Value loss coefficient
    "max_grad_norm": 0.5,  # Gradient clipping

    # Training parameters
    "batch_size": 64,
    "n_epochs": 10,  # PPO epochs per update
    "n_steps": 2048,  # Steps per rollout

    # Regularization
    "weight_decay": 0.0001,

    # Device
    "device": "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu",
}

# Training configuration
TRAIN_KWARGS = {
    # Multi-agent training
    "num_agents": 4,

    # Training duration
    "total_timesteps": 1_000_000,
    "eval_freq": 10_000,
    "checkpoint_freq": 50_000,

    # Logging
    "log_freq": 1000,
    "tensorboard": True,

    # Data refresh
    "refresh_interval": 300,  # Refresh market data every 5 minutes
}

# Inference configuration
INFERENCE_KWARGS = {
    # Polling intervals (seconds)
    "latest_interval": 30,
    "five_min_interval": 60,

    # Decision making
    "min_confidence": 0.7,  # Minimum action confidence to execute
    "deterministic": False,  # Use deterministic policy

    # Safety
    "max_daily_trades": 1000,
    "max_daily_loss_pct": 0.10,  # Stop trading if down 10%
}
