#!/usr/bin/env python3
"""
Configuration for PPO Flipper

Contains environment and PPO hyperparameters used across training and inference.
"""

import torch

# Environment configuration
ENV_KWARGS = {
    # Starting capital
    "initial_cash": 1_000_000,  # 1M GP

    # Episode settings
    # Note: episode_length + LOOKBACK_PERIODS (24) must be <= available hourly data points
    # With 365 hours of 1h data, max episode_length is ~300-340
    "episode_length": 168,  # 1 week of hourly steps (168 hours)
    
    # Top N items to trade
    "top_n_items": 50,  # Number of items to trade
    
    # Database/cache settings - NO DATABASE, use cache only
    "db_path": None,  # No SQLite database
    "cache_file": "training_cache.json",  # Use cache file instead of database
    
    # GE limit multiplier
    "ge_limit_multiplier": 1.0,  # Scale GE limits (1.0 = realistic)
    
    # Volume constraint
    "include_volume_constraint": True,  # Include volume constraints
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
    "num_agents": 16,  # Number of agents to run in parallel (optimized for h100 sxm with 20 vcpus)

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
