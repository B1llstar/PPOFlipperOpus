"""
Multiprocess PPO Configuration

MAXIMUM UTILIZATION settings for H100 SXM 80GB.
Optimized for 95-100% GPU utilization + high CPU utilization.

Strategy for max utilization:
- Fewer workers (4) = less barrier sync overhead, each worker does MORE work
- HUGE rollouts (16384 steps) = more data per GPU update, fewer syncs
- MASSIVE minibatches (4096) = saturate H100 tensor cores
- Many PPO epochs (16) = maximize GPU compute per rollout
- Large network (1024 hidden, 4 layers) = more GPU compute per forward/backward

Target: ~50M total environment steps
- 4 workers × 12.8M steps each = 51.2M total steps
- With 16384 rollout steps, this is ~782 updates per worker
- Each update does 16 epochs of training on 16384 samples
"""

# Environment Configuration
ENV_KWARGS = {
    "cache_file": "training_cache.json",  # JSON file containing cached historical price data
    "initial_cash": 1_000_000,  # Starting capital (gp) for each trading episode
    "episode_length": 864,  # Number of time steps per episode (864 = 3 days at 5min intervals)
    "top_n_items": 200,  # Focus on top 200 traded items for better learning signal
}

# PPO Agent Configuration - MAXIMUM GPU UTILIZATION
PPO_KWARGS = {
    "hidden_size": 1024,  # Large network for more GPU compute
    "num_layers": 4,  # Deep network
    "lr": 3e-4,  # Slightly lower LR for stability with huge batches
    "gamma": 0.99,  # Discount factor for future rewards
    "gae_lambda": 0.95,  # Lambda parameter for GAE
    "clip_epsilon": 0.2,  # PPO clipping range
    "entropy_coef": 0.05,  # Lower entropy (less exploration needed with more data)
    "value_coef": 0.5,  # Value function loss coefficient
    "price_bins": 20,  # Number of discrete price levels
    "quantity_bins": 10,  # Number of discrete quantity levels
    "wait_steps_bins": 10,  # Number of discrete wait time options
    "risk_tolerance": 0.3,  # Maximum portfolio allocation per item
    # === KEY CHANGES FOR MAX GPU UTILIZATION ===
    "rollout_steps": 16384,  # 8x larger rollouts (was 2048) - MUCH more data per sync
    "minibatch_size": 4096,  # 16x larger batches (was 256) - saturate H100 tensor cores
    "ppo_epochs": 16,  # 4x more epochs (was 4) - maximize GPU work per rollout
}

# Training Configuration - MAXIMUM UTILIZATION
TRAIN_KWARGS = {
    # FEWER workers = less barrier overhead, each worker collects more before sync
    "num_workers": 4,  # 4 workers (was 10) - faster barrier sync, less waiting
    "max_steps_per_worker": 12_800_000,  # 12.8M steps each = 51.2M total (same total)
    "save_every_steps": 655_360,  # Save every ~2.6M total steps (every 10 updates)
    "log_every_steps": 16_384,  # Log every rollout
    "eval_every_steps": 1_310_720,  # Evaluate every ~5.2M total steps
    "use_shared_cache": True,  # Share price data cache across workers
    "gpu_distribution": "single",  # All on single H100
    "use_shared_model": True,  # Enable shared model training
    "max_checkpoints": 20,  # Keep checkpoints for analysis
}

# Evaluation Configuration
EVAL_KWARGS = {
    "num_episodes": 10,  # Number of episodes to run during each evaluation
    "deterministic": True,  # Use deterministic policy (no exploration) for evaluation
}
