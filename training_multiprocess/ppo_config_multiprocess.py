"""
Multiprocess PPO Configuration

EXTREME UTILIZATION settings for H100 SXM 80GB.
Target: 70-90% VRAM usage, maximum GPU compute utilization.

Strategy:
- MASSIVE network (2048 hidden, 6 layers) = 4x more parameters
- HUGE rollouts (65536 steps) = tons of data per update
- ENORMOUS minibatches (16384) = max batch for H100
- Many PPO epochs (32) = extreme GPU compute per rollout
- 2 workers = minimal sync overhead, each worker is a beast

Target: ~50M total environment steps
- 2 workers × 25.6M steps each = 51.2M total steps
- With 65536 rollout steps, this is ~391 updates per worker
- Each update does 32 epochs of training on 65536 samples
"""

# Environment Configuration
ENV_KWARGS = {
    "cache_file": "training_cache.json",  # JSON file containing cached historical price data
    "initial_cash": 1_000_000,  # Starting capital (gp) for each trading episode
    "episode_length": 864,  # Number of time steps per episode (864 = 3 days at 5min intervals)
    "top_n_items": 200,  # Focus on top 200 traded items for better learning signal
}

# PPO Agent Configuration - EXTREME GPU UTILIZATION FOR H100 80GB
PPO_KWARGS = {
    "hidden_size": 2048,  # 2x larger network (was 1024) = 4x more compute
    "num_layers": 6,  # Deeper network (was 4) = more GPU work per pass
    "lr": 1e-4,  # Lower LR for stability with massive batches
    "gamma": 0.99,  # Discount factor for future rewards
    "gae_lambda": 0.95,  # Lambda parameter for GAE
    "clip_epsilon": 0.2,  # PPO clipping range
    "entropy_coef": 0.02,  # Lower entropy for huge batch stability
    "value_coef": 0.5,  # Value function loss coefficient
    "price_bins": 20,  # Number of discrete price levels
    "quantity_bins": 10,  # Number of discrete quantity levels
    "wait_steps_bins": 10,  # Number of discrete wait time options
    "risk_tolerance": 0.3,  # Maximum portfolio allocation per item
    # === EXTREME GPU UTILIZATION ===
    "rollout_steps": 65536,  # 32x original (was 2048) - massive data per sync
    "minibatch_size": 16384,  # 64x original (was 256) - fill H100 memory
    "ppo_epochs": 32,  # 8x original (was 4) - tons of GPU work per rollout
}

# Training Configuration - EXTREME UTILIZATION
TRAIN_KWARGS = {
    # Only 2 workers = minimal barrier overhead, maximum work per worker
    "num_workers": 2,  # 2 workers - almost no sync overhead
    "max_steps_per_worker": 25_600_000,  # 25.6M steps each = 51.2M total
    "save_every_steps": 1_310_720,  # Save every ~2.6M total steps
    "log_every_steps": 65_536,  # Log every rollout
    "eval_every_steps": 2_621_440,  # Evaluate every ~5.2M total steps
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
