"""
Multiprocess PPO Configuration

MAXIMUM UTILIZATION settings for H100 SXM 80GB.
Target: 60-80% VRAM usage, maximum GPU compute utilization.

Key insight: obs_dim = 2610 (200 items × 13 features + 10)
This is small, so we need a MASSIVE network to utilize H100.

Strategy:
- HUGE network (4096 hidden, 8 layers) = ~500M parameters
- MASSIVE rollouts (131072 steps) = 64x original
- ENORMOUS minibatches (32768) = fills H100 tensor cores
- Many PPO epochs (16) = 64 gradient updates per rollout
- 1 worker = ZERO barrier overhead, pure GPU throughput

Target: ~50M total environment steps
- 1 worker × 51.2M steps = 51.2M total steps
- With 131072 rollout steps, this is ~391 updates total
- Each update does 16 epochs × 4 batches = 64 gradient steps

Expected VRAM usage:
- Model: ~2GB (500M params × 4 bytes)
- Rollout buffer: 131072 × 2610 × 4 = ~1.3GB
- Minibatch: 32768 × 2610 × 4 = ~340MB per batch
- Optimizer states: ~4GB
- Gradients + activations: ~10-20GB
- Total: ~20-30GB active, should show 40-60% VRAM
"""

# Environment Configuration
ENV_KWARGS = {
    "cache_file": "training_cache.json",  # JSON file containing cached historical price data
    "initial_cash": 1_000_000,  # Starting capital (gp) for each trading episode
    "episode_length": 864,  # Number of time steps per episode (864 = 3 days at 5min intervals)
    "top_n_items": 200,  # Focus on top 200 traded items for better learning signal
}

# PPO Agent Configuration - MAXIMUM GPU UTILIZATION FOR H100 80GB
# Key insight: obs_dim = 200*13+10 = 2610, so we need MASSIVE hidden layers
PPO_KWARGS = {
    "hidden_size": 4096,  # 4x larger (was 1024) = 16x more compute per layer
    "num_layers": 8,  # 2x deeper (was 4) = way more GPU work
    "lr": 5e-5,  # Very low LR for stability with huge model
    "gamma": 0.99,  # Discount factor for future rewards
    "gae_lambda": 0.95,  # Lambda parameter for GAE
    "clip_epsilon": 0.2,  # PPO clipping range
    "entropy_coef": 0.01,  # Lower entropy for huge batch stability
    "value_coef": 0.5,  # Value function loss coefficient
    "price_bins": 20,  # Number of discrete price levels
    "quantity_bins": 10,  # Number of discrete quantity levels
    "wait_steps_bins": 10,  # Number of discrete wait time options
    "risk_tolerance": 0.3,  # Maximum portfolio allocation per item
    # === MAXIMUM GPU UTILIZATION ===
    "rollout_steps": 131072,  # 64x original - fills memory with data
    "minibatch_size": 32768,  # 128x original - max batch H100 can handle
    "ppo_epochs": 16,  # 4x original - lots of GPU work per rollout
}

# Training Configuration - MAXIMUM UTILIZATION
TRAIN_KWARGS = {
    # Single worker = NO barrier overhead at all, pure GPU throughput
    "num_workers": 1,  # 1 worker - zero sync overhead, max GPU focus
    "max_steps_per_worker": 51_200_000,  # 51.2M steps total
    "save_every_steps": 2_621_440,  # Save every ~2.6M steps
    "log_every_steps": 131_072,  # Log every rollout
    "eval_every_steps": 5_242_880,  # Evaluate every ~5.2M steps
    "use_shared_cache": True,  # Share price data cache
    "gpu_distribution": "single",  # Single H100
    "use_shared_model": False,  # No shared model needed with 1 worker
    "max_checkpoints": 20,  # Keep checkpoints for analysis
}

# Evaluation Configuration
EVAL_KWARGS = {
    "num_episodes": 10,  # Number of episodes to run during each evaluation
    "deterministic": True,  # Use deterministic policy (no exploration) for evaluation
}
