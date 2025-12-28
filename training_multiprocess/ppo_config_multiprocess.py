"""
Multiprocess PPO Configuration

Optimized settings for parallel training.

Hardware Profiles:
- H100 SXM 80GB: 20 workers, 150 items, 2048 rollout
- RTX 3090 24GB: 5 workers, 80 items, 1024 rollout (for local testing)
"""

# Hardware profile selection
USE_3090_PROFILE = True  # Set to True for RTX 3090 testing, False for H100

# Environment Configuration
ENV_KWARGS = {
    "cache_file": "training_cache.json",
    "initial_cash": 1_000_000,
    "episode_length": 168,  # 1 week of hourly data
    "top_n_items": 20 if USE_3090_PROFILE else 999999999,  # 3090: 80 items | H100: 150 items
}

# PPO Agent Configuration
PPO_KWARGS = {
    "hidden_size": 512,
    "num_layers": 3,
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "price_bins": 20,
    "quantity_bins": 10,
    "wait_steps_bins": 10,
    "risk_tolerance": 0.3,
    "rollout_steps": 2048,      # Steps before update
    "minibatch_size": 64,       # Batch size for training
    "ppo_epochs": 10,           # Epochs per update
}

# Training Configuration
TRAIN_KWARGS = {
    # 3090: 5 workers × 3.5GB = ~17.5GB (73% of 24GB, safe buffer)
    # H100: 20 workers × 3.5GB = ~70GB (87.5% of 80GB, 15% buffer)
    "num_workers": 2 if USE_3090_PROFILE else 20,
    "max_steps_per_worker": 100_000 if USE_3090_PROFILE else 1_000_000,  # 3090: shorter for testing
    "save_every_steps": 10_000 if USE_3090_PROFILE else 50_000,  # 3090: more frequent saves
    "log_every_steps": 500 if USE_3090_PROFILE else 1_000,  # 3090: more frequent logs
    "eval_every_steps": 5_000 if USE_3090_PROFILE else 25_000,
    "use_shared_cache": True,       # Use shared memory for cache
    "gpu_distribution": "round-robin",  # How to assign GPUs (single GPU: all workers share GPU 0)
}

# Evaluation Configuration
EVAL_KWARGS = {
    "num_episodes": 10,
    "deterministic": True,
}
