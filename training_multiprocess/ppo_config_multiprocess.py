"""
Multiprocess PPO Configuration

Optimized settings for parallel training on H100 GPU.
"""

# Environment Configuration
ENV_KWARGS = {
    "cache_file": "training_cache.json",
    "initial_cash": 1_000_000,
    "episode_length": 168,  # 1 week of hourly data
    "top_n_items": 150,     # More items for diverse experiences across workers
    "transaction_fee": 0.01,
    "max_inventory_slots": 500,
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
    "num_workers": 20,              # 20 workers = ~70GB VRAM (87.5%), 15% buffer for spikes
    "max_steps_per_worker": 1_000_000,  # Max steps per worker
    "save_every_steps": 50_000,     # Save checkpoint frequency
    "log_every_steps": 1_000,       # Log frequency
    "eval_every_steps": 25_000,     # Evaluation frequency
    "use_shared_cache": True,       # Use shared memory for cache
    "gpu_distribution": "round-robin",  # How to assign GPUs (single GPU: all workers share GPU 0)
}

# Evaluation Configuration
EVAL_KWARGS = {
    "num_episodes": 10,
    "deterministic": True,
}
