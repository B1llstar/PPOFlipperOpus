"""
Multiprocess PPO Configuration

Production settings for H100 SXM 80GB parallel training.
Optimized for convergence with 200 items.

Target: ~50M total environment steps for robust convergence
- 10 workers × 5.12M steps each = 51.2M total steps
- With 2048 rollout steps, this is ~2,500 updates per worker
- Total training time estimate: 4-8 hours on H100

Key optimizations:
- Higher learning rate (5e-4) for faster initial learning
- Larger rollout steps (2048) for more stable gradients
- 10 workers for stable barrier synchronization
- Very long training (5.12M steps per worker) for proper convergence
- Higher entropy coefficient (0.1) for better exploration with large action space
"""

# Environment Configuration
ENV_KWARGS = {
    "cache_file": "training_cache.json",  # JSON file containing cached historical price data
    "initial_cash": 1_000_000,  # Starting capital (gp) for each trading episode
    "episode_length": 864,  # Number of time steps per episode (864 = 3 days at 5min intervals)
    "top_n_items": 200,  # Focus on top 200 traded items for better learning signal
}

# PPO Agent Configuration
PPO_KWARGS = {
    "hidden_size": 1024,  # Number of neurons in each hidden layer of the neural network
    "num_layers": 4,  # Number of hidden layers in the policy and value networks
    "lr": 5e-4,  # Higher learning rate for faster convergence
    "gamma": 0.99,  # Discount factor for future rewards (0.99 = highly values future)
    "gae_lambda": 0.95,  # Lambda parameter for Generalized Advantage Estimation
    "clip_epsilon": 0.2,  # PPO clipping range to prevent large policy updates
    "entropy_coef": 0.1,  # Higher entropy for exploration with large action space
    "value_coef": 0.5,  # Coefficient for value function loss in total loss
    "price_bins": 20,  # Number of discrete price levels for action space
    "quantity_bins": 10,  # Number of discrete quantity levels for action space
    "wait_steps_bins": 10,  # Number of discrete wait time options for action space
    "risk_tolerance": 0.3,  # Maximum portfolio allocation per single item (30%)
    "rollout_steps": 2048,  # Larger rollouts for more stable gradients
    "minibatch_size": 256,  # Larger batches for H100
    "ppo_epochs": 4,  # Fewer epochs with larger batches
}

# Training Configuration
TRAIN_KWARGS = {
    # Reduced to 10 workers for more stable training (less barrier contention)
    "num_workers": 10,  # 10 parallel workers - more stable than 16
    "max_steps_per_worker": 5_120_000,  # 5.12M steps per worker = 51.2M total steps
    "save_every_steps": 204_800,  # Save every ~3.2M total steps (every 100 updates)
    "log_every_steps": 2_048,  # Log every rollout
    "eval_every_steps": 409_600,  # Evaluate every ~6.5M total steps
    "use_shared_cache": True,  # Share price data cache across workers via shared memory
    "gpu_distribution": "single",  # All on single H100 for shared memory efficiency
    "use_shared_model": True,  # Enable shared model training with gradient aggregation
    "max_checkpoints": 20,  # Keep more checkpoints for analysis
}

# Evaluation Configuration
EVAL_KWARGS = {
    "num_episodes": 10,  # Number of episodes to run during each evaluation
    "deterministic": True,  # Use deterministic policy (no exploration) for evaluation
}
