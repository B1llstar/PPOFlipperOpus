"""
Multiprocess PPO Configuration

Production settings for H100 SXM 80GB parallel training.
20 workers, 150 items, optimized for convergence.
"""

# Environment Configuration
ENV_KWARGS = {
    "cache_file": "training_cache.json",  # JSON file containing cached historical price data
    "initial_cash": 1_000_000,  # Starting capital (gp) for each trading episode
    "episode_length": 864,  # Number of time steps per episode (864 = 3 days at 5min intervals)
    "top_n_items": 999_999_999,  # Maximum number of items to include (effectively unlimited)
}

# PPO Agent Configuration
PPO_KWARGS = {
    "hidden_size": 1024,  # Number of neurons in each hidden layer of the neural network
    "num_layers": 4,  # Number of hidden layers in the policy and value networks
    "lr": 3e-4,  # Learning rate for Adam optimizer (0.0003)
    "gamma": 0.99,  # Discount factor for future rewards (0.99 = highly values future)
    "gae_lambda": 0.95,  # Lambda parameter for Generalized Advantage Estimation
    "clip_epsilon": 0.2,  # PPO clipping range to prevent large policy updates
    "entropy_coef": 0.08,  # Coefficient for entropy bonus (encourages exploration)
    "value_coef": 0.5,  # Coefficient for value function loss in total loss
    "price_bins": 20,  # Number of discrete price levels for action space
    "quantity_bins": 10,  # Number of discrete quantity levels for action space
    "wait_steps_bins": 10,  # Number of discrete wait time options for action space
    "risk_tolerance": 0.3,  # Maximum portfolio allocation per single item (30%)
    "rollout_steps": 2048,  # Number of environment steps collected before policy update
    "minibatch_size": 64,  # Batch size for minibatch SGD during policy optimization
    "ppo_epochs": 10,  # Number of epochs to train on each batch of rollout data
}

# Training Configuration
TRAIN_KWARGS = {
    # H100: 20 workers × 3.5GB = ~70GB (87.5% of 80GB, safe buffer)
    "num_workers": 20,  # Number of parallel environment workers for data collection
    "max_steps_per_worker": 5120,  # Maximum training steps per worker (10M total)
    "save_every_steps": 100,  # Save model checkpoint every N steps
    "log_every_steps": 1_000,  # Log training metrics (loss, rewards) every N steps
    "eval_every_steps": 25_000,  # Run evaluation episodes every N steps
    "use_shared_cache": True,  # Share price data cache across workers via shared memory
    "gpu_distribution": "round-robin",  # Strategy for distributing workers across GPUs
    "use_shared_model": True,  # Enable shared model training with gradient aggregation (recommended)
    "max_checkpoints": 5,  # Maximum number of checkpoints to keep (oldest deleted automatically, 0 = unlimited)
}

# Evaluation Configuration
EVAL_KWARGS = {
    "num_episodes": 10,  # Number of episodes to run during each evaluation
    "deterministic": True,  # Use deterministic policy (no exploration) for evaluation
}
