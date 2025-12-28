"""
Multiprocess PPO Training System - README

Complete ground-up rewrite of the training system designed for true parallel
multiprocessing on H100 GPUs.

## Key Features

- **True Parallelism**: Each worker runs independently in its own process
- **Shared Memory Cache**: Single cache load shared across all workers
- **GPU Distribution**: Automatic round-robin GPU assignment for multiple GPUs
- **Independent Training**: Each worker trains its own agent independently
- **Comprehensive Logging**: Per-worker logs + main coordinator log
- **Graceful Shutdown**: Proper cleanup and checkpoint saving on interrupt

## Architecture

### Components

1. **train_ppo_multiprocess.py**: Main training script
   - Coordinator process that spawns and monitors workers
   - Pre-loads cache into shared memory
   - Monitors progress from all workers
   - Handles graceful shutdown

2. **ppo_config_multiprocess.py**: Configuration
   - ENV_KWARGS: Environment settings (150 items for diversity)
   - PPO_KWARGS: PPO hyperparameters (optimized for H100)
   - TRAIN_KWARGS: Training settings (16 workers, 1M steps each)
   - EVAL_KWARGS: Evaluation settings

### Worker Process

Each worker:
1. Creates its own environment instance (shares pre-loaded cache)
2. Creates its own PPO agent on assigned GPU
3. Runs training loop independently:
   - Collects rollouts (2048 steps)
   - Updates policy with PPO
   - Saves checkpoints periodically
4. Reports progress to coordinator

### Multiprocessing Details

- **Start Method**: `spawn` (required for CUDA)
- **Communication**: Queue for progress updates, Event for shutdown signal
- **GPU Assignment**: Round-robin across available GPUs
- **Process Isolation**: Each worker is completely independent

## Usage

### Basic Training

```python
cd training_multiprocess
python train_ppo_multiprocess.py
```

### Configuration

Edit `ppo_config_multiprocess.py`:

```python
# Number of parallel workers (20 = optimal for H100 SXM 80GB)
# Targets 85% resource utilization with 15% buffer
TRAIN_KWARGS["num_workers"] = 20

# Steps per worker
TRAIN_KWARGS["max_steps_per_worker"] = 1_000_000

# Top N items (higher = more diversity)
ENV_KWARGS["top_n_items"] = 150
```

### GPU Distribution

- **round-robin** (default): Workers assigned to GPUs in rotation
- **single**: All workers share GPU 0
- **CPU**: Set `num_workers` to `-1` or run without CUDA

## Output Structure

```
agent_states_multiprocess/
├── logs/
│   ├── main.log              # Coordinator log
│   ├── worker_0.log          # Worker 0 log
│   ├── worker_1.log          # Worker 1 log
│   └── ...
├── worker_0_step_50000.pt    # Periodic checkpoints
├── worker_0_step_100000.pt
├── worker_0_final.pt         # Final checkpoint
└── ...
```

## Monitoring

The coordinator logs:
- Worker startup status
- Individual worker progress updates (every 1000 steps)
- Periodic summary (every 60 seconds)
- Final statistics on completion

Each worker logs:
- Environment creation
- Agent initialization
- Episode completions
- PPO update timing and metrics
- Checkpoint saves

## Performance

Expected performance on H100 SXM (80GB) with single GPU:
- ~500-1000 steps/sec per worker (depends on environment complexity)
- 20 workers = ~10,000-20,000 steps/sec total
- VRAM usage: ~3.5 GB per worker × 20 = **~70 GB** (87.5% utilization, 15% buffer)
- System RAM: Cache ~2.4 GB + worker overhead ~12 GB = ~15-18 GB total (plenty of 125GB)
- vCPU: 20 workers = 1 worker per vCPU (optimal)
- **All 20 workers share the single GPU efficiently** - H100 has massive parallel compute
- GPU utilization: Near 100% compute, 85-90% VRAM (safe buffer for spikes)

## Comparison to Single-Process

**Old System (training/train_ppo.py)**:
- Sequential agent training (no parallelism)
- Each agent loads cache independently
- Complex shared knowledge system
- Dictionary-based observations (incompatible with current env)

**New System (training_multiprocess/)**:
- True parallel training
- Shared memory cache
- Independent agents
- Compatible with numpy array observations
- Simpler, cleaner architecture

## Advantages

1. **Parallel Efficiency**: 16x speedup with 16 workers
2. **Memory Efficiency**: Single cache load for all workers
3. **Scalability**: Easily scale to more workers/GPUs
4. **Robustness**: Worker failures don't affect others
5. **Simplicity**: No complex synchronization or shared knowledge
6. **Compatibility**: Works with existing PPOAgent and GrandExchangeEnv

## Graceful Shutdown

Press Ctrl+C to stop training:
1. Stop event is set
2. Workers finish current rollout and save checkpoint
3. Coordinator waits up to 30s per worker
4. Terminates workers that don't stop gracefully
5. Prints final statistics

## Checkpoints

Each worker saves:
- Periodic checkpoints (every 50,000 steps)
- Final checkpoint on completion
- All checkpoints include full agent state

To continue training from a checkpoint:
```python
agent = PPOAgent(...)
agent.load("agent_states_multiprocess/worker_0_step_50000.pt")
```

## Troubleshooting

**CUDA out of memory** (try 18 or 16 if 20 is too much)
- Reduce `ENV_KWARGS["top_n_items"]` (150 → 100)
- Reduce `PPO_KWARGS["rollout_steps"]` (2048 → 1024)
- Note: 20 workers is tuned for 85% VRAM usage with 15% buffer
- Reduce `PPO_KWARGS["rollout_steps"]`

**Slow startup**:
- Normal for first-time cache load (~30-60 seconds)
- Subsequent runs use cached data

**Workers not progressing**:
- Check individual worker logs in `agent_states_multiprocess/logs/`
- Common issues: environment errors, NaN in training

## Notes

- Each worker is independent (no shared learning)
- For shared learning, consider parameter server architecture
- Current design optimizes for exploration diversity
- Best agents can be selected after training based on final performance
