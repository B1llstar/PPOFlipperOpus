#!/usr/bin/env python3
"""
Multiprocess PPO Training System

Ground-up rewrite designed for true parallel training on H100 GPU.
Each worker runs independently with its own environment and agent.
"""

import numpy as np
import torch
import torch.multiprocessing as mp
import os
import sys
import logging
import time
import signal
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from collections import deque
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_multiprocess.ppo_config_multiprocess import ENV_KWARGS, PPO_KWARGS, TRAIN_KWARGS, EVAL_KWARGS
from ppo_agent import PPOAgent
from training.ge_environment import GrandExchangeEnv


def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Setup logger for a process."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers
    
    formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.propagate = False
    return logger


def worker_process(
    worker_id: int,
    device_id: int,
    save_dir: Path,
    progress_queue: mp.Queue,
    stop_event: mp.Event,
    config: Dict[str, Any]
):
    """
    Worker process that trains a single agent independently.
    
    Args:
        worker_id: Unique worker identifier
        device_id: CUDA device to use (-1 for CPU)
        save_dir: Directory to save checkpoints
        progress_queue: Queue for sending progress updates
        stop_event: Event to signal worker to stop
        config: Combined configuration dictionary
    """
    # Setup logging for this worker
    log_dir = save_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"worker_{worker_id}.log"
    logger = setup_logger(f"Worker-{worker_id}", str(log_file))
    
    try:
        logger.info("="*60)
        logger.info(f"WORKER {worker_id} STARTING")
        logger.info("="*60)
        
        # Set device
        if device_id >= 0 and torch.cuda.is_available():
            device = torch.device(f"cuda:{device_id}")
            logger.info(f"Using GPU {device_id}: {torch.cuda.get_device_name(device_id)}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        
        # Create environment (cache will be pre-loaded and shared)
        logger.info("Creating environment...")
        env_kwargs = config['env_kwargs']
        env = GrandExchangeEnv(
            cache_file=env_kwargs["cache_file"],
            initial_cash=env_kwargs["initial_cash"],
            episode_length=env_kwargs["episode_length"],
            top_n_items=env_kwargs["top_n_items"],
            seed=worker_id * 1000 + np.random.randint(0, 1000)
        )
        logger.info(f"Environment created: {len(env.tradeable_items)} tradeable items")
        
        # Create agent
        logger.info("Creating PPO agent...")
        ppo_kwargs = config['ppo_kwargs']
        agent = PPOAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            hidden_size=ppo_kwargs["hidden_size"],
            num_layers=ppo_kwargs["num_layers"],
            lr=ppo_kwargs["lr"],
            gamma=ppo_kwargs["gamma"],
            gae_lambda=ppo_kwargs["gae_lambda"],
            clip_epsilon=ppo_kwargs["clip_epsilon"],
            entropy_coef=ppo_kwargs["entropy_coef"],
            value_coef=ppo_kwargs["value_coef"],
            price_bins=ppo_kwargs["price_bins"],
            quantity_bins=ppo_kwargs["quantity_bins"],
            wait_steps_bins=ppo_kwargs["wait_steps_bins"],
            risk_tolerance=ppo_kwargs["risk_tolerance"],
            buffer_size=ppo_kwargs["rollout_steps"]
        )
        agent.agent_id = worker_id
        logger.info(f"Agent created with {agent.obs_dim} obs dim, {agent.n_items} items")
        
        # Training configuration
        train_kwargs = config['train_kwargs']
        max_steps = train_kwargs["max_steps_per_worker"]
        save_every = train_kwargs["save_every_steps"]
        log_every = train_kwargs["log_every_steps"]
        rollout_steps = ppo_kwargs["rollout_steps"]
        ppo_epochs = ppo_kwargs["ppo_epochs"]
        minibatch_size = ppo_kwargs["minibatch_size"]
        
        # Training state
        step = 0
        episode = 0
        episode_reward = 0.0
        episode_length = 0
        recent_rewards = deque(maxlen=100)
        recent_episode_lengths = deque(maxlen=100)
        
        obs, info = env.reset()
        logger.info("Starting training loop...")
        
        start_time = time.time()
        last_log_time = start_time
        
        # Progress bar for this worker
        pbar = tqdm(
            total=max_steps,
            desc=f"Worker {worker_id}",
            position=worker_id,
            leave=True,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        while step < max_steps and not stop_event.is_set():
            # Collect rollout
            for _ in range(rollout_steps):
                # Sample action from policy
                action_dict, log_prob = agent.sample_action(obs, deterministic=False)
                
                # Get value estimate
                value = agent.get_value(obs)
                
                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(action_dict)
                done = terminated or truncated
                
                # Store transition
                agent.store_transition(obs, action_dict, reward, done, value, log_prob)
                
                # Update tracking
                episode_reward += reward
                episode_length += 1
                step += 1
                pbar.update(1)
                pbar.set_postfix({
                    'ep': episode,
                    'ep_rew': f'{episode_reward:.1f}',
                    'avg_rew': f'{np.mean(recent_rewards) if recent_rewards else 0:.1f}'
                })
                
                # Handle episode end
                if done:
                    recent_rewards.append(episode_reward)
                    recent_episode_lengths.append(episode_length)
                    
                    if step % log_every < rollout_steps:
                        logger.info(
                            f"Episode {episode} complete: "
                            f"reward={episode_reward:.2f}, length={episode_length}, "
                            f"step={step}"
                        )
                    
                    episode += 1
                    episode_reward = 0.0
                    episode_length = 0
                    obs, info = env.reset()
                else:
                    obs = next_obs
                
                # Check if should stop
                if step >= max_steps or stop_event.is_set():
                    break
            
            # Update policy
            logger.info(f"Step {step}: Running PPO update...")
            update_start = time.time()
            loss_info = agent.update(
                last_obs=obs,
                last_done=done,
                n_epochs=ppo_epochs,
                batch_size=minibatch_size
            )
            update_time = time.time() - update_start
            logger.info(f"PPO update complete in {update_time:.2f}s")
            
            # Log progress
            if step % log_every < rollout_steps or step >= max_steps:
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed if elapsed > 0 else 0
                
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
                avg_length = np.mean(recent_episode_lengths) if recent_episode_lengths else 0
                
                log_msg = (
                    f"Step {step}/{max_steps} ({step/max_steps*100:.1f}%)\n"
                    f"  Episodes: {episode}\n"
                    f"  Avg Reward (last 100): {avg_reward:.2f}\n"
                    f"  Avg Length (last 100): {avg_length:.1f}\n"
                    f"  Speed: {steps_per_sec:.1f} steps/sec\n"
                    f"  Loss: {loss_info['loss']:.4f}\n"
                    f"  PG Loss: {loss_info['pg_loss']:.4f}\n"
                    f"  Value Loss: {loss_info['value_loss']:.4f}\n"
                    f"  Entropy: {loss_info['entropy_loss']:.4f}\n"
                    f"  Clip Fraction: {loss_info['clip_fraction']:.3f}"
                )
                logger.info(f"\n{log_msg}")
                
                # Send progress update to main process
                progress_queue.put({
                    'worker_id': worker_id,
                    'step': step,
                    'episode': episode,
                    'avg_reward': avg_reward,
                    'avg_length': avg_length,
                    'loss_info': loss_info,
                    'steps_per_sec': steps_per_sec
                })
            
          Close progress bar
        pbar.close()
        
        #   # Save checkpoint
            if step % save_every < rollout_steps or step >= max_steps:
                checkpoint_path = save_dir / f"worker_{worker_id}_step_{step}.pt"
                agent.save_checkpoint(str(checkpoint_path))
                logger.info(f"[OK] Checkpoint saved: {checkpoint_path.name}")
        
        # Save final checkpoint
        final_path = save_dir / f"worker_{worker_id}_final.pt"
        agent.save_checkpoint(str(final_path))
        
        # Final summary
        total_time = time.time() - start_time
        logger.info("="*60)
        logger.info(f"WORKER {worker_id} COMPLETE")
        logger.info(f"  Total Steps: {step}")
        logger.info(f"  Total Episodes: {episode}")
        logger.info(f"  Total Time: {total_time/60:.1f} minutes")
        logger.info(f"  Avg Speed: {step/total_time:.1f} steps/sec")
        logger.info(f"  Final checkpoint: {final_path.name}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Worker {worker_id} failed with error: {e}", exc_info=True)
        progress_queue.put({
            'worker_id': worker_id,
            'error': str(e)
        })
        raise


def main():
    """Main training coordinator."""
    # Setup main logger
    save_dir = Path("agent_states_multiprocess")
    save_dir.mkdir(exist_ok=True)
    
    log_dir = save_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logger = setup_logger("Main", str(log_dir / "main.log"))
    
    logger.info("="*80)
    logger.info("PPO MULTIPROCESS TRAINING SYSTEM")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True)
        logger.info("[OK] Multiprocessing start method: spawn")
    except RuntimeError:
        logger.info("[OK] Multiprocessing start method already set to spawn")
    
    logger.info(f"[OK] Save directory: {save_dir}")
    
    # Pre-load cache into shared memory
    cache_file = ENV_KWARGS["cache_file"]
    logger.info(f"Pre-loading cache: {cache_file}")
    cache_start = time.time()
    
    if TRAIN_KWARGS.get("use_shared_cache", True):
        from training.cached_market_loader import load_cache
        load_cache(cache_file, use_shared_memory=True)
        cache_time = time.time() - cache_start
        logger.info(f"[OK] Cache loaded in shared memory ({cache_time:.1f}s)")
    else:
        logger.info("[OK] Shared cache disabled, each worker will load independently")
    
    # Hardware info
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    logger.info(f"[OK] Available GPUs: {num_gpus}")
    if num_gpus > 0:
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"    GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
    
    # Configuration
    num_workers = TRAIN_KWARGS["num_workers"]
    logger.info(f"[OK] Number of workers: {num_workers}")
    logger.info(f"[OK] Steps per worker: {TRAIN_KWARGS['max_steps_per_worker']:,}")
    logger.info(f"[OK] Rollout steps: {PPO_KWARGS['rollout_steps']:,}")
    logger.info(f"[OK] Top N items: {ENV_KWARGS['top_n_items']}")
    
    # Combined config for workers
    config = {
        'env_kwargs': ENV_KWARGS,
        'ppo_kwargs': PPO_KWARGS,
        'train_kwargs': TRAIN_KWARGS,
        'eval_kwargs': EVAL_KWARGS
    }
    
    # Create communication channels
    progress_queue = mp.Queue()
    stop_event = mp.Event()
    
    # Start worker processes
    logger.info("="*80)
    logger.info("STARTING WORKERS")
    logger.info("="*80)
    
    workers = []
    for worker_id in range(num_workers):
        # Assign GPU (with single GPU, all workers share it efficiently)
        if num_gpus > 0:
            if TRAIN_KWARGS.get("gpu_distribution") == "single":
                device_id = 0  # All on GPU 0
            else:  # round-robin (single GPU: all get device 0)
                device_id = worker_id % num_gpus
        else:
            device_id = -1  # CPU
        
        p = mp.Process(
            target=worker_process,
            args=(worker_id, device_id, save_dir, progress_queue, stop_event, config),
            daemon=False
        )
        p.start()
        workers.append(p)
        logger.info(f"[OK] Started worker {worker_id} (PID: {p.pid}, Device: {'GPU '+str(device_id) if device_id >= 0 else 'CPU'})")
        time.sleep(0.5)  # Stagger starts slightly
    
    logger.info("="*80)
    logger.info("ALL WORKERS STARTED - MONITORING PROGRESS")
    logger.info("="*80)
    
    # Monitor progress
    worker_status = {i: {'step': 0, 'last_update': time.time()} for i in range(num_workers)}
    last_summary = time.time()
    summary_interval = 60  # Print summary every 60 seconds
    
    try:
        while any(p.is_alive() for p in workers):
            # Check for progress updates
            while not progress_queue.empty():
                update = progress_queue.get_nowait()
                worker_id = update.get('worker_id')
                
                if 'error' in update:
                    logger.error(f"Worker {worker_id} reported error: {update['error']}")
                    continue
                
                worker_status[worker_id].update({
                    'step': update.get('step', 0),
                    'episode': update.get('episode', 0),
                    'avg_reward': update.get('avg_reward', 0),
                    'last_update': time.time()
                })
                
                # Log individual worker update
                logger.info(
                    f"Worker {worker_id}: "
                    f"step={update['step']:,}, "
                    f"episode={update['episode']}, "
                    f"reward={update.get('avg_reward', 0):.2f}, "
                    f"speed={update.get('steps_per_sec', 0):.1f} steps/s"
                )
            
            # Print summary periodically
            if time.time() - last_summary > summary_interval:
                logger.info("-"*80)
                logger.info("PROGRESS SUMMARY")
                total_steps = sum(s['step'] for s in worker_status.values())
                active_workers = sum(1 for p in workers if p.is_alive())
                logger.info(f"Active Workers: {active_workers}/{num_workers}")
                logger.info(f"Total Steps: {total_steps:,}")
                for worker_id, status in worker_status.items():
                    logger.info(
                        f"  Worker {worker_id}: "
                        f"{status['step']:,} steps, "
                        f"episode {status.get('episode', 0)}, "
                        f"reward {status.get('avg_reward', 0):.2f}"
                    )
                logger.info("-"*80)
                last_summary = time.time()
            
            # Sleep briefly
            time.sleep(2)
        
        logger.info("="*80)
        logger.info("ALL WORKERS COMPLETED")
        logger.info("="*80)
        
    except KeyboardInterrupt:
        logger.info("="*80)
        logger.info("INTERRUPTED BY USER - STOPPING WORKERS")
        logger.info("="*80)
        stop_event.set()
        
        # Wait for workers to finish gracefully
        logger.info("Waiting for workers to stop gracefully...")
        for i, p in enumerate(workers):
            p.join(timeout=30)
            if p.is_alive():
                logger.warning(f"Worker {i} (PID: {p.pid}) didn't stop gracefully, terminating...")
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    logger.error(f"Worker {i} (PID: {p.pid}) still alive, killing...")
                    p.kill()
    
    # Final summary
    logger.info("="*80)
    logger.info("TRAINING SESSION COMPLETE")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Checkpoints saved in: {save_dir}")
    logger.info(f"Logs saved in: {log_dir}")
    
    # Summary statistics
    total_steps = sum(s['step'] for s in worker_status.values())
    logger.info(f"Total steps across all workers: {total_steps:,}")
    for worker_id, status in worker_status.items():
        logger.info(
            f"  Worker {worker_id}: {status['step']:,} steps, "
            f"{status.get('episode', 0)} episodes"
        )
    
    logger.info("="*80)


if __name__ == "__main__":
    main()
