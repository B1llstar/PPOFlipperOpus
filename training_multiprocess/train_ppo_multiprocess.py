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
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from collections import deque
from tqdm import tqdm

# Firebase imports for async checkpoint upload
try:
    import firebase_admin
    from firebase_admin import credentials, storage
    FIREBASE_AVAILABLE = True
    FIREBASE_IMPORT_ERROR = None
except ImportError as e:
    FIREBASE_AVAILABLE = False
    FIREBASE_IMPORT_ERROR = str(e)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_multiprocess.ppo_config_multiprocess import ENV_KWARGS, PPO_KWARGS, TRAIN_KWARGS, EVAL_KWARGS
from ppo_agent import PPOAgent
from training.ge_environment import GrandExchangeEnv

# Global Firebase Storage bucket reference (initialized once)
_firebase_bucket = None
_firebase_init_lock = threading.Lock()


def init_firebase_storage(logger: logging.Logger) -> bool:
    """
    Initialize Firebase Storage for checkpoint uploads.

    Returns:
        True if initialization successful, False otherwise
    """
    global _firebase_bucket

    logger.info("[Firebase] Starting Firebase Storage initialization...")
    logger.info(f"[Firebase] FIREBASE_AVAILABLE = {FIREBASE_AVAILABLE}")

    if not FIREBASE_AVAILABLE:
        logger.warning(f"[Firebase] SDK not installed. Import error: {FIREBASE_IMPORT_ERROR}")
        logger.warning("[Firebase] To install: pip install firebase-admin")
        logger.warning("[Firebase] Checkpoints will only be saved locally.")
        return False

    logger.info(f"[Firebase] firebase_admin version: {firebase_admin.__version__}")

    with _firebase_init_lock:
        if _firebase_bucket is not None:
            logger.info("[Firebase] Already initialized, reusing existing bucket")
            return True

        try:
            # Find service account file
            base_path = Path(__file__).parent.parent
            service_account_paths = [
                base_path / "ppoflipperopus-firebase-adminsdk-fbsvc-907a50dae7.json",
                base_path / "config" / "ppoflipperopus-firebase-adminsdk-fbsvc-907a50dae7.json",
                Path(os.environ.get("FIREBASE_SERVICE_ACCOUNT", "")),
            ]

            logger.info(f"[Firebase] Searching for service account file...")
            logger.info(f"[Firebase] Base path: {base_path}")

            service_account_path = None
            for path in service_account_paths:
                exists = path.exists() if str(path) else False
                logger.info(f"[Firebase]   Checking: {path} -> {'EXISTS' if exists else 'NOT FOUND'}")
                if exists:
                    service_account_path = path
                    break

            if service_account_path is None:
                logger.warning("[Firebase] Service account file not found in any location!")
                logger.warning("[Firebase] Expected locations:")
                for path in service_account_paths[:2]:  # Skip empty env var path
                    logger.warning(f"[Firebase]   - {path}")
                logger.warning("[Firebase] Checkpoints will only be saved locally.")
                return False

            logger.info(f"[Firebase] Using service account: {service_account_path}")

            # Initialize Firebase if not already done
            if not firebase_admin._apps:
                logger.info("[Firebase] No existing Firebase app, initializing new one...")
                cred = credentials.Certificate(str(service_account_path))
                firebase_admin.initialize_app(cred, {
                    'storageBucket': 'ppoflipperopus.firebasestorage.app'
                })
                logger.info("[Firebase] Firebase app initialized successfully")
            else:
                logger.info(f"[Firebase] Using existing Firebase app: {list(firebase_admin._apps.keys())}")

            _firebase_bucket = storage.bucket()
            logger.info(f"[Firebase] ✓ Storage initialized successfully!")
            logger.info(f"[Firebase]   Bucket name: {_firebase_bucket.name}")
            logger.info(f"[Firebase]   Bucket path: gs://{_firebase_bucket.name}/checkpoints/")
            return True

        except Exception as e:
            logger.error(f"[Firebase] Failed to initialize: {e}")
            import traceback
            logger.error(f"[Firebase] Traceback:\n{traceback.format_exc()}")
            return False


def async_upload_checkpoint(checkpoint_path: Path, logger: logging.Logger):
    """
    Asynchronously upload a checkpoint to Firebase Storage.

    Args:
        checkpoint_path: Local path to the checkpoint file
        logger: Logger instance
    """
    def upload_task():
        global _firebase_bucket

        if _firebase_bucket is None:
            logger.warning(f"[Firebase] Skipping upload - bucket not initialized")
            return

        try:
            # Get file size for logging
            file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
            logger.info(f"[Firebase] Starting upload: {checkpoint_path.name} ({file_size_mb:.1f} MB)")

            # Upload to /checkpoints/ directory in Firebase Storage
            blob_name = f"checkpoints/{checkpoint_path.name}"
            blob = _firebase_bucket.blob(blob_name)

            # Upload with metadata
            blob.metadata = {
                'uploaded_at': datetime.now().isoformat(),
                'source': 'multiprocess_training'
            }

            start_time = time.time()
            blob.upload_from_filename(str(checkpoint_path))
            upload_time = time.time() - start_time

            logger.info(f"[Firebase] ✓ Uploaded checkpoint to gs://{_firebase_bucket.name}/{blob_name}")
            logger.info(f"[Firebase]   Size: {file_size_mb:.1f} MB, Time: {upload_time:.1f}s, Speed: {file_size_mb/upload_time:.1f} MB/s")

        except Exception as e:
            logger.error(f"[Firebase] Failed to upload checkpoint {checkpoint_path.name}: {e}")
            import traceback
            logger.error(f"[Firebase] Upload traceback:\n{traceback.format_exc()}")

    # Run upload in background thread
    logger.info(f"[Firebase] Queuing async upload for {checkpoint_path.name}")
    thread = threading.Thread(target=upload_task, daemon=True)
    thread.start()


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


def manage_checkpoints(save_dir: Path, worker_id: int, max_checkpoints: int, logger: logging.Logger):
    """
    Manage checkpoint files by keeping only the most recent N checkpoints.
    
    Args:
        save_dir: Directory containing checkpoints
        worker_id: Worker ID (for worker-specific checkpoints) or None for shared model
        max_checkpoints: Maximum number of checkpoints to keep (0 = unlimited)
        logger: Logger instance
    """
    if max_checkpoints <= 0:
        return  # Unlimited checkpoints
    
    # Find all checkpoint files for this worker/model
    if worker_id is not None:
        pattern = f"worker_{worker_id}_step_*.pt"
    else:
        # Match both old and new checkpoint formats
        pattern = "shared_model_step_*.pt"
    
    checkpoints = list(save_dir.glob(pattern))
    
    if len(checkpoints) <= max_checkpoints:
        return  # Within limit
    
    # Sort by modification time (oldest first)
    checkpoints.sort(key=lambda p: p.stat().st_mtime)
    
    # Delete oldest checkpoints
    num_to_delete = len(checkpoints) - max_checkpoints
    for checkpoint in checkpoints[:num_to_delete]:
        try:
            checkpoint.unlink()
            logger.info(f"Deleted old checkpoint: {checkpoint.name}")
        except Exception as e:
            logger.warning(f"Failed to delete checkpoint {checkpoint.name}: {e}")


def find_latest_checkpoint(save_dir: Path, logger: logging.Logger) -> Optional[Path]:
    """
    Find the most recent shared model checkpoint.
    
    Args:
        save_dir: Directory containing checkpoints
        logger: Logger instance
        
    Returns:
        Path to latest checkpoint or None if none found
    """
    # Check for final checkpoint first
    final_checkpoint = save_dir / "shared_model_final.pt"
    if final_checkpoint.exists():
        logger.info(f"Found final checkpoint: {final_checkpoint.name}")
        return final_checkpoint
    
    # Look for step-based checkpoints (new format)
    step_checkpoints = list(save_dir.glob("shared_model_step_*.pt"))
    
    if step_checkpoints:
        # Parse step numbers from filenames and find the highest
        def get_step_number(path: Path) -> int:
            try:
                # Extract number from "shared_model_step_12345.pt"
                return int(path.stem.split('_')[-1])
            except:
                return -1
        
        # Sort by step number (highest first)
        step_checkpoints.sort(key=get_step_number, reverse=True)
        latest = step_checkpoints[0]
        step_num = get_step_number(latest)
        logger.info(f"Found latest checkpoint: {latest.name} (step {step_num})")
        return latest
    
    # Look for update-based checkpoints (old format, for backward compatibility)
    update_checkpoints = list(save_dir.glob("shared_model_update_*.pt"))
    
    if update_checkpoints:
        # Sort by modification time (newest first)
        update_checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest = update_checkpoints[0]
        logger.info(f"Found latest checkpoint: {latest.name}")
        return latest
    
    logger.info("No existing checkpoints found - starting fresh")
    return None


def gradient_coordinator(
    num_workers: int,
    shared_model: PPOAgent,
    gradient_queue: mp.Queue,
    param_update_event: mp.Event,
    barrier: mp.Barrier,
    stop_event: mp.Event,
    save_dir: Path,
    logger: logging.Logger,
    max_checkpoints: int = 0,
    initial_update_count: int = 0,
    initial_timesteps: int = 0,
    save_every_updates: int = 10,
    rollout_steps: int = 2048
):
    """
    Coordinator process that aggregates gradients from workers and updates shared model.

    Args:
        num_workers: Number of worker processes
        shared_model: Shared PPOAgent model to update
        gradient_queue: Queue receiving gradients from workers
        param_update_event: Event to signal parameter updates
        barrier: Barrier for synchronizing workers
        stop_event: Event to signal shutdown
        save_dir: Directory for saving coordinated checkpoints
        logger: Logger instance
        max_checkpoints: Maximum number of checkpoints to keep (0 = unlimited)
        initial_update_count: Starting update count (for resumed training)
        initial_timesteps: Starting timestep count (for resumed training)
        save_every_updates: Save checkpoint every N updates (derived from config)
        rollout_steps: Steps per rollout (for logging)
    """
    logger.info("[Coordinator] Starting gradient coordinator")

    # Initialize Firebase Storage for async checkpoint uploads
    firebase_enabled = init_firebase_storage(logger)
    if firebase_enabled:
        logger.info("[Coordinator] Checkpoints will be uploaded to Firebase Storage")
    else:
        logger.info("[Coordinator] Checkpoints will only be saved locally")

    if initial_timesteps > 0:
        logger.info(f"[Coordinator] *** RESUMING TRAINING ***")
        logger.info(f"[Coordinator] Starting from step {initial_timesteps} (update {initial_update_count})")
    else:
        logger.info(f"[Coordinator] Starting fresh training from step 0")

    update_count = initial_update_count
    total_timesteps = initial_timesteps

    # Initialize shared model tracking (important for first save)
    shared_model.n_updates = update_count
    shared_model.total_timesteps = total_timesteps
    
    try:
        while not stop_event.is_set():
            # Wait at barrier for all workers to submit gradients
            logger.info("[Coordinator] Waiting at barrier for workers to submit gradients...")
            try:
                barrier.wait(timeout=600)  # 10 minute timeout to match workers
                logger.info("[Coordinator] All workers reached barrier")
            except Exception as e:
                logger.error(f"[Coordinator] Barrier wait failed (timeout or broken): {e}")
                logger.error("[Coordinator] One or more workers may have crashed or hung")
                logger.error("[Coordinator] Check individual worker logs in logs/ directory")
                stop_event.set()  # Signal all workers to stop
                break
            
            # Collect gradients from all workers
            gradients_batch = []
            loss_infos = []
            total_steps_this_update = 0
            
            for _ in range(num_workers):
                try:
                    grad_data = gradient_queue.get(timeout=5)
                    gradients_batch.append(grad_data['gradients'])
                    loss_infos.append(grad_data['loss_info'])
                    total_steps_this_update += grad_data.get('steps', 0)
                except Exception as e:
                    logger.error(f"[Coordinator] Error receiving gradients: {e}")
                    break
            
            if len(gradients_batch) != num_workers:
                logger.warning(f"[Coordinator] Only received {len(gradients_batch)}/{num_workers} gradient sets")
                continue
            
            # Average gradients
            logger.info(f"[Coordinator] Aggregating gradients from {num_workers} workers")
            aggregated_grads = {}
            for key in gradients_batch[0].keys():
                grads = [g[key] for g in gradients_batch if key in g]
                if grads:
                    aggregated_grads[key] = torch.stack(grads).mean(dim=0)
            
            # Apply aggregated gradients to shared model
            shared_model.optimizer.zero_grad()
            for name, param in shared_model.actor.named_parameters():
                key = f'actor.{name}'
                if key in aggregated_grads:
                    param.grad = aggregated_grads[key].to(param.device)
            
            for name, param in shared_model.critic.named_parameters():
                key = f'critic.{name}'
                if key in aggregated_grads:
                    param.grad = aggregated_grads[key].to(param.device)
            
            for name, param in shared_model.feature_extractor.named_parameters():
                key = f'feature_extractor.{name}'
                if key in aggregated_grads:
                    param.grad = aggregated_grads[key].to(param.device)
            
            # Update parameters
            shared_model.optimizer.step()
            update_count += 1
            total_timesteps += total_steps_this_update

            # Step learning rate scheduler
            if shared_model.scheduler is not None:
                shared_model.scheduler.step()
                current_lr = shared_model.scheduler.get_last_lr()[0]
                if update_count % 10 == 0:  # Log LR every 10 updates
                    logger.info(f"[Coordinator] Learning rate: {current_lr:.6f}")

            # Update shared model tracking variables BEFORE saving
            shared_model.n_updates = update_count
            shared_model.total_timesteps = total_timesteps
            
            # Compute gradient norm for monitoring
            total_grad_norm = 0.0
            for param in shared_model.all_parameters:
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item() ** 2
            total_grad_norm = total_grad_norm ** 0.5

            # Log aggregated metrics
            avg_loss = np.mean([info['loss'] for info in loss_infos])
            avg_pg_loss = np.mean([info['pg_loss'] for info in loss_infos])
            avg_value_loss = np.mean([info['value_loss'] for info in loss_infos])
            avg_entropy = np.mean([info.get('entropy_loss', 0) for info in loss_infos])
            avg_clip_frac = np.mean([info.get('clip_fraction', 0) for info in loss_infos])
            logger.info(
                f"[Coordinator] Update {update_count}: "
                f"loss={avg_loss:.4f}, pg={avg_pg_loss:.4f}, val={avg_value_loss:.4f}, "
                f"ent={avg_entropy:.4f}, clip={avg_clip_frac:.3f}, grad_norm={total_grad_norm:.4f}, "
                f"steps={total_timesteps:,}"
            )
            
            # Signal workers that parameters are updated
            param_update_event.set()
            logger.info("[Coordinator] Parameters updated, signaled workers")

            # Wait at second barrier for all workers to load updated parameters
            try:
                barrier.wait(timeout=60)
                logger.info("[Coordinator] All workers loaded updated parameters")
            except Exception as e:
                logger.warning(f"[Coordinator] Post-update barrier failed: {e}")

            # Save shared model checkpoint periodically
            if update_count % save_every_updates == 0:
                checkpoint_path = save_dir / f"shared_model_step_{total_timesteps}.pt"
                shared_model.save(str(checkpoint_path))
                logger.info(f"[Coordinator] ✓ SHARED MODEL CHECKPOINT SAVED: {checkpoint_path.name} (steps={total_timesteps}, updates={update_count})")

                # Async upload to Firebase Storage
                async_upload_checkpoint(checkpoint_path, logger)

                # Manage checkpoint count
                if max_checkpoints > 0:
                    manage_checkpoints(save_dir, None, max_checkpoints, logger)
    
    except Exception as e:
        logger.error(f"[Coordinator] Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("[Coordinator] Shutting down")


def worker_process(
    worker_id: int,
    device_id: int,
    save_dir: Path,
    progress_queue: mp.Queue,
    stop_event: mp.Event,
    config: Dict[str, Any],
    shared_model: Optional[PPOAgent] = None,
    gradient_queue: Optional[mp.Queue] = None,
    param_update_event: Optional[mp.Event] = None,
    barrier: Optional[mp.Barrier] = None
):
    """
    Worker process that collects experiences and contributes gradients to shared model.
    
    Args:
        worker_id: Unique worker identifier
        device_id: CUDA device to use (-1 for CPU)
        save_dir: Directory to save checkpoints
        progress_queue: Queue for sending progress updates
        stop_event: Event to signal worker to stop
        config: Combined configuration dictionary
        shared_model: Shared PPOAgent model (optional, for parameter sharing)
        gradient_queue: Queue for sending gradients to coordinator
        param_update_event: Event to signal parameters have been updated
        barrier: Barrier for synchronizing workers
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
        
        # Construct price ranges and buy limits from environment data
        price_ranges = {}
        buy_limits = {}
        for item_id in env.tradeable_items:
            metadata = env.item_metadata.get(item_id, {})
            # Use reasonable defaults based on typical OSRS item values
            price_ranges[item_id] = (metadata.get('low_price', 100), metadata.get('high_price', 10000))
            buy_limits[item_id] = metadata.get('buy_limit', 1000)
        
        agent = PPOAgent(
            item_list=env.tradeable_items,
            price_ranges=price_ranges,
            buy_limits=buy_limits,
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
            buffer_size=ppo_kwargs["rollout_steps"],
            agent_id=worker_id
        )
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

        # CRITICAL: Sync parameters from shared model BEFORE collecting any data
        # This ensures all workers start with the same weights
        if shared_model is not None:
            logger.info(f"[Worker {worker_id}] Syncing initial parameters from shared model...")
            agent.actor.load_state_dict(shared_model.actor.state_dict())
            agent.critic.load_state_dict(shared_model.critic.state_dict())
            agent.feature_extractor.load_state_dict(shared_model.feature_extractor.state_dict())
            logger.info(f"[Worker {worker_id}] ✓ Initial parameters synced from SHARED MODEL")

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
            rollout_start_step = step
            logger.info(f"[Worker {worker_id}] Starting rollout collection (steps {step}-{step+rollout_steps})")
            for rollout_idx in range(rollout_steps):
                # Sample action from policy
                action_dict, log_prob = agent.sample_action(obs, deterministic=False)
                
                # Get value estimate
                value = agent.get_value(obs)
                
                # Convert action dict to numpy array format expected by environment
                # Environment expects (n_items, 3) where each row is [action_type, item_idx, price]
                # For single action agent, we create an array with one action at the selected item
                action_array = np.zeros((len(env.tradeable_items), 3), dtype=np.float32)
                if action_dict['type'] != 'hold':
                    item_idx = action_dict['_item_idx']
                    action_type = 1.0 if action_dict['type'] == 'buy' else 2.0  # 0=hold, 1=buy, 2=sell
                    action_array[item_idx] = [action_type, action_dict['price'], action_dict['quantity']]
                
                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(action_array)
                done = terminated or truncated
                
                # Store transition
                agent.store_transition(obs, action_dict, reward, done, value, log_prob)
                
                # Update tracking
                episode_reward += reward
                episode_length += 1
                step += 1
                pbar.update(1)
                
                # Log progress during rollout (helps identify slow workers)
                if rollout_idx > 0 and rollout_idx % 100 == 0:
                    logger.debug(f"[Worker {worker_id}] Rollout progress: {rollout_idx}/{rollout_steps} steps")
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
            
            # Log rollout completion
            logger.info(f"[Worker {worker_id}] ✓ Rollout complete: {rollout_steps} steps collected")
            
            # Update policy
            logger.info(f"Step {step}: Running PPO update...")
            update_start = time.time()
            
            # If using shared model, compute gradients and send to coordinator
            if shared_model is not None and gradient_queue is not None:
                # Compute gradients locally
                loss_info = agent.compute_gradients(
                    last_obs=obs,
                    last_done=done,
                    n_epochs=ppo_epochs,
                    batch_size=minibatch_size
                )
                
                # Extract gradients
                gradients = {}
                for name, param in agent.actor.named_parameters():
                    if param.grad is not None:
                        gradients[f'actor.{name}'] = param.grad.clone().cpu()
                for name, param in agent.critic.named_parameters():
                    if param.grad is not None:
                        gradients[f'critic.{name}'] = param.grad.clone().cpu()
                for name, param in agent.feature_extractor.named_parameters():
                    if param.grad is not None:
                        gradients[f'feature_extractor.{name}'] = param.grad.clone().cpu()
                
                # Send gradients to coordinator
                gradient_queue.put({
                    'worker_id': worker_id,
                    'gradients': gradients,
                    'loss_info': loss_info,
                    'steps': rollout_steps  # Steps collected in this rollout
                })
                logger.info(f"[Worker {worker_id}] Gradients sent to coordinator")
                
                # Wait at barrier for all workers to submit gradients
                if barrier is not None:
                    logger.info(f"[Worker {worker_id}] ⏳ Approaching barrier (step {step})...")
                    try:
                        barrier.wait(timeout=600)  # 10 minute timeout
                        logger.info(f"[Worker {worker_id}] ✓ Barrier passed, waiting for parameter update...")
                    except Exception as e:
                        logger.error(f"Barrier wait failed: {e}. Worker will exit.")
                        raise RuntimeError(f"Worker {worker_id} barrier synchronization failed") from e

                # Wait for parameter update from coordinator
                param_update_event.wait()
                logger.info("Parameter update event received")

                # Load updated parameters from shared model
                agent.actor.load_state_dict(shared_model.actor.state_dict())
                agent.critic.load_state_dict(shared_model.critic.state_dict())
                agent.feature_extractor.load_state_dict(shared_model.feature_extractor.state_dict())
                logger.info("✓ Loaded updated parameters from SHARED MODEL")

                # Second barrier to ensure all workers have loaded params before clearing event
                try:
                    barrier.wait(timeout=60)
                except Exception as e:
                    logger.warning(f"Post-update barrier failed: {e}")

                # Only worker 0 clears the event to avoid race condition
                if worker_id == 0:
                    param_update_event.clear()
                
            else:
                # Independent training (original behavior)
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
            
            # Save checkpoint (only if NOT using shared model)
            if shared_model is None and (step % save_every < rollout_steps or step >= max_steps):
                checkpoint_path = save_dir / f"worker_{worker_id}_step_{step}.pt"
                agent.save(str(checkpoint_path))
                logger.info(f"[OK] Checkpoint saved: {checkpoint_path.name}")
                
                # Manage checkpoint count
                max_checkpoints = config['train_kwargs'].get('max_checkpoints', 0)
                if max_checkpoints > 0:
                    manage_checkpoints(save_dir, worker_id, max_checkpoints, logger)
        
        # Close progress bar
        pbar.close()
        
        # Save final checkpoint (only if NOT using shared model)
        if shared_model is None:
            final_path = save_dir / f"worker_{worker_id}_final.pt"
            agent.save(str(final_path))
            logger.info(f"[OK] Final checkpoint saved: {final_path.name}")
        else:
            logger.info("Using shared model - skipping worker checkpoint (coordinator handles saves)")
        
        # Final summary
        total_time = time.time() - start_time
        logger.info("="*60)
        logger.info(f"WORKER {worker_id} COMPLETE")
        logger.info(f"  Total Steps: {step}")
        logger.info(f"  Total Episodes: {episode}")
        logger.info(f"  Total Time: {total_time/60:.1f} minutes")
        logger.info(f"  Avg Speed: {step/max(total_time, 0.001):.1f} steps/sec")
        if shared_model is None:
            logger.info(f"  Final checkpoint: {final_path.name}")
        else:
            logger.info(f"  Checkpoints: managed by coordinator")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Worker {worker_id} failed with error: {e}", exc_info=True)
        try:
            progress_queue.put({
                'worker_id': worker_id,
                'error': str(e),
                'step': step if 'step' in locals() else 0,
                'fatal': True
            }, timeout=5)
        except:
            pass  # Queue might be full or closed
        # NOTE: Do NOT set stop_event here - let main process decide whether to stop
        # based on how many workers have failed. This prevents cascade failures
        # where one worker crash kills all others.
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
    
    # Set environment variable to indicate cache is pre-loaded
    os.environ['CACHE_PRELOADED'] = '1'
    
    shm_name = None
    if TRAIN_KWARGS.get("use_shared_cache", True):
        from training.cached_market_loader import load_cache, get_shared_memory_name
        load_cache(cache_file, use_shared_memory=True)
        shm_name = get_shared_memory_name()
        cache_time = time.time() - cache_start
        logger.info(f"[OK] Cache loaded in shared memory ({cache_time:.1f}s)")
        logger.info(f"[OK] Shared memory name: {shm_name}")
        os.environ['CACHE_SHM_NAME'] = shm_name if shm_name else ''
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
    
    # Check if using shared model mode
    use_shared_model = TRAIN_KWARGS.get("use_shared_model", True)
    logger.info(f"[OK] Shared model training: {'enabled' if use_shared_model else 'disabled'}")
    
    # Create communication channels
    progress_queue = mp.Queue()
    stop_event = mp.Event()
    
    # Shared model components (only if enabled)
    shared_model = None
    gradient_queue = None
    param_update_event = None
    barrier = None
    coordinator = None
    
    if use_shared_model:
        logger.info("="*80)
        logger.info("INITIALIZING SHARED MODEL")
        logger.info("="*80)
        
        # Create shared model (on GPU 0 if available)
        device = 'cuda:0' if num_gpus > 0 else 'cpu'
        logger.info(f"Creating shared model on {device}...")
        
        # Need to create a dummy environment to get item metadata
        from training.ge_environment import GrandExchangeEnv
        dummy_env = GrandExchangeEnv(**ENV_KWARGS)
        
        # Construct price ranges and buy limits from environment data
        price_ranges = {}
        buy_limits = {}
        for item_id in dummy_env.tradeable_items:
            metadata = dummy_env.item_metadata.get(item_id, {})
            price_ranges[item_id] = (metadata.get('low_price', 100), metadata.get('high_price', 10000))
            buy_limits[item_id] = metadata.get('buy_limit', 1000)
        
        shared_model = PPOAgent(
            item_list=dummy_env.tradeable_items,
            price_ranges=price_ranges,
            buy_limits=buy_limits,
            device=device,
            hidden_size=PPO_KWARGS["hidden_size"],
            num_layers=PPO_KWARGS["num_layers"],
            lr=PPO_KWARGS["lr"],
            gamma=PPO_KWARGS["gamma"],
            gae_lambda=PPO_KWARGS["gae_lambda"],
            clip_epsilon=PPO_KWARGS["clip_epsilon"],
            entropy_coef=PPO_KWARGS["entropy_coef"],
            value_coef=PPO_KWARGS["value_coef"],
            price_bins=PPO_KWARGS["price_bins"],
            quantity_bins=PPO_KWARGS["quantity_bins"],
            wait_steps_bins=PPO_KWARGS["wait_steps_bins"],
            buffer_size=PPO_KWARGS["rollout_steps"],
            agent_id=0  # Shared model is agent 0
        )
        
        # Check for existing checkpoint to resume training
        latest_checkpoint = find_latest_checkpoint(save_dir, logger)
        initial_update_count = 0
        initial_timesteps = 0
        
        if latest_checkpoint is not None:
            logger.info("="*80)
            logger.info(f"RESUMING FROM CHECKPOINT: {latest_checkpoint.name}")
            logger.info(f"Full path: {latest_checkpoint}")
            logger.info("="*80)
            try:
                logger.info(f"Loading checkpoint file: {latest_checkpoint}")
                checkpoint_data = torch.load(latest_checkpoint, map_location=device)
                shared_model.feature_extractor.load_state_dict(checkpoint_data["feature_extractor"])
                shared_model.actor.load_state_dict(checkpoint_data["actor"])
                shared_model.critic.load_state_dict(checkpoint_data["critic"])
                shared_model.optimizer.load_state_dict(checkpoint_data["optimizer"])
                shared_model.n_updates = checkpoint_data.get("n_updates", 0)
                shared_model.total_timesteps = checkpoint_data.get("total_timesteps", 0)
                initial_update_count = shared_model.n_updates
                initial_timesteps = shared_model.total_timesteps
                logger.info(f"✓ Checkpoint loaded successfully from {latest_checkpoint.name}")
                logger.info(f"  Previous updates: {shared_model.n_updates}")
                logger.info(f"  Previous timesteps: {shared_model.total_timesteps}")
                logger.info(f"  Resuming from step {initial_timesteps}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                logger.warning("Starting fresh training instead")
                initial_update_count = 0
                initial_timesteps = 0
        
        logger.info(f"[OK] Shared model created: obs_dim={shared_model.obs_dim}, n_items={shared_model.n_items}")
        logger.info(f"[OK] Training will start from step {initial_timesteps}")

        # Setup learning rate scheduler for better convergence
        # Linear warmup for first 1% of training, then cosine annealing
        total_updates = (TRAIN_KWARGS["max_steps_per_worker"] // PPO_KWARGS["rollout_steps"]) * num_workers
        from torch.optim.lr_scheduler import CosineAnnealingLR
        shared_model.scheduler = CosineAnnealingLR(
            shared_model.optimizer,
            T_max=total_updates,
            eta_min=PPO_KWARGS["lr"] * 0.1  # Decay to 10% of initial LR
        )
        logger.info(f"[OK] Learning rate scheduler: CosineAnnealingLR over {total_updates} updates")

        # Share model parameters across processes
        shared_model.actor.share_memory()
        shared_model.critic.share_memory()
        shared_model.feature_extractor.share_memory()
        logger.info("[OK] Shared model created and moved to shared memory")
        
        # Create gradient coordination components
        gradient_queue = mp.Queue()
        param_update_event = mp.Event()
        barrier = mp.Barrier(num_workers + 1)  # +1 for coordinator
        
        # Start gradient coordinator process
        max_checkpoints = TRAIN_KWARGS.get('max_checkpoints', 0)

        # Calculate save interval in updates from config
        # save_every_steps / (rollout_steps * num_workers) = updates between saves
        rollout_steps = PPO_KWARGS["rollout_steps"]
        save_every_steps = TRAIN_KWARGS["save_every_steps"]
        save_every_updates = max(1, save_every_steps // (rollout_steps * num_workers))
        logger.info(f"[OK] Save checkpoint every {save_every_updates} updates (~{save_every_steps:,} total steps)")

        coordinator = mp.Process(
            target=gradient_coordinator,
            args=(num_workers, shared_model, gradient_queue, param_update_event,
                  barrier, stop_event, save_dir, setup_logger("Coordinator", str(log_dir / "coordinator.log")),
                  max_checkpoints, initial_update_count, initial_timesteps, save_every_updates, rollout_steps),
            daemon=False
        )
        coordinator.start()
        logger.info(f"[OK] Gradient coordinator started (PID: {coordinator.pid})")
    
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
            args=(worker_id, device_id, save_dir, progress_queue, stop_event, config,
                  shared_model, gradient_queue, param_update_event, barrier),
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
                    if update.get('fatal', False):
                        logger.error(f"Worker {worker_id} encountered fatal error - stopping all workers")
                        stop_event.set()
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
            
            # Check for dead workers
            dead_workers = [i for i, p in enumerate(workers) if not p.is_alive()]
            alive_workers = num_workers - len(dead_workers)

            # For shared model training, ALL workers must stay alive due to barrier synchronization
            # For independent training, we can tolerate some worker deaths
            if TRAIN_KWARGS.get("use_shared_model", False):
                # Shared model mode - ANY worker death breaks the barrier, must stop
                if dead_workers:
                    logger.error(f"⚠️  WORKERS DIED: {dead_workers}")
                    logger.error("Shared model training requires ALL workers (barrier sync)")
                    logger.error("Check individual worker logs for error details:")
                    for dead_id in dead_workers:
                        logger.error(f"  - logs/worker_{dead_id}.log")
                    logger.error("Stopping remaining workers...")
                    stop_event.set()
                    break
            else:
                # Independent training - allow graceful degradation
                # Only stop if too many workers have died (less than 50% remaining)
                min_workers_required = max(1, num_workers // 2)

                if dead_workers and alive_workers < min_workers_required:
                    logger.error(f"⚠️  TOO MANY WORKERS DIED: {dead_workers}")
                    logger.error(f"Only {alive_workers}/{num_workers} workers remaining (need at least {min_workers_required})")
                    logger.error("Check individual worker logs for error details")
                    logger.error("Stopping remaining workers...")
                    stop_event.set()
                    break
                elif dead_workers:
                    # Some workers died but we can continue with reduced capacity
                    logger.warning(f"⚠️  Workers died: {dead_workers}, but continuing with {alive_workers} workers")
            
            # Check if coordinator died (only if using shared model)
            if TRAIN_KWARGS.get("use_shared_model", False):
                if coordinator is not None and not coordinator.is_alive():
                    logger.error("⚠️  COORDINATOR DIED!")
                    logger.error("Check coordinator.log for error details")
                    logger.error("Stopping all workers...")
                    stop_event.set()
                    break
            
            # Print summary periodically
            if time.time() - last_summary > summary_interval:
                logger.info("-"*80)
                logger.info("PROGRESS SUMMARY")
                total_steps = sum(s['step'] for s in worker_status.values())
                active_workers = sum(1 for p in workers if p.is_alive())
                logger.info(f"Active Workers: {active_workers}/{num_workers}")
                logger.info(f"Total Steps: {total_steps:,}")
                
                # Show which workers are alive/dead
                worker_health = []
                for i, p in enumerate(workers):
                    status = "✓ ALIVE" if p.is_alive() else "✗ DEAD"
                    worker_health.append(f"{i}:{status}")
                logger.info(f"Worker Health: {' | '.join(worker_health)}")
                
                # Show worker progress
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
    
    # Stop coordinator if running
    if coordinator is not None:
        logger.info("Stopping gradient coordinator...")
        stop_event.set()
        coordinator.join(timeout=10)
        if coordinator.is_alive():
            logger.warning("Coordinator didn't stop gracefully, terminating...")
            coordinator.terminate()
        logger.info("[OK] Coordinator stopped")
    
    # Save final shared model if enabled
    if use_shared_model and shared_model is not None:
        final_model_path = save_dir / "shared_model_final.pt"
        shared_model.save(str(final_model_path))
        logger.info(f"[OK] Final shared model saved: {final_model_path.name}")
    
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
