"""
Training Controller Module

This module provides a TrainingController class that wraps train_ppo.py
and allows external control and monitoring of the training process.
"""

import os
import sys
import threading
import queue
import time
import json
from datetime import datetime
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum

import torch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppo_agent import PPOAgent
from ppo_config import ENV_KWARGS, PPO_KWARGS, TRAIN_KWARGS
from training.ge_environment import GrandExchangeEnv
from training.shared_knowledge import SharedKnowledgeRepository


class TrainingState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"


@dataclass
class TrainingStats:
    """Training statistics."""
    episode: int = 0
    total_steps: int = 0
    avg_reward: float = 0.0
    best_reward: float = float('-inf')
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    learning_rate: float = 0.0
    elapsed_time: float = 0.0


class TrainingController:
    """
    Controls and monitors PPO training with support for pause/resume/stop.
    Similar to Kohya SS training controller.
    """
    
    def __init__(self, db_path: str = "ge_prices.db", callback: Optional[Callable] = None):
        """
        Initialize training controller.
        
        Args:
            db_path: Path to the SQLite database with GE price data
            callback: Optional callback function for updates, called with (agent_id, stats_dict)
        """
        self.db_path = db_path
        self.callback = callback
        
        # Training state
        self.state = TrainingState.IDLE
        self.training_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.pause_event.set()  # Not paused initially
        
        # Training components
        self.envs = []
        self.agents = []
        self.num_agents = TRAIN_KWARGS.get('num_agents', 4)
        
        # Statistics
        self.stats: Dict[int, TrainingStats] = {}
        for i in range(self.num_agents):
            self.stats[i] = TrainingStats()
        
        self.start_time: Optional[datetime] = None
        
        # Thread-safe queue for updates
        self.update_queue = queue.Queue()
        
    def start(self) -> bool:
        """Start training."""
        if self.state != TrainingState.IDLE:
            return False
            
        self.state = TrainingState.RUNNING
        self.stop_event.clear()
        self.pause_event.set()
        self.start_time = datetime.now()
        
        # Reset statistics
        for i in range(self.num_agents):
            self.stats[i] = TrainingStats()
        
        # Start training thread
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()
        
        return True
    
    def stop(self) -> bool:
        """Stop training."""
        if self.state not in [TrainingState.RUNNING, TrainingState.PAUSED]:
            return False
            
        self.state = TrainingState.STOPPING
        self.stop_event.set()
        self.pause_event.set()  # Unpause to allow thread to exit
        
        return True
    
    def pause(self) -> bool:
        """Pause training."""
        if self.state != TrainingState.RUNNING:
            return False
            
        self.state = TrainingState.PAUSED
        self.pause_event.clear()
        
        return True
    
    def resume(self) -> bool:
        """Resume training."""
        if self.state != TrainingState.PAUSED:
            return False
            
        self.state = TrainingState.RUNNING
        self.pause_event.set()
        
        return True
    
    def get_state(self) -> str:
        """Get current training state."""
        return self.state.value
    
    def get_stats(self, agent_id: int) -> Dict:
        """Get statistics for a specific agent."""
        if agent_id in self.stats:
            stats = self.stats[agent_id]
            return asdict(stats)
        return {}
    
    def get_all_stats(self) -> Dict[int, Dict]:
        """Get statistics for all agents."""
        return {
            agent_id: asdict(stats)
            for agent_id, stats in self.stats.items()
        }
    
    def _emit_update(self, agent_id: int, update_type: str, data: Dict):
        """Emit an update via callback."""
        if self.callback:
            try:
                self.callback(agent_id, {
                    'type': update_type,
                    'timestamp': datetime.now().isoformat(),
                    'data': data
                })
            except Exception as e:
                print(f"Error in callback: {e}")
    
    def _training_loop(self):
        """Main training loop."""
        try:
            print(f"Starting training with database: {self.db_path}")
            print(f"Database exists: {os.path.exists(self.db_path)}")
            
            if not os.path.exists(self.db_path):
                raise FileNotFoundError(f"Database not found at {self.db_path}")
            
            device = torch.device(PPO_KWARGS.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
            print(f"Using device: {device}")
            
            # Create environments and agents
            for i in range(self.num_agents):
                print(f"Creating environment {i+1}/{self.num_agents}...")
                
                env = GrandExchangeEnv(
                    db_path=self.db_path,
                    initial_cash=ENV_KWARGS.get('starting_gp', 10_000_000),
                    episode_length=ENV_KWARGS.get('max_steps', 168),
                    top_n_items=ENV_KWARGS.get('top_n_items', 50),
                )
                self.envs.append(env)
                print(f"Environment {i+1} created. Tradeable items: {len(env.tradeable_items)}")
                
                # Build agent configuration from environment
                item_list = [f"Item_{item_id}" for item_id in env.tradeable_items]
                price_ranges = {}
                buy_limits = {}
                
                for item_id in env.tradeable_items:
                    meta = env.item_metadata.get(item_id, {})
                    history = env.market_history.get(item_id, [])
                    
                    if history:
                        min_price = min(s.low_price for s in history)
                        max_price = max(s.high_price for s in history)
                    else:
                        min_price, max_price = 1, 1000000
                    
                    price_ranges[f"Item_{item_id}"] = (min_price, max_price)
                    buy_limits[f"Item_{item_id}"] = meta.get('ge_limit', 10000)
                
                agent = PPOAgent(
                    item_list=item_list,
                    price_ranges=price_ranges,
                    buy_limits=buy_limits,
                    device=device,
                    hidden_size=PPO_KWARGS.get('hidden_size', 256),
                    num_layers=PPO_KWARGS.get('num_layers', 3),
                    price_bins=PPO_KWARGS.get('price_bins', 21),
                    quantity_bins=PPO_KWARGS.get('quantity_bins', 11),
                    lr=PPO_KWARGS.get('lr', 3e-4),
                    gamma=PPO_KWARGS.get('gamma', 0.99),
                    gae_lambda=PPO_KWARGS.get('gae_lambda', 0.95),
                    clip_epsilon=PPO_KWARGS.get('clip_epsilon', 0.2),
                    entropy_coef=PPO_KWARGS.get('entropy_coef', 0.01),
                    value_coef=PPO_KWARGS.get('value_coef', 0.5),
                    max_grad_norm=PPO_KWARGS.get('max_grad_norm', 0.5),
                    agent_id=i,
                )
                self.agents.append(agent)
                print(f"Agent {i+1} created with {len(item_list)} items")
            
            # Initialize observations
            observations = [env.reset()[0] for env in self.envs]
            episode_rewards = [0.0] * self.num_agents
            episode_steps = [0] * self.num_agents
            
            step = 0
            episode = 0
            
            print("Starting training loop...")
            
            while not self.stop_event.is_set():
                # Check for pause
                self.pause_event.wait()
                
                if self.stop_event.is_set():
                    break
                
                # Run one step for each agent
                for i, (env, agent, obs) in enumerate(zip(self.envs, self.agents, observations)):
                    # Get action from agent
                    action = env.action_space.sample()  # For now, random actions
                    
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    episode_rewards[i] += reward
                    episode_steps[i] += 1
                    
                    # Update statistics
                    stats = self.stats[i]
                    stats.total_steps = step
                    stats.episode = episode
                    stats.avg_reward = episode_rewards[i] / max(episode_steps[i], 1)
                    
                    if self.start_time:
                        stats.elapsed_time = (datetime.now() - self.start_time).total_seconds()
                    
                    # Emit update
                    self._emit_update(i, 'step', {
                        'step': step,
                        'episode': episode,
                        'reward': reward,
                        'episode_reward': episode_rewards[i],
                        'cash': env.cash,
                        'portfolio_value': info.get('total_value', env.cash) - env.cash,
                        'total_assets': info.get('total_value', env.cash),
                    })
                    
                    if done:
                        # Episode complete
                        if episode_rewards[i] > stats.best_reward:
                            stats.best_reward = episode_rewards[i]
                        
                        self._emit_update(i, 'episode_complete', {
                            'episode': episode,
                            'total_reward': episode_rewards[i],
                            'steps': episode_steps[i],
                        })
                        
                        # Reset
                        observations[i] = env.reset()[0]
                        episode_rewards[i] = 0.0
                        episode_steps[i] = 0
                    else:
                        observations[i] = next_obs
                
                step += 1
                
                # Check if all agents completed an episode
                if all(s == 0 for s in episode_steps):
                    episode += 1
                
                # Small delay to prevent CPU spinning
                time.sleep(0.01)
            
            print("Training loop finished")
            
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            self.state = TrainingState.IDLE
            print("Training stopped")
    
    def cleanup(self):
        """Clean up resources."""
        self.stop()
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=5)


if __name__ == "__main__":
    # Test the controller
    def callback(agent_id, data):
        print(f"Agent {agent_id}: {data}")
    
    controller = TrainingController(callback=callback)
    
    print("Starting training...")
    controller.start()
    
    try:
        time.sleep(10)
        print("\nPausing training...")
        controller.pause()
        time.sleep(3)
        
        print("\nResuming training...")
        controller.resume()
        time.sleep(10)
        
        print("\nStopping training...")
        controller.stop()
    except KeyboardInterrupt:
        print("\nInterrupted!")
        controller.stop()
    
    print("\nFinal stats:")
    for agent_id, stats in controller.get_all_stats().items():
        print(f"Agent {agent_id}: {stats}")
