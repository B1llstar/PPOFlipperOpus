#!/usr/bin/env python3
"""
PPO Flipper Training Dashboard Backend

FastAPI server with WebSocket support for real-time training monitoring and control.
"""

import os
import sys
import json
import asyncio
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ppo_config import ENV_KWARGS, PPO_KWARGS, TRAIN_KWARGS


class TrainingState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"


@dataclass
class AgentMetrics:
    """Metrics for a single agent."""
    agent_id: int
    cash: float = 0.0
    portfolio_value: float = 0.0
    total_assets: float = 0.0
    holdings: Dict[str, int] = field(default_factory=dict)
    pending_orders: List[Dict] = field(default_factory=list)
    episode_reward: float = 0.0
    total_reward: float = 0.0
    trades_executed: int = 0
    profitable_trades: int = 0
    taxes_paid: float = 0.0
    current_action: Optional[str] = None
    last_trade: Optional[Dict] = None
    episode: int = 0
    step: int = 0


@dataclass
class TrainingMetrics:
    """Global training metrics."""
    state: TrainingState = TrainingState.IDLE
    start_time: Optional[str] = None
    elapsed_seconds: float = 0.0
    total_episodes: int = 0
    total_steps: int = 0
    avg_reward: float = 0.0
    best_reward: float = float('-inf')
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    learning_rate: float = 0.0


class TrainingManager:
    """Manages training state and metrics."""

    def __init__(self):
        self.state = TrainingState.IDLE
        self.metrics = TrainingMetrics()
        self.agents: Dict[int, AgentMetrics] = {}
        self.num_agents = TRAIN_KWARGS.get('num_agents', 4)

        # Initialize agents
        for i in range(self.num_agents):
            self.agents[i] = AgentMetrics(
                agent_id=i,
                cash=ENV_KWARGS.get('starting_gp', 10_000_000)
            )

        # History for charts (keep last 1000 points)
        self.reward_history: deque = deque(maxlen=1000)
        self.portfolio_history: deque = deque(maxlen=1000)
        self.loss_history: deque = deque(maxlen=1000)

        # Trade log (keep last 500 trades)
        self.trade_log: deque = deque(maxlen=500)

        # Training thread
        self.training_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.pause_event.set()  # Not paused initially

        # WebSocket connections
        self.connections: List[WebSocket] = []

        # Lock for thread-safe updates
        self.lock = threading.Lock()

        self.start_time: Optional[datetime] = None

    def get_state_dict(self) -> Dict:
        """Get complete state as dictionary."""
        with self.lock:
            return {
                'training': {
                    'state': self.state.value,
                    'start_time': self.metrics.start_time,
                    'elapsed_seconds': self.metrics.elapsed_seconds,
                    'total_episodes': self.metrics.total_episodes,
                    'total_steps': self.metrics.total_steps,
                    'avg_reward': self.metrics.avg_reward,
                    'best_reward': self.metrics.best_reward if self.metrics.best_reward != float('-inf') else 0,
                    'policy_loss': self.metrics.policy_loss,
                    'value_loss': self.metrics.value_loss,
                    'entropy': self.metrics.entropy,
                    'learning_rate': self.metrics.learning_rate,
                },
                'agents': {
                    agent_id: asdict(agent)
                    for agent_id, agent in self.agents.items()
                },
                'history': {
                    'rewards': list(self.reward_history),
                    'portfolios': list(self.portfolio_history),
                    'losses': list(self.loss_history),
                },
                'recent_trades': list(self.trade_log),
                'config': {
                    'env': ENV_KWARGS,
                    'ppo': PPO_KWARGS,
                    'train': TRAIN_KWARGS,
                }
            }

    def update_agent(self, agent_id: int, **kwargs):
        """Update agent metrics."""
        with self.lock:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                for key, value in kwargs.items():
                    if hasattr(agent, key):
                        setattr(agent, key, value)

                # Update total assets
                agent.total_assets = agent.cash + agent.portfolio_value

    def update_training(self, **kwargs):
        """Update training metrics."""
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self.metrics, key):
                    setattr(self.metrics, key, value)

            # Update elapsed time
            if self.start_time and self.state == TrainingState.RUNNING:
                self.metrics.elapsed_seconds = (datetime.now() - self.start_time).total_seconds()

    def add_reward_point(self, episode: int, reward: float, agent_id: int):
        """Add a reward data point."""
        with self.lock:
            self.reward_history.append({
                'episode': episode,
                'reward': reward,
                'agent_id': agent_id,
                'timestamp': datetime.now().isoformat()
            })

    def add_portfolio_point(self, episode: int, values: Dict[int, float]):
        """Add portfolio value data point."""
        with self.lock:
            self.portfolio_history.append({
                'episode': episode,
                'values': values,
                'timestamp': datetime.now().isoformat()
            })

    def add_loss_point(self, step: int, policy_loss: float, value_loss: float):
        """Add loss data point."""
        with self.lock:
            self.loss_history.append({
                'step': step,
                'policy_loss': policy_loss,
                'value_loss': value_loss,
                'timestamp': datetime.now().isoformat()
            })

    def log_trade(self, agent_id: int, trade: Dict):
        """Log a trade."""
        with self.lock:
            trade['agent_id'] = agent_id
            trade['timestamp'] = datetime.now().isoformat()
            self.trade_log.append(trade)

    async def broadcast(self, message: Dict):
        """Broadcast message to all connected clients."""
        disconnected = []
        for connection in self.connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)

        for conn in disconnected:
            self.connections.remove(conn)

    def start_training(self):
        """Start training."""
        if self.state != TrainingState.IDLE:
            return False

        self.state = TrainingState.RUNNING
        self.metrics.state = TrainingState.RUNNING
        self.start_time = datetime.now()
        self.metrics.start_time = self.start_time.isoformat()
        self.stop_event.clear()
        self.pause_event.set()

        # Reset agents
        for agent in self.agents.values():
            agent.cash = ENV_KWARGS.get('starting_gp', 10_000_000)
            agent.portfolio_value = 0.0
            agent.total_assets = agent.cash
            agent.holdings = {}
            agent.pending_orders = []
            agent.episode_reward = 0.0
            agent.total_reward = 0.0
            agent.trades_executed = 0
            agent.profitable_trades = 0
            agent.taxes_paid = 0.0

        # Start training thread
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()

        return True

    def stop_training(self):
        """Stop training."""
        if self.state not in [TrainingState.RUNNING, TrainingState.PAUSED]:
            return False

        self.state = TrainingState.STOPPING
        self.metrics.state = TrainingState.STOPPING
        self.stop_event.set()
        self.pause_event.set()  # Unpause to allow thread to exit

        return True

    def pause_training(self):
        """Pause training."""
        if self.state != TrainingState.RUNNING:
            return False

        self.state = TrainingState.PAUSED
        self.metrics.state = TrainingState.PAUSED
        self.pause_event.clear()

        return True

    def resume_training(self):
        """Resume training."""
        if self.state != TrainingState.PAUSED:
            return False

        self.state = TrainingState.RUNNING
        self.metrics.state = TrainingState.RUNNING
        self.pause_event.set()

        return True

    def _training_loop(self):
        """Main training loop (runs in separate thread)."""
        try:
            # Import training components
            from training.ge_environment import GrandExchangeEnv
            from ppo_agent import PPOAgent
            from ppo_config import ENV_KWARGS, PPO_KWARGS

            import torch
            import numpy as np

            device = PPO_KWARGS.get('device', 'cpu')
            print(f"Using device: {device}")

            # Create environments and agents for each agent
            envs = []
            agents = []

            # Database path - relative to this file's location
            # server.py is in dashboard/backend/, so go up 2 levels to project root
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            db_path = os.path.join(project_root, 'ge_prices.db')
            print(f"Database path: {db_path}")
            print(f"Database exists: {os.path.exists(db_path)}")

            if not os.path.exists(db_path):
                raise FileNotFoundError(f"Database not found at {db_path}")

            for i in range(self.num_agents):
                print(f"Creating environment {i+1}/{self.num_agents}...")
                env = GrandExchangeEnv(
                    db_path=db_path,
                    initial_cash=ENV_KWARGS.get('starting_gp', 10_000_000),
                    episode_length=ENV_KWARGS.get('max_steps', 168),
                    top_n_items=50,
                )
                print(f"Environment {i+1} created. Tradeable items: {len(env.tradeable_items)}")
                envs.append(env)

                # Build item_list, price_ranges, buy_limits from environment
                item_list = [f"Item_{item_id}" for item_id in env.tradeable_items]
                price_ranges = {}
                buy_limits = {}
                for item_id in env.tradeable_items:
                    meta = env.item_metadata.get(item_id, {})
                    # Get price range from market history
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
                    device=torch.device(device),
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
                agents.append(agent)
                print(f"Agent {i+1} created with {len(item_list)} items")

            # Initialize observations
            observations = [env.reset()[0] for env in envs]
            episode_rewards = [0.0] * self.num_agents
            episode_steps = [0] * self.num_agents

            step = 0
            episode = 0

            while not self.stop_event.is_set():
                # Check for pause
                self.pause_event.wait()

                if self.stop_event.is_set():
                    break

                # Run one step for each agent
                for i, (env, agent, obs) in enumerate(zip(envs, agents, observations)):
                    # Get action from policy - sample from action space for now
                    # The environment expects actions of shape (n_items, 3)
                    action = env.action_space.sample()

                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    episode_rewards[i] += reward
                    episode_steps[i] += 1

                    # Get holdings from positions
                    holdings = {}
                    for item_id, position in env.positions.items():
                        if position.quantity > 0:
                            holdings[f"Item_{item_id}"] = position.quantity

                    # Update agent metrics
                    self.update_agent(
                        i,
                        cash=env.cash,
                        portfolio_value=info.get('total_value', env.cash) - env.cash,
                        holdings=holdings,
                        pending_orders=[],
                        step=episode_steps[i],
                        episode=episode,
                        current_action=f"step {episode_steps[i]}",
                    )

                    # Track trades from info
                    if info.get('total_trades', 0) > self.agents[i].trades_executed:
                        new_trades = info['total_trades'] - self.agents[i].trades_executed
                        self.update_agent(i, trades_executed=info['total_trades'])

                        # Estimate profitable trades from win rate
                        win_rate = info.get('win_rate', 0)
                        profitable = int(info['total_trades'] * win_rate)
                        self.update_agent(i, profitable_trades=profitable)

                    if done:
                        # Episode complete for this agent
                        self.update_agent(
                            i,
                            episode_reward=episode_rewards[i],
                            total_reward=self.agents[i].total_reward + episode_rewards[i],
                        )
                        self.add_reward_point(episode, episode_rewards[i], i)

                        # Reset
                        observations[i] = env.reset()[0]
                        episode_rewards[i] = 0.0
                        episode_steps[i] = 0
                    else:
                        observations[i] = next_obs

                step += 1

                # Update training metrics periodically
                if step % 100 == 0:
                    portfolio_values = {
                        i: self.agents[i].total_assets
                        for i in range(self.num_agents)
                    }
                    self.add_portfolio_point(episode, portfolio_values)

                    avg_reward = np.mean([a.episode_reward for a in self.agents.values()])
                    best_reward = max(self.metrics.best_reward, max(a.total_reward for a in self.agents.values()))

                    self.update_training(
                        total_steps=step,
                        total_episodes=episode,
                        avg_reward=avg_reward,
                        best_reward=best_reward,
                        learning_rate=PPO_KWARGS.get('lr', 3e-4),
                    )

                # Check if all agents completed an episode
                if all(s == 0 for s in episode_steps):
                    episode += 1

                # Small delay to prevent CPU spinning
                time.sleep(0.01)

        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.state = TrainingState.IDLE
            self.metrics.state = TrainingState.IDLE

    def _load_items(self) -> Dict:
        """Load items for training."""
        import sqlite3

        db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'ge_prices.db'
        )

        items = {}

        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                # Get items with price data
                cursor.execute("""
                    SELECT item_id,
                           AVG(high_price) as avg_high,
                           AVG(low_price) as avg_low,
                           MIN(low_price) as min_price,
                           MAX(high_price) as max_price
                    FROM timeseries
                    WHERE high_price IS NOT NULL AND low_price IS NOT NULL
                    GROUP BY item_id
                    LIMIT 50
                """)

                for row in cursor.fetchall():
                    item_id, avg_high, avg_low, min_price, max_price = row
                    base_price = int((avg_high + avg_low) / 2)
                    items[f"Item_{item_id}"] = {
                        'id': item_id,
                        'base_price': base_price,
                        'buy_limit': 10000,
                        'min_price': int(min_price * 0.9),
                        'max_price': int(max_price * 1.1),
                    }

                conn.close()
            except Exception as e:
                print(f"Error loading items: {e}")

        # Fallback items
        if not items:
            items = {
                'Dragon bones': {'base_price': 2500, 'buy_limit': 13000, 'min_price': 2000, 'max_price': 3500},
                'Nature rune': {'base_price': 150, 'buy_limit': 25000, 'min_price': 100, 'max_price': 200},
                'Cannonball': {'base_price': 200, 'buy_limit': 11000, 'min_price': 150, 'max_price': 300},
            }

        return items


# Create FastAPI app and training manager
app = FastAPI(title="PPO Flipper Dashboard", version="1.0.0")
manager = TrainingManager()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check."""
    return {"status": "ok", "message": "PPO Flipper Dashboard API"}


@app.get("/api/state")
async def get_state():
    """Get complete training state."""
    return manager.get_state_dict()


@app.get("/api/config")
async def get_config():
    """Get training configuration."""
    return {
        'env': ENV_KWARGS,
        'ppo': PPO_KWARGS,
        'train': TRAIN_KWARGS,
    }


@app.post("/api/training/start")
async def start_training():
    """Start training."""
    if manager.start_training():
        return {"status": "ok", "message": "Training started"}
    raise HTTPException(400, "Cannot start training - already running or stopping")


@app.post("/api/training/stop")
async def stop_training():
    """Stop training."""
    if manager.stop_training():
        return {"status": "ok", "message": "Training stopping"}
    raise HTTPException(400, "Cannot stop training - not running")


@app.post("/api/training/pause")
async def pause_training():
    """Pause training."""
    if manager.pause_training():
        return {"status": "ok", "message": "Training paused"}
    raise HTTPException(400, "Cannot pause training - not running")


@app.post("/api/training/resume")
async def resume_training():
    """Resume training."""
    if manager.resume_training():
        return {"status": "ok", "message": "Training resumed"}
    raise HTTPException(400, "Cannot resume training - not paused")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates."""
    await websocket.accept()
    manager.connections.append(websocket)

    try:
        # Send initial state
        await websocket.send_json({
            'type': 'init',
            'data': manager.get_state_dict()
        })

        # Keep connection alive and send updates
        while True:
            # Send state updates every 500ms
            await asyncio.sleep(0.5)
            await websocket.send_json({
                'type': 'update',
                'data': manager.get_state_dict()
            })

    except WebSocketDisconnect:
        manager.connections.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in manager.connections:
            manager.connections.remove(websocket)


def main():
    """Run the server."""
    print("Starting PPO Flipper Dashboard on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
