import asyncio
import logging
import os
import sys
import time
import json
import numpy as np
import torch
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ppo_websocket_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ppo_websocket_test")

# Import the modules to test
from ppo_websocket_integration import PPOWebSocketIntegration, EventTypes
from ge_env import GrandExchangeEnv
from ppo_agent import PPOAgent
from volume_analysis import VolumeAnalyzer, create_volume_analyzer
from market_order_manager import MarketOrderManager

class MockHistoricalClient:
    """Mock historical client for testing"""
    
    def __init__(self, id_to_name_map, name_to_id_map):
        self._id_to_name_map = id_to_name_map
        self._name_to_id_map = name_to_id_map
        self.latest_data = {}
        
    def get_latest(self):
        """Get latest price data"""
        return self.latest_data
        
    def advance(self):
        """Advance to next data point (no-op in mock)"""
        pass
        
    def update_prices(self, prices):
        """Update the latest prices"""
        self.latest_data = {
            self._name_to_id_map[item]: {"high": price, "low": int(price * 0.95)}
            for item, price in prices.items()
            if item in self._name_to_id_map
        }

async def run_simulation():
    """Run a simulation of the PPO agent with websocket integration"""
    logger.info("Starting PPO WebSocket integration simulation")
    
    # Create test items for GE environment
    test_items = {
        "Abyssal whip": {
            "base_price": 1500000,
            "min_price": 1000000,
            "max_price": 2000000,
            "buy_limit": 10
        },
        "Dragon bones": {
            "base_price": 3000,
            "min_price": 2000,
            "max_price": 4000,
            "buy_limit": 100
        },
        "Nature rune": {
            "base_price": 250,
            "min_price": 200,
            "max_price": 300,
            "buy_limit": 1000
        },
        "Cannonball": {
            "base_price": 200,
            "min_price": 150,
            "max_price": 250,
            "buy_limit": 5000
        },
        "Zulrah's scales": {
            "base_price": 180,
            "min_price": 150,
            "max_price": 220,
            "buy_limit": 20000
        }
    }
    
    # Load actual mappings from file
    try:
        with open("endpoints/mapping.txt", 'r') as f:
            mapping_data = json.load(f)
            
        # Create ID to name mapping
        id_to_name_map = {}
        name_to_id_map = {}
        for item in mapping_data:
            item_id = str(item.get('id'))
            item_name = item.get('name')
            if item_id and item_name:
                id_to_name_map[item_id] = item_name
                name_to_id_map[item_name] = item_id
        
        logger.info(f"Loaded mapping data for {len(id_to_name_map)} items")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load mapping data: {e}")
        # Fallback to mock mappings
        id_to_name_map = {
            "1": "Abyssal whip",
            "2": "Dragon bones",
            "3": "Nature rune",
            "4": "Cannonball",
            "5": "Zulrah's scales"
        }
        name_to_id_map = {v: k for k, v in id_to_name_map.items()}
    
    # Create a volume analyzer
    volume_analyzer = VolumeAnalyzer(id_to_name_map, name_to_id_map)
    
    # Add some test data to volume analyzer
    current_time = int(time.time())
    volume_analyzer.volume_history_1h = {
        "1": [(current_time, 100, 50)],  # timestamp, high_vol, low_vol
        "2": [(current_time, 200, 150)],
        "3": [(current_time, 300, 250)],
        "4": [(current_time, 1000, 800)],
        "5": [(current_time, 5000, 4000)]
    }
    volume_analyzer.price_history_1h = {
        "1": [(current_time, 1500000, 1450000)],  # timestamp, high_price, low_price
        "2": [(current_time, 3000, 2900)],
        "3": [(current_time, 250, 240)],
        "4": [(current_time, 200, 190)],
        "5": [(current_time, 180, 170)]
    }
    
    # Create a mock historical client
    historical_client = MockHistoricalClient(id_to_name_map, name_to_id_map)
    historical_client.update_prices({
        "Abyssal whip": 1500000,
        "Dragon bones": 3000,
        "Nature rune": 250,
        "Cannonball": 200,
        "Zulrah's scales": 180
    })
    
    # Create a GE environment
    env = GrandExchangeEnv(
        items=test_items,
        starting_gp=10000000,
        tick_duration=5,
        max_ticks=1000,
        price_fluctuation_pct=0.01,
        buy_limit_reset_ticks=240,
        enable_margin_experimentation=True,
        max_margin_attempts=10,
        min_margin_pct=0.8,
        max_margin_pct=0.40,
        margin_wait_steps=3,
        volume_analyzer=volume_analyzer,
        historical_client=historical_client,
        name_to_id_map=name_to_id_map,
        id_to_name_map=id_to_name_map
    )
    
    # Reset the environment
    obs = env.reset()
    
    # Create a PPO agent
    item_list = list(test_items.keys())
    price_ranges = {item: (data["min_price"], data["max_price"]) 
                   for item, data in test_items.items()}
    buy_limits = {item: data["buy_limit"] for item, data in test_items.items()}
    
    agent = PPOAgent(
        item_list=item_list,
        price_ranges=price_ranges,
        buy_limits=buy_limits,
        device="cpu",
        hidden_size=128,
        price_bins=10,
        quantity_bins=10,
        wait_steps_bins=10,
        volume_analyzer=volume_analyzer
    )
    
    # Load a pre-trained model if available, otherwise use random initialization
    try:
        agent.load_all("test_models/agent_0.pth")
        logger.info("Loaded pre-trained model")
    except Exception as e:
        logger.warning(f"Could not load pre-trained model: {e}")
        logger.info("Using randomly initialized model")
    
    # Create a PPO WebSocket Integration instance
    integration = PPOWebSocketIntegration(
        websocket_url="http://localhost:5178",
        max_slots=8
    )
    
    # Connect to the WebSocket server (this will be mocked)
    with open("ppo_websocket_test_events.json", "w") as f:
        f.write("[]")  # Initialize empty events array
        
    # Mock the connect and submit_event methods
    original_connect = integration.connect
    original_submit_event = integration.submit_event
    
    async def mock_connect():
        integration.session = "mock_session"
        integration.connected = True
        logger.info("Mock connection established")
        return True
        
    async def mock_submit_event(event_type, data, priority=0):
        # Save event to a file for inspection
        try:
            with open("ppo_websocket_test_events.json", "r") as f:
                events = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            events = []
            
        event = {
            "type": event_type,
            "data": data,
            "timestamp": int(time.time()),
            "priority": priority
        }
        events.append(event)
        
        with open("ppo_websocket_test_events.json", "w") as f:
            json.dump(events, f, indent=2)
            
        logger.debug(f"Submitted event: {event_type}")
        return True
    
    integration.connect = mock_connect
    integration.submit_event = mock_submit_event
    
    # Connect to the WebSocket server
    await integration.connect()
    
    # Hook the agent and environment
    integration.hook_agent(agent, "agent_0")
    integration.hook_env(env, "env_0")
    
    # Run the simulation
    logger.info("Starting simulation")
    
    # Track metrics
    total_reward = 0
    episode_rewards = []
    trades_executed = 0
    
    # Run for a fixed number of steps
    num_steps = 100
    for step in range(num_steps):
        # Get action from agent
        action, _ = agent.sample_action(obs)
        
        # Take step in environment
        next_obs, reward, done, info = env.step(action)
        
        # Update metrics
        total_reward += reward
        episode_rewards.append(reward)
        
        if action['type'] in ['buy', 'sell']:
            trades_executed += 1
            
        # Log step information
        logger.info(f"Step {step}: Action={action['type']}, Item={action['item']}, " +
                   f"Price={action['price']}, Quantity={action['quantity']}, Reward={reward:.4f}")
        
        # Update observation
        obs = next_obs
        
        # Simulate price changes
        if step % 10 == 0:
            # Update prices with some random fluctuations
            new_prices = {}
            for item in test_items:
                base_price = test_items[item]['base_price']
                fluctuation = np.random.uniform(-0.05, 0.05)  # -5% to +5%
                new_price = int(base_price * (1 + fluctuation))
                new_prices[item] = new_price
                
            # Update historical client
            historical_client.update_prices(new_prices)
            
            # Update volume analyzer
            current_time = int(time.time())
            for item_id, item_name in id_to_name_map.items():
                high_vol = np.random.randint(50, 500)
                low_vol = np.random.randint(50, 500)
                high_price = new_prices[item_name]
                low_price = int(high_price * 0.95)
                
                if item_id not in volume_analyzer.volume_history_1h:
                    volume_analyzer.volume_history_1h[item_id] = []
                if item_id not in volume_analyzer.price_history_1h:
                    volume_analyzer.price_history_1h[item_id] = []
                    
                volume_analyzer.volume_history_1h[item_id].append((current_time, high_vol, low_vol))
                volume_analyzer.price_history_1h[item_id].append((current_time, high_price, low_price))
        
        # Break if episode is done
        if done:
            logger.info("Episode finished")
            break
    
    # Unhook the agent and environment
    integration.unhook_agent("agent_0")
    integration.unhook_env("env_0")
    
    # Disconnect from the WebSocket server
    await integration.disconnect()
    
    # Restore original methods
    integration.connect = original_connect
    integration.submit_event = original_submit_event
    
    # Print summary
    logger.info("=== Simulation Summary ===")
    logger.info(f"Steps: {num_steps}")
    logger.info(f"Total reward: {total_reward:.4f}")
    logger.info(f"Average reward per step: {total_reward / num_steps:.4f}")
    logger.info(f"Trades executed: {trades_executed}")
    logger.info(f"Final GP: {obs['gp']}")
    logger.info(f"Final inventory: {obs['inventory']}")
    
    # Calculate total portfolio value
    portfolio_value = obs['gp']
    for item, quantity in obs['inventory'].items():
        portfolio_value += quantity * obs['prices'][item]
    logger.info(f"Final portfolio value: {portfolio_value}")
    
    # Return summary data
    return {
        "steps": num_steps,
        "total_reward": total_reward,
        "avg_reward": total_reward / num_steps,
        "trades_executed": trades_executed,
        "final_gp": obs['gp'],
        "final_inventory": obs['inventory'],
        "portfolio_value": portfolio_value
    }

def analyze_events():
    """Analyze the events captured during simulation"""
    try:
        with open("ppo_websocket_test_events.json", "r") as f:
            events = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        logger.error("No events file found or invalid JSON")
        return
    
    # Count events by type
    event_counts = {}
    for event in events:
        event_type = event["type"]
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
    
    logger.info("=== Event Analysis ===")
    logger.info(f"Total events: {len(events)}")
    for event_type, count in event_counts.items():
        logger.info(f"{event_type}: {count}")
    
    # Analyze trade events
    trade_events = [e for e in events if e["type"] == EventTypes.TRADE_EXECUTED]
    if trade_events:
        total_profit = sum(e["data"].get("profit", 0) for e in trade_events)
        logger.info(f"Total profit from trades: {total_profit}")
        
        # Count trades by item
        item_counts = {}
        for event in trade_events:
            item = event["data"]["item"]
            item_counts[item] = item_counts.get(item, 0) + 1
        
        logger.info("Trades by item:")
        for item, count in item_counts.items():
            logger.info(f"{item}: {count}")
    
    # Analyze margin updates
    margin_events = [e for e in events if e["type"] == EventTypes.MARGIN_UPDATE]
    if margin_events:
        avg_margin = sum(e["data"]["margin"] for e in margin_events) / len(margin_events)
        logger.info(f"Average margin: {avg_margin}")
        
        # Count margin updates by item
        item_counts = {}
        for event in margin_events:
            item = event["data"]["item"]
            item_counts[item] = item_counts.get(item, 0) + 1
        
        logger.info("Margin updates by item:")
        for item, count in item_counts.items():
            logger.info(f"{item}: {count}")
    
    return event_counts

if __name__ == "__main__":
    # Run the simulation
    loop = asyncio.get_event_loop()
    summary = loop.run_until_complete(run_simulation())
    
    # Analyze the events
    event_counts = analyze_events()
    
    print("\nSimulation completed successfully!")
    print(f"Total reward: {summary['total_reward']:.4f}")
    print(f"Trades executed: {summary['trades_executed']}")
    print(f"Final portfolio value: {summary['portfolio_value']}")