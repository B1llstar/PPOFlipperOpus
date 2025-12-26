import asyncio
import logging
import os
import time
import torch
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("real_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("real_trading")

# Import required modules
from ge_env import GrandExchangeEnv
from ppo_agent import PPOAgent
from config import ENV_KWARGS, PPO_KWARGS
from real_time_ge_client import RealTimeGrandExchangeClient
from ppo_websocket_integration import PPOWebSocketIntegration
from ge_rest_client import GrandExchangeClient
from shared_knowledge import SharedKnowledgeRepository
from volume_analysis import VolumeAnalyzer, create_volume_analyzer

async def run_real_trading():
    """
    Run the PPO agent in real trading mode.
    This function initializes the real-time client, sets up the environment,
    loads the trained PPO agent, and runs it in real trading mode.
    """
    logger.info("Starting real trading mode")
    
    # Set device for PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize WebSocket integration
    websocket_url = "http://localhost:5178"  # Default WebSocket server URL
    websocket_integration = PPOWebSocketIntegration(websocket_url=websocket_url)
    await websocket_integration.connect()
    logger.info(f"Connected to WebSocket server at {websocket_url}")
    
    # Initialize real-time client with WebSocket integration
    real_time_client = RealTimeGrandExchangeClient(
        websocket_integration=websocket_integration,
        update_interval=300,  # 5 minutes in seconds
        cache_size=128,
        max_retries=3,
        backoff_factor=1.0
    )
    await real_time_client.start()
    logger.info("Real-time client started")
    
    # Send system status event
    await websocket_integration.system_status(
        "real_trading_started",
        "Real trading mode started"
    )
    
    # Load item mappings
    id_name_map, buy_limits_map = read_mapping_file()
    
    # Initialize REST client for initial data
    rest_client = GrandExchangeClient()
    
    # Fetch marketplace data
    marketplace_data = fetch_marketplace_data(rest_client)
    
    # Build items dictionary
    items = build_items_dict(id_name_map, buy_limits_map, marketplace_data)
    item_list, price_ranges, buy_limits = get_item_lists(items)
    
    # Initialize volume analyzer
    volume_analyzer = initialize_volume_analyzer(rest_client, id_name_map)
    
    # Create environment with real trading mode enabled
    env_kwargs = dict(ENV_KWARGS)
    env_kwargs["items"] = items
    env_kwargs["real_trading_mode"] = True
    env_kwargs["real_trading_client"] = real_time_client
    env_kwargs["volume_analyzer"] = volume_analyzer
    
    # Create environment
    env = GrandExchangeEnv(**env_kwargs)
    logger.info("Environment created with real trading mode enabled")
    
    # Initialize shared knowledge repository
    shared_knowledge = SharedKnowledgeRepository(id_name_map, {name: id for id, name in id_name_map.items()})
    
    # Load trained agent
    agent = load_trained_agent(
        item_list=item_list,
        price_ranges=price_ranges,
        buy_limits=buy_limits,
        device=device,
        volume_analyzer=volume_analyzer,
        shared_knowledge=shared_knowledge
    )
    logger.info("Trained agent loaded")
    
    # Hook agent and environment to WebSocket integration
    websocket_integration.hook_agent(agent, agent_id="real_trading_agent")
    websocket_integration.hook_env(env, env_id="real_trading_env")
    logger.info("Agent and environment hooked to WebSocket integration")
    
    # Reset environment to start trading
    obs = env.reset()
    logger.info(f"Environment reset. Starting GP: {obs['gp']}")
    
    # Main trading loop
    try:
        logger.info("Starting real trading loop")
        
        # Send initial portfolio update
        total_value = obs['gp'] + sum(obs['prices'][item] * obs['inventory'][item] for item in obs['inventory'])
        await websocket_integration.portfolio_update(
            agent_id="real_trading_agent",
            gp=obs['gp'],
            inventory=obs['inventory'],
            total_value=total_value
        )
        
        # Trading loop
        while True:
            # Sample action from agent
            action, _ = agent.sample_action(obs)
            
            # Log action
            logger.info(f"Agent action: {action['type']} | Item: {action['item']} | Price: {action['price']} | Quantity: {action['quantity']}")
            
            # Execute action in environment
            next_obs, reward, done, info = env.step(action)
            
            # Log step result
            logger.info(f"Step result: Reward: {reward:.2f} | Done: {done} | Info: {info}")
            
            # Update observation
            obs = next_obs
            
            # Log portfolio status
            total_value = obs['gp'] + sum(obs['prices'][item] * obs['inventory'][item] for item in obs['inventory'])
            logger.info(f"Portfolio: GP: {obs['gp']} | Total value: {total_value}")
            
            # Wait for next trading interval (5 minutes)
            await asyncio.sleep(300)  # 5 minutes in seconds
            
            # Update market data
            await env.update_real_market_data()
            
            # Check for any completed orders
            env._update_real_orders()
            
    except KeyboardInterrupt:
        logger.info("Trading interrupted by user")
    except Exception as e:
        logger.error(f"Error in trading loop: {e}", exc_info=True)
    finally:
        # Clean up
        websocket_integration.unhook_agent("real_trading_agent")
        websocket_integration.unhook_env("real_trading_env")
        await real_time_client.stop()
        await websocket_integration.disconnect()
        logger.info("Real trading mode stopped")

def read_mapping_file():
    """Read the mapping file to get item IDs, names, and buy limits."""
    mapping_path = "endpoints/mapping.txt"
    id_name_map = {}
    buy_limits_map = {}
    
    try:
        import json
        with open(mapping_path, 'r') as f:
            mapping_data = json.load(f)
            for item in mapping_data:
                item_id = str(item.get('id'))
                item_name = item.get('name')
                if item_id and item_name:
                    id_name_map[item_id] = item_name
                    # Buy limits might be in a different format, adjust as needed
                    buy_limits_map[item_id] = item.get('limit', 5000)  # Default to 5000 if not specified
        logger.info(f"Loaded mapping data for {len(id_name_map)} items")
    except Exception as e:
        logger.error(f"Error reading mapping file: {e}")
        # Provide some minimal default data if file can't be read
        id_name_map = {"554": "Fire rune", "555": "Water rune", "556": "Air rune"}
        buy_limits_map = {"554": 10000, "555": 10000, "556": 10000}
    
    return id_name_map, buy_limits_map

def fetch_marketplace_data(client):
    """Fetch marketplace data from the GrandExchangeClient."""
    try:
        # Get latest prices
        latest_data = client.get_latest()
        logger.info(f"Fetched latest marketplace data for {len(latest_data)} items")
        return latest_data
    except Exception as e:
        logger.error(f"Error fetching marketplace data: {e}")
        # Return minimal default data if API fails
        return {
            "554": {"high": 5, "low": 4, "highTime": int(time.time()), "lowTime": int(time.time())},
            "555": {"high": 5, "low": 4, "highTime": int(time.time()), "lowTime": int(time.time())},
            "556": {"high": 5, "low": 4, "highTime": int(time.time()), "lowTime": int(time.time())}
        }

def build_items_dict(id_name_map, buy_limits_map, marketplace_data):
    """Build a dictionary of items using mapping data and marketplace data."""
    # Read strict mode and high volume items path directly from config
    strict_mode = ENV_KWARGS.get("strict_mode", False)
    high_vol_items_path = ENV_KWARGS.get("high_vol_items_path", "high_vol_items.txt")

    items = {}

    # Read high volume items if strict mode is enabled
    high_vol_items = set()
    if strict_mode:
        try:
            with open(high_vol_items_path, 'r') as f:
                high_vol_items = {line.strip() for line in f if line.strip()}
            logger.info(f"Strict mode enabled: Loaded {len(high_vol_items)} high volume items from {high_vol_items_path}")
        except Exception as e:
            logger.error(f"Error reading high volume items file {high_vol_items_path}: {e}")

    # First build from marketplace data
    for item_id, item_name in id_name_map.items():
        # In strict mode, only include items from the high volume list
        if strict_mode and item_name not in high_vol_items:
            continue

        if item_id in marketplace_data:
            market_info = marketplace_data[item_id]
            high_price = market_info.get('high', 100)
            low_price = market_info.get('low', 90)
            
            # Calculate min and max price based on high/low with some margin
            min_price = max(1, int(low_price * 0.9))
            max_price = max(min_price + 1, int(high_price * 1.1))  # Ensure max_price > min_price
            
            # Get buy limit from mapping or use default
            buy_limit = buy_limits_map.get(item_id, 5000)
            
            # Ensure base_price is never 0
            base_price = max(1, (high_price + low_price) // 2)
            items[item_name] = {
                'base_price': base_price,
                'buy_limit': buy_limit,
                'min_price': min_price,
                'max_price': max_price
            }
    
    # Ensure we have at least some items
    if not items and not strict_mode:
        items = {
            "Fire rune": {
                "base_price": 5,
                "buy_limit": 10000,
                "min_price": 4,
                "max_price": 6,
            },
            "Water rune": {
                "base_price": 5,
                "buy_limit": 10000,
                "min_price": 4,
                "max_price": 6,
            },
            "Air rune": {
                "base_price": 5,
                "buy_limit": 10000,
                "min_price": 4,
                "max_price": 6,
            }
        }
    
    # Log the number of items after all filtering and processing
    logger.info(f"Final items dictionary built with {len(items)} items (Strict Mode: {strict_mode})")
    
    # Take a subset of the items for real trading to manage risk
    if items:
        import random
        num_items_to_select = max(1, int(len(items) * 0.05))  # 5% of items for real trading
        selected_keys = random.sample(list(items.keys()), num_items_to_select)
        items = {k: items[k] for k in selected_keys}
        logger.info(f"Selected subset of {len(items)} items for real trading")
    
    return items

def get_item_lists(items):
    """Extract item lists, price ranges, and buy limits from items dictionary."""
    item_list = list(items.keys())
    price_ranges = {item: (items[item]['min_price'], items[item]['max_price']) for item in item_list}
    buy_limits = {item: items[item]['buy_limit'] for item in item_list}
    return item_list, price_ranges, buy_limits

def initialize_volume_analyzer(client, id_to_name_map):
    """Initialize volume analyzer with initial data."""
    # Create volume analyzer
    volume_analyzer = create_volume_analyzer(id_to_name_map)
    
    # Update volume analyzer with initial data
    try:
        data_5m = client.get_5m()
        data_1h = client.get_1h()
        volume_analyzer.update_volume_data(data_5m, data_1h)
        logger.info(f"Initialized volume analyzer with {len(data_5m)} 5m items and {len(data_1h)} 1h items")
    except Exception as e:
        logger.error(f"Error initializing volume analyzer: {e}")
    
    return volume_analyzer

def load_trained_agent(item_list, price_ranges, buy_limits, device, volume_analyzer=None, shared_knowledge=None):
    """Load the trained PPO agent from saved state."""
    # Create agent with parameters from config
    agent = PPOAgent(
        item_list=item_list,
        price_ranges=price_ranges,
        buy_limits=buy_limits,
        device=device,
        hidden_size=PPO_KWARGS["hidden_size"],
        price_bins=PPO_KWARGS["price_bins"],
        quantity_bins=PPO_KWARGS["quantity_bins"],
        wait_steps_bins=PPO_KWARGS["wait_steps_bins"],
        lr=PPO_KWARGS["lr"],
        volume_analyzer=volume_analyzer,
        shared_knowledge=shared_knowledge
    )
    
    # Try to load best model
    base_save_dir = "agent_states"
    best_actor_path = os.path.join(base_save_dir, "actor_best.pth")
    best_critic_path = os.path.join(base_save_dir, "critic_best.pth")
    
    if os.path.exists(best_actor_path) and os.path.exists(best_critic_path):
        try:
            agent.load_actor(best_actor_path)
            agent.load_critic(best_critic_path)
            logger.info(f"Loaded best agent from {best_actor_path} and {best_critic_path}")
            return agent
        except Exception as e:
            logger.error(f"Error loading best agent: {e}")
    
    # If best model not found, try to find the latest model
    latest_step = 0
    latest_actor_path = None
    latest_critic_path = None
    
    # Check all agent directories
    for agent_idx in range(5):  # Assuming up to 5 agents
        agent_dir = os.path.join(base_save_dir, f"agent{agent_idx + 1}")
        if os.path.exists(agent_dir):
            # Find latest step directory
            step_dirs = []
            for dirname in os.listdir(agent_dir):
                if dirname.startswith("step_"):
                    step_num = int(dirname.split("_")[1])
                    step_dirs.append((step_num, dirname))
            
            if step_dirs:
                step_dirs.sort()  # Sort by step number
                step_num, step_dir = step_dirs[-1]  # Get latest
                
                if step_num > latest_step:
                    latest_step = step_num
                    latest_actor_path = os.path.join(agent_dir, step_dir, "actor.pth")
                    latest_critic_path = os.path.join(agent_dir, step_dir, "critic.pth")
    
    # Load latest model if found
    if latest_actor_path and latest_critic_path and os.path.exists(latest_actor_path) and os.path.exists(latest_critic_path):
        try:
            agent.load_actor(latest_actor_path)
            agent.load_critic(latest_critic_path)
            logger.info(f"Loaded latest agent from {latest_actor_path} and {latest_critic_path}")
            return agent
        except Exception as e:
            logger.error(f"Error loading latest agent: {e}")
    
    # If no model found, return the newly created agent
    logger.warning("No saved agent found, using newly initialized agent")
    return agent

if __name__ == "__main__":
    # Run the real trading mode
    asyncio.run(run_real_trading())