import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("run_ppo_websocket.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("run_ppo_websocket")

# Import PPO components
try:
    from ge_env import GrandExchangeEnv
    from ppo_agent import PPOAgent
    from shared_knowledge import SharedKnowledgeRepository
    from volume_analysis import VolumeAnalyzer, create_volume_analyzer
    from ge_rest_client import GrandExchangeClient
    from ppo_websocket_integration import PPOWebSocketIntegration
    logger.info("Successfully imported PPO components")
except ImportError as e:
    logger.error(f"Error importing PPO components: {str(e)}")
    raise

# Configuration
WEBSOCKET_PORT = 5178
API_PORT = 6928  # Default API server port
NUM_AGENTS = 1  # Number of agents to create
SIMULATION_STEPS = 100  # Number of steps to simulate


async def start_websocket_server():
    """Start the WebSocket server as a subprocess."""
    logger.info("Starting WebSocket server...")
    
    # Start the server
    process = subprocess.Popen(
        [sys.executable, "websocket_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to start
    logger.info("Waiting for WebSocket server to start...")
    await asyncio.sleep(2)
    
    # Check if process is still running
    if process.poll() is not None:
        stdout, stderr = process.communicate()
        logger.error(f"WebSocket server failed to start: {stderr}")
        return None
    
    logger.info(f"WebSocket server started on port {WEBSOCKET_PORT}")
    return process


def load_item_data():
    """Load item data from the mapping file."""
    logger.info("Loading item data...")
    
    try:
        # Load mapping file
        import json
        with open("endpoints/mapping.txt", "r") as f:
            mapping_data = json.load(f)
        
        # Extract item data
        id_to_name_map = {}
        name_to_id_map = {}
        buy_limits = {}
        
        for item in mapping_data:
            if "id" in item and "name" in item:
                item_id = str(item["id"])
                item_name = item["name"]
                id_to_name_map[item_id] = item_name
                name_to_id_map[item_name] = item_id
                
                # Extract buy limit if available
                if "limit" in item:
                    buy_limits[item_name] = item["limit"]
        
        logger.info(f"Loaded {len(id_to_name_map)} items from mapping file")
        return id_to_name_map, name_to_id_map, buy_limits
    
    except Exception as e:
        logger.error(f"Error loading item data: {str(e)}")
        
        # Return minimal default data
        id_to_name_map = {"554": "Fire rune", "555": "Water rune", "556": "Air rune"}
        name_to_id_map = {"Fire rune": "554", "Water rune": "555", "Air rune": "556"}
        buy_limits = {"Fire rune": 10000, "Water rune": 10000, "Air rune": 10000}
        
        return id_to_name_map, name_to_id_map, buy_limits


def create_environment(id_to_name_map, name_to_id_map, items):
    """Create a GrandExchangeEnv instance."""
    logger.info("Creating environment...")
    
    try:
        # Create client
        client = GrandExchangeClient()
        
        # Create volume analyzer
        volume_analyzer = create_volume_analyzer(id_to_name_map)
        
        # Update volume analyzer with data
        data_5m = client.get_5m()
        data_1h = client.get_1h()
        volume_analyzer.update_volume_data(data_5m, data_1h)
        
        # Create environment
        env = GrandExchangeEnv(
            items=items,
            starting_gp=10000000,  # 10M starting GP
            tick_duration=5,
            max_ticks=1440,
            price_fluctuation_pct=0.01,
            buy_limit_reset_ticks=240,
            random_seed=42,
            high_vol_items_path="high_vol_items.txt",
            historical_client=None,
            name_to_id_map=name_to_id_map,
            id_to_name_map=id_to_name_map,
            enable_margin_experimentation=True,
            volume_analyzer=volume_analyzer
        )
        
        logger.info("Environment created")
        return env, volume_analyzer
    
    except Exception as e:
        logger.error(f"Error creating environment: {str(e)}")
        raise


def create_agents(num_agents, item_list, price_ranges, buy_limits, volume_analyzer):
    """Create PPO agents."""
    logger.info(f"Creating {num_agents} agents...")
    
    agents = []
    
    try:
        # Create shared knowledge repository
        shared_knowledge = SharedKnowledgeRepository(
            id_to_name_map={v: k for k, v in name_to_id_map.items()},
            name_to_id_map=name_to_id_map
        )
        
        # Create agents
        for i in range(num_agents):
            agent = PPOAgent(
                item_list=item_list,
                price_ranges=price_ranges,
                buy_limits=buy_limits,
                device="cpu",
                hidden_size=128,
                price_bins=10,
                quantity_bins=10,
                wait_steps_bins=10,
                lr=3e-4,
                volume_blacklist=None,
                volume_analyzer=volume_analyzer,
                shared_knowledge=shared_knowledge,
                agent_id=f"agent_{i}"
            )
            
            agents.append(agent)
            logger.info(f"Created agent {i}")
        
        logger.info(f"Created {len(agents)} agents")
        return agents, shared_knowledge
    
    except Exception as e:
        logger.error(f"Error creating agents: {str(e)}")
        raise


def build_items_dict(marketplace_data, buy_limits):
    """Build a dictionary of items using marketplace data."""
    logger.info("Building items dictionary...")
    
    items = {}
    
    try:
        # Process marketplace data
        for item_id, item_data in marketplace_data.items():
            if item_id in id_to_name_map:
                item_name = id_to_name_map[item_id]
                
                # Get high and low prices
                high_price = item_data.get("high", 100)
                low_price = item_data.get("low", 90)
                
                # Calculate min and max price
                min_price = max(1, int(low_price * 0.9))
                max_price = max(min_price + 1, int(high_price * 1.1))
                
                # Get buy limit
                buy_limit = buy_limits.get(item_name, 5000)
                
                # Calculate base price
                base_price = max(1, (high_price + low_price) // 2)
                
                # Add to items dictionary
                items[item_name] = {
                    "base_price": base_price,
                    "buy_limit": buy_limit,
                    "min_price": min_price,
                    "max_price": max_price
                }
        
        logger.info(f"Built items dictionary with {len(items)} items")
        
        # Take a subset of items for faster simulation
        import random
        if len(items) > 10:
            selected_items = random.sample(list(items.keys()), 10)
            items = {k: items[k] for k in selected_items}
            logger.info(f"Selected subset of {len(items)} items for simulation")
        
        return items
    
    except Exception as e:
        logger.error(f"Error building items dictionary: {str(e)}")
        
        # Return minimal default data
        return {
            "Fire rune": {
                "base_price": 5,
                "buy_limit": 10000,
                "min_price": 4,
                "max_price": 6
            },
            "Water rune": {
                "base_price": 5,
                "buy_limit": 10000,
                "min_price": 4,
                "max_price": 6
            },
            "Air rune": {
                "base_price": 5,
                "buy_limit": 10000,
                "min_price": 4,
                "max_price": 6
            }
        }


def get_item_lists(items):
    """Get item lists from items dictionary."""
    item_list = list(items.keys())
    price_ranges = {item: (items[item]['min_price'], items[item]['max_price']) for item in item_list}
    buy_limits = {item: items[item]['buy_limit'] for item in item_list}
    return item_list, price_ranges, buy_limits


async def run_simulation(env, agents, integration, steps=100):
    """Run a simulation with the environment and agents."""
    logger.info(f"Running simulation for {steps} steps...")
    
    # Reset environment
    obs = env.reset()
    
    # Run simulation
    for step in range(steps):
        # For each agent
        for i, agent in enumerate(agents):
            # Sample action
            action, _ = agent.sample_action(obs)
            
            # Take step
            next_obs, reward, done, info = env.step(action)
            
            # Update observation
            obs = next_obs
            
            # Log step
            if step % 10 == 0:
                logger.info(f"Step {step}: Agent {i} took action {action['type']} for item {action['item']} at price {action['price']} with quantity {action['quantity']}")
                logger.info(f"Step {step}: Agent {i} has GP: {obs['gp']}")
        
        # Small delay to avoid overwhelming the WebSocket server
        await asyncio.sleep(0.1)
    
    logger.info("Simulation completed")


async def main():
    # Start WebSocket server
    server_process = await start_websocket_server()
    if not server_process:
        return
    
    try:
        # Load item data
        global id_to_name_map, name_to_id_map
        id_to_name_map, name_to_id_map, buy_limits = load_item_data()
        
        # Get marketplace data
        client = GrandExchangeClient()
        marketplace_data = client.get_latest()
        
        # Build items dictionary
        items = build_items_dict(marketplace_data, buy_limits)
        
        # Get item lists
        item_list, price_ranges, buy_limits = get_item_lists(items)
        
        # Create environment
        env, volume_analyzer = create_environment(id_to_name_map, name_to_id_map, items)
        
        # Create agents
        agents, shared_knowledge = create_agents(NUM_AGENTS, item_list, price_ranges, buy_limits, volume_analyzer)
        
        # Create integration
        integration = PPOWebSocketIntegration(websocket_url=f"http://localhost:{WEBSOCKET_PORT}")
        
        # Connect to WebSocket server
        await integration.connect()
        
        # Hook agents and environment
        for i, agent in enumerate(agents):
            integration.hook_agent(agent, f"agent_{i}")
        
        integration.hook_env(env, "env_0")
        
        # Run simulation
        await run_simulation(env, agents, integration, SIMULATION_STEPS)
        
        # Unhook all components
        integration.unhook_all()
        
        # Disconnect from WebSocket server
        await integration.disconnect()
        
        # Send system status event
        await integration.system_status("simulation_completed", "Simulation completed")
    
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
    
    finally:
        # Stop WebSocket server
        if server_process:
            logger.info("Stopping WebSocket server...")
            server_process.terminate()
            server_process.wait()
            logger.info("WebSocket server stopped")


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())