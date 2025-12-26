import asyncio
import argparse
import logging
import json
import time
import os
from datetime import datetime
from typing import Dict, Any, Optional

from ge_rest_client import GrandExchangeClient

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("real_marketplace_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("real_marketplace_trading")

# Import components
try:
    from ppo_agent import PPOAgent
    from ge_env import GrandExchangeEnv
    from ppo_inferencing_client import PPOWebSocketIntegration
    from real_trading_config import RealTradingConfig
    from ge_rest_client import GrandExchangeClient
    from config import ENV_KWARGS, PPO_KWARGS
    logger.info("Successfully imported components")
except ImportError as e:
    logger.error(f"Error importing components: {str(e)}")
    raise

# Constants
DEFAULT_MODEL_PATH = "C:/Users/B1llstar/Documents/Github/PPOFlipper/PPOFlipperTemplate/PPOFlipper/agent_states/best_model.pth"
MAX_LOSS_PERCENT = 0.10  # 10% maximum loss before stopping
RECONNECT_ATTEMPTS = 3
HEALTH_CHECK_INTERVAL = 60  # 1 minute
REST_API_URL = "https://prices.runescape.wiki/api/v1/osrs"  # Default REST API URL

# Add detailed docstring explaining how the script works
"""
Run Real Marketplace Trading Script

This script connects the PPO agent to a real-time marketplace through WebSocket and REST API.
It works as follows:

1. Initialization:
   - Loads configuration from real_trading_config.py
   - Checks if real trading mode is enabled
   - Creates the environment with real_trading_mode=True
   - Loads the PPO agent model
   - Creates the marketplace integration

2. Main Loop:
   - The script starts the integration, which begins the agent loop
   - The agent loop runs in the background, taking steps every 5 minutes
   - Each step involves:
     - Fetching latest market data from the REST client
     - Having the agent make a decision based on the data
     - Executing the decision through the WebSocket client
   - The script also performs health checks to monitor performance

3. Shutdown:
   - When the script is interrupted (Ctrl+C) or encounters an error
   - It stops the integration, which:
     - Cancels any pending orders
     - Disconnects from the WebSocket server
     - Logs final performance metrics

The script processes real-time 5-minute data from the REST client by:
1. Fetching the latest data every 5 minutes
2. Updating the environment's prices and other market metrics
3. Using this data for agent decision-making
4. Sending relevant updates through the WebSocket client
"""

async def run_real_trading(args):
    """
    Run real-time trading with the marketplace.
    
    Args:
        args: Command-line arguments
    """
    # Load configuration
    config_manager = RealTradingConfig()
    
    # Check if real trading mode is enabled
    if not config_manager.is_real_trading_enabled() and not args.force_enable:
        logger.error("Real trading mode is not enabled. Use --force-enable to override.")
        return
    
    # If force enable is specified, enable real trading mode
    if args.force_enable and not config_manager.is_real_trading_enabled():
        logger.info("Forcing real trading mode to be enabled")
        config_manager.enable_real_trading()
    
    # Update configuration if specified in arguments
    if args.websocket_url:
        config_manager.set_websocket_url(args.websocket_url)
    
    if args.max_slots:
        config_manager.set_max_slots(args.max_slots)
    
    if args.update_interval:
        config_manager.set_update_interval(args.update_interval)
    
    if args.agent_id:
        config_manager.set_agent_id(args.agent_id)
    
    # Get configuration
    websocket_url = config_manager.get_websocket_url()
    max_slots = config_manager.get_max_slots()
    experiment_timeout = config_manager.get_experiment_timeout()
    final_timeout = config_manager.get_final_timeout()
    update_interval = config_manager.get_update_interval()
    agent_id = config_manager.get_agent_id()
    
    # Log configuration
    logger.info("="*80)
    logger.info(f"Starting real marketplace trading at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Configuration:")
    logger.info(f"  WebSocket URL: {websocket_url}")
    logger.info(f"  Max Slots: {max_slots}")
    logger.info(f"  Experiment Timeout: {experiment_timeout}s")
    logger.info(f"  Final Timeout: {final_timeout}s")
    logger.info(f"  Update Interval: {update_interval}s")
    logger.info(f"  Agent ID: {agent_id}")
    logger.info(f"  Model Path: {args.model_path}")
    logger.info("="*80)
    
    # Create REST client first to get items data
    rest_client = GrandExchangeClient()
    
    # Get latest data and mapping
    try:
        latest_data = rest_client.get_latest()
        mapping = rest_client.get_mapping()
        
        # Build name to ID mappings
        name_to_id = {item['name']: str(item['id']) for item in mapping}
        id_to_name = {str(item['id']): item['name'] for item in mapping}
        
        # Build items dictionary
        items = {}
        for item_data in mapping:
            item_id = str(item_data['id'])
            if item_id in latest_data:
                price_data = latest_data[item_id]
                items[item_data['name']] = {
                    'base_price': price_data.get('high', 1),
                    'buy_limit': 10000,  # Default buy limit
                    'min_price': int(price_data.get('high', 1) * 0.9),
                    'max_price': int(price_data.get('high', 1) * 1.1)
                }
        
        # Set mappings in REST client
        rest_client._name_to_id_map = name_to_id
        rest_client._id_to_name_map = id_to_name
        
        logger.info(f"Loaded {len(items)} items from REST API")
    except Exception as e:
        logger.error(f"Failed to get items data from REST API: {e}")
        return
    
    # Create environment
    env_kwargs = ENV_KWARGS.copy()
    env_kwargs["real_trading_mode"] = True
    env_kwargs["max_order_slots"] = max_slots
    env_kwargs["items"] = items  # Add items to env_kwargs
    # Remove parameters not accepted by GrandExchangeEnv
    env_kwargs.pop("strict_mode", None)
    env_kwargs.pop("use_historical_data", None)
    # Add mappings to env_kwargs
    env_kwargs["name_to_id_map"] = name_to_id
    env_kwargs["id_to_name_map"] = id_to_name
    env_kwargs.pop("min_volume_threshold", None) # Remove unexpected argument before env creation
    env = GrandExchangeEnv(**env_kwargs)
    logger.info("Created environment with real trading mode enabled")
    
    # Create agent with proper parameters
    item_list = list(items.keys())
    price_ranges = {item: (items[item]['min_price'], items[item]['max_price'])
                   for item in items}
    buy_limits = {item: items[item]['buy_limit'] for item in items}
    
    agent = PPOAgent(
        item_list=item_list,
        price_ranges=price_ranges,
        buy_limits_info=buy_limits, # Corrected parameter name
        hidden_size=128,
        price_bins=10,
        quantity_bins=10,
        wait_steps_bins=10,
        lr=3e-4
    )
    
    # Load model if specified
    if args.model_path and os.path.exists(args.model_path):
        try:
            agent.load_all(args.model_path)
            logger.info(f"Loaded full model from {args.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return
    else:
        logger.warning(f"Model path {args.model_path} not found, using untrained agent")
    
    # Create integration, passing the loaded item maps
    integration = PPOWebSocketIntegration(
        websocket_url=websocket_url,
        max_slots=max_slots,
        id_to_name_map=id_to_name,  # Pass the loaded map
        name_to_id_map=name_to_id   # Pass the loaded map
    )

    # Hook agent and environment
    await integration.hook_agent(agent, agent_id=agent_id)
    # await integration.hook_env(env, env_id="env_0") # Removed: hook_env does not exist/is not needed for integration

    # Connect to WebSocket server
    logger.info("Connecting to PPO WebSocket integration")
    if not await integration.connect():
        logger.error("Failed to connect to PPO WebSocket integration")
        return

    # Portfolio building is now handled by the integration
    logger.info("Portfolio building will be handled by the PPO WebSocket integration")
    # Set up health monitoring
    start_time = time.time()
    start_gp = env.gp
    consecutive_errors = 0
    max_consecutive_errors = 5

    # Run until interrupted
    try:
        while True:
            try:
                # Perform health check
                current_time = time.time()
                elapsed_time = current_time - start_time

                # Check for excessive loss
                current_gp = env.gp
                gp_change = current_gp - start_gp
                loss_percent = abs(gp_change) / start_gp if gp_change < 0 else 0

                if loss_percent > MAX_LOSS_PERCENT:
                    logger.warning(f"Stopping trading due to excessive loss: {loss_percent:.2%} (threshold: {MAX_LOSS_PERCENT:.2%})")
                    break

                # Log status
                if args.verbose:
                    total_value = env.gp + sum(env.prices[item] * env.inventory[item] for item in env.inventory)
                    logger.info(f"Status: GP={env.gp}, Total Value={total_value}, Running Time={elapsed_time:.0f}s")

                    # Log inventory
                    if env.inventory:
                        inventory_str = ", ".join([f"{item}: {qty}" for item, qty in env.inventory.items() if qty > 0])
                        logger.info(f"Inventory: {inventory_str}")

                    # Log active orders
                    active_orders = len(integration.order_manager.active_orders)
                    logger.info(f"Active Orders: {active_orders}/{max_slots}")

                # Reset consecutive errors counter
                consecutive_errors = 0

                # Sleep until next health check
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)

            except Exception as e:
                logger.error(f"Error in health check: {str(e)}")
                consecutive_errors += 1

                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"Stopping trading due to {consecutive_errors} consecutive errors")
                    break

                # Sleep before retrying
                await asyncio.sleep(10)

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, stopping trading")

    finally:
        # Disconnect integration
        logger.info("Disconnecting PPO WebSocket integration")
        await integration.disconnect()

        # Log final status
        end_time = time.time()
        elapsed_time = end_time - start_time

        logger.info("="*80)
        logger.info(f"Trading session ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Session duration: {elapsed_time:.0f} seconds ({elapsed_time/3600:.2f} hours)")

        # Calculate profit/loss
        end_gp = env.gp
        gp_change = end_gp - start_gp
        percent_change = (gp_change / start_gp) * 100 if start_gp > 0 else 0

        logger.info(f"Starting GP: {start_gp}")
        logger.info(f"Ending GP: {end_gp}")
        logger.info(f"Change: {gp_change:+} ({percent_change:+.2f}%)")
        logger.info("="*80)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run real-time trading with the marketplace")
    
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH,
                        help=f"Path to the trained model (default: {DEFAULT_MODEL_PATH})")
    
    parser.add_argument("--force-enable", action="store_true",
                        help="Force enable real trading mode even if disabled in config")
    
    parser.add_argument("--websocket-url", type=str,
                        help="WebSocket URL for the marketplace")
    
    parser.add_argument("--max-slots", type=int,
                        help="Maximum number of order slots")
    
    parser.add_argument("--update-interval", type=int,
                        help="Update interval in seconds")
    
    parser.add_argument("--agent-id", type=str,
                        help="Agent ID for the marketplace")
                        
    parser.add_argument("--rest-api-url", type=str, default=REST_API_URL,
                        help=f"URL for the REST API (default: {REST_API_URL})")
    
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Run real trading
    asyncio.run(run_real_trading(args))

if __name__ == "__main__":
    main()