import asyncio
import logging
import os
import time
import torch
import numpy as np
import argparse
import json
import sys
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Configure logging with rotating file handler
import logging.handlers

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Configure real trading logger with rotating file handler
logger = logging.getLogger("real_trading")
logger.setLevel(logging.INFO)
handler = logging.handlers.RotatingFileHandler(
    "logs/real_trading.log",
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Configure trade logger for detailed trade information
trade_logger = logging.getLogger("real_trading.trades")
trade_logger.setLevel(logging.INFO)
trade_handler = logging.FileHandler("logs/real_trading_trades.log")
trade_handler.setFormatter(formatter)
trade_logger.addHandler(trade_handler)

# Configure error logger for detailed error information
error_logger = logging.getLogger("real_trading.errors")
error_logger.setLevel(logging.ERROR)
error_handler = logging.FileHandler("logs/real_trading_errors.log")
error_handler.setFormatter(formatter)
error_logger.addHandler(error_handler)

# Configure performance logger for monitoring performance
performance_logger = logging.getLogger("real_trading.performance")
performance_logger.setLevel(logging.INFO)
performance_handler = logging.FileHandler("logs/real_trading_performance.log")
performance_handler.setFormatter(formatter)
performance_logger.addHandler(performance_handler)

# Import required modules
from ge_env import GrandExchangeEnv
from ppo_agent import PPOAgent
from config import ENV_KWARGS, PPO_KWARGS
from real_time_ge_client import RealTimeGrandExchangeClient
from ppo_websocket_integration import PPOWebSocketIntegration
from ge_rest_client import GrandExchangeClient
from shared_knowledge import SharedKnowledgeRepository
from volume_analysis import VolumeAnalyzer, create_volume_analyzer

# Constants for safeguards
MAX_TRADE_VALUE_PERCENT = 0.10  # Maximum 10% of portfolio in a single trade
MAX_LOSS_PERCENT = 0.05  # Maximum 5% loss before stopping trading
MAX_CONSECUTIVE_ERRORS = 5  # Maximum consecutive errors before stopping trading
RECONNECT_ATTEMPTS = 3  # Number of reconnection attempts
RECONNECT_DELAY = 5  # Delay between reconnection attempts in seconds
async def run_real_trading(args):
    """
    Run the PPO agent in real trading mode.
    This function initializes the real-time client, sets up the environment,
    loads the trained PPO agent, and runs it in real trading mode.
    
    Args:
        args: Command-line arguments
    """
    start_time = time.time()
    logger.info("="*80)
    logger.info(f"Starting real trading mode at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Configuration: WebSocket URL: {args.websocket_url}, Update Interval: {args.update_interval}s, Model Path: {args.model_path}, Strict Mode: {args.strict_mode}")
    logger.info("="*80)
    
    # Initialize performance monitoring
    performance_stats = {
        "start_time": start_time,
        "trades_executed": 0,
        "successful_trades": 0,
        "failed_trades": 0,
        "total_profit": 0,
        "errors": 0,
        "consecutive_errors": 0,
        "market_updates": 0,
        "last_portfolio_value": 0,
    }
    
    # Set device for PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize WebSocket integration with reconnection logic
    websocket_integration = None
    for attempt in range(RECONNECT_ATTEMPTS):
        try:
            websocket_integration = PPOWebSocketIntegration(websocket_url=args.websocket_url)
            connection_result = await websocket_integration.connect()
            if connection_result:
                logger.info(f"Connected to WebSocket server at {args.websocket_url} (Attempt {attempt+1}/{RECONNECT_ATTEMPTS})")
                break
            else:
                logger.warning(f"Failed to connect to WebSocket server at {args.websocket_url} (Attempt {attempt+1}/{RECONNECT_ATTEMPTS})")
                if attempt < RECONNECT_ATTEMPTS - 1:
                    logger.info(f"Retrying in {RECONNECT_DELAY} seconds...")
                    await asyncio.sleep(RECONNECT_DELAY)
        except Exception as e:
            error_logger.error(f"Error connecting to WebSocket server: {e}", exc_info=True)
            if attempt < RECONNECT_ATTEMPTS - 1:
                logger.info(f"Retrying in {RECONNECT_DELAY} seconds...")
                await asyncio.sleep(RECONNECT_DELAY)
    
    if not websocket_integration or not websocket_integration.connected:
        logger.error(f"Failed to connect to WebSocket server after {RECONNECT_ATTEMPTS} attempts. Exiting.")
        return
    
    # Initialize real-time client with WebSocket integration
    real_time_client = RealTimeGrandExchangeClient(
        websocket_integration=websocket_integration,
        update_interval=args.update_interval,
        cache_size=args.cache_size,
        max_retries=args.max_retries,
        backoff_factor=args.backoff_factor
    )
    
    try:
        await real_time_client.start()
        logger.info("Real-time client started")
        
        # Send system status event
        await websocket_integration.system_status(
            "real_trading_started",
            f"Real trading mode started with update interval {args.update_interval}s"
        )
        
        # Load item mappings
        logger.info("Loading item mappings...")
        id_name_map, buy_limits_map = read_mapping_file()
        
        # Initialize REST client for initial data
        rest_client = GrandExchangeClient()
        
        # Fetch marketplace data
        logger.info("Fetching marketplace data...")
        marketplace_data = fetch_marketplace_data(rest_client)
        
        # Build items dictionary
        logger.info("Building items dictionary...")
        items = build_items_dict(id_name_map, buy_limits_map, marketplace_data, strict_mode=args.strict_mode)
        item_list, price_ranges, buy_limits = get_item_lists(items)
        
        # Initialize volume analyzer
        logger.info("Initializing volume analyzer...")
        volume_analyzer = initialize_volume_analyzer(rest_client, id_name_map)
        
        # Create environment with real trading mode enabled
        logger.info("Creating environment with real trading mode enabled...")
        env_kwargs = dict(ENV_KWARGS)
        env_kwargs["items"] = items
        env_kwargs["real_trading_mode"] = True
        env_kwargs["real_trading_client"] = real_time_client
        env_kwargs["volume_analyzer"] = volume_analyzer
        env_kwargs["strict_mode"] = args.strict_mode
        
        # Create environment
        env = GrandExchangeEnv(**env_kwargs)
        logger.info(f"Environment created with real trading mode enabled (Strict Mode: {args.strict_mode})")
        
        # Initialize shared knowledge repository
        logger.info("Initializing shared knowledge repository...")
        shared_knowledge = SharedKnowledgeRepository(id_name_map, {name: id for id, name in id_name_map.items()})
        
        # Load trained agent
        logger.info(f"Loading trained agent from {args.model_path}...")
        agent = load_trained_agent(
            item_list=item_list,
            price_ranges=price_ranges,
            buy_limits=buy_limits,
            device=device,
            volume_analyzer=volume_analyzer,
            shared_knowledge=shared_knowledge,
            model_path=args.model_path
        )
        logger.info("Trained agent loaded")
        
        # Hook agent and environment to WebSocket integration
        websocket_integration.hook_agent(agent, agent_id="real_trading_agent")
        websocket_integration.hook_env(env, env_id="real_trading_env")
        logger.info("Agent and environment hooked to WebSocket integration")
        
        # Reset environment to start trading
        obs = env.reset()
        logger.info(f"Environment reset. Starting GP: {obs['gp']}")
        
        # Save initial portfolio value for loss tracking
        initial_portfolio_value = obs['gp']
        performance_stats["last_portfolio_value"] = initial_portfolio_value
        
        # Create a file to store portfolio snapshots
        portfolio_file = os.path.join("logs", "real_trading_portfolio.json")
        save_portfolio_snapshot(portfolio_file, {
            "timestamp": time.time(),
            "gp": obs['gp'],
            "inventory": obs['inventory'],
            "prices": obs['prices'],
            "total_value": obs['gp']
        })
        
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
            consecutive_errors = 0
            last_market_update_time = time.time()
            
            while True:
                loop_start_time = time.time()
                
                try:
                    # Check for excessive loss
                    current_portfolio_value = obs['gp'] + sum(obs['prices'][item] * obs['inventory'][item] for item in obs['inventory'])
                    loss_percent = (initial_portfolio_value - current_portfolio_value) / initial_portfolio_value
                    
                    if loss_percent > MAX_LOSS_PERCENT:
                        logger.warning(f"Stopping trading due to excessive loss: {loss_percent:.2%} (threshold: {MAX_LOSS_PERCENT:.2%})")
                        await websocket_integration.system_status(
                            "trading_stopped",
                            f"Trading stopped due to excessive loss: {loss_percent:.2%}"
                        )
                        break
                    
                    # Sample action from agent
                    action, logits = agent.sample_action(obs)
                    
                    # Apply safeguards to the action
                    action = apply_safeguards(action, obs, MAX_TRADE_VALUE_PERCENT)
                    
                    # Log action with detailed information
                    trade_logger.info(f"Action: {action['type']} | Item: {action['item']} | Price: {action['price']} | Quantity: {action['quantity']} | GP: {obs['gp']} | Portfolio Value: {current_portfolio_value}")
                    
                    # Execute action in environment
                    next_obs, reward, done, info = env.step(action)
                    
                    # Update performance stats
                    if info.get('trade_executed', False):
                        performance_stats["trades_executed"] += 1
                        if reward > 0:
                            performance_stats["successful_trades"] += 1
                            performance_stats["total_profit"] += reward
                        else:
                            performance_stats["failed_trades"] += 1
                    
                    # Log step result with detailed information
                    trade_logger.info(f"Step result: Reward: {reward:.2f} | Done: {done} | Trade executed: {info.get('trade_executed', False)} | Trade profit: {info.get('profit', 0)}")
                    
                    # Reset consecutive errors counter on successful step
                    consecutive_errors = 0
                    
                    # Update observation
                    obs = next_obs
                    
                    # Log portfolio status
                    total_value = obs['gp'] + sum(obs['prices'][item] * obs['inventory'][item] for item in obs['inventory'])
                    performance_stats["last_portfolio_value"] = total_value
                    
                    # Save portfolio snapshot every hour
                    current_time = time.time()
                    if current_time - last_market_update_time >= 3600:  # 1 hour
                        save_portfolio_snapshot(portfolio_file, {
                            "timestamp": current_time,
                            "gp": obs['gp'],
                            "inventory": obs['inventory'],
                            "prices": obs['prices'],
                            "total_value": total_value
                        })
                        last_market_update_time = current_time
                    
                    # Log performance metrics every 10 trades
                    if performance_stats["trades_executed"] % 10 == 0 and performance_stats["trades_executed"] > 0:
                        log_performance_metrics(performance_stats, total_value)
                    
                    # Wait for next trading interval
                    logger.info(f"Waiting {args.update_interval} seconds until next trading cycle...")
                    await asyncio.sleep(args.update_interval)
                    
                    # Update market data
                    logger.info("Updating market data...")
                    await env.update_real_market_data()
                    performance_stats["market_updates"] += 1
                    
                    # Check for any completed orders
                    logger.info("Checking for completed orders...")
                    env._update_real_orders()
                    
                except asyncio.CancelledError:
                    logger.info("Trading loop cancelled")
                    break
                except Exception as e:
                    error_logger.error(f"Error in trading loop: {e}", exc_info=True)
                    performance_stats["errors"] += 1
                    consecutive_errors += 1
                    
                    # Send error event to WebSocket
                    if websocket_integration and websocket_integration.connected:
                        await websocket_integration.error_event(
                            "trading_error",
                            f"Error in trading loop: {str(e)}",
                            {"traceback": traceback.format_exc()}
                        )
                    
                    # Check for too many consecutive errors
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        logger.error(f"Stopping trading due to {consecutive_errors} consecutive errors")
                        await websocket_integration.system_status(
                            "trading_stopped",
                            f"Trading stopped due to {consecutive_errors} consecutive errors"
                        )
                        break
                    
                    # Wait before retrying
                    await asyncio.sleep(10)
                
                # Log loop execution time
                loop_execution_time = time.time() - loop_start_time
                performance_logger.info(f"Trading loop execution time: {loop_execution_time:.2f}s")
                
        except KeyboardInterrupt:
            logger.info("Trading interrupted by user")
        finally:
            # --- Save Margin Experimentation Data ---
            if websocket_integration and websocket_integration.margin_handler and websocket_integration.margin_handler.margin_exp:
                logger.info("Saving margin experimentation success rates...")
                try:
                    websocket_integration.margin_handler.margin_exp.save_data()
                except Exception as save_err:
                    error_logger.error(f"Error saving margin data: {save_err}", exc_info=True)
            else:
                logger.warning("Could not save margin data: Integration or handler not available.")
            # -----------------------------------------

            # Save final portfolio snapshot
            # Ensure obs is defined, might not be if setup failed early
            if 'obs' in locals() and obs:
                save_portfolio_snapshot(portfolio_file, {
                    "timestamp": time.time(),
                "gp": obs['gp'],
                "inventory": obs['inventory'],
                "prices": obs['prices'],
                "total_value": obs['gp'] + sum(obs['prices'][item] * obs['inventory'][item] for item in obs['inventory'])
            })
            
            # Log final performance metrics
            total_value = obs['gp'] + sum(obs['prices'][item] * obs['inventory'][item] for item in obs['inventory'])
            log_performance_metrics(performance_stats, total_value, final=True)
            
            # Clean up
            logger.info("Cleaning up resources...")
            websocket_integration.unhook_agent("real_trading_agent")
            websocket_integration.unhook_env("real_trading_env")
            await real_time_client.stop()
            await websocket_integration.disconnect()
            logger.info("Real trading mode stopped")
    except Exception as e:
        error_logger.error(f"Critical error in real trading setup: {e}", exc_info=True)
        if websocket_integration and websocket_integration.connected:
            await websocket_integration.error_event(
                "critical_error",
                f"Critical error in real trading setup: {str(e)}",
                {"traceback": traceback.format_exc()}
            )
            await websocket_integration.disconnect()
        logger.error("Real trading mode failed to start")
def apply_safeguards(action, obs, max_trade_value_percent):
    """
    Apply safeguards to the action to prevent excessive trading or losses.
    
    Args:
        action: The action to apply safeguards to
        obs: The current observation
        max_trade_value_percent: Maximum percentage of portfolio value for a single trade
        
    Returns:
        The action with safeguards applied
    """
    # Make a copy of the action to avoid modifying the original
    safe_action = action.copy()
    
    # Calculate portfolio value
    portfolio_value = obs['gp'] + sum(obs['prices'][item] * obs['inventory'][item] for item in obs['inventory'])
    
    # Calculate maximum trade value
    max_trade_value = portfolio_value * max_trade_value_percent
    
    # Calculate current trade value
    trade_value = safe_action['price'] * safe_action['quantity']
    
    # If trade value exceeds maximum, reduce quantity
    if trade_value > max_trade_value:
        original_quantity = safe_action['quantity']
        safe_action['quantity'] = max(1, int(max_trade_value / safe_action['price']))
        logger.warning(f"Reduced trade quantity from {original_quantity} to {safe_action['quantity']} due to value limit ({trade_value} > {max_trade_value})")
    
    # For buy actions, ensure we have enough GP
    if safe_action['type'] == 'buy':
        max_affordable = obs['gp'] // safe_action['price']
        if safe_action['quantity'] > max_affordable:
            original_quantity = safe_action['quantity']
            safe_action['quantity'] = max_affordable
            logger.warning(f"Reduced buy quantity from {original_quantity} to {safe_action['quantity']} due to insufficient GP")
    
    # For sell actions, ensure we have enough inventory
    elif safe_action['type'] == 'sell':
        available_quantity = obs['inventory'].get(safe_action['item'], 0)
        if safe_action['quantity'] > available_quantity:
            original_quantity = safe_action['quantity']
            safe_action['quantity'] = available_quantity
            logger.warning(f"Reduced sell quantity from {original_quantity} to {safe_action['quantity']} due to insufficient inventory")
    
    return safe_action

def save_portfolio_snapshot(file_path, portfolio_data):
    """
    Save a snapshot of the portfolio to a JSON file.
    
    Args:
        file_path: Path to the JSON file
        portfolio_data: Portfolio data to save
    """
    try:
        # Convert inventory and prices to serializable format
        serializable_data = {
            "timestamp": portfolio_data["timestamp"],
            "gp": portfolio_data["gp"],
            "inventory": {k: int(v) for k, v in portfolio_data["inventory"].items()},
            "prices": {k: int(v) for k, v in portfolio_data["prices"].items()},
            "total_value": portfolio_data["total_value"]
        }
        
        # Load existing data if file exists
        existing_data = []
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Error reading portfolio file {file_path}, creating new file")
        
        # Append new data
        existing_data.append(serializable_data)
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(existing_data, f, indent=2)
        
        logger.info(f"Portfolio snapshot saved to {file_path}")
    except Exception as e:
        error_logger.error(f"Error saving portfolio snapshot: {e}", exc_info=True)

def log_performance_metrics(stats, current_value, final=False):
    """
    Log performance metrics.
    
    Args:
        stats: Performance statistics
        current_value: Current portfolio value
        final: Whether this is the final log
    """
    runtime = time.time() - stats["start_time"]
    hours = runtime // 3600
    minutes = (runtime % 3600) // 60
    seconds = runtime % 60
    
    profit_percent = 0
    if stats["last_portfolio_value"] > 0:
        profit_percent = (current_value - stats["last_portfolio_value"]) / stats["last_portfolio_value"] * 100
    
    success_rate = 0
    if stats["trades_executed"] > 0:
        success_rate = stats["successful_trades"] / stats["trades_executed"] * 100
    
    log_prefix = "Final" if final else "Current"
    
    performance_logger.info(f"{log_prefix} Performance Metrics:")
    performance_logger.info(f"  Runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    performance_logger.info(f"  Trades Executed: {stats['trades_executed']}")
    performance_logger.info(f"  Successful Trades: {stats['successful_trades']} ({success_rate:.2f}%)")
    performance_logger.info(f"  Failed Trades: {stats['failed_trades']}")
    performance_logger.info(f"  Total Profit: {stats['total_profit']:.2f}")
    performance_logger.info(f"  Current Portfolio Value: {current_value}")
    performance_logger.info(f"  Profit Since Last Update: {profit_percent:.2f}%")
    performance_logger.info(f"  Errors: {stats['errors']}")
    performance_logger.info(f"  Market Updates: {stats['market_updates']}")

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
        error_logger.error(f"Error reading mapping file: {e}", exc_info=True)
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
        error_logger.error(f"Error fetching marketplace data: {e}", exc_info=True)
        # Return minimal default data if API fails
        return {
            "554": {"high": 5, "low": 4, "highTime": int(time.time()), "lowTime": int(time.time())},
            "555": {"high": 5, "low": 4, "highTime": int(time.time()), "lowTime": int(time.time())},
            "556": {"high": 5, "low": 4, "highTime": int(time.time()), "lowTime": int(time.time())}
        }

def build_items_dict(id_name_map, buy_limits_map, marketplace_data, strict_mode=False):
    """Build a dictionary of items using mapping data and marketplace data."""
    # Read high volume items path directly from config
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
            error_logger.error(f"Error reading high volume items file {high_vol_items_path}: {e}", exc_info=True)

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
        error_logger.error(f"Error initializing volume analyzer: {e}", exc_info=True)
    
    return volume_analyzer

def load_trained_agent(item_list, price_ranges, buy_limits, device, volume_analyzer=None, shared_knowledge=None, model_path="agent_states"):
    """
    Load the trained PPO agent from saved state.
    
    Args:
        item_list: List of items
        price_ranges: Dictionary of price ranges for each item
        buy_limits: Dictionary of buy limits for each item
        device: PyTorch device
        volume_analyzer: Volume analyzer
        shared_knowledge: Shared knowledge repository
        model_path: Path to the model weights
        
    Returns:
        Loaded PPO agent
    """
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
    best_actor_path = os.path.join(model_path, "actor_best.pth")
    best_critic_path = os.path.join(model_path, "critic_best.pth")
    
    if os.path.exists(best_actor_path) and os.path.exists(best_critic_path):
        try:
            agent.load_actor(best_actor_path)
            agent.load_critic(best_critic_path)
            logger.info(f"Loaded best agent from {best_actor_path} and {best_critic_path}")
            return agent
        except Exception as e:
            error_logger.error(f"Error loading best agent: {e}", exc_info=True)
    
    # If best model not found, try to find the latest model
    latest_step = 0
    latest_actor_path = None
    latest_critic_path = None
    
    # Check all agent directories
    for agent_idx in range(5):  # Assuming up to 5 agents
        agent_dir = os.path.join(model_path, f"agent{agent_idx + 1}")
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
            error_logger.error(f"Error loading latest agent: {e}", exc_info=True)
    
    # If no model found, return the newly created agent
    logger.warning("No saved agent found, using newly initialized agent")
    return agent

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run PPO agent in real trading mode")
    
    parser.add_argument("--websocket-url", type=str, default="http://localhost:5178",
                        help="URL of the WebSocket server (default: http://localhost:5178)")
    
    parser.add_argument("--update-interval", type=int, default=300,
                        help="Interval in seconds between market data updates (default: 300)")
    
    parser.add_argument("--model-path", type=str, default="agent_states",
                        help="Path to the trained model weights (default: agent_states)")
    
    parser.add_argument("--strict-mode", action="store_true",
                        help="Enable strict mode to only trade high-volume items")
    
    parser.add_argument("--cache-size", type=int, default=128,
                        help="Size of the cache for API responses (default: 128)")
    
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Maximum number of retries for API requests (default: 3)")
    
    parser.add_argument("--backoff-factor", type=float, default=1.0,
                        help="Backoff factor for retries (default: 1.0)")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    
    # Run the real trading mode
    asyncio.run(run_real_trading(args))