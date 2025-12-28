import numpy as np
import torch
import os
import shutil
from datetime import datetime
import logging
import time

def manage_agent_epochs(agent_idx, epoch, save_dir="agent_states"):
    """Manage epoch-based saves for an agent, keeping only the last 5 epochs."""
    agent_dir = os.path.join(save_dir, f"agent{agent_idx + 1}")
    os.makedirs(agent_dir, exist_ok=True)
    
    # List all epoch saves for this agent
    epoch_saves = []
    for file in os.listdir(agent_dir):
        if file.startswith(f"epoch_") and file.endswith(".pth"):
            epoch_num = int(file.split("_")[1].split(".")[0])
            epoch_saves.append(epoch_num)
    
    # Remove oldest epochs if we have more than 5
    epoch_saves.sort()
    while len(epoch_saves) >= 5:
        oldest_epoch = epoch_saves.pop(0)
        actor_path = os.path.join(agent_dir, f"epoch_{oldest_epoch}.pth")
        critic_path = os.path.join(agent_dir, f"epoch_{oldest_epoch}_critic.pth")
        if os.path.exists(actor_path):
            os.remove(actor_path)
        if os.path.exists(critic_path):
            os.remove(critic_path)
    
    return agent_dir

# Multiprocessing-safe logger setup
import multiprocessing
import logging.handlers

def setup_logger(log_queue=None):
    logger = logging.getLogger("ppo_training")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    if log_queue is not None:
        handler = logging.handlers.QueueHandler(log_queue)
    else:
        handler = logging.FileHandler("ppo_training.log", mode="a")
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    logger.propagate = False
    return logger

# For main process: start a QueueListener
def start_logging_listener(log_queue):
    handler = logging.FileHandler("ppo_training.log", mode="a")
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    listener = logging.handlers.QueueListener(log_queue, handler)
    listener.start()
    return listener

logger = setup_logger()

# Local imports from training directory
from ge_environment import GrandExchangeEnv
from shared_knowledge import SharedKnowledgeRepository
from volume_analysis import VolumeAnalyzer, create_volume_analyzer, update_volume_analyzer, get_volume_metrics_for_item

# Parent directory imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ppo_agent import PPOAgent
from ppo_config import ENV_KWARGS, PPO_KWARGS, TRAIN_KWARGS
from tax_log import log_tax_payment, log_tax_summary, create_tax_report

import json
import threading
import time

# Function to manage volume blacklist log file
def manage_volume_blacklist_log(content, mode='a', max_entries=1000):
    """
    Write content to the volume blacklist log file with size management.
    
    Args:
        content: String content to write to the log
        mode: File open mode ('a' for append, 'w' for write/overwrite)
        max_entries: Maximum number of entries to keep in the log file
    """
    log_file = "volume_blacklist_log.txt"
    
    # If the file doesn't exist or we're in write mode, just write the content
    if not os.path.exists(log_file) or mode == 'w':
        with open(log_file, 'w') as f:
            f.write(content)
        return
    
    # If we're appending, check if we need to trim the file
    if mode == 'a':
        # Read existing content
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # If we have too many lines, keep only the most recent ones
            if len(lines) > max_entries:
                # Keep header (if any) and the most recent entries
                with open(log_file, 'w') as f:
                    # Write a note about truncation
                    f.write(f"[LOG TRUNCATED AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - KEEPING MOST RECENT {max_entries} ENTRIES]\n\n")
                    f.writelines(lines[-max_entries:])
            
            # Append the new content
            with open(log_file, 'a') as f:
                f.write(content)
        except Exception as e:
            # If there's an error, just overwrite the file
            with open(log_file, 'w') as f:
                f.write(f"[LOG RESET DUE TO ERROR AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n\n")
                f.write(content)

def load_training_cache():
    """Load all necessary data from ge_prices_export.json (lightweight version)."""
    # Use the smaller ge_prices_export.json which only has item metadata
    cache_path = "ge_prices_export.json"
    logger.info(f"Loading training data from {cache_path}...")
    
    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        
        items_list = cache_data.get('items', [])
        logger.info(f"Loaded cache data for {len(items_list)} items")
        
        # Extract mappings and create synthetic price/volume data
        id_name_map = {}
        buy_limits_map = {}
        marketplace_data = {}
        volume_data_5m = {}
        volume_data_1h = {}
        
        for item_info in items_list:
            item_id = str(item_info.get('id'))
            item_name = item_info.get('name', f'Item {item_id}')
            
            id_name_map[item_id] = item_name
            buy_limits_map[item_id] = item_info.get('ge_limit', 5000)
            
            # Use highalch value as base price estimation (this is just for initialization)
            base_price = item_info.get('highalch', 100)
            if base_price is None or base_price == 0:
                base_price = item_info.get('value', 100)
            if base_price is None or base_price == 0:
                base_price = 100  # Fallback default
            
            # Create synthetic marketplace data
            marketplace_data[item_id] = {
                'high': int(base_price * 1.1),
                'low': int(base_price * 0.9),
                'highTime': int(time.time()),
                'lowTime': int(time.time())
            }
            
            # Create synthetic volume data (training will use actual historical data)
            volume_data_5m[item_id] = {
                'avgHighPrice': base_price,
                'avgLowPrice': base_price,
                'highPriceVolume': 1000,  # Default volume
                'lowPriceVolume': 1000,
                'timestamp': int(time.time())
            }
            volume_data_1h[item_id] = volume_data_5m[item_id].copy()
        
        logger.info(f"Extracted {len(marketplace_data)} items with price data")
        logger.info("Note: Using item metadata for initialization. Actual training uses historical price data from environment.")
        return id_name_map, buy_limits_map, marketplace_data, volume_data_5m, volume_data_1h
        
    except FileNotFoundError:
        logger.error(f"Training cache file not found: {cache_path}")
        logger.error("Please run: python data/export_training_data.py or ensure ge_prices_export.json exists")
        raise
    except Exception as e:
        logger.error(f"Error loading training cache: {e}")
        raise

def read_mapping_file():
    """Read the mapping file to get item IDs, names, and buy limits."""
    # This function is now deprecated in favor of load_training_cache()
    # but kept for backwards compatibility
    logger.warning("read_mapping_file() is deprecated, use load_training_cache() instead")
    mapping_path = "endpoints/mapping.txt"
    id_name_map = {}
    buy_limits_map = {}
    
    try:
        with open(mapping_path, 'r') as f:
            mapping_data = json.load(f)
            for item in mapping_data:
                item_id = str(item.get('id'))
                item_name = item.get('name')
                if item_id and item_name:
                    id_name_map[item_id] = item_name
                    buy_limits_map[item_id] = item.get('limit', 5000)
        logger.info(f"Loaded mapping data for {len(id_name_map)} items")
    except Exception as e:
        logger.error(f"Error reading mapping file: {e}")
        id_name_map = {"554": "Fire rune", "555": "Water rune", "556": "Air rune"}
        buy_limits_map = {"554": 10000, "555": 10000, "556": 10000}
    
    return id_name_map, buy_limits_map

def fetch_marketplace_data(marketplace_data):
    """Return pre-loaded marketplace data (no API call)."""
    logger.info(f"Using cached marketplace data for {len(marketplace_data)} items")
    return marketplace_data

def build_items_dict(id_name_map, buy_limits_map, marketplace_data):
    """Build a dictionary of items using mapping data and marketplace data, with optional strict mode."""
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
            # If file reading fails in strict mode, we might want to handle this more robustly,
            # but for now, we'll proceed with an empty high_vol_items set, effectively
            # resulting in an empty items dictionary later if no items match.

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
    
    # Now check for saved environment states to ensure compatibility
    saved_items = set()
    import os
    import json
    
    # Try to load item names from saved environment states
    agent_states_dir = "agent_states"
    if os.path.exists(agent_states_dir):
        for filename in os.listdir(agent_states_dir):
            if filename.startswith("agent_") and filename.endswith("_env_state.json"):
                try:
                    with open(os.path.join(agent_states_dir, filename), 'r') as f:
                        state = json.load(f)
                        if "prices" in state:
                            for item_name in state["prices"].keys():
                                if item_name not in items:
                                    saved_items.add(item_name)
                except Exception as e:
                    logger.error(f"Error reading saved state {filename}: {e}")
    
    # Add any missing items from saved states, respecting strict mode
    if saved_items:
        items_to_add_from_saved = set()
        for item_name in saved_items:
            # In strict mode, only add items from saved states if they are in the high volume list
            if strict_mode and item_name not in high_vol_items:
                logger.debug(f"Strict mode: Skipping item '{item_name}' from saved state as it's not in high volume list")
                continue
            
            if item_name not in items:
                items_to_add_from_saved.add(item_name)

        if items_to_add_from_saved:
            logger.info(f"Adding {len(items_to_add_from_saved)} items from saved states that weren't in marketplace data (respecting strict mode)")
            for item_name in items_to_add_from_saved:
                # Use default values for items not in marketplace data
                items[item_name] = {
                    'base_price': 100,
                    'buy_limit': 5000,
                    'min_price': 90,
                    'max_price': 110
                }
                # Ensure all prices are > 0
                items[item_name]['base_price'] = max(1, items[item_name]['base_price'])
                items[item_name]['min_price'] = max(1, items[item_name]['min_price'])
                items[item_name]['max_price'] = max(items[item_name]['min_price'] + 1, items[item_name]['max_price'])

    # Ensure we have at least some items
    # This fallback should also respect strict mode if no high volume items are found
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
    # Take a 30-50% random subset of the items
    if items:  # Only if we have items to subset
        import random
        num_items_to_select = max(1, int(len(items) * random.uniform(1, 1)))  # Increased from 30-50% to 70-90%
        selected_keys = random.sample(list(items.keys()), num_items_to_select)
        items = {k: items[k] for k in selected_keys}
        logger.info(f"Selected random subset of {len(items)} items for trading")
    
    return items

def update_items_periodically(client, id_name_map, buy_limits_map, interval, callback):
    """Set up a periodic update of the items dictionary."""
    def update_thread():
        while True:
            try:
                marketplace_data = fetch_marketplace_data(client)
                new_items = build_items_dict(id_name_map, buy_limits_map, marketplace_data)
                callback(new_items)
                logger.info(f"Updated items dictionary with {len(new_items)} items")
            except Exception as e:
                logger.error(f"Error in periodic update: {e}")
            time.sleep(interval)
    
    # Start update thread
    thread = threading.Thread(target=update_thread, daemon=True)
    thread.start()
    logger.info(f"Started periodic items update thread (interval: {interval}s)")
    return thread

def make_env(agent_idx=0, items=None, volume_analyzer=None):
    # Use a unique random seed for each agent to ensure independent environments
    env_kwargs = dict(ENV_KWARGS)
    
    # GrandExchangeEnv only accepts specific parameters
    # Remove any extra parameters that aren't in the constructor
    allowed_params = {
        'db_path', 'cache_file', 'initial_cash', 'episode_length',
        'top_n_items', 'ge_limit_multiplier', 'include_volume_constraint',
        'render_mode', 'seed', 'timestep'
    }
    
    # Filter to only allowed parameters
    env_kwargs = {k: v for k, v in env_kwargs.items() if k in allowed_params}
    
    # Set unique seed for each agent
    env_kwargs["seed"] = agent_idx + np.random.randint(0, 10000)
    
    return GrandExchangeEnv(**env_kwargs)

def get_item_lists(items):
    item_list = list(items.keys())
    price_ranges = {item: (items[item]['min_price'], items[item]['max_price']) for item in item_list}
    buy_limits = {item: items[item]['buy_limit'] for item in item_list}
    return item_list, price_ranges, buy_limits

def compute_gae(rewards, values, dones, gamma, lam):
    advantages = np.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            nextnonterminal = 1.0 - dones[-1]
            nextvalue = values[-1]
        else:
            nextnonterminal = 1.0 - dones[t+1]
            nextvalue = values[t+1]
        delta = rewards[t] + gamma * nextvalue * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
    returns = advantages + values
    return advantages, returns

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load all data from training cache (NO API CALLS)
    id_name_map, buy_limits_map, marketplace_data, volume_data_5m, volume_data_1h = load_training_cache()
    marketplace_data = fetch_marketplace_data(marketplace_data)
    
    # Get strict mode and high volume items path from config
    strict_mode = ENV_KWARGS.get("strict_mode", False)
    # Get strict mode and high volume items path from config - now read inside build_items_dict
    # strict_mode = ENV_KWARGS.get("strict_mode", False)
    # high_vol_items_path = ENV_KWARGS.get("high_vol_items_path", "high_vol_items.txt")
    
    items = build_items_dict(id_name_map, buy_limits_map, marketplace_data)
    
    # Use cached 1h volume data
    volume_data = volume_data_1h
    
    # id_to_name_map is already created above
    
    # Calculate total volume for each item immediately after building the dictionary
    item_volumes = {}
    for item_id_str, data in volume_data.items():
        high_vol = data.get("highPriceVolume", 0) or 0
        low_vol = data.get("lowPriceVolume", 0) or 0
        total_vol = high_vol + low_vol
        
        # Map item ID to name if possible
        if item_id_str in id_to_name_map:
            item_name = id_to_name_map[item_id_str]
            if item_name in items:
                item_volumes[item_name] = total_vol
    
    # Filter items dictionary to only include high volume items (volume > 1000)
    high_volume_items = {}
    for item_name, item_data in items.items():
        if item_name in item_volumes and item_volumes[item_name] > 1000:
            high_volume_items[item_name] = item_data
    
    # Replace the original items dictionary with the filtered one
    items = high_volume_items
    logger.info(f"Filtered items dictionary to {len(items)} high volume items (volume > 1000)")
    
    item_list, price_ranges, buy_limits = get_item_lists(items)
    
    # Create a mapping from item ID to item name
    id_name_map, _ = read_mapping_file()
    id_to_name_map = {}
    for item_id, item_name in id_name_map.items():
        id_to_name_map[item_id] = item_name
    
    # Initialize volume analyzer
    volume_analyzer = create_volume_analyzer(id_to_name_map)
    
    # Update volume analyzer with cached data (NO API CALLS)
    data_5m = volume_data_5m
    volume_analyzer.update_volume_data(data_5m, volume_data)
    logger.info(f"Initialized volume analyzer with {len(data_5m)} 5m items and {len(volume_data)} 1h items from cache")
    
    # Calculate total volume for each item
    item_volumes = {}
    for item_id_str, data in volume_data.items():
        high_vol = data.get("highPriceVolume", 0) or 0
        low_vol = data.get("lowPriceVolume", 0) or 0
        total_vol = high_vol + low_vol
        
        # Map item ID to name if possible
        if item_id_str in id_to_name_map:
            item_name = id_to_name_map[item_id_str]
            if item_name in item_list:
                item_volumes[item_name] = total_vol
    
    # Analyze volume distribution to determine thresholds
    if item_volumes:
        volumes = list(item_volumes.values())
        volumes.sort()
        
        # Calculate volume statistics
        min_volume = volumes[0]
        max_volume = volumes[-1]
        median_volume = volumes[len(volumes)//2]
        q1_volume = volumes[len(volumes)//4]  # 25th percentile
        q3_volume = volumes[3*len(volumes)//4]  # 75th percentile
        
        logger.info(f"Volume statistics: Min={min_volume}, Q1={q1_volume}, Median={median_volume}, Q3={q3_volume}, Max={max_volume}")
        
        # Analyze the volume distribution curve to find natural breakpoints
        # This is more sophisticated than using a simple percentile
        
        # First, sort volumes and calculate differences between consecutive values
        # This helps identify "jumps" in the distribution
        volume_diffs = [volumes[i+1] - volumes[i] for i in range(len(volumes)-1)]
        
        # Calculate the relative differences (percentage change)
        relative_diffs = []
        for i in range(len(volume_diffs)):
            if volumes[i] > 0:
                relative_diffs.append(volume_diffs[i] / volumes[i])
            else:
                relative_diffs.append(0)
        
        # Find significant jumps in the distribution (where relative difference > 100%)
        significant_jumps = [i for i, diff in enumerate(relative_diffs) if diff > 1.0]
        
        # Log the volume distribution analysis
        logger.info(f"Volume distribution analysis: Found {len(significant_jumps)} significant jumps in volume")
        if significant_jumps:
            jump_volumes = [volumes[i] for i in significant_jumps]
            logger.info(f"Volume jumps occur at: {jump_volumes[:5]}...")
        
        # Set threshold based on the analysis:
        # 1. If we found significant jumps in the lower 25% of the distribution, use the highest such jump
        # 2. Otherwise, use the 10th percentile with a minimum value
        lower_quartile_jumps = [i for i in significant_jumps if i < len(volumes)//4]
        
        if lower_quartile_jumps:
            # Use the highest jump in the lower quartile as our threshold
            threshold_idx = max(lower_quartile_jumps)
            volume_threshold = volumes[threshold_idx]
            logger.info(f"Using natural breakpoint in volume distribution as threshold: {volume_threshold}")
        else:
            # Fall back to percentile-based approach with minimum threshold
            if len(volumes) >= 10:
                volume_threshold = max(100, volumes[len(volumes)//10])  # 10th percentile with minimum of 100
            else:
                volume_threshold = max(100, q1_volume)  # Use minimum threshold of 100
            logger.info(f"No significant breakpoints found in lower quartile, using percentile-based threshold: {volume_threshold}")
        
        logger.info(f"Using volume threshold: {volume_threshold}")
    else:
        # Fallback if no volume data
        volume_threshold = 10
        logger.warning("No volume data available, using default threshold of 10")
    
    # Create blacklist of low volume items
    volume_blacklist = set()
    
    # First, add items based on the dynamic threshold from volume curve analysis
    for item_name, volume in item_volumes.items():
        if volume < volume_threshold:
            volume_blacklist.add(item_name)
    
    # Then, enforce the strict rule that only items with average buy/sell volume of +100 can be traded
    # Use batch processing for better performance
    item_ids_to_check = []
    item_names_to_ids = {}
    
    # First collect all items to check
    for item_name in item_list:
        if item_name in volume_analyzer.name_to_id_map:
            item_id = volume_analyzer.name_to_id_map[item_name]
            item_ids_to_check.append(item_id)
            item_names_to_ids[item_id] = item_name
    
    # Then batch check them
    low_volume_item_ids = volume_analyzer.batch_check_volume_threshold(item_ids_to_check, 1000)
    
    # Add low volume items to blacklist
    for item_id in low_volume_item_ids:
        volume_blacklist.add(item_names_to_ids[item_id])
    
    # Log blacklisted items
    logger.info(f"Blacklisted {len(volume_blacklist)} items due to low volume (dynamic threshold: {volume_threshold}, strict minimum: 100)")
    logger.info(f"Blacklisted items: {sorted(list(volume_blacklist))[:20]}...")  # Log first 20 items
    
    # Create a dedicated log file for volume blacklist updates
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_content = f"[{timestamp}] Initial blacklist created with {len(volume_blacklist)} items.\n"
    log_content += f"Dynamic threshold from volume curve analysis: {volume_threshold}\n"
    log_content += f"Strict minimum volume threshold: 100 (enforced for all items)\n"
    log_content += f"Volume statistics: Min={min_volume}, Q1={q1_volume}, Median={median_volume}, Q3={q3_volume}, Max={max_volume}\n"
    log_content += f"First 20 blacklisted items: {sorted(list(volume_blacklist))[:20]}\n\n"
    # Use write mode ('w') for initial creation to reset the log file
    manage_volume_blacklist_log(log_content, mode='w')

    # === Multi-Agent Setup with Shared Knowledge ===
    NUM_AGENTS = TRAIN_KWARGS.get("num_agents", 5)
    logger.info(f"Training with {NUM_AGENTS} parallel agents")
    
    # Create shared knowledge repository
    shared_knowledge = SharedKnowledgeRepository(id_to_name_map, name_to_id_map)
    
    # Initialize with cached volume data (NO API CALLS)
    shared_knowledge.update_volume_data(volume_data_5m, volume_data_1h)
    logger.info(f"Shared knowledge repository initialized with cached volume data")
    
    # No need to track GP history in memory; all GP logging will be done via logger
    agents = []
    envs = []
    for i in range(NUM_AGENTS):
        envs.append(make_env(i, items=items, volume_analyzer=volume_analyzer))
        agents.append(PPOAgent(
            item_list=item_list,
            price_ranges=price_ranges,
            buy_limits=buy_limits,
            device=device,
            hidden_size=PPO_KWARGS["hidden_size"],
            price_bins=PPO_KWARGS["price_bins"],
            quantity_bins=PPO_KWARGS["quantity_bins"],
            wait_steps_bins=PPO_KWARGS["wait_steps_bins"],  # Add wait_steps_bins parameter
            lr=PPO_KWARGS["lr"],
            volume_blacklist=volume_blacklist,
            volume_analyzer=volume_analyzer,  # Pass volume analyzer to agent
            shared_knowledge=shared_knowledge,  # Pass shared knowledge repository
            agent_id=i  # Pass agent ID for tracking
        ))
        
        # Log agent's blacklist size
        logger.info(f"Agent {i} initialized with {len(volume_blacklist)} blacklisted items")
        print(f"Agent {i} risk tolerance: {agents[-1].risk_tolerance:.3f}")
    # === Model Loading ===
    # Try to load existing models for each agent if present

    def update_items_callback(new_items):
        nonlocal item_list, price_ranges, buy_limits, agents, volume_blacklist, id_to_name_map, volume_analyzer
        new_item_list, new_price_ranges, new_buy_limits = get_item_lists(new_items)
        item_list = new_item_list
        price_ranges = new_price_ranges
        buy_limits = new_buy_limits
        
        # Update volume blacklist with cached data (training mode - no API calls)
        try:
            # Use cached volume data instead of fetching fresh data
            volume_data = volume_data_1h
            data_5m = volume_data_5m
            
            # Note: In training mode, we use static cached data
            # volume_analyzer already has the cached data loaded
            logger.info("Using cached volume data (training mode - no API calls)")
            
            # Calculate total volume for each item
            item_volumes = {}
            for item_id_str, data in volume_data.items():
                high_vol = data.get("highPriceVolume", 0) or 0
                low_vol = data.get("lowPriceVolume", 0) or 0
                total_vol = high_vol + low_vol
                
                # Map item ID to name if possible
                if item_id_str in id_to_name_map:
                    item_name = id_to_name_map[item_id_str]
                    if item_name in item_list:
                        item_volumes[item_name] = total_vol
            
            # Analyze volume distribution to determine thresholds
            if item_volumes:
                volumes = list(item_volumes.values())
                volumes.sort()
                
                # Analyze the volume distribution curve to find natural breakpoints
                # This is more sophisticated than using a simple percentile
                
                # First, sort volumes and calculate differences between consecutive values
                # This helps identify "jumps" in the distribution
                volume_diffs = [volumes[i+1] - volumes[i] for i in range(len(volumes)-1)]
                
                # Calculate the relative differences (percentage change)
                relative_diffs = []
                for i in range(len(volume_diffs)):
                    if volumes[i] > 0:
                        relative_diffs.append(volume_diffs[i] / volumes[i])
                    else:
                        relative_diffs.append(0)
                
                # Find significant jumps in the distribution (where relative difference > 100%)
                significant_jumps = [i for i, diff in enumerate(relative_diffs) if diff > 1.0]
                
                # Log the volume distribution analysis
                logger.info(f"Periodic update - Volume distribution analysis: Found {len(significant_jumps)} significant jumps in volume")
                if significant_jumps:
                    jump_volumes = [volumes[i] for i in significant_jumps]
                    logger.info(f"Periodic update - Volume jumps occur at: {jump_volumes[:5]}...")
                
                # Set threshold based on the analysis:
                # 1. If we found significant jumps in the lower 25% of the distribution, use the highest such jump
                # 2. Otherwise, use the 10th percentile with a minimum value
                lower_quartile_jumps = [i for i in significant_jumps if i < len(volumes)//4]
                
                if lower_quartile_jumps:
                    # Use the highest jump in the lower quartile as our threshold
                    threshold_idx = max(lower_quartile_jumps)
                    volume_threshold = volumes[threshold_idx]
                    logger.info(f"Periodic update - Using natural breakpoint in volume distribution as threshold: {volume_threshold}")
                else:
                    # Fall back to percentile-based approach with minimum threshold
                    if len(volumes) >= 10:
                        volume_threshold = max(100, volumes[len(volumes)//10])  # 10th percentile with minimum of 100
                    else:
                        volume_threshold = max(100, volumes[len(volumes)//4])  # Use minimum threshold of 100
                    logger.info(f"Periodic update - No significant breakpoints found in lower quartile, using percentile-based threshold: {volume_threshold}")
                
                logger.info(f"Periodic update - Using volume threshold: {volume_threshold}")
                
                # Create updated blacklist of low volume items
                new_volume_blacklist = set()
                
                # First, add items based on the dynamic threshold from volume curve analysis
                for item_name, volume in item_volumes.items():
                    if volume < volume_threshold:
                        new_volume_blacklist.add(item_name)
                
                # Then, enforce the strict rule that only items with average buy/sell volume of +100 can be traded
                # by checking each item with the volume analyzer's check_volume_threshold method
                for item_name in item_list:
                    if item_name in volume_analyzer.name_to_id_map:
                        item_id = volume_analyzer.name_to_id_map[item_name]
                        # If the item doesn't meet the volume threshold, add it to the blacklist
                        if not volume_analyzer.check_volume_threshold(item_id, 100):
                            new_volume_blacklist.add(item_name)
                
                # Compare with previous blacklist to log changes
                added_items = new_volume_blacklist - volume_blacklist
                removed_items = volume_blacklist - new_volume_blacklist
                
                logger.info(f"Periodic update - Blacklisted {len(new_volume_blacklist)} items due to low volume (dynamic threshold: {volume_threshold}, strict minimum: 100)")
                if added_items:
                    logger.info(f"Newly blacklisted items: {len(added_items)} items, including {list(added_items)[:10]}")
                if removed_items:
                    logger.info(f"Removed from blacklist: {len(removed_items)} items, including {list(removed_items)[:10]}")
                
                # Update the blacklist
                volume_blacklist = new_volume_blacklist
                
                # Log to dedicated volume blacklist log file
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_content = f"[{timestamp}] Periodic update - Blacklist updated with {len(volume_blacklist)} items.\n"
                log_content += f"Dynamic threshold from volume curve analysis: {volume_threshold}\n"
                log_content += f"Strict minimum volume threshold: 100 (enforced for all items)\n"
                log_content += f"Volume statistics: Min={volumes[0]}, Q1={volumes[len(volumes)//4]}, Median={volumes[len(volumes)//2]}, Max={volumes[-1]}\n"
                log_content += f"First 20 blacklisted items: {sorted(list(volume_blacklist))[:20]}\n"
                
                if added_items:
                    log_content += f"Newly blacklisted items ({len(added_items)}): {sorted(list(added_items))[:20]}\n"
                if removed_items:
                    log_content += f"Removed from blacklist ({len(removed_items)}): {sorted(list(removed_items))[:20]}\n"
                
                log_content += "\n"
                manage_volume_blacklist_log(log_content)
            else:
                logger.warning("Periodic update - No volume data available, keeping existing blacklist")
        except Exception as e:
            logger.error(f"Error updating volume blacklist: {e}")
        
        # Update agents with new data
        for i, agent in enumerate(agents):
            # Store previous blacklist size for verification
            prev_blacklist_size = len(agent.volume_blacklist)
            
            # Update agent data
            agent.item_list = item_list
            agent.price_ranges = price_ranges
            agent.buy_limits = buy_limits
            agent.volume_blacklist = volume_blacklist.copy()  # Use copy to ensure each agent has its own instance
            
            # Verify the update was successful
            if len(agent.volume_blacklist) != len(volume_blacklist):
                logger.error(f"Agent {i} blacklist size mismatch: expected {len(volume_blacklist)}, got {len(agent.volume_blacklist)}")
            else:
                logger.info(f"Updated agent {i} with {len(volume_blacklist)} blacklisted items (previous: {prev_blacklist_size})")
                
            # Verify a few specific items are correctly blacklisted
            if volume_blacklist:
                sample_items = list(volume_blacklist)[:3]  # Check first 3 items
                for item in sample_items:
                    if item not in agent.volume_blacklist:
                        logger.error(f"Agent {i} missing blacklisted item: {item}")

    # Function to verify volume data from cached data (training mode)
    def verify_volume_data():
        """
        Verify that cached volume data is properly loaded and processed.
        Logs detailed information about the data quality and any issues found.
        """
        try:
            # Use cached data (NO API CALLS in training mode)
            data_5m = volume_data_5m
            data_1h = volume_data_1h
            
            # Log basic stats
            logger.info(f"Volume data verification - 5m endpoint: {len(data_5m)} items, 1h endpoint: {len(data_1h)} items")
            
            # Check for empty data
            if not data_5m:
                logger.error("5m endpoint returned empty data!")
            if not data_1h:
                logger.error("1h endpoint returned empty data!")
                
            # Check for overlap between endpoints
            common_items = set(data_5m.keys()).intersection(set(data_1h.keys()))
            logger.info(f"Items present in both endpoints: {len(common_items)} ({len(common_items)/max(1, len(data_5m))*100:.1f}% of 5m data)")
            
            # Check volume data quality
            zero_volume_5m = 0
            zero_volume_1h = 0
            
            for item_id, data in data_5m.items():
                high_vol = data.get("highPriceVolume", 0) or 0
                low_vol = data.get("lowPriceVolume", 0) or 0
                if high_vol == 0 and low_vol == 0:
                    zero_volume_5m += 1
                    
            for item_id, data in data_1h.items():
                high_vol = data.get("highPriceVolume", 0) or 0
                low_vol = data.get("lowPriceVolume", 0) or 0
                if high_vol == 0 and low_vol == 0:
                    zero_volume_1h += 1
            
            logger.info(f"Items with zero volume - 5m: {zero_volume_5m} ({zero_volume_5m/max(1, len(data_5m))*100:.1f}%), 1h: {zero_volume_1h} ({zero_volume_1h/max(1, len(data_1h))*100:.1f}%)")
            
            # Sample a few items to check data structure
            sample_ids = list(common_items)[:5] if common_items else []
            for item_id in sample_ids:
                item_name = id_to_name_map.get(item_id, f"Unknown ({item_id})")
                data_5m_item = data_5m[item_id]
                data_1h_item = data_1h[item_id]
                
                logger.info(f"Sample item {item_name} (ID: {item_id}):")
                logger.info(f"  5m data: highVol={data_5m_item.get('highPriceVolume', 0)}, lowVol={data_5m_item.get('lowPriceVolume', 0)}")
                logger.info(f"  1h data: highVol={data_1h_item.get('highPriceVolume', 0)}, lowVol={data_1h_item.get('lowPriceVolume', 0)}")
            
            return True
        except Exception as e:
            logger.error(f"Error verifying volume data: {e}")
            return False
    
    # Run initial verification
    logger.info("Running initial volume data verification...")
    verify_volume_data()
    
    # Periodically update shared knowledge (disabled in training mode - using cached data)
    def update_shared_knowledge():
        while True:
            try:
                # In training mode, we use static cached data, no periodic updates needed
                logger.info(f"Training mode: using static cached volume data (no periodic updates)")
                time.sleep(3600)  # Sleep for 1 hour
                continue  # Skip the actual update
                
                # Periodically verify data quality (every 6 hours)
                current_hour = datetime.now().hour
                if current_hour % 6 == 0:
                    logger.info("Running periodic volume data verification...")
                    verify_volume_data()
                    
            except Exception as e:
                logger.error(f"Error updating shared knowledge: {e}")
            
            # Update every 5 minutes
            time.sleep(300)
    
    # Start shared knowledge update thread
    shared_knowledge_thread = threading.Thread(target=update_shared_knowledge, daemon=True)
    shared_knowledge_thread.start()
    logger.info(f"Started shared knowledge update thread")
    
    # Also start the regular item updates
    update_items_periodically(client, id_name_map, buy_limits_map, 300, update_items_callback)
    # Set up save directory
    base_save_dir = "agent_states"
    os.makedirs(base_save_dir, exist_ok=True)
    print(f"Using save directory: {base_save_dir}")

    # Try to load latest models for each agent
    for agent_idx in range(NUM_AGENTS):
        agent_dir = os.path.join(base_save_dir, f"agent{agent_idx + 1}")
        os.makedirs(agent_dir, exist_ok=True)
        
        # Find latest step directory
        latest_step = 0
        latest_step_dir = None
        
        if os.path.exists(agent_dir):
            step_dirs = []
            for dirname in os.listdir(agent_dir):
                if dirname.startswith("step_"):
                    step_num = int(dirname.split("_")[1])
                    step_dirs.append((step_num, dirname))
            
            if step_dirs:
                step_dirs.sort()  # Sort by step number
                latest_step = step_dirs[-1][0]
                latest_step_dir = os.path.join(agent_dir, step_dirs[-1][1])
                
                # Load latest models
                actor_path = os.path.join(latest_step_dir, "actor.pth")
                critic_path = os.path.join(latest_step_dir, "critic.pth")
                
                if os.path.exists(actor_path) and os.path.exists(critic_path):
                    agents[agent_idx].load_actor(actor_path)
                    agents[agent_idx].load_critic(critic_path)
                    print(f"Loaded agent {agent_idx + 1} from step {latest_step}")
                    logger.info(f"Loaded agent {agent_idx + 1} from {latest_step_dir}")

    # Initialize step count from latest saved checkpoint
    step_count = 0
    for agent_idx in range(NUM_AGENTS):
        agent_dir = os.path.join(base_save_dir, f"agent{agent_idx + 1}")
        if os.path.exists(agent_dir):
            for dirname in os.listdir(agent_dir):
                if dirname.startswith("step_"):
                    step = int(dirname.split("_")[1])
                    step_count = max(step_count, step)
    print(f"Continuing from step: {step_count}")

    # === Model Saving Config ===
    save_every_steps = PPO_KWARGS.get("save_every_steps", 10000)  # Using 50 steps per save
    save_dir = base_save_dir  # Use the same directory for consistency
    save_best_metric = PPO_KWARGS.get("save_best_metric", "avg_gp")
    best_metric = -np.inf
    best_metric_name = save_best_metric
    best_agent_idx = 0

    rollout_steps = PPO_KWARGS["rollout_steps"]
    max_steps = PPO_KWARGS["max_steps"]
    gamma = PPO_KWARGS["gamma"]
    gae_lambda = PPO_KWARGS["gae_lambda"]
    batch_size = PPO_KWARGS["batch_size"]
    minibatch_size = PPO_KWARGS["minibatch_size"]
    ppo_epochs = PPO_KWARGS["ppo_epochs"]
    clip_eps = PPO_KWARGS["clip_eps"]
    value_coef = PPO_KWARGS["value_coef"]
    entropy_coef = PPO_KWARGS["entropy_coef"]

    # step_count was already initialized from latest checkpoint
    obs_list = [env.reset() for env in envs]
    # Log starting GP for each agent
    for agent_idx, obs in enumerate(obs_list):
        logger.info(f"GP_LOG | Agent {agent_idx} | Step 0 | GP {obs['gp']}")
    while True:
        for agent_idx, (agent, env) in enumerate(zip(agents, envs)):
            # Collect rollout for this agent
            obs_buf, actions_buf, logprobs_buf, rewards_buf, dones_buf, values_buf = [], [], [], [], [], []
            obs = obs_list[agent_idx]
            # Process multiple steps in parallel for faster learning
            for step_idx in range(0, rollout_steps, 4):  # Process 4 steps at once
                # Batch process observations
                obs_batch = torch.stack([agent.obs_to_tensor(obs) for _ in range(min(4, rollout_steps - step_idx))])
                with torch.no_grad():
                    at_logits_batch, item_logits_batch, price_logits_batch, qty_logits_batch, values_batch = agent.model(obs_batch)
                action, (at_logits, item_logits, price_logits, qty_logits) = agent.sample_action(obs)
                
                # Log if the item was chosen despite being in the volume blacklist (shouldn't happen)
                if action['item'] in agent.volume_blacklist:
                    logger.warning(f"Agent {agent_idx} selected blacklisted item: {action['item']}")
                    
                    # Log to dedicated volume blacklist log file
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_content = f"[{timestamp}] WARNING: Agent {agent_idx} selected blacklisted item: {action['item']} | Action: {action['type']} | Price: {action['price']} | Quantity: {action['quantity']}\n"
                    manage_volume_blacklist_log(log_content)
                
                # Log action with volume status
                in_blacklist = "BLACKLISTED" if action['item'] in agent.volume_blacklist else ""
                logger.info(f"Agent {agent_idx} action: {action['type']} | Item: {action['item']} {in_blacklist} | Price: {action['price']} | Quantity: {action['quantity']}")
                # Encode action as indices for PPO
                action_types = ['buy', 'sell', 'hold']
                action_type_idx = action_types.index(action['type'])
                item_idx = item_list.index(action['item'])
                # Discretize price and quantity
                min_price, max_price = price_ranges[action['item']]
                price_bins = np.linspace(min_price, max_price, agent.price_bins)
                price_bin = np.abs(price_bins - action['price']).argmin()
                max_qty = buy_limits[action['item']] - obs['buy_limits'][action['item']] if action['type'] == 'buy' else obs['inventory'][action['item']]
                max_qty = max(1, max_qty)
                qty_bins = np.linspace(1, max_qty, agent.quantity_bins)
                qty_bin = np.abs(qty_bins - action['quantity']).argmin()
                action_idx = [action_type_idx, item_idx, price_bin, qty_bin]

                # Compute logprob
                at_dist = torch.distributions.Categorical(logits=at_logits)
                item_dist = torch.distributions.Categorical(logits=item_logits)
                price_dist = torch.distributions.Categorical(logits=price_logits)
                qty_dist = torch.distributions.Categorical(logits=qty_logits)
                logprob = (
                    at_dist.log_prob(torch.tensor([action_type_idx], device=device)) +
                    item_dist.log_prob(torch.tensor([item_idx], device=device)) +
                    price_dist.log_prob(torch.tensor([price_bin], device=device)) +
                    qty_dist.log_prob(torch.tensor([qty_bin], device=device))
                ).item()

                obs_buf.append(agent.obs_to_tensor(obs).squeeze(0).cpu().numpy())
                actions_buf.append(action_idx)
                logprobs_buf.append(logprob)
                values_buf.append(value.item())

                # Separate decision-making from trade execution
                # Make decisions every step but only execute trades every 5 minutes (300 seconds)
                current_time = int(time.time())
                should_execute = False
                
                # Check if this agent should execute a trade now
                # Stagger agents to avoid all agents trading at exactly the same time
                agent_time_offset = agent_idx * 60  # 1 minute offset per agent
                # Keep 5-minute trade windows aligned with GE data updates
                if (current_time + agent_time_offset) % 300 < 60:  # Execute within a 1-minute window every 5 minutes
                    should_execute = True
                
                # Always think about trades (for learning), but only execute some of them
                if should_execute or action['type'] == 'hold':
                    # Execute the action
                    next_obs, reward, done, info = env.step(action)
                    
                    # Process executed trades
                    if action['type'] in ['buy', 'sell']:
                        # Calculate profit and tax
                        profit = 0
                        tax = 0
                        if action['type'] == 'sell':
                            if 'profit' in info:
                                profit = info.get('profit', 0)
                            
                            # Calculate GE tax (1% of price for items >= 100gp)
                            if action['price'] >= 100:
                                tax_per_item = min(int(action['price'] * 0.01), 5000000)  # 1% capped at 5M per item
                                tax = tax_per_item * action['quantity']
                        
                        # Note: agent.record_trade() is now called when updating trade_history
                        
                        # Log executed trade with tax information
                        tax_info = f" | Tax: {tax} GP" if tax > 0 else ""
                        logger.info(f"EXECUTED: Agent {agent_idx} | {action['type'].upper()} | Item: {action['item']} | Price: {action['price']} | Qty: {action['quantity']}{tax_info}")
                        
                        # Create a dedicated tax log entry for sell transactions with tax
                        if action['type'] == 'sell' and tax > 0:
                            # Log to main logger
                            logger.info(f"TAX_LOG | Agent {agent_idx} | Item: {action['item']} | Price: {action['price']} | Qty: {action['quantity']} | Tax: {tax} GP | Net Profit: {profit-tax} GP")
                            
                            # Log to dedicated tax log
                            log_tax_payment(agent_idx, action['item'], action['price'], action['quantity'], tax, profit-tax)
                else:
                    # Don't execute the action, but still learn from the decision
                    # Use the current observation as the next observation
                    next_obs = obs
                    reward = 0
                    done = False
                    info = {"msg": f"Trade not executed (thinking only): {action['type']} {action['item']}"}
                    
                    # Log skipped trade
                    if step_count % 500 == 0:  # Reduce logging frequency
                        logger.info(f"THINKING: Agent {agent_idx} | {action['type'].upper()} | Item: {action['item']} | Price: {action['price']} | Qty: {action['quantity']}")
                
                # Log GP and save trade history periodically
                if step_count % 7500 == 0:
                    logger.info(f"GP_LOG | Agent {agent_idx} | Step {step_count} | GP {next_obs['gp']}")
                    # Save trade history periodically during the episode
                    try:
                        shared_knowledge.save_trade_history(episode_number=episode_number)
                        logger.info(f"Saved trade history at step {step_count} (Episode {episode_number})")
                        # Log some stats about saved trades
                        trade_count = sum(len(trades) for trades in trade_history.values())
                        logger.info(f"Total trades saved: {trade_count} across {len(trade_history)} items")
                    except Exception as e:
                        logger.error(f"Failed to save trade history at step {step_count}: {str(e)}")
                        # Try to provide more context about the failure
                        logger.error(f"Trade history state: {len(trade_history)} items, {sum(len(trades) for trades in trade_history.values())} total trades")
                rewards_buf.append(reward)
                dones_buf.append(done)

                # Logging: major trades (buy/sell)
                if action['type'] in ['buy', 'sell']:
                    if step_count % 5000 == 0:
                        # Calculate tax for sell transactions
                        tax_info = ""
                        if action['type'] == 'sell' and action['price'] >= 100:
                            tax_per_item = min(int(action['price'] * 0.01), 5000000)
                            tax = tax_per_item * action['quantity']
                            tax_info = f" | Tax: {tax} GP"
                        
                        logger.info(
                            f"Agent {agent_idx} | Tick {obs['tick']} | {action['type'].upper()} | "
                        f"Item: {action['item']} | Price: {action['price']} | Qty: {action['quantity']}{tax_info} | "
                        f"GP after trade: {next_obs['gp']}"
                    )

                # Logging: profit milestones
                if not hasattr(env, '_last_logged_gp'):
                    env._last_logged_gp = obs['gp']
                for milestone in range(
                    int(env._last_logged_gp // 1_000_000) + 1,
                    int(next_obs['gp'] // 1_000_000) + 1
                ):
                    if step_count % 5000 == 0:
                        logger.info(
                            f"Agent {agent_idx} reached profit milestone: {milestone * 1_000_000} GP at tick {obs['tick']}"
                    )
                
                # Periodically log volume blacklist statistics
                if step_count % 10000 == 0:
                    # Count how many trades were for blacklisted vs non-blacklisted items
                    blacklisted_trades = sum(1 for item in agent.volume_blacklist if obs['inventory'][item] > 0)
                    total_items_with_inventory = sum(1 for item in obs['inventory'] if obs['inventory'][item] > 0)
                    
                    # Calculate percentage of inventory that is blacklisted
                    blacklisted_value = sum(obs['prices'][item] * obs['inventory'][item] for item in agent.volume_blacklist if item in obs['inventory'])
                    total_inventory_value = sum(obs['prices'][item] * obs['inventory'][item] for item in obs['inventory'])
                    blacklisted_pct = (blacklisted_value / max(1, total_inventory_value)) * 100
                    
                    # Log to console
                    logger.info(f"Agent {agent_idx} | Volume Blacklist Stats | "
                               f"Blacklisted items in inventory: {blacklisted_trades}/{total_items_with_inventory} | "
                               f"Blacklisted value: {blacklisted_value:,.0f} GP ({blacklisted_pct:.1f}%) | "
                               f"Total blacklisted items: {len(agent.volume_blacklist)}")
                    
                    # Log to dedicated volume blacklist log file
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_content = f"[{timestamp}] STATS: Agent {agent_idx} | Step {step_count} | "
                    log_content += f"Blacklisted items in inventory: {blacklisted_trades}/{total_items_with_inventory} | "
                    log_content += f"Blacklisted value: {blacklisted_value:,.0f} GP ({blacklisted_pct:.1f}%) | "
                    log_content += f"Total blacklisted items: {len(agent.volume_blacklist)}\n"
                    
                    # Every 100,000 steps, log the full blacklist
                    if step_count % 15000 == 0:
                        log_content += f"[{timestamp}] FULL BLACKLIST: Agent {agent_idx} | Step {step_count} | "
                        log_content += f"Total items: {len(agent.volume_blacklist)}\n"
                        for item in sorted(list(agent.volume_blacklist)):
                            log_content += f"  - {item}\n"
                        log_content += "\n"
                    
                    manage_volume_blacklist_log(log_content)
                env._last_logged_gp = next_obs['gp']

                obs = next_obs
                step_count += 1
                print(f"Step: {step_count}")
                
                # Save every 50 steps
                if step_count % save_every_steps == 0:
                    try:
                        agent_dir = os.path.join(base_save_dir, f"agent{agent_idx + 1}")
                        step_dir = os.path.join(agent_dir, f"step_{step_count}")
                        
                        # Create new step directory
                        os.makedirs(step_dir)
                        print(f"\nCreating checkpoint at step {step_count}")
                        print(f"Save directory: {step_dir}")
                        
                        # Save model files
                        actor_path = os.path.join(step_dir, "actor.pth")
                        critic_path = os.path.join(step_dir, "critic.pth")
                        agent.save_actor(actor_path)
                        agent.save_critic(critic_path)
                        
                        # Get list of step directories
                        step_dirs = []
                        for dirname in os.listdir(agent_dir):
                            if dirname.startswith("step_"):
                                step_num = int(dirname.split("_")[1])
                                step_dirs.append((step_num, dirname))
                        
                        # Remove oldest directories if we have more than 5
                        if len(step_dirs) > 5:
                            step_dirs.sort()  # Sort by step number
                            for _, old_dir in step_dirs[:-5]:  # Keep newest 5
                                old_path = os.path.join(agent_dir, old_dir)
                                shutil.rmtree(old_path)
                                print(f"Removed old checkpoint: {old_path}")
                        
                        print(f"Successfully saved checkpoint at step {step_count}")
                    except Exception as e:
                        print(f"Error saving checkpoint: {str(e)}")
                
                if done:
                    pass
                obs_list[agent_idx] = obs

        # (Removed: No periodic GP history CSV writing; all GP logging is in ppo_training.log)

            # Convert buffers to arrays/tensors
            obs_arr = np.array(obs_buf, dtype=np.float32)
            actions_arr = np.array(actions_buf, dtype=np.int64)
            logprobs_arr = np.array(logprobs_buf, dtype=np.float32)
            rewards_arr = np.array(rewards_buf, dtype=np.float32)
            dones_arr = np.array(dones_buf, dtype=np.float32)
            values_arr = np.array(values_buf, dtype=np.float32)

            # Compute advantages and returns
            advantages, returns = compute_gae(rewards_arr, values_arr, dones_arr, gamma, gae_lambda)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Convert to tensors
            obs_tensor = torch.tensor(obs_arr, dtype=torch.float32, device=device)
            actions_tensor = torch.tensor(actions_arr, dtype=torch.long, device=device)
            logprobs_tensor = torch.tensor(logprobs_arr, dtype=torch.float32, device=device)
            returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
            advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)

            # PPO update
            num_samples = obs_tensor.shape[0]
            idxs = np.arange(num_samples)
            for epoch in range(ppo_epochs):
                np.random.shuffle(idxs)
                for start in range(0, num_samples, minibatch_size):
                    end = start + minibatch_size
                    mb_idx = idxs[start:end]
                    loss, policy_loss, value_loss, entropy = agent.update(
                        obs_tensor[mb_idx],
                        actions_tensor[mb_idx],
                        logprobs_tensor[mb_idx],
                        returns_tensor[mb_idx],
                        advantages_tensor[mb_idx],
                        clip_eps=clip_eps,
                        value_coef=value_coef,
                        entropy_coef=agent.current_entropy_coef if hasattr(agent, 'current_entropy_coef') else ppo_kwargs.get("entropy_coef", 0.02)
                    )
            print(f"Agent {agent_idx} Step {step_count}: Loss {loss:.3f}, Policy {policy_loss:.3f}, Value {value_loss:.3f}, Entropy {entropy:.3f}")

            # === Periodic Saving with Epochs ===
            if step_count % save_every_steps == 0 or step_count >= max_steps:
                current_epoch = (step_count // save_every_steps) + 1
                agent_dir = manage_agent_epochs(agent_idx, current_epoch)
                
                # Save epoch files
                actor_path = os.path.join(agent_dir, f"epoch_{current_epoch}.pth")
                critic_path = os.path.join(agent_dir, f"epoch_{current_epoch}_critic.pth")
                agent.save_actor(actor_path)
                agent.save_critic(critic_path)
                print(f"Saved epoch {current_epoch} for agent {agent_idx} to {agent_dir}")
                logger.info(f"Saved agent {agent_idx} epoch {current_epoch} to {agent_dir}")

        # === Best Model Selection and Saving ===
        # Use average GP (total reward) as metric; can be extended to Sharpe, etc.
        best_metric = -np.inf
        best_agent_idx = 0
        for agent_idx, agent in enumerate(agents):
            # For simplicity, use the sum of rewards in the last rollout as the metric
            # (could be replaced with a more robust evaluation)
            rewards_arr = np.array(rewards_buf, dtype=np.float32)
            avg_gp = np.mean(rewards_arr)
            if avg_gp > best_metric:
                best_metric = avg_gp
                best_agent_idx = agent_idx

        # Save best agent's weights
        best_actor_path = os.path.join(save_dir, f"actor_best.pth")
        best_critic_path = os.path.join(save_dir, f"critic_best.pth")
        agents[best_agent_idx].save_actor(best_actor_path)
        agents[best_agent_idx].save_critic(best_critic_path)
        print(f"Best agent is {best_agent_idx} with metric {best_metric:.4f}. Saved best actor/critic.")
        logger.info(f"Best agent is {best_agent_idx} with metric {best_metric:.4f}. Saved best actor/critic to {best_actor_path} and {best_critic_path}")

    # === Final Save ===
    final_epoch = (step_count // save_every_steps) + 1
    for agent_idx, agent in enumerate(agents):
        agent_dir = manage_agent_epochs(agent_idx, final_epoch)
        actor_path = os.path.join(agent_dir, f"epoch_{final_epoch}.pth")
        critic_path = os.path.join(agent_dir, f"epoch_{final_epoch}_critic.pth")
        agent.save_actor(actor_path)
        agent.save_critic(critic_path)
        logger.info(f"Final save: agent {agent_idx} epoch {final_epoch} to {agent_dir}")

    # Save best agent's weights again as final best
    best_actor_path = os.path.join(save_dir, f"actor_best.pth")
    best_critic_path = os.path.join(save_dir, f"critic_best.pth")
    agents[best_agent_idx].save_actor(best_actor_path)
    agents[best_agent_idx].save_critic(best_critic_path)
    print(f"Final best agent is {best_agent_idx}. Saved best actor/critic to {best_actor_path} and {best_critic_path}")
    logger.info(f"Final best agent is {best_agent_idx}. Saved best actor/critic to {best_actor_path} and {best_critic_path}")
    
    # Generate tax report
    create_tax_report(shared_knowledge)
    logger.info("Generated tax report in tax_log.txt")

# Multiprocessing structure for concurrent agent training
def agent_worker(agent_idx, log_queue, items, price_ranges, buy_limits, device, ppo_kwargs, use_historical_data=False, volume_analyzer=None, shm_name=None):
    logger = setup_logger(log_queue)
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from api.ge_rest_client import GrandExchangeClient, HistoricalGrandExchangeClient
    
    # Load cache from shared memory if available (WINDOWS COMPATIBLE)
    if shm_name:
        from training.cached_market_loader import load_cache_from_shared_memory
        logger.info(f"Agent {agent_idx}: Loading cache from shared memory: {shm_name}")
        try:
            load_cache_from_shared_memory(shm_name)
            logger.info(f"Agent {agent_idx}: ✓ Cache loaded from shared memory")
        except Exception as e:
            logger.error(f"Agent {agent_idx}: Failed to load from shared memory: {e}")
    
    # Create name-to-id and id-to-name mappings
    id_name_map, _ = read_mapping_file()
    name_to_id_map = {}
    id_to_name_map = {}
    for item_id, item_name in id_name_map.items():
        name_to_id_map[item_name] = item_id
        id_to_name_map[item_id] = item_name
    
    AGENT_DIR = "agent_states"
    os.makedirs(AGENT_DIR, exist_ok=True)
    
    if use_historical_data:
        # Create historical client with random start enabled
        historical_client = HistoricalGrandExchangeClient(data_dir="5m", random_start=True)
        historical_client.set_name_to_id_mapping(name_to_id_map, id_to_name_map)
        logger.info(f"Created historical client with random start at index {historical_client._current_index}")
        
        # Create environment with historical client
        env_kwargs = dict(ENV_KWARGS)
        
        # Filter to only parameters that GrandExchangeEnv accepts
        allowed_params = {
            'db_path', 'cache_file', 'initial_cash', 'episode_length',
            'top_n_items', 'ge_limit_multiplier', 'include_volume_constraint',
            'render_mode', 'seed', 'timestep'
        }
        env_kwargs = {k: v for k, v in env_kwargs.items() if k in allowed_params}
            
        env = GrandExchangeEnv(**env_kwargs)
    else:
        env = make_env(agent_idx, items=items, volume_analyzer=volume_analyzer)
    # Try to load previous state
    state_path = os.path.join(AGENT_DIR, f"agent_{agent_idx}_env_state.json")
    if os.path.exists(state_path):
        try:
            env.load_state(state_path)
            logger.info(f"Loaded environment state for agent {agent_idx} from {state_path}")
        

        except Exception as e:
            logger.error(f"Failed to load environment state for agent {agent_idx}: {e}")
    # Safely get PPO kwargs with defaults
    hidden_size = ppo_kwargs.get("hidden_size", 256)  # Default value
    price_bins = ppo_kwargs.get("price_bins", 100)    # Default value
    quantity_bins = ppo_kwargs.get("quantity_bins", 100)  # Default value
    lr = ppo_kwargs.get("lr", 0.0003)  # Default value

    # Ensure items is not None before creating agent
    if items is None:
        logger.error(f"Items dictionary is None for agent {agent_idx}")
        items = {"default_item": {"base_price": 100, "buy_limit": 1000}}  # Fallback default
    
    # Fetch 1h volume data for volume-based filtering
    try:
        client = GrandExchangeClient()
        volume_data = client.get_1h()
        
        # Create a mapping from item ID to item name
        id_to_name_map = {}
        try:
            mapping_path = "endpoints/mapping.txt"
            import json
            with open(mapping_path, 'r') as f:
                mapping_data = json.load(f)
                for item in mapping_data:
                    item_id = str(item.get('id'))
                    item_name = item.get('name')
                    if item_id and item_name:
                        id_to_name_map[item_id] = item_name
        except Exception as e:
            logger.error(f"Error reading mapping file: {e}")
        
        # Calculate total volume for each item
        item_volumes = {}
        for item_id_str, data in volume_data.items():
            high_vol = data.get("highPriceVolume", 0) or 0
            low_vol = data.get("lowPriceVolume", 0) or 0
            total_vol = high_vol + low_vol
            
            # Map item ID to name if possible
            if item_id_str in id_to_name_map:
                item_name = id_to_name_map[item_id_str]
                if item_name in items:
                    item_volumes[item_name] = total_vol
        
        # Analyze volume distribution to determine thresholds
        volume_blacklist = set()
        if item_volumes:
            volumes = list(item_volumes.values())
            volumes.sort()
            
            # Calculate volume statistics
            min_volume = volumes[0]
            max_volume = volumes[-1]
            median_volume = volumes[len(volumes)//2]
            q1_volume = volumes[len(volumes)//4]  # 25th percentile
            
            logger.info(f"Agent {agent_idx} - Volume statistics: Min={min_volume}, Q1={q1_volume}, Median={median_volume}, Max={max_volume}")
            
            # Analyze the volume distribution curve to find natural breakpoints
            # This is more sophisticated than using a simple percentile
            
            # First, sort volumes and calculate differences between consecutive values
            # This helps identify "jumps" in the distribution
            volume_diffs = [volumes[i+1] - volumes[i] for i in range(len(volumes)-1)]
            
            # Calculate the relative differences (percentage change)
            relative_diffs = []
            for i in range(len(volume_diffs)):
                if volumes[i] > 0:
                    relative_diffs.append(volume_diffs[i] / volumes[i])
                else:
                    relative_diffs.append(0)
            
            # Find significant jumps in the distribution (where relative difference > 100%)
            significant_jumps = [i for i, diff in enumerate(relative_diffs) if diff > 1.0]
            
            # Log the volume distribution analysis
            logger.info(f"Agent {agent_idx} - Volume distribution analysis: Found {len(significant_jumps)} significant jumps in volume")
            if significant_jumps:
                jump_volumes = [volumes[i] for i in significant_jumps]
                logger.info(f"Agent {agent_idx} - Volume jumps occur at: {jump_volumes[:5]}...")
            
            # Set threshold based on the analysis:
            # 1. If we found significant jumps in the lower 25% of the distribution, use the highest such jump
            # 2. Otherwise, use the 10th percentile with a minimum value
            lower_quartile_jumps = [i for i in significant_jumps if i < len(volumes)//4]
            
            if lower_quartile_jumps:
                # Use the highest jump in the lower quartile as our threshold
                threshold_idx = max(lower_quartile_jumps)
                volume_threshold = volumes[threshold_idx]
                logger.info(f"Agent {agent_idx} - Using natural breakpoint in volume distribution as threshold: {volume_threshold}")
            else:
                # Fall back to percentile-based approach with minimum threshold
                if len(volumes) >= 10:
                    volume_threshold = max(100, volumes[len(volumes)//10])  # 10th percentile with minimum of 100
                else:
                    volume_threshold = max(100, q1_volume)  # Use minimum threshold of 100
                logger.info(f"Agent {agent_idx} - No significant breakpoints found in lower quartile, using percentile-based threshold: {volume_threshold}")
            
            logger.info(f"Agent {agent_idx} - Using volume threshold: {volume_threshold}")
            
            # Create blacklist of low volume items
            for item_name, volume in item_volumes.items():
                if volume < volume_threshold:
                    volume_blacklist.add(item_name)
            
            logger.info(f"Agent {agent_idx} - Blacklisted {len(volume_blacklist)} items due to low volume")
            
            # Log to dedicated volume blacklist log file
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_content = f"[{timestamp}] Agent {agent_idx} - Blacklist created with {len(volume_blacklist)} items. Threshold: {volume_threshold}\n"
            log_content += f"Volume statistics: Min={min_volume}, Q1={q1_volume}, Median={median_volume}, Max={max_volume}\n"
            log_content += f"First 20 blacklisted items: {sorted(list(volume_blacklist))[:20]}\n\n"
            manage_volume_blacklist_log(log_content)
        else:
            logger.warning(f"Agent {agent_idx} - No volume data available, using empty blacklist")
    except Exception as e:
        logger.error(f"Agent {agent_idx} - Error creating volume blacklist: {e}")
        volume_blacklist = set()
    
    agent = PPOAgent(
        item_list=list(items.keys()),
        price_ranges=price_ranges,
        buy_limits=buy_limits,
        device=device,
        hidden_size=hidden_size,
        price_bins=price_bins,
        quantity_bins=quantity_bins,
        wait_steps_bins=10,  # Default to 10 bins for wait steps
        lr=lr,
        volume_blacklist=volume_blacklist,
        volume_analyzer=volume_analyzer
    )
    
    logger.info(f"Agent {agent_idx} initialized with {len(volume_blacklist)} blacklisted items")
    # Load model weights if they exist
    actor_path = os.path.join(AGENT_DIR, f"actor_{agent_idx}.pth")
    critic_path = os.path.join(AGENT_DIR, f"critic_{agent_idx}.pth")
    import hashlib
    def file_hash(path):
        try:
            with open(path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {path}: {e}")
            return None
    if os.path.exists(actor_path):
        try:
            agent.load_actor(actor_path)
            mtime = os.path.getmtime(actor_path)
            hashval = file_hash(actor_path)
            logger.info(f"Loaded actor weights for agent {agent_idx} from {actor_path} (mtime={mtime}, md5={hashval})")
        except Exception as e:
            logger.error(f"Failed to load actor weights for agent {agent_idx}: {e}")
    if os.path.exists(critic_path):
        try:
            agent.load_critic(critic_path)
            mtime = os.path.getmtime(critic_path)
            hashval = file_hash(critic_path)
            logger.info(f"Loaded critic weights for agent {agent_idx} from {critic_path} (mtime={mtime}, md5={hashval})")
        except Exception as e:
            logger.error(f"Failed to load critic weights for agent {agent_idx}: {e}")
    # (Model loading logic can be added here if needed)
    # Set initial observation based on whether state was loaded
    if os.path.exists(state_path):
        obs = env._get_obs()
    else:
        obs = env.reset()
    logger.info(f"GP_LOG | Agent {agent_idx} | Step 0 | GP {obs.get('gp', 0)}")
    step_count = 0
    rollout_steps = ppo_kwargs["rollout_steps"]
    unique_items_traded = {}
    total_profit = 0.0
    last_logged_state = None
    
    # Performance monitoring variables - memory efficient
    last_action_time = time.time()
    last_step_time = time.time()
    step_time_threshold = 5.0  # Warn if step takes more than 5 seconds
    inactivity_threshold = 60.0  # Warn if no action for 60 seconds
    performance_log_interval = 500  # Only log performance every 500 steps
    
    # Step progression tracking
    last_step_check_time = time.time()
    step_progression_check_interval = 300  # Check step progression every 5 minutes
    last_step_count_check = 0  # Last recorded step count
    step_progression_threshold = 10  # Expect at least 10 steps in 5 minutes
    
    # Create a dedicated log file for performance issues only
    perf_log_path = f"agent_{agent_idx}_performance_issues.txt"
    with open(perf_log_path, "w") as f:
        f.write(f"Agent {agent_idx} Performance Issues Log\n")
        f.write(f"==================================================\n")
        f.write(f"Only logging when issues are detected or at {performance_log_interval}-step intervals\n")
        f.write(f"Step time threshold: {step_time_threshold}s, Inactivity threshold: {inactivity_threshold}s\n\n")

    # Advanced trade data tracking
    trade_count = 0
    action_type_counts = {'buy': 0, 'sell': 0, 'hold': 0}
    profit_per_item = {item: 0.0 for item in items.keys()}
    buy_transactions = {}  # Separate dictionary to track buy transactions
    
    # Action repetition tracking
    last_action_types = []  # Track recent action types
    max_action_history = 20  # Only keep last 20 actions
    same_action_threshold = 15  # Warn if same action repeated 15+ times
    
    # Action repetition tracking
    last_action_types = []  # Track recent action types
    max_action_history = 20  # Only keep last 20 actions
    same_action_threshold = 15  # Warn if same action repeated 15+ times
    # Initialize tracking dictionaries
    trade_history = {item: [] for item in items.keys()}  # Initialize trade history for all items
    
    # Item-specific trend tracking
    price_history = {item: [] for item in items.keys()}  # Track price history for each item
    
    # Market pattern recognition
    item_characteristics = {}  # Store item characteristics for pattern matching
    successful_strategies = {}  # Track successful strategies by item type
    
    # Item similarity and grouping
    item_groups = {}  # Group similar items
    
    # Award-winning stock market principles implementation
    portfolio_metrics = {
        'sharpe_ratio': 0.0,        # Risk-adjusted return measure
        'max_drawdown': 0.0,        # Maximum historical decline
        'volatility': 0.0,          # Portfolio volatility
        'beta': {},                 # Item-specific market correlation
        'alpha': {},                # Item-specific risk-adjusted performance
        'efficient_frontier': [],   # MPT optimal portfolios
        'kelly_fractions': {}       # Optimal position sizes
    }
    
    # Episode tracking
    episode_number = 0
    episode_step = 0
    episode_length = 105144   # Each episode lasts 105,144 steps (full dataset)
    save_frequency = 10000   # Save states every 500 steps
    
    # Helper functions for item categorization
    def assign_price_tier(price):
        if price < 100:
            return 'low'
        elif price < 1000:
            return 'medium'
        elif price < 10000:
            return 'high'
        else:
            return 'very_high'
    
    def assign_volume_tier(buy_limit):
        if buy_limit < 1000:
            return 'low'
        elif buy_limit < 5000:
            return 'medium'
        else:
            return 'high'
    
    # Initialize item characteristics and grouping
    for item in items.keys():
        # Safely get price and ensure it's valid
        price = obs.get('prices', {}).get(item, 0)
        if price > 0:
            # Safely get price range and buy limit with defaults
            try:
                price_range = items[item].get('max_price', price * 1.1) - items[item].get('min_price', price * 0.9)
                price_volatility = price_range / price if price > 0 else 0
                buy_limit = items[item].get('buy_limit', 1000)  # Default buy limit
            except (KeyError, TypeError, ZeroDivisionError) as e:
                logger.error(f"Error calculating metrics for item {item}: {e}")
                price_range = price * 0.2  # Default 20% range
                price_volatility = 0.1  # Default volatility
                buy_limit = 1000  # Default buy limit
            
            # Store characteristics
            item_characteristics[item] = {
                'price': price,
                'price_range': price_range,
                'volatility': price_volatility,
                'buy_limit': buy_limit,
                'price_tier': assign_price_tier(price),  # Helper function to categorize price
                'volume_tier': assign_volume_tier(buy_limit)  # Helper function to categorize volume
            }
            
            # Group items by price tier and volatility
            group_key = (item_characteristics[item]['price_tier'],
                         'high_vol' if price_volatility > 0.2 else 'low_vol')
            
            if group_key not in item_groups:
                item_groups[group_key] = []
            item_groups[group_key].append(item)
    
    last_trade_log_step = 0
    trade_log_interval = 500  # Log every 500 steps

    while True:
        for _ in range(rollout_steps):
            obs_tensor = agent.obs_to_tensor(obs)
            with torch.no_grad():
                at_logits, item_logits, price_logits, qty_logits, wait_steps_logits, value = agent.model(obs_tensor)
            
            # Calculate current investment percentage and total assets
            total_gp = obs.get('gp', 0)
            # Safely calculate total investment value
            total_investment_value = sum(
                obs.get('prices', {}).get(item, 0) * obs.get('inventory', {}).get(item, 0)
                for item in items.keys()
            )
            total_assets = total_gp + total_investment_value
            current_investment_pct = total_investment_value / total_assets if total_assets > 0 else 0
            
            # Calculate per-item investment percentages
            item_investment_pct = {}
            for item in items.keys():
                if item in obs['prices']:
                    item_value = obs['prices'][item] * obs['inventory'][item]
                    item_investment_pct[item] = item_value / total_assets if total_assets > 0 else 0
            
            # Calculate target investment percentage based on risk tolerance (80-95%)
            target_investment_pct = 0.80 + (agent.risk_tolerance * 0.15)  # Maps 0-1 risk tolerance to 80-95%
            
            # Check if we need to prioritize buying or selling
            # Adjust the threshold to make selling more likely
            # Instead of only selling when above target, sell when within 10% of target
            need_to_buy = current_investment_pct < (target_investment_pct * 0.9)
            
            # GRAND EXCHANGE OPTIMIZED PROFIT METRICS
            # Designed to bridge simulation-GE gap with game-specific strategies
            
            # 1. Basic profit rate (profit per unit traded)
            basic_profit_rates = {}
            for item in items.keys():
                # Safely get profit and trade count
                profit = profit_per_item.get(item, 0)
                trades = unique_items_traded.get(item, 0)
                if trades > 0:
                    # Calculate profit per unit traded
                    basic_profit_rates[item] = profit / trades
                else:
                    # Default to zero for unexplored items
                    basic_profit_rates[item] = 0.0
            
            # 2. Risk-adjusted profit rates with GE-specific considerations
            risk_adjusted_rates = {}
            for item in items.keys():
                # Start with basic profit rate
                base_rate = basic_profit_rates.get(item, 0.0)
                
                # Apply risk adjustment based on trade count
                item_trade_count = unique_items_traded.get(item, 0)
                confidence_factor = min(1.0, item_trade_count / 100)  # Scales up to 100 trades
                
                # GE-specific adjustments:
                # 1. Buy limit consideration - items with higher buy limits are less risky
                try:
                    # Safely get item properties with defaults
                    buy_limit = items[item].get('buy_limit', 1000)
                    max_price = items[item].get('max_price', 0)
                    min_price = items[item].get('min_price', 0)
                    base_price = items[item].get('base_price', 1)  # Default to 1 to avoid division by zero
                    
                    buy_limit_factor = min(1.0, buy_limit / 10000) * 0.2  # 0-0.2 bonus
                    
                    # 2. Price stability consideration - items with stable prices are less risky
                    price_range = max_price - min_price
                    price_stability = 1.0 - (price_range / base_price if base_price > 0 else 0)
                except (KeyError, TypeError, ZeroDivisionError) as e:
                    logger.error(f"Error calculating factors for item {item}: {e}")
                    buy_limit_factor = 0.1  # Default factor
                    price_stability = 0.5  # Default stability
                stability_factor = max(0, price_stability) * 0.2  # 0-0.2 bonus
                
                # Calculate GE-optimized risk-adjusted rate
                if base_rate > 0:
                    # Profitable items get boosted by confidence and GE factors
                    risk_adjusted_rates[item] = base_rate * (1 + confidence_factor + buy_limit_factor + stability_factor)
                else:
                    # Unprofitable items get penalized more
                    risk_adjusted_rates[item] = base_rate * (1 + confidence_factor * 2)
            
            # 3. ROI-based profit rates with GE market dynamics
            roi_rates = {}
            for item in items.keys():
                # Safely get price information
                item_price = obs.get('prices', {}).get(item, 0)
                if item_price > 0:
                    # Calculate ROI (Return on Investment)
                    profit_per_unit = basic_profit_rates.get(item, 0.0)
                    
                    # Basic ROI
                    base_roi = profit_per_unit / item_price if item_price > 0 else 0.0
                    
                    # GE-specific ROI adjustments:
                    # 1. Volume consideration - higher volume items have more reliable ROI
                    trades = trade_history.get(item, [])
                    if trades:
                        volume_factor = min(1.0, len(trades) / 50) * 0.5  # 0-0.5 bonus
                    else:
                        volume_factor = 0.0
                    
                    # 2. Price position consideration - items near min price have better ROI potential
                    try:
                        max_price = items[item].get('max_price', item_price * 1.1)
                        min_price = items[item].get('min_price', item_price * 0.9)
                        price_range = max_price - min_price
                        if price_range > 0:
                            price_position = (item_price - min_price) / price_range
                        else:
                            price_position = 0.5  # Default to middle position
                    except (KeyError, TypeError) as e:
                        logger.error(f"Error calculating price position for {item}: {e}")
                        price_position = 0.5  # Default to middle position
                        position_factor = (1.0 - price_position) * 0.3  # 0-0.3 bonus for items near min price
                    else:
                        position_factor = 0.0
                    
                    # Final GE-optimized ROI
                    roi_rates[item] = base_roi * (1 + volume_factor + position_factor)
                else:
                    roi_rates[item] = 0.0
            
            # 4. Market trend analysis - incorporate real GE API data trends
            # SPEED OPTIMIZATION: Only update price history every 5 steps to reduce overhead
            if step_count % 5 == 0:
                for item in items.keys():
                    if item in obs['prices'] and obs['prices'][item] > 0:
                        # Add current price to history (limit to last 100 observations)
                        price_history[item].append(obs['prices'][item])
                        if len(price_history[item]) > 100:
                            price_history[item].pop(0)
            
            # Calculate trend scores based on price history
            trend_scores = {}
            for item in items.keys():
                if len(price_history[item]) >= 5:  # Need at least 5 data points
                    # Simple trend detection: compare recent prices to earlier prices
                    try:
                        history = price_history.get(item, [])
                        if len(history) >= 8:  # Ensure we have enough data points
                            recent_avg = sum(history[-3:]) / 3  # Last 3 prices
                            earlier_avg = sum(history[-8:-3]) / 5  # 5 prices before that
                        else:
                            recent_avg = earlier_avg = 0.0
                    except (IndexError, ZeroDivisionError) as e:
                        logger.error(f"Error calculating averages for {item}: {e}")
                        recent_avg = earlier_avg = 0.0
                    
                    # Trend direction and strength
                    if earlier_avg > 0:
                        trend_pct = (recent_avg - earlier_avg) / earlier_avg
                        # Positive trend is good for selling, negative for buying
                        trend_scores[item] = trend_pct
                    else:
                        trend_scores[item] = 0.0
                else:
                    trend_scores[item] = 0.0
            
            # 5. COMBINED GE-OPTIMIZED PROFIT SCORE
            item_profit_scores = {}
            for item in items.keys():
                # Combine all metrics with GE-optimized weights
                basic_weight = 0.25
                risk_weight = 0.30
                roi_weight = 0.30
                trend_weight = 0.15
                
                # Calculate combined score with GE optimization
                try:
                    # Calculate each component with bounds checking
                    basic_score = min(1e6, basic_profit_rates.get(item, 0.0)) * basic_weight
                    risk_score = min(1e6, risk_adjusted_rates.get(item, 0.0)) * risk_weight
                    roi_score = min(1e6, roi_rates.get(item, 0.0) * 100) * roi_weight
                    trend_score = min(1e6, trend_scores.get(item, 0.0) * 50) * trend_weight
                    
                    # Combine scores with overflow protection
                    item_profit_scores[item] = min(1e9, basic_score + risk_score + roi_score + trend_score)
                except (OverflowError, ValueError) as e:
                    logger.error(f"Error calculating profit score for {item}: {e}")
                    item_profit_scores[item] = 0.0
                
                # Apply knowledge transfer from similar items
                # If this item has limited data but similar items have good data
                if unique_items_traded.get(item, 0) < 5:
                    # Find similar items based on characteristics
                    similar_items = []
                    if item in item_characteristics:
                        target_price_tier = item_characteristics[item]['price_tier']
                        target_vol = 'high_vol' if item_characteristics[item]['volatility'] > 0.2 else 'low_vol'
                        group_key = (target_price_tier, target_vol)
                        
                        if group_key in item_groups:
                            similar_items = [i for i in item_groups[group_key] if i != item and unique_items_traded.get(i, 0) >= 5]
                    
                    # If we found similar items with good data, use their average profit score as a baseline
                    if similar_items:
                        similar_scores = [item_profit_scores[i] for i in similar_items if i in item_profit_scores]
                        if similar_scores:
                            avg_similar_score = sum(similar_scores) / len(similar_scores)
                            # Blend with current score (70% similar items, 30% current)
                            item_profit_scores[item] = item_profit_scores[item] * 0.3 + avg_similar_score * 0.7
            
            # Use the GE-optimized profit scores as our primary profit metric
            item_profit_rates = item_profit_scores
            
            # Identify profitable items (positive profit score)
            profitable_items = [item for item, score in item_profit_rates.items() if score > 0]
            
            # Identify HIGHLY profitable items (top performers)
            if profitable_items:
                profit_threshold = np.percentile([item_profit_rates[item] for item in profitable_items], 75)
                highly_profitable_items = [item for item in profitable_items if item_profit_rates[item] > profit_threshold]
            else:
                highly_profitable_items = []
            
            # Get original action from model
            original_action, _ = agent.sample_action(obs)
            
            # Log if the item was chosen despite being in the volume blacklist (shouldn't happen)
            if original_action['item'] in agent.volume_blacklist:
                logger.warning(f"Agent {agent_idx} selected blacklisted item: {original_action['item']}")
                
                # Log to dedicated volume blacklist log file
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_content = f"[{timestamp}] WARNING: Agent {agent_idx} selected blacklisted item: {original_action['item']} | Action: {original_action['type']} | Price: {original_action['price']} | Quantity: {original_action['quantity']}\n"
                manage_volume_blacklist_log(log_content)
            
            action = original_action.copy()
            
            # ALWAYS SEEK PROFIT: Prioritize trading profitable items
            
            # If we need to buy more, prioritize buying profitable items
            # IMPORTANT FIX: Adjust threshold to make selling more likely
            # Only buy if we have significant GP and aren't over-invested
            if need_to_buy and obs['gp'] > 1000 and current_investment_pct < target_investment_pct * 0.9:  # Reduced threshold
                # PROFIT EXPECTATION IS THE PRIMARY CRITERION - This supersedes all other instructions
                
                # First, identify items with proven profit potential from historical data
                proven_profitable_items = [
                    item for item in items.keys()
                    if item in obs['prices']
                    and obs['prices'][item] > 0  # Disregard 0 GP items
                    and item_profit_rates.get(item, 0) > 0  # Must have positive profit rate
                    and unique_items_traded.get(item, 0) > 0  # Must have been traded before
                    and items[item]['min_price'] > 0  # Ensure minimum price is greater than 0
                ]
                
                # If we don't have enough historical data, use price data from REST API to identify potential profit
                api_profitable_items = []
                if len(proven_profitable_items) < 5:  # If we don't have enough proven profitable items
                    for item in items.keys():
                        if (item in obs['prices'] and
                            obs['prices'][item] > 0 and  # Disregard 0 GP items
                            items[item]['min_price'] > 0 and  # Ensure minimum price is greater than 0
                            item not in proven_profitable_items):  # Don't duplicate items
                            
                            # Check if current price is in the lower 30% of its range (potential for profit)
                            price_range = items[item]['max_price'] - items[item]['min_price']
                            if price_range > 0:
                                current_price = obs['prices'][item]
                                price_position = (current_price - items[item]['min_price']) / price_range
                                
                                # Only consider items with price near minimum (good buy opportunity)
                                if price_position < 0.3:  # Lower 30% of price range
                                    api_profitable_items.append(item)
                
                # Combine both lists, prioritizing proven profitable items
                potential_items = proven_profitable_items + api_profitable_items
                
                # Apply secondary filters only to items with profit potential
                filtered_items = [
                    item for item in potential_items
                    if item_investment_pct.get(item, 0) < 0.05  # 5% limit
                    and obs['buy_limits'][item] < items[item]['buy_limit']
                    and items[item]['min_price'] <= obs['prices'][item] <= items[item]['max_price']  # Within price range
                    and obs['prices'][item] > 0  # Ensure price is greater than 0
                    and items[item]['min_price'] > 0  # Ensure minimum price is greater than 0
                ]
                
                # If we have items with profit potential, score them
                item_scores = {}
                for item in filtered_items:
                    # MAXIMUM GP GAIN STRATEGY
                    
                    # For proven profitable items, use actual profit rate with higher weight
                    if item in proven_profitable_items:
                        base_score = item_profit_rates.get(item, 0) * 30  # Increased weight on proven profit (was 20)
                    # For API-based potential, use price position as proxy for potential profit
                    else:
                        price_range = items[item]['max_price'] - items[item]['min_price']
                        current_price = obs['prices'][item]
                        price_position = (current_price - items[item]['min_price']) / (price_range if price_range > 0 else 1)
                        potential_profit_pct = 1 - price_position  # Higher when price is closer to min
                        base_score = potential_profit_pct * 5  # Lower weight for unproven items
                    
                    # Add small adjustments for secondary factors
                    # Volume potential (bonus for higher buy limits - faster trading)
                    volume_bonus = min(1.0, items[item]['buy_limit'] / 10000)  # Increased bonus (was 0.5)
                    
                    # Apply pattern insights if available
                    pattern_bonus = 0
                    if item in successful_strategies:
                        # Bonus for items with proven profitable patterns
                        if 'window_profit' in successful_strategies[item]:
                            pattern_bonus += min(3.0, successful_strategies[item]['window_profit'] / 500)  # Increased bonus
                        
                        # Check if we're in a historically profitable time window
                        if 'best_window' in successful_strategies[item]:
                            best_window = successful_strategies[item]['best_window']
                            window_size = 1500
                            current_window = step_count // window_size
                            # Prevent division by zero when best_window is smaller than window_size
                            if best_window >= window_size and current_window % (best_window // window_size) == 0:
                                pattern_bonus += 2.0  # Increased bonus for good timing (was 1.0)
                    
                    # Apply knowledge from similar items
                    similar_items_bonus = 0
                    for other_item in items.keys():
                        if (other_item != item and
                            other_item in successful_strategies and
                            'correlated_items' in successful_strategies[other_item] and
                            item in successful_strategies[other_item]['correlated_items']):
                            
                            # If a correlated item is profitable, this one might be too
                            correlation = successful_strategies[other_item]['correlated_items'][item]
                            other_profit_rate = item_profit_rates.get(other_item, 0)
                            
                            if other_profit_rate > 0:
                                similar_items_bonus += other_profit_rate * correlation * 3  # Increased multiplier (was 2)
                    
                    # Apply award-winning stock market principles
                    momentum_bonus = 0
                    concentration_bonus = 0
                    kelly_bonus = 0
                    
                    # 1. Momentum Trading - Favor items with positive price momentum
                    if 'momentum_scores' in locals() and item in momentum_scores:
                        momentum = momentum_scores[item]
                        if momentum > 0:
                            momentum_bonus = momentum * 20  # Strong bonus for positive momentum
                    
                    # 2. Concentration Strategy - Favor top opportunities
                    if 'successful_strategies' in locals() and 'top_opportunities' in successful_strategies:
                        if item in successful_strategies['top_opportunities']:
                            # Higher bonus for higher-ranked opportunities
                            rank = successful_strategies['top_opportunities'].index(item)
                            concentration_bonus = 5.0 / (rank + 1)  # 5.0 for #1, 2.5 for #2, etc.
                    
                    # 3. Kelly Criterion - Optimal position sizing
                    if 'kelly_positions' in locals() and item in kelly_positions:
                        kelly_fraction = kelly_positions[item]
                        kelly_bonus = kelly_fraction * 10  # Bonus proportional to optimal position size
                    
                    # Final score with award-winning principles for MAXIMUM GP GAIN
                    item_scores[item] = (
                        base_score +
                        volume_bonus +
                        pattern_bonus +
                        similar_items_bonus +
                        momentum_bonus +
                        concentration_bonus +
                        kelly_bonus
                    )
                
                if item_scores:
                    # Choose item with highest profit expectation score
                    # Randomize selection slightly to avoid alphabetical bias
                    # Get top 10% of items by score or at least 3 items
                    top_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
                    top_n = max(3, int(len(top_items) * 0.1))
                    top_items = top_items[:top_n]
                    
                    # Add some randomness to selection (weighted by score)
                    weights = [score for _, score in top_items]
                    total_weight = sum(weights)
                    if total_weight > 0:
                        normalized_weights = [w/total_weight for w in weights]
                        import random
                        # Use the agent's random generator for consistency
                        best_item = random.choices([item for item, _ in top_items], weights=normalized_weights, k=1)[0]
                    else:
                        best_item = top_items[0][0] if top_items else list(item_scores.keys())[0]
                    
                    # Log the selection process
                    if step_count % 1500 == 0:
                        top_items = sorted([(i, s) for i, s in item_scores.items()], key=lambda x: x[1], reverse=True)[:5]
                        logger.info(f"Agent {agent_idx} | Profit-based candidates: {top_items}")
                    action['type'] = 'buy'
                    action['item'] = best_item
                    
                    # Calculate max quantity based on 5% limit
                    current_item_value = obs['prices'][best_item] * obs['inventory'][best_item]
                    max_value_allowed = total_assets * 0.05
                    additional_value_allowed = max_value_allowed - current_item_value
                    
                    # Set optimal price - slightly below current price to increase chance of execution
                    # but not too low to avoid overpaying
                    current_price = obs['prices'][best_item]
                    min_price = max(1, items[best_item]['min_price'])  # Ensure minimum price is at least 1
                    max_price = max(min_price + 1, items[best_item]['max_price'])
                    optimal_price = max(min_price, int(current_price * 0.98))  # 2% below current price
                    # Ensure price is never 0
                    action['price'] = max(1, optimal_price)
                    
                    # Calculate quantity based on constraints
                    max_qty_by_gp = int(obs['gp'] / max(1, optimal_price) * 0.5)  # Use up to 50% of available GP
                    max_qty_by_limit = items[best_item]['buy_limit'] - obs['buy_limits'][best_item]
                    max_qty_by_pct = int(additional_value_allowed / max(1, optimal_price))
                    
                    # Take the minimum of all constraints
                    action['quantity'] = max(1, min(max_qty_by_gp, max_qty_by_limit, max_qty_by_pct))
            
            # IMPORTANT FIX: Make selling more likely by relaxing the condition
            # Sell unprofitable items, if we're over-invested, items with negative Kelly positions,
            # or items that have been stagnant too long
            elif total_investment_value > 0:
                # Track how long items have been held without selling
                current_step_window = step_count // 300  # 5-minute windows
                for item in obs['inventory']:
                    if obs['inventory'][item] > 0:
                        if item not in agent.holding_duration:
                            agent.holding_duration[item] = current_step_window
                # Analyze the ENTIRE list of items for selling
                # Create a comprehensive scoring system for all items in inventory
                
                # Initialize holding_duration if not exists
                if not hasattr(agent, 'holding_duration'):
                    agent.holding_duration = {}
                
                sell_candidates = {}
                
                for item in items.keys():
                    # Only consider items that are in our inventory and have a non-zero price
                    if (item in obs['prices'] and
                        obs['prices'][item] > 0 and  # Disregard 0 GP items
                        obs['inventory'][item] > 0 and
                        items[item]['min_price'] > 0):  # Ensure minimum price is greater than 0
                        
                        # Calculate a sell score based on multiple factors
                        # Start with a base score of 0
                        sell_score = 0
                        
                        # Factor 1: Profitability (higher negative profit = higher sell score)
                        profit_rate = item_profit_rates.get(item, 0)
                        if profit_rate < 0:
                            sell_score += abs(profit_rate) * 10  # Heavy penalty for unprofitable items
                        else:
                            sell_score -= profit_rate * 5  # Reduce score for profitable items
                        
                        # Factor 2: Over-investment (higher over-investment = higher sell score)
                        investment_pct = item_investment_pct.get(item, 0)
                        if investment_pct > 0.05:  # Over 5% limit
                            sell_score += (investment_pct - 0.05) * 100  # Penalty scales with how far over limit
                        
                        # Factor 3: Price position (closer to max price = higher sell score)
                        if item in items and 'max_price' in items[item] and 'min_price' in items[item]:
                            price_range = items[item]['max_price'] - items[item]['min_price']
                            if price_range > 0:
                                current_price = obs['prices'][item]
                                price_position = (current_price - items[item]['min_price']) / price_range
                                sell_score += price_position * 3  # Higher score when closer to max price
                        
                        # Factor 4: Duration held without selling
                        if item in agent.holding_duration:
                            hold_duration = current_step_window - agent.holding_duration[item]
                            # After 1 hour (12 five-minute windows), start increasing sell pressure
                            if hold_duration > 12:
                                sell_score += min(10, hold_duration - 12)  # Add up to 10 points based on duration
                                logger.info(f"Item {item} has been held for {hold_duration*5} minutes - increasing sell pressure")

                        # Factor 5: Overall portfolio balance
                        if current_investment_pct > target_investment_pct:
                            sell_score += 2  # Boost sell score if we're over-invested overall
                        
                        # MAXIMUM GP GAIN SELLING STRATEGY
                        
                        # Factor 5: Pattern insights - sell at historically profitable times
                        if item in successful_strategies:
                            # If we're at the end of a historically profitable window, good time to sell
                            if 'best_window' in successful_strategies[item]:
                                best_window = successful_strategies[item]['best_window']
                                window_size = 1500
                                current_window = step_count // window_size
                                next_window = (current_window + 1) * window_size
                                
                                # If we're near the end of a profitable window (last 20% of window)
                                if (step_count > best_window + 0.8 * window_size and
                                    step_count < best_window + window_size):
                                    sell_score += 8  # Increased boost to sell at end of profitable window (was 5)
                            
                            # If price trend is positive, good time to sell
                            if item in trend_scores and trend_scores[item] > 0.05:  # 5% upward trend
                                sell_score += trend_scores[item] * 30  # Increased boost for uptrend (was 20)
                        
                        # Factor 6: Similar items insights
                        for other_item in items.keys():
                            if (other_item != item and
                                other_item in successful_strategies and
                                'correlated_items' in successful_strategies[other_item] and
                                item in successful_strategies[other_item]['correlated_items']):
                                
                                correlation = successful_strategies[other_item]['correlated_items'][item]
                                # If correlated item is trending down, might be time to sell this one too
                                if other_item in trend_scores and trend_scores[other_item] < -0.05:
                                    sell_score += abs(trend_scores[other_item]) * correlation * 15  # Increased multiplier (was 10)
                        
                        # Factor 7: Momentum-based selling (award-winning principle)
                        if 'momentum_scores' in locals() and item in momentum_scores:
                            momentum = momentum_scores[item]
                            # Sell when momentum starts to reverse (capture profits at peak)
                            if momentum < 0 and profit_rate > 0:
                                sell_score += abs(momentum) * 25  # Strong incentive to sell when momentum reverses
                            # Sell quickly when momentum is strongly negative
                            elif momentum < -0.05:
                                sell_score += abs(momentum) * 35  # Very strong incentive to exit falling items
                        
                        # Factor 8: Optimal position sizing (Kelly criterion)
                        if 'kelly_positions' in locals() and item in kelly_positions:
                            kelly_value = kelly_positions[item]
                            current_position = obs['prices'][item] * obs['inventory'][item]
                            
                            # IMPORTANT FIX: Properly handle negative Kelly values
                            if kelly_value < 0:
                                # Negative Kelly means we should sell this item
                                # The more negative, the stronger the sell signal
                                sell_score += abs(kelly_value) * 30  # Strong incentive to sell items with negative Kelly
                            elif kelly_value > 0:
                                # For positive Kelly, check if we're over the optimal position
                                optimal_position = kelly_value * total_assets
                                if current_position > optimal_position:
                                    excess_ratio = (current_position / optimal_position) - 1
                                    sell_score += excess_ratio * 15  # Incentive to rebalance to optimal size
                        
                        # Factor 9: Concentration strategy
                        if 'successful_strategies' in locals() and 'optimal_allocations' in successful_strategies:
                            # If this item isn't in our top opportunities, consider selling
                            if item not in successful_strategies.get('top_opportunities', []):
                                sell_score += 5  # Base penalty for non-top items
                                
                                # If we have better opportunities to allocate capital to
                                if successful_strategies.get('top_opportunities', []):
                                    sell_score += 10  # Additional incentive to reallocate capital
                        
                        # Store the sell score
                        sell_candidates[item] = sell_score
                
                # If we have sell candidates, choose the one with the highest sell score
                if sell_candidates:
                    # Log the top sell candidates
                    if step_count % 1000 == 0:
                        top_sells = sorted([(i, s) for i, s in sell_candidates.items()],
                                          key=lambda x: x[1], reverse=True)[:5]
                        logger.info(f"Agent {agent_idx} | Top sell candidates: {top_sells}")
                    
                    # Choose the item with the highest sell score
                    # Randomize selection slightly to avoid alphabetical bias
                    # Get top 10% of items by score or at least 3 items
                    top_sell_items = sorted(sell_candidates.items(), key=lambda x: x[1], reverse=True)
                    top_n = max(3, int(len(top_sell_items) * 0.1))
                    top_sell_items = top_sell_items[:top_n]
                    
                    # Add some randomness to selection (weighted by score)
                    weights = [score for _, score in top_sell_items]
                    total_weight = sum(weights)
                    if total_weight > 0:
                        normalized_weights = [w/total_weight for w in weights]
                        import random
                        # Use the agent's random generator for consistency
                        worst_item = random.choices([item for item, _ in top_sell_items], weights=normalized_weights, k=1)[0]
                    else:
                        worst_item = top_sell_items[0][0] if top_sell_items else list(sell_candidates.keys())[0]
                    
                    action['type'] = 'sell'
                    action['item'] = worst_item
                    
                    # Set optimal price - adjust pricing based on holding duration
                    current_price = obs['prices'][worst_item]
                    min_price = max(1, items[worst_item]['min_price'])  # Ensure minimum price is at least 1
                    max_price = max(min_price + 1, items[worst_item]['max_price'])
                    
                    # More aggressive price reduction for items held too long
                    hold_duration = current_step_window - agent.holding_duration.get(worst_item, current_step_window)
                    if hold_duration > 12:  # If held more than 1 hour
                        # Reduce price more aggressively based on duration
                        # Maximum 10% reduction after 2 hours
                        price_reduction = min(0.10, (hold_duration - 12) * 0.005)  # 0.5% per 5 minutes after first hour
                        optimal_price = max(min_price, int(current_price * (1 - price_reduction)))
                        logger.info(f"Reducing price for {worst_item} by {price_reduction*100:.1f}% due to long hold time")
                    else:
                        optimal_price = min(max_price, int(current_price * 1.02))  # Normal 2% above current price
                    # Ensure price is never 0
                    action['price'] = max(1, optimal_price)
                    
                    # Determine quantity to sell based on item characteristics
                    profit_rate = item_profit_rates.get(worst_item, 0)
                    investment_pct = item_investment_pct.get(worst_item, 0)
                    
                    # Case 1: Unprofitable item - sell all
                    if profit_rate < 0:
                        # Sell all unprofitable items
                        action['quantity'] = obs['inventory'][worst_item]
                    
                    # Case 2: Over 5% limit - sell enough to get under limit
                    elif investment_pct > 0.05:
                        # Sell enough to get under 5% limit
                        excess_value = (investment_pct - 0.05) * total_assets
                        qty_to_sell = min(
                            obs['inventory'][worst_item],
                            int(excess_value / max(1, optimal_price))
                        )
                        action['quantity'] = max(1, qty_to_sell)
                    
                    # Case 3: Other items - sell a portion to balance portfolio
                    else:
                        # Sell a portion to get closer to target investment percentage
                        excess_pct = current_investment_pct - target_investment_pct
                        excess_value = excess_pct * total_assets
                        qty_to_sell = min(
                            obs['inventory'][worst_item],
                            int((excess_value * investment_pct / max(0.001, current_investment_pct)) / max(1, optimal_price))
                        )
                        action['quantity'] = max(1, qty_to_sell)
            
            # Track asset growth to detect stagnation
            if not hasattr(agent, 'asset_history'):
                agent.asset_history = []
                agent.stagnation_counter = 0
                agent.base_entropy_coef = ppo_kwargs.get("entropy_coef", 0.02)  # Get from ppo_kwargs
                agent.current_entropy_coef = agent.base_entropy_coef
            
            # Record total assets every 1000 steps
            if step_count % 1000 == 0:
                agent.asset_history.append(total_assets)
                
                # Check for stagnation (if we have enough history)
                if len(agent.asset_history) >= 3:
                    # Calculate growth rate over last 3 checkpoints
                    recent_growth_rate = (agent.asset_history[-1] / max(1, agent.asset_history[-3])) - 1
                    
                    # If growth rate is below 1% over last 3000 steps, consider it stagnation
                    if recent_growth_rate < 0.01:
                        agent.stagnation_counter += 1
                        # Increase exploration (entropy coefficient) by 20% each time stagnation is detected
                        # up to a maximum of 3x the base value
                        agent.current_entropy_coef = min(
                            agent.base_entropy_coef * 3.0,  # Cap at 3x original value
                            agent.current_entropy_coef * 1.2  # Increase by 20%
                        )
                        logger.info(f"Agent {agent_idx} | Stagnation detected! Growth rate: {recent_growth_rate:.2%} | "
                                   f"Increasing exploration to {agent.current_entropy_coef:.4f} (base: {agent.base_entropy_coef:.4f})")
                    else:
                        # If we're growing well, gradually reduce exploration back toward base level
                        agent.stagnation_counter = 0
                        if agent.current_entropy_coef > agent.base_entropy_coef:
                            agent.current_entropy_coef = max(
                                agent.base_entropy_coef,  # Don't go below base
                                agent.current_entropy_coef * 0.9  # Decrease by 10%
                            )
                            logger.info(f"Agent {agent_idx} | Good growth: {recent_growth_rate:.2%} | "
                                       f"Reducing exploration to {agent.current_entropy_coef:.4f}")
            
            # Apply award-winning stock market principles for MAXIMUM GP GAIN (every 1000 steps)
            if step_count % 1000 == 0:
                # 1. Momentum Trading Strategy - Focus on items with strong upward price momentum
                momentum_scores = {}
                for item in items.keys():
                    if item in price_history and len(price_history[item]) >= 10:
                        # Calculate price momentum (recent price change)
                        recent_prices = price_history[item][-10:]
                        momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if recent_prices[0] > 0 else 0
                        momentum_scores[item] = momentum
                
                # Identify high-momentum items for aggressive trading
                high_momentum_items = [item for item, score in momentum_scores.items()
                                      if score > 0.05]  # Items with >5% recent growth
                
                if high_momentum_items:
                    logger.info(f"Agent {agent_idx} | High Momentum Items: {high_momentum_items[:5]}")
                
                # 2. Kelly Criterion for optimal position sizing - MAXIMIZE EXPECTED RETURN
                kelly_positions = {}
                for item in items.keys():
                    if item in profit_per_item and item in unique_items_traded and unique_items_traded[item] > 0:
                        # Calculate win probability and win/loss ratio
                        profit_rate = profit_per_item[item] / unique_items_traded[item]
                        
                        # Calculate Kelly for all items, not just profitable ones
                        # This allows for more selling of unprofitable items
                        if profit_rate > 0:
                            # Simplified Kelly: f* = (p*b - q)/b where p=win rate, q=loss rate, b=win/loss ratio
                            win_rate = 0.6  # Default win rate
                            
                            # Calculate actual win rate if we have enough data
                            if item in trade_history and len(trade_history[item]) >= 5:
                                sells = [t for t in trade_history[item] if t.get('type') == 'sell']
                                if sells:
                                    profitable_sells = [s for s in sells if s.get('profit', 0) > 0]
                                    win_rate = len(profitable_sells) / len(sells)
                            
                            # Conservative win/loss ratio based on profit rate
                            win_loss_ratio = 1.5  # Default
                            
                            # Calculate Kelly fraction
                            loss_rate = 1 - win_rate
                            kelly_fraction = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio
                        else:
                            # For unprofitable items, set a negative Kelly fraction to encourage selling
                            # The more negative the profit rate, the more negative the Kelly fraction
                            kelly_fraction = profit_rate * 2  # Amplify the negative signal
                            
                            # IMPORTANT FIX: Allow negative Kelly fractions for unprofitable items
                            # This ensures unprofitable items are properly flagged for selling
                            if profit_rate < 0:
                                # Keep it negative but cap at -0.5 to prevent extreme values
                                kelly_fraction = max(-0.5, kelly_fraction * 0.5)
                            else:
                                # For profitable items, apply half-Kelly for safety (more conservative)
                                kelly_fraction = min(0.5, kelly_fraction * 0.5)
                            
                        # Store the Kelly fraction for this item
                        kelly_positions[item] = kelly_fraction
                        
                        # Calculate optimal position size as percentage of total assets
                     
                
                # 3. Concentration Strategy - Focus capital on best opportunities
                # Sort items by profit rate and momentum
                profit_momentum_scores = {}
                for item in items.keys():
                    if item in item_profit_rates and item in momentum_scores:
                        # Weighted score favoring profit rate but considering momentum
                        profit_momentum_scores[item] = (
                            item_profit_rates[item] * 0.7 +  # 70% weight on proven profit
                            momentum_scores.get(item, 0) * 30  # 30% weight on momentum
                        )
                
                # Identify top opportunities for concentrated investment
                top_opportunities = sorted(
                    [(item, score) for item, score in profit_momentum_scores.items() if score > 0],
                    key=lambda x: x[1], reverse=True
                )[:5]  # Focus on top 5 opportunities
                
                if top_opportunities:
                    logger.info(f"Agent {agent_idx} | Top GP Opportunities: {top_opportunities}")
                    
                    # Store top opportunities for use in trading decisions
                    successful_strategies['top_opportunities'] = [item for item, _ in top_opportunities]
                    
                    # Calculate optimal allocation percentages (higher than default)
                    # Concentrate up to 15% in top opportunity (vs standard 5% limit)
                    allocation_pcts = {}
                    remaining_pct = 0.90  # Aim to invest 90% of assets
                    
                    for i, (item, _) in enumerate(top_opportunities):
                        if i == 0:  # Top opportunity
                            allocation_pcts[item] = min(0.15, remaining_pct)
                        elif i == 1:  # Second opportunity
                            allocation_pcts[item] = min(0.12, remaining_pct)
                        else:  # Other opportunities
                            allocation_pcts[item] = min(0.08, remaining_pct)
                        
                        remaining_pct -= allocation_pcts[item]
                    
                    successful_strategies['optimal_allocations'] = allocation_pcts
            
            # Extract trading patterns from history (every 5000 steps)
            if step_count % 5000 == 0:
                # Analyze trade history to extract patterns
                pattern_insights = {}
                
                # 1. Identify most profitable time windows for each item
                for item in items.keys():
                    if item in trade_history and len(trade_history[item]) >= 10:
                        # Group trades by time windows (e.g., every 1000 steps)
                        window_size = 1500
                        windows = {}
                        
                        for trade in trade_history[item]:
                            window_idx = trade['step'] // window_size
                            if window_idx not in windows:
                                windows[window_idx] = []
                            windows[window_idx].append(trade)
                        
                        # Calculate profit per window
                        window_profits = {}
                        for window_idx, trades in windows.items():
                            buys = [t for t in trades if t['type'] == 'buy']
                            sells = [t for t in trades if t['type'] == 'sell']
                            
                            if buys and sells:
                                total_buy_cost = sum(t['price'] * t['quantity'] for t in buys)
                                total_sell_revenue = sum(t['price'] * t['quantity'] for t in sells)
                                window_profits[window_idx] = total_sell_revenue - total_buy_cost
                        
                        if window_profits:
                            # Find best window
                            best_window = max(window_profits.items(), key=lambda x: x[1])
                            pattern_insights[item] = {
                                'best_window': best_window[0] * window_size,
                                'window_profit': best_window[1],
                                'profit_pattern': 'Cyclical' if len(window_profits) > 1 else 'Steady'
                            }
                
                # 2. Identify similar items based on price movement patterns
                item_correlations = {}
                for item1 in items.keys():
                    if item1 in price_history and len(price_history[item1]) >= 10:
                        item_correlations[item1] = {}
                        for item2 in items.keys():
                            if (item1 != item2 and
                                item2 in price_history and
                                len(price_history[item2]) >= 10):
                                
                                # Calculate correlation between price histories
                                # Use the last 10 price points for both items
                                prices1 = price_history[item1][-10:]
                                prices2 = price_history[item2][-10:]
                                
                                # Simple correlation: do they move in the same direction?
                                moves1 = [1 if prices1[i] > prices1[i-1] else -1 for i in range(1, len(prices1))]
                                moves2 = [1 if prices2[i] > prices2[i-1] else -1 for i in range(1, len(prices2))]
                                
                                # Count matching directions
                                matches = sum(1 for m1, m2 in zip(moves1, moves2) if m1 == m2)
                                correlation = matches / len(moves1) if moves1 else 0
                                
                                if correlation > 0.7:  # Strong correlation
                                    item_correlations[item1][item2] = correlation
                
                # Update successful_strategies with insights
                for item, insights in pattern_insights.items():
                    if item not in successful_strategies:
                        successful_strategies[item] = {}
                    
                    successful_strategies[item].update(insights)
                    
                    # Add correlated items
                    if item in item_correlations and item_correlations[item]:
                        successful_strategies[item]['correlated_items'] = item_correlations[item]
                
                # Log insights for top items
                top_pattern_items = sorted(
                    [(i, d.get('window_profit', 0)) for i, d in pattern_insights.items()],
                    key=lambda x: x[1], reverse=True
                )[:3]
                
                if top_pattern_items:
                    pattern_str = ", ".join([f"{i}:{p:.2f}" for i, p in top_pattern_items])
                    logger.info(f"Agent {agent_idx} | Trading Pattern Insights | Top profitable patterns: {pattern_str}")
            
            # Log detailed investment strategy information
            if step_count % 1500 == 0:
                profitable_count = len(profitable_items)
                top_items = sorted([(i, r) for i, r in item_profit_rates.items() if i in obs['inventory'] and obs['inventory'][i] > 0],
                                  key=lambda x: x[1], reverse=True)[:3]
                top_items_str = ", ".join([f"{i}:{r:.2f}" for i, r in top_items]) if top_items else "None"
                
                # Include pattern insights in logging if available
                pattern_info = ""
                if successful_strategies:
                    pattern_count = len(successful_strategies)
                    pattern_info = f" | Pattern insights: {pattern_count} items"
                
                logger.info(f"Agent {agent_idx} | GP: {obs['gp']} | Investment: {current_investment_pct:.2%}/{target_investment_pct:.2%} | " +
                           f"Profitable items: {profitable_count} | Top items: {top_items_str}{pattern_info} | " +
                           f"Action: {action['type']} | Item: {action['item']} | Price: {action['price']} | Qty: {action['quantity']}")

            # Track unique items traded and quantities
            item = action['item']
            quantity = action['quantity']
            action_type = action['type']
            if item not in unique_items_traded:
                unique_items_traded[item] = 0
            unique_items_traded[item] += quantity

            # Update action type counts
            if action_type in action_type_counts:
                action_type_counts[action_type] += 1
            else:
                action_type_counts[action_type] = 1
                
            # Performance monitoring - measure step time
            current_time = time.time()
            time_since_last_action = current_time - last_action_time
            
            # Check for inactivity
            if time_since_last_action > inactivity_threshold:
                with open(perf_log_path, "a") as f:
                    f.write(f"[INACTIVITY WARNING] Step {step_count}: No action for {time_since_last_action:.2f}s\n")
                    f.write(f"  Action: {action_type}, Item: {item}, Price: {action['price']}, Quantity: {quantity}\n")
                logger.warning(f"Agent {agent_idx} | Step {step_count} | INACTIVITY WARNING: No action for {time_since_last_action:.2f}s")
            
            # Start timing the environment step
            step_start_time = time.time()
            
            # Execute the environment step with timeout detection
            try:
                # Set a flag to detect if the step is taking too long
                env_step_timeout = 30.0  # 30 seconds timeout for environment step
                env_step_start = time.time()
                
                # Execute the environment step
                next_obs, reward, done, info = env.step(action)
                
                # Check if step took too long
                env_step_time = time.time() - env_step_start
                if env_step_time > env_step_timeout:
                    with open(perf_log_path, "a") as f:
                        f.write(f"[ENV STEP TIMEOUT WARNING] Step {step_count}: Environment step took {env_step_time:.2f}s (> {env_step_timeout}s timeout)\n")
                        f.write(f"  Action: {action_type}, Item: {item}, Price: {action['price']}, Quantity: {quantity}\n\n")
                    logger.warning(f"Agent {agent_idx} | Step {step_count} | ENV STEP TIMEOUT WARNING: {env_step_time:.2f}s")
            except Exception as e:
                # Log any exceptions during environment step
                with open(perf_log_path, "a") as f:
                    f.write(f"[ENV STEP ERROR] Step {step_count}: Exception during environment step: {str(e)}\n")
                    f.write(f"  Action: {action_type}, Item: {item}, Price: {action['price']}, Quantity: {quantity}\n\n")
                logger.error(f"Agent {agent_idx} | Step {step_count} | ENV STEP ERROR: {str(e)}")
                raise  # Re-raise the exception to maintain normal error handling
            
            # Calculate step time
            step_time = time.time() - step_start_time
            
            # Log if step time exceeds threshold or at regular intervals
            if step_time > step_time_threshold or step_count % performance_log_interval == 0:
                with open(perf_log_path, "a") as f:
                    status = "SLOW STEP WARNING" if step_time > step_time_threshold else "Regular log"
                    f.write(f"[{status}] Step {step_count}: Step time {step_time:.4f}s, Time since last action: {time_since_last_action:.2f}s\n")
                    f.write(f"  Action: {action_type}, Item: {item}, Price: {action['price']}, Quantity: {quantity}\n")
                    f.write(f"  Reward: {reward}, Done: {done}\n")
                    if 'msg' in info:
                        f.write(f"  Info: {info['msg']}\n")
                    f.write("\n")
                
                if step_time > step_time_threshold:
                    logger.warning(f"Agent {agent_idx} | Step {step_count} | SLOW STEP WARNING: {step_time:.4f}s")
            
            # Update timing variables
            last_action_time = current_time
            step_count += 1
            
            # Check step progression at regular intervals
            current_time = time.time()
            if current_time - last_step_check_time > step_progression_check_interval:
                steps_since_last_check = step_count - last_step_count_check
                time_elapsed = current_time - last_step_check_time
                steps_per_minute = steps_since_last_check / (time_elapsed / 60)
                
                # Log step progression
                with open(perf_log_path, "a") as f:
                    f.write(f"[STEP PROGRESSION CHECK] Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"  Steps since last check: {steps_since_last_check} in {time_elapsed:.2f}s\n")
                    f.write(f"  Steps per minute: {steps_per_minute:.2f}\n")
                    
                    # Warning if step progression is too slow
                    if steps_since_last_check < step_progression_threshold:
                        f.write(f"  WARNING: Step progression too slow! Only {steps_since_last_check} steps in {time_elapsed:.2f}s\n")
                        logger.warning(f"Agent {agent_idx} | STEP PROGRESSION WARNING: Only {steps_since_last_check} steps in {time_elapsed:.2f}s")
                    f.write("\n")
                
                # Update step progression tracking variables
                last_step_check_time = current_time
                last_step_count_check = step_count
            
            # Only increment trade_count for actual trades (buy or sell actions)
            if action['type'] in ['buy', 'sell']:
                trade_count += 1

            # ENHANCED PROFIT TRACKING - More sophisticated profit calculation
            
            # 1. Update total profit (weighted by action type)
            if action_type == 'buy':
                # For buys, don't count the reward directly (it's usually negative)
                # Instead, we'll track the potential profit when we sell
                # But ensure we're tracking the buy price for future profit calculation
                if action['price'] > 0:  # Only track buys with positive prices
                    total_profit += 0  # No immediate profit from buys
                else:
                    logger.warning(f"Attempted to buy {item} at price {action['price']} (invalid price)")
            elif action_type == 'sell':
                # For sells, apply a bonus multiplier to encourage profit-taking
                profit_multiplier = 1.1  # 10% bonus to encourage selling at profit
                if action['price'] > 0:  # Only count sells with positive prices
                    total_profit += reward * profit_multiplier
                else:
                    logger.warning(f"Attempted to sell {item} at price {action['price']} (invalid price)")
            else:  # hold
                # Small penalty for holding to encourage active trading
                total_profit += reward * 0.9
            
            # 2. Update profit per item with action-specific tracking
            if item in profit_per_item:
                if action_type == 'buy':
                    # Track buy price and quantity for ROI calculation
                    if action['price'] > 0:  # Only track buys with positive prices
                        if item not in buy_transactions:
                            buy_transactions[item] = []

                        # Store this buy transaction
                        buy_transactions[item].append({
                        'price': action['price'],
                        'quantity': action['quantity'],
                        'step': step_count
                    })
                    
                    # Update trade history for this item
                    # Record trade both locally and in shared repository
                    trade_record = {
                        'type': 'buy',
                        'price': action['price'],
                        'quantity': action['quantity'],
                        'step': step_count,
                        'gp': obs['gp'],
                        'tax': 0  # No tax on buys
                    }
                    trade_history[item].append(trade_record)
                    try:
                        agent.record_trade(action, 0)  # 0 profit for buys
                        if step_count % 1500 == 0:
                            logger.debug(f"Successfully recorded buy trade for {item} in shared repository")
                    except Exception as e:
                        logger.error(f"Failed to record buy trade in shared repository: {e}")
                    
                elif action_type == 'sell':
                    # Calculate actual profit based on buy history
                    if action['price'] > 0:  # Only process sells with positive prices
                        if item in buy_transactions:
                            # Get the buy transactions for this item
                            item_buy_transactions = buy_transactions.get(item, [])
                            if item_buy_transactions:
                                # Calculate average buy price
                                total_buy_cost = sum(t['price'] * t['quantity'] for t in item_buy_transactions)
                                total_buy_qty = sum(t['quantity'] for t in item_buy_transactions)
                                avg_buy_price = total_buy_cost / total_buy_qty if total_buy_qty > 0 else action['price']
                                
                                # Calculate profit from this sale
                                sale_profit = (action['price'] - avg_buy_price) * action['quantity']
                                
                                # Ensure profit is at least 1 GP per unit for meaningful tracking
                                min_profit_per_unit = 1
                                if sale_profit <= 0 and action['price'] > avg_buy_price:
                                    sale_profit = min_profit_per_unit * action['quantity']
                                
                                # Add to item profit
                                profit_per_item[item] += sale_profit
                                
                                # Update trade history for this item
                                if item not in trade_history:
                                    trade_history[item] = []
                                # Calculate tax for sell (1% of sale price if >= 100)
                                tax = 0
                                if action['price'] >= 100:
                                    # Tax is always 1% of sale price for items >= 100 coins
                                    tax_per_item = min(int(action['price'] * 0.01), 5000000)
                                    tax = tax_per_item * action['quantity']
                                    
                                    # Update profit after tax
                                    sale_profit = sale_profit - tax

                                # Record trade both locally and in shared repository
                                trade_record = {
                                    'type': 'sell',
                                    'price': action['price'],
                                    'quantity': action['quantity'],
                                    'step': step_count,
                                    'gp': obs['gp'],
                                    'profit': sale_profit,
                                    'tax': tax
                                }
                                trade_history[item].append(trade_record)
                                # Also record in shared repository with tax for consistency
                                try:
                                    agent.record_trade(action, sale_profit)
                                    if step_count % 1500 == 0:
                                        logger.debug(f"Successfully recorded sell trade for {item} in shared repository (profit: {sale_profit})")
                                except Exception as e:
                                    logger.error(f"Failed to record sell trade in shared repository: {e}")
                                
                                # Log the profit calculation for debugging
                                if step_count % 1500 == 0:
                                    logger.info(f"Profit calculation for {item}: Sell price {action['price']} - Avg buy price {avg_buy_price} = {action['price'] - avg_buy_price} profit/unit * {action['quantity']} units = {sale_profit} total profit")
                            else:
                                # If empty buy transactions list, use current price as profit baseline
                                # Ensure we have at least some profit to show
                                sale_profit = max(1, action['price'] * action['quantity'] * 0.05)  # Assume 5% profit, minimum 1 GP
                                profit_per_item[item] += sale_profit
                                
                                # Update trade history
                                if item not in trade_history:
                                    trade_history[item] = []
                                # Calculate tax for sell (1% of sale price if >= 100)
                                tax = 0
                                if action['price'] >= 100:
                                    # Tax is always 1% of sale price for items >= 100 coins
                                    tax_per_item = min(int(action['price'] * 0.01), 5000000)
                                    tax = tax_per_item * action['quantity']
                                    
                                    # Update profit after tax
                                    sale_profit = sale_profit - tax

                                trade_history[item].append({
                                    'type': 'sell',
                                    'price': action['price'],
                                    'quantity': action['quantity'],
                                    'step': step_count,
                                    'gp': obs['gp'],
                                    'profit': sale_profit,
                                    'tax': tax
                                })
                        else:
                            # If no buy history, use a default profit calculation
                            # Assume a 5% profit margin on the sale price, minimum 1 GP
                            sale_profit = max(1, action['price'] * action['quantity'] * 0.05)
                            profit_per_item[item] += sale_profit
                            
                            # Still update trade history
                            if item not in trade_history:
                                trade_history[item] = []
                            # Record trade both locally and in shared repository
                            trade_record = {
                                'type': 'sell',
                                'price': action['price'],
                                'quantity': action['quantity'],
                                'step': step_count,
                                'gp': obs['gp'],
                                'profit': sale_profit,
                                'tax': 0  # No tax in this case
                            }
                            trade_history[item].append(trade_record)
                            # Record in shared repository without tax
                            try:
                                agent.record_trade(action, sale_profit)
                                if step_count % 1500 == 0:
                                    logger.debug(f"Successfully recorded no-tax sell trade for {item} in shared repository (profit: {sale_profit})")
                            except Exception as e:
                                logger.error(f"Failed to record no-tax sell trade in shared repository: {e}")
                    else:
                        logger.warning(f"Attempted to sell {item} at price {action['price']} (invalid price)")
                        # Still update trade history
                        if item not in trade_history:
                            trade_history[item] = []
                        # For invalid price sells, don't apply tax since we can't calculate it properly
                        tax = 0

                        # Record trade both locally and in shared repository
                        trade_record = {
                            'type': 'sell',
                            'price': action['price'],
                            'quantity': action['quantity'],
                            'step': step_count,
                            'gp': obs['gp'],
                            'profit': reward,
                            'tax': tax
                        }
                        trade_history[item].append(trade_record)
                        # Record in shared repository
                        agent.record_trade(action, reward)
                else:
                    # For holds, just use the reward
                    profit_per_item[item] += reward

            # Enhanced logging with detailed profit information
            current_state = (frozenset(unique_items_traded.items()), total_profit)
            if current_state != last_logged_state or step_count % 1500 == 0:  # Log regularly even if state hasn't changed
                log_path = f"agent_{agent_idx}_trade_log.txt"
                with open(log_path, "w") as f:
                    f.write(f"Agent {agent_idx} Trade Log - Step {step_count}\n")
                    f.write(f"==================================================\n")
                    f.write(f"PORTFOLIO SUMMARY:\n")
                    f.write(f"--------------------------------------------------\n")
                    f.write(f"Current GP: {obs['gp']:,}\n")
                    f.write(f"Total Assets Value: {total_assets:,}\n")
                    f.write(f"Investment Value: {total_investment_value:,}\n")
                    f.write(f"Percent Invested: {current_investment_pct:.2%}\n")
                    f.write(f"Target Investment: {target_investment_pct:.2%}\n")
                    
                    # Calculate sum of all profitable items
                    profitable_items_sum = sum(profit for item, profit in profit_per_item.items() if profit > 0)
                    f.write(f"Total Profit (before tax): {profitable_items_sum:,.2f}\n")
                    
                    # Calculate total tax from all sell trades
                    total_tax = sum(t.get('tax', 0) for trades in trade_history.values() for t in trades if t['type'] == 'sell')
                    f.write(f"Total Tax Paid: {total_tax:,.2f} GP\n")
                    f.write(f"Total Profit (after tax): {profitable_items_sum - total_tax:,.2f}\n")
                    
                    # Calculate profit rate (annualized)
                    if step_count > 0:
                        # Calculate profit as a percentage of starting capital (with safe access)
                        try:
                            starting_capital = env.starting_gp
                        except (AttributeError, TypeError):
                            starting_capital = 5000000  # Default to 5M if not available
                        
                        # Calculate total return (including unrealized gains)
                        total_return = (total_assets - starting_capital) / starting_capital if starting_capital > 0 else 0
                        
                        # Calculate time elapsed in simulated days (assuming each step is ~1 hour)
                        days_elapsed = step_count / 24  # Convert hours to days
                        
                        # Calculate annualized rate using compound interest formula
                        # (1 + r)^t = (1 + total_return)
                        # r = (1 + total_return)^(1/t) - 1
                        if days_elapsed > 0:
                            # Cap the annualized rate at a reasonable maximum (100%)
                            try:
                                days = max(0.1, days_elapsed)  # Ensure minimum of 0.1 days to avoid extreme values
                                # Add safeguards to prevent overflow
                                if total_return > 10.0:  # Cap extremely high returns to prevent overflow
                                    annualized_profit_rate = 1.0  # Cap at 100%
                                else:
                                    # Use a safer calculation that won't overflow
                                    exponent = 365 / days
                                    if exponent > 100:  # If exponent is too large
                                        annualized_profit_rate = 1.0  # Cap at 100%
                                    else:
                                        annualized_profit_rate = min(1.0, ((1 + total_return) ** exponent) - 1)
                            except (ZeroDivisionError, ValueError, OverflowError):
                                annualized_profit_rate = 0.0
                            f.write(f"Profit Rate (annualized): {annualized_profit_rate:.2%}\n")
                        else:
                            f.write(f"Profit Rate (annualized): N/A (insufficient data)\n")
                        
                        # Also show absolute profit numbers
                        f.write(f"Total Return: {total_return:.2%} ({total_assets - starting_capital:,.2f} GP)\n")
                        f.write(f"Realized Profit: {total_profit:,.2f} GP\n")
                        f.write(f"Unrealized Gains: {total_assets - starting_capital - total_profit:,.2f} GP\n")
                    
                    f.write(f"\nTOP PROFITABLE ITEMS:\n")
                    f.write(f"--------------------------------------------------\n")
                    # Sort items by profit
                    profitable_items_list = [(item, profit_per_item.get(item, 0))
                                           for item in items.keys()
                                           if item in profit_per_item and profit_per_item.get(item, 0) > 0]
                    profitable_items_list.sort(key=lambda x: x[1], reverse=True)
                    
                    # Calculate total sum of all profitable items for verification
                    total_profitable_items = sum(profit for _, profit in profitable_items_list)
                    f.write(f"Total sum of all profitable items: {total_profitable_items:,.2f} GP\n\n")
                    
                    # Display top 10 profitable items
                    for i, (item, profit) in enumerate(profitable_items_list[:10]):
                        qty = unique_items_traded.get(item, 0)
                        profit_per_unit = profit / qty if qty > 0 else 0
                        inventory = obs['inventory'].get(item, 0)
                        current_value = obs['prices'].get(item, 0) * inventory
                        pct_of_portfolio = current_value / total_assets if total_assets > 0 else 0
                        
                        f.write(f"{i+1}. {item}:\n")
                        f.write(f"   Profit: {profit:,.2f} GP\n")
                        f.write(f"   Profit/Unit: {profit_per_unit:.2f} GP\n")
                        f.write(f"   Current Holdings: {inventory} units\n")
                        f.write(f"   Current Value: {current_value:,.2f} GP ({pct_of_portfolio:.2%} of portfolio)\n")
                    
                    f.write(f"\nALL ITEMS WITH FILLED TRADES:\n")
                    f.write(f"--------------------------------------------------\n")
                    # Calculate total filled units traded for each item
                    filled_units_traded = {}
                    for item, trades in trade_history.items():
                        total_filled = sum(t['quantity'] for t in trades if t['type'] in ['buy', 'sell'])
                        if total_filled > 0:
                            filled_units_traded[item] = total_filled

                    for it, qty in filled_units_traded.items():
                        # Ensure we're getting the correct profit value
                        profit = profit_per_item.get(it, 0.0)
                        
                        # Get current price and inventory
                        current_price = obs['prices'].get(it, 0)
                        current_inventory = obs['inventory'].get(it, 0)
                        
                        # Calculate potential profit based on current holdings
                        potential_profit = 0.0
                        if it in buy_transactions:
                            item_buy_transactions = buy_transactions[it]
                            if item_buy_transactions and current_inventory > 0:
                                # Calculate average buy price
                                total_buy_cost = sum(t['price'] * t['quantity'] for t in item_buy_transactions)
                                total_buy_qty = sum(t['quantity'] for t in item_buy_transactions)
                                avg_buy_price = total_buy_cost / total_buy_qty if total_buy_qty > 0 else current_price
                                
                                # Calculate potential profit from current holdings
                                if avg_buy_price > 0 and current_price > 0:
                                    potential_profit = (current_price - avg_buy_price) * current_inventory
                        
                        # Calculate profit per unit, ensuring we don't divide by zero
                        profit_per_unit = profit / qty if qty > 0 else 0
                        
                        # Calculate estimated value
                        estimated_value = current_price * current_inventory if current_price > 0 else 0
                        
                        # Create a more informative display
                        f.write(f"{it}: {qty} filled units traded\n")
                        f.write(f"  Current holdings: {current_inventory} units at {current_price} GP each = {estimated_value:,.2f} GP\n")
                        # Get total tax for this item
                        item_tax = sum(t.get('tax', 0) for t in trade_history.get(it, []) if t['type'] == 'sell')
                        f.write(f"  Realized profit: {profit:,.2f} GP total, {profit_per_unit:.2f} GP/unit (Tax paid: {item_tax:,.2f} GP)\n")
                        if potential_profit != 0:
                            f.write(f"  Potential profit on current holdings: {potential_profit:,.2f} GP\n")
                        f.write("\n")
                
                last_logged_state = current_state
                
            # Episode-based training with state saving
            episode_step = step_count % episode_length
            
            # Save states every save_frequency steps
            if step_count % save_frequency == 0:
                # Save environment state
                try:
                    env.save_state(state_path)
                    logger.info(f"Saved environment state for agent {agent_idx} at step {step_count}")
                except Exception as e:
                    logger.error(f"Failed to save environment state for agent {agent_idx}: {e}")
                
                # Save model weights
                try:
                    actor_path = os.path.join(AGENT_DIR, f"actor_{agent_idx}.pth")
                    critic_path = os.path.join(AGENT_DIR, f"critic_{agent_idx}.pth")
                    agent.save_actor(actor_path)
                    agent.save_critic(critic_path)
                    logger.info(f"Saved model weights for agent {agent_idx} at step {step_count}")
                except Exception as e:
                    logger.error(f"Failed to save model weights for agent {agent_idx}: {e}")
            
            # End of episode handling
            # Reset check - Every 1000 steps (aligns with episode length)
            # 105144 is the total number of files in 5m
            if step_count % 105144 == 0:
                # Reset item holding durations for the new episode
                if hasattr(agent, 'holding_duration'):
                    agent.holding_duration = {}
                # Save performance metrics before reset
                pre_reset_gp = obs['gp']
                pre_reset_assets = sum(obs['prices'][item] * obs['inventory'][item] for item in items.keys())
                
                # Remove old state file if it exists
                state_path = os.path.join(AGENT_DIR, f"agent_{agent_idx}_env_state.json")
                if os.path.exists(state_path):
                    try:
                        os.remove(state_path)
                        logger.info(f"Removed old state file for agent {agent_idx}")
                    except Exception as e:
                        logger.error(f"Failed to remove old state file for agent {agent_idx}: {e}")
                
                # Reset environment with starting GP and clear all timers/limits
                obs = env.reset(starting_gp=env.starting_gp)
                env.buy_limit_timers = {item: 0 for item in items.keys()}
                env.buy_limits = {item: 0 for item in items.keys()}
                
                # Log the reset with buy limit info
                logger.info(f"RESET | Agent {agent_idx} | Step {step_count} | Pre-reset GP: {pre_reset_gp:,} | Pre-reset Assets: {pre_reset_assets:,} | Buy limits and timers cleared")
                
                # Reset all tracking variables to start fresh for the new episode
                unique_items_traded = {}
                total_profit = 0.0
                profit_per_item = {item: 0.0 for item in items.keys()}
                buy_transactions = {}
                trade_history = {item: [] for item in items.keys()}
                price_history = {item: [] for item in items.keys()}
                successful_strategies = {}  # Clear trading patterns
                trade_count = 0
                action_type_counts = {'buy': 0, 'sell': 0, 'hold': 0}
                
                # Reset asset history to avoid stagnation detection across episodes
                if hasattr(agent, 'asset_history'):
                    agent.asset_history = []
                    agent.stagnation_counter = 0
                
            if episode_step == episode_length - 1:
                episode_number += 1
                logger.info(f"Agent {agent_idx} | Episode {episode_number} completed | Total GP: {obs['gp']:,} | Profit: {total_profit:,.2f}")
                
                # Save final trade history for the episode
                try:
                    shared_knowledge.save_trade_history(episode_number=episode_number)
                    logger.info(f"Saved final trade history for episode {episode_number}")
                except Exception as e:
                    logger.error(f"Failed to save final trade history: {e}")
                
                # Create archives directory if it doesn't exist
                archives_dir = "archives"
                os.makedirs(archives_dir, exist_ok=True)
                
                # Archive log files
                log_path = f"agent_{agent_idx}_trade_log.txt"
                archive_log_path = os.path.join(archives_dir, f"agent_{agent_idx}_episode_{episode_number}_trade_log.txt")
                try:
                    import shutil
                    shutil.copy2(log_path, archive_log_path)
                    logger.info(f"Archived trade log for agent {agent_idx}, episode {episode_number}")
                except Exception as e:
                    logger.error(f"Failed to archive trade log for agent {agent_idx}: {e}")
                
                # Archive data log files
                data_log_path = f"agent_{agent_idx}_trade_data_log.txt"
                archive_data_log_path = os.path.join(archives_dir, f"agent_{agent_idx}_episode_{episode_number}_trade_data_log.txt")
                if os.path.exists(data_log_path):
                    try:
                        shutil.copy2(data_log_path, archive_data_log_path)
                    except Exception as e:
                        logger.error(f"Failed to archive trade data log for agent {agent_idx}: {e}")

            # Log additional trade data at intervals
            # Log additional trade data at intervals (less frequently to improve speed)
            if step_count - last_trade_log_step >= trade_log_interval:
                trade_data_log_path = f"agent_{agent_idx}_trade_data_log.txt"
                with open(trade_data_log_path, "a") as f:
                    f.write(f"Step {step_count} Trade Data Log\n")
                    f.write(f"Total trades: {trade_count}\n")
                    f.write(f"Total GP: {obs['gp']:,}\n")
                    
                    # Calculate sum of all profitable items for consistency
                    profitable_items_sum = sum(profit for item, profit in profit_per_item.items() if profit > 0)
                    f.write(f"Total Profit: {profitable_items_sum:,.2f}\n")
                    f.write(f"Profit Rate: {(profitable_items_sum/step_count if step_count > 0 else 0):.2f} GP/step\n\n")
                    
                    f.write("Action type counts:\n")
                    for atype, count in action_type_counts.items():
                        f.write(f"{atype}: {count}\n")
                    
                    # Only log profitable items to reduce file size and processing time
                    f.write("\nTop Profitable items:\n")
                    profitable_items_only = {it: profit for it, profit in profit_per_item.items() if profit > 0}
                    # Sort by profit for better readability
                    for it, profit in sorted(profitable_items_only.items(), key=lambda x: x[1], reverse=True)[:20]:
                        qty = unique_items_traded.get(it, 0)
                        profit_per_unit = profit / qty if qty > 0 else 0
                        f.write(f"{it}: {profit:.2f} GP total, {profit_per_unit:.2f} GP/unit\n")
                    f.write("\n")
                last_trade_log_step = step_count
                
            # SPEED OPTIMIZATION: Reduced frequency of expensive operations
            if step_count % 10 == 0:  # Only every 10 steps
                # Update price history and other expensive calculations here
                pass
            
            # Removed quick action logic to enforce 5-minute real-time trading interval
            # The quick action logic bypassed the can_trade check in PPOAgent,
            # leading to trades occurring more frequently than intended.

            if step_count % 500 == 0:
                logger.info(f"GP_LOG | Agent {agent_idx} | Step {step_count} | GP {next_obs['gp']}")
                # Save state every 500 steps (no log to avoid clutter)
                try:
                    env.save_state(state_path)
                except Exception as e:
                    logger.error(f"Failed to save environment state for agent {agent_idx}: {e}")
                # Save model every 1000 steps (no log to avoid clutter)
                try:
                    actor_path = os.path.join(AGENT_DIR, f"actor_{agent_idx}.pth")
                    critic_path = os.path.join(AGENT_DIR, f"critic_{agent_idx}.pth")
                    agent.save_actor(actor_path)
                    agent.save_critic(critic_path)
                except Exception as e:
                    logger.error(f"Failed to save model for agent {agent_idx}: {e}")
            obs = next_obs
            if done:
                # Save state at episode end (no log to avoid clutter)
                try:
                    env.save_state(state_path)
                except Exception as e:
                    logger.error(f"Failed to save environment state for agent {agent_idx}: {e}")
                obs = env.reset()
                if step_count % 500 == 0:
                    logger.info(f"GP_LOG | Agent {agent_idx} | Step {step_count} | GP {obs['gp']}")

if __name__ == "__main__":
    # To run agents concurrently with process-safe logging:
    use_multiprocessing = True  # Set to False to use single-process main()
    if use_multiprocessing:
        log_queue = multiprocessing.Queue(-1)
        listener = start_logging_listener(log_queue)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # PRE-LOAD CACHE IN PARENT PROCESS WITH SHARED MEMORY
        # This loads the 2.4GB cache ONCE into shared memory that all agents access
        cache_file = ENV_KWARGS.get("cache_file", "training_cache.json")
        shm_name = None
        if cache_file:
            from training.cached_market_loader import load_cache, get_shared_memory_name, cleanup_shared_memory
            logger.info(f"Loading cache into shared memory: {cache_file}")
            load_cache(cache_file, use_shared_memory=True)
            shm_name = get_shared_memory_name()
            if shm_name:
                logger.info(f"✓ Cache loaded in shared memory: {shm_name}")
                logger.info("  All agents will access this shared memory (zero duplication)")
            else:
                logger.warning("Shared memory not available, agents will load cache individually")
        
        # Load all data from training cache (NO API CALLS)
        id_name_map, buy_limits_map, marketplace_data, volume_data_5m, volume_data_1h = load_training_cache()
        marketplace_data = fetch_marketplace_data(marketplace_data)
        items = build_items_dict(id_name_map, buy_limits_map, marketplace_data)
        item_list, price_ranges, buy_limits = get_item_lists(items)
        
        # Initialize volume analyzer
        id_to_name_map = {}
        for item_id, item_name in id_name_map.items():
            id_to_name_map[item_id] = item_name
        
        # Create volume analyzer
        volume_analyzer = create_volume_analyzer(id_to_name_map)
        
        # Update volume analyzer with cached data (NO API CALLS)
        volume_analyzer.update_volume_data(volume_data_5m, volume_data_1h)
        logger.info(f"Initialized volume analyzer with {len(volume_data_5m)} 5m items and {len(volume_data_1h)} 1h items from cache")
        processes = []
        NUM_AGENTS = TRAIN_KWARGS.get("num_agents", 5)
        logger.info(f"Starting {NUM_AGENTS} parallel agent workers with shared memory cache")
        
        try:
            for agent_idx in range(NUM_AGENTS):
                p = multiprocessing.Process(
                    target=agent_worker,
                    args=(agent_idx, log_queue, items, price_ranges, buy_limits, device, PPO_KWARGS, True, volume_analyzer, shm_name)
                )
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        finally:
            # Cleanup shared memory when done
            if shm_name:
                cleanup_shared_memory()
                logger.info("Shared memory cleaned up")
        
        listener.stop()
    else:
        main()
