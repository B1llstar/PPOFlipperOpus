import os
import json
import logging
from ge_rest_client import HistoricalGrandExchangeClient
from ge_env import GrandExchangeEnv

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_historical_data")

def read_mapping_file():
    """Read the mapping file to get item IDs, names, and buy limits."""
    mapping_path = "endpoints/mapping.txt"
    id_name_map = {}
    name_to_id_map = {}
    
    try:
        with open(mapping_path, 'r') as f:
            mapping_data = json.load(f)
            for item in mapping_data:
                item_id = str(item.get('id'))
                item_name = item.get('name')
                if item_id and item_name:
                    id_name_map[item_id] = item_name
                    name_to_id_map[item_name] = item_id
        logger.info(f"Loaded mapping data for {len(id_name_map)} items")
    except Exception as e:
        logger.error(f"Error reading mapping file: {e}")
        # Provide some minimal default data if file can't be read
        id_name_map = {"554": "Fire rune", "555": "Water rune", "556": "Air rune"}
        name_to_id_map = {"Fire rune": "554", "Water rune": "555", "Air rune": "556"}
    
    return id_name_map, name_to_id_map

def test_historical_client():
    """Test the HistoricalGrandExchangeClient functionality."""
    logger.info("=== Testing HistoricalGrandExchangeClient ===")
    
    # Initialize the client with random_start=False for predictable testing
    client = HistoricalGrandExchangeClient(data_dir="5m", random_start=False)
    logger.info(f"Initialized client with {len(client._data_files)} data files")
    
    # Get ID-name mappings
    id_name_map, name_to_id_map = read_mapping_file()
    client.set_name_to_id_mapping(name_to_id_map, id_name_map)
    
    # Test initial data loading
    initial_data = client.get_latest()
    logger.info(f"Initial data contains {len(initial_data)} items")
    
    # Test a few specific items
    test_items = ["Fire rune", "Water rune", "Air rune"]
    for item_name in test_items:
        item_id = client.get_id_for_name(item_name)
        if item_id and item_id in initial_data:
            price_data = initial_data[item_id]
            logger.info(f"{item_name} (ID: {item_id}): High price = {price_data.get('high')}, Low price = {price_data.get('low')}")
        else:
            logger.warning(f"Item {item_name} not found in initial data")
    
    # Test advancing through files
    logger.info("Testing file advancement...")
    current_index = client._current_index
    for i in range(5):  # Test 5 advancements
        success = client.advance()
        if success:
            new_index = client._current_index
            logger.info(f"Advanced from file {current_index} to {new_index}")
            current_index = new_index
            
            # Get data after advancement
            data = client.get_latest()
            logger.info(f"Data after advancement contains {len(data)} items")
            
            # Check a specific item after advancement
            if test_items and len(test_items) > 0:
                item_name = test_items[0]
                item_id = client.get_id_for_name(item_name)
                if item_id and item_id in data:
                    price_data = data[item_id]
                    logger.info(f"{item_name} after advancement: High price = {price_data.get('high')}, Low price = {price_data.get('low')}")
        else:
            logger.warning("Failed to advance to next file")
    
    # Test reset (should go back to index 0 since random_start=False)
    client.reset()
    logger.info(f"Reset to first file, index is now {client._current_index}")
    
    # Test looping behavior
    logger.info("Testing looping behavior...")
    # First, advance to the last file
    while client._current_index < len(client._data_files) - 1:
        client.advance()
    logger.info(f"Advanced to last file, index is now {client._current_index}")
    
    # Now advance one more time, which should loop back to the beginning
    client.advance()
    logger.info(f"After advancing past the end, index is now {client._current_index}")
    
    # Test with random start
    logger.info("Testing with random start...")
    random_client = HistoricalGrandExchangeClient(data_dir="5m", random_start=True)
    logger.info(f"Random start client initialized at index {random_client._current_index}")
    
    return client

def test_environment_with_historical_data(client):
    """Test the GrandExchangeEnv with historical data."""
    logger.info("=== Testing GrandExchangeEnv with Historical Data ===")
    
    # Create a simple items dictionary for testing
    id_name_map, name_to_id_map = read_mapping_file()
    
    # Select a few items for testing
    test_items = ["Fire rune", "Water rune", "Air rune", "Nature rune", "Death rune"]
    items = {}
    
    # Get initial data to set up item prices
    initial_data = client.get_latest()
    
    for item_name in test_items:
        item_id = name_to_id_map.get(item_name)
        if item_id and item_id in initial_data:
            price_data = initial_data[item_id]
            high_price = price_data.get('high', 100)
            low_price = price_data.get('low', 90)
            
            # Calculate min and max price based on high/low with some margin
            min_price = max(1, int(low_price * 0.9))
            max_price = max(min_price + 1, int(high_price * 1.1))
            
            # Use a default buy limit
            buy_limit = 10000
            
            # Ensure base_price is never 0
            base_price = max(1, (high_price + low_price) // 2)
            
            items[item_name] = {
                'base_price': base_price,
                'buy_limit': buy_limit,
                'min_price': min_price,
                'max_price': max_price
            }
    
    logger.info(f"Created test environment with {len(items)} items")
    
    # Initialize environment with historical client
    env = GrandExchangeEnv(
        items=items,
        starting_gp=1000000,
        tick_duration=5,  # 5-minute ticks
        max_ticks=100,
        historical_client=client,
        name_to_id_map=name_to_id_map,
        id_to_name_map=id_name_map
    )
    
    # Reset environment
    obs = env.reset()
    logger.info("Environment reset")
    
    # Log initial prices
    logger.info("Initial prices:")
    for item_name, price in obs['prices'].items():
        logger.info(f"{item_name}: {price} GP")
    
    # Test a few steps to see price changes
    logger.info("Testing price changes over steps...")
    for i in range(10):
        # Use a hold action to just observe price changes
        action = {
            'type': 'hold',
            'item': test_items[0],
            'price': 0,
            'quantity': 0
        }
        
        # Step the environment
        obs, reward, done, info = env.step(action)
        
        # Log prices for test items
        logger.info(f"Step {i+1} prices:")
        for item_name in test_items:
            if item_name in obs['prices']:
                logger.info(f"{item_name}: {obs['prices'][item_name]} GP")
    
    # Test buy/sell functionality
    logger.info("Testing buy/sell functionality...")
    
    # Try to buy an item
    buy_item = test_items[0]
    buy_price = obs['prices'][buy_item]
    buy_action = {
        'type': 'buy',
        'item': buy_item,
        'price': buy_price,
        'quantity': 100
    }
    
    obs, reward, done, info = env.step(buy_action)
    logger.info(f"Buy action result: {info.get('msg', 'No message')}")
    logger.info(f"Inventory after buy: {obs['inventory'].get(buy_item, 0)} {buy_item}")
    
    # Try to sell the item
    if obs['inventory'].get(buy_item, 0) > 0:
        sell_price = obs['prices'][buy_item]
        sell_action = {
            'type': 'sell',
            'item': buy_item,
            'price': sell_price,
            'quantity': obs['inventory'][buy_item]
        }
        
        obs, reward, done, info = env.step(sell_action)
        logger.info(f"Sell action result: {info.get('msg', 'No message')}")
        logger.info(f"Inventory after sell: {obs['inventory'].get(buy_item, 0)} {buy_item}")
    
    return env

if __name__ == "__main__":
    logger.info("Starting historical data tests")
    
    # Test the historical client
    client = test_historical_client()
    
    # Test the environment with historical data
    env = test_environment_with_historical_data(client)
    
    logger.info("All tests completed")