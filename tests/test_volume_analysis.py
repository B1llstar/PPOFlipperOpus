import logging
import json
from ge_rest_client import GrandExchangeClient
from volume_analysis import create_volume_analyzer, update_volume_analyzer, get_volume_metrics_for_item

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger("test_volume_analysis")

def read_mapping_file():
    """Read the mapping file to get item IDs, names, and buy limits."""
    mapping_path = "endpoints/mapping.txt"
    id_name_map = {}
    
    try:
        with open(mapping_path, 'r') as f:
            mapping_data = json.load(f)
            for item in mapping_data:
                item_id = str(item.get('id'))
                item_name = item.get('name')
                if item_id and item_name:
                    id_name_map[item_id] = item_name
        logger.info(f"Loaded mapping data for {len(id_name_map)} items")
    except Exception as e:
        logger.error(f"Error reading mapping file: {e}")
        # Provide some minimal default data if file can't be read
        id_name_map = {"554": "Fire rune", "555": "Water rune", "556": "Air rune"}
    
    return id_name_map

def main():
    # Initialize client
    client = GrandExchangeClient()
    
    # Read mapping file
    id_name_map = read_mapping_file()
    
    # Create volume analyzer
    volume_analyzer = create_volume_analyzer(id_name_map)
    
    # Fetch market data
    data_5m = client.get_5m()
    data_1h = client.get_1h()
    
    # Update volume analyzer with data
    volume_analyzer.update_volume_data(data_5m, data_1h)
    logger.info(f"Updated volume analyzer with {len(data_5m)} 5m items and {len(data_1h)} 1h items")
    
    # Test 1: Calculate volume-weighted price metrics
    logger.info("\n=== Test 1: Volume-Weighted Price Metrics ===")
    for item_id in list(data_1h.keys())[:5]:  # Test first 5 items
        item_name = id_name_map.get(item_id, f"Unknown ({item_id})")
        vwap_1h = volume_analyzer.calculate_volume_weighted_price(item_id, "1h")
        vwap_5m = volume_analyzer.calculate_volume_weighted_price(item_id, "5m")
        logger.info(f"Item: {item_name} (ID: {item_id})")
        logger.info(f"  VWAP (1h): {vwap_1h}")
        logger.info(f"  VWAP (5m): {vwap_5m}")
    
    # Test 2: Analyze volume differences to detect market activity
    logger.info("\n=== Test 2: Volume Differences and Market Activity ===")
    for item_id in list(data_1h.keys())[:5]:  # Test first 5 items
        item_name = id_name_map.get(item_id, f"Unknown ({item_id})")
        imbalance_1h = volume_analyzer.calculate_buy_sell_imbalance(item_id, "1h")
        imbalance_5m = volume_analyzer.calculate_buy_sell_imbalance(item_id, "5m")
        momentum = volume_analyzer.calculate_volume_momentum(item_id)
        logger.info(f"Item: {item_name} (ID: {item_id})")
        logger.info(f"  Buy/Sell Imbalance (1h): {imbalance_1h}")
        logger.info(f"  Buy/Sell Imbalance (5m): {imbalance_5m}")
        logger.info(f"  Volume Momentum: {momentum}")
    
    # Test 3: Real-time market activity detection
    logger.info("\n=== Test 3: Real-Time Market Activity Detection ===")
    market_changes = volume_analyzer.detect_real_time_market_changes(threshold=0.2)
    logger.info(f"Detected {len(market_changes)} significant market changes")
    for i, change in enumerate(market_changes[:5]):  # Show top 5 changes
        logger.info(f"Change {i+1}: {change['item_name']} (ID: {change['item_id']})")
        logger.info(f"  Volume Change: {change['volume_change']:.2f}")
        logger.info(f"  Buy/Sell Ratio Change: {change['ratio_change']:.2f}")
        logger.info(f"  Current Volume: {change['current_volume']}")
        logger.info(f"  Buy Volume: {change['buy_volume']}, Sell Volume: {change['sell_volume']}")
    
    # Test 4: Identify profit opportunities
    logger.info("\n=== Test 4: Profit Opportunities ===")
    opportunities = volume_analyzer.identify_profit_opportunities()
    logger.info(f"Identified {len(opportunities)} potential profit opportunities")
    for i, opp in enumerate(opportunities[:10]):  # Show top 10 opportunities
        logger.info(f"Opportunity {i+1}: {opp['item_name']} (ID: {opp['item_id']})")
        logger.info(f"  Score: {opp['score']:.2f}")
        logger.info(f"  Current Price: {opp['current_price']}, VWAP: {opp['vwap']}")
        logger.info(f"  Imbalance: {opp['imbalance']:.2f}, Momentum: {opp['momentum']:.2f}")
        logger.info(f"  Volume: {opp['volume']}")
    
    # Test 5: Get volume metrics for specific items
    logger.info("\n=== Test 5: Volume Metrics for Specific Items ===")
    popular_items = ["Dragon bones", "Cannonball", "Nature rune", "Zulrah's scales", "Abyssal whip"]
    for item_name in popular_items:
        # Find item ID
        item_id = None
        for id, name in id_name_map.items():
            if name == item_name:
                item_id = id
                break
        
        if item_id:
            metrics = get_volume_metrics_for_item(volume_analyzer, item_name)
            logger.info(f"Item: {item_name} (ID: {item_id})")
            for metric_name, metric_value in metrics.items():
                logger.info(f"  {metric_name}: {metric_value}")
        else:
            logger.info(f"Item not found: {item_name}")

if __name__ == "__main__":
    main()