import logging
import time
from ge_env import GrandExchangeEnv
from margin_comparison import MarginComparisonAnalyzer, validate_and_rollback_failed_experiments
from volume_analysis import VolumeAnalyzer, create_volume_analyzer

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_margin_experimentation")

def load_test_items():
    """Load a small set of test items for experimentation"""
    test_items = {
        "Nature rune": {
            "base_price": 200,
            "buy_limit": 10000,
            "min_price": 100,  # Lower min_price to allow margin experimentation
            "max_price": 300,  # Higher max_price to allow margin experimentation
        },
        "Law rune": {
            "base_price": 300,
            "buy_limit": 8000,
            "min_price": 270,
            "max_price": 330,
        },
        "Death rune": {
            "base_price": 250,
            "buy_limit": 7000,
            "min_price": 225,
            "max_price": 275,
        },
    }
    return test_items

def test_margin_experimentation():
    """Test margin experimentation functionality"""
    logger.info("Testing margin experimentation...")
    
    # Create environment with margin experimentation enabled
    items = load_test_items()
    
    env = GrandExchangeEnv(
        items=items,
        starting_gp=1000000,
        tick_duration=5,
        max_ticks=1440,
        price_fluctuation_pct=0.05,  # Higher fluctuation for testing
        buy_limit_reset_ticks=240,
        random_seed=42,
        high_vol_items_path="high_vol_items.txt",
        enable_margin_experimentation=True,
        max_margin_attempts=5,  # 5 steps to reach 2% margin
        min_margin_pct=0.02,  # 2% minimum margin
        max_margin_pct=0.40,  # 40% maximum margin
        margin_wait_steps=2   # Wait 2 steps before trying a new margin
    )
    
    env.reset()
    
    # Test buy order
    logger.info("\nTesting buy order with margin experimentation:")
    item = "Nature rune"
    env.prices[item] = 200  # Set market price
    
    # Place a buy order with a price below market price and custom wait steps
    action = {
        "type": "buy",
        "item": item,
        "price": 190,  # 5% below market price
        "quantity": 100,
        "wait_steps": 4  # Custom wait steps (longer than default)
    }
    
    # Execute the action
    _, reward, done, info = env.step(action)
    logger.info(f"Buy order placed: {info['msg']}")
    
    # Run more steps to see margin experimentation in action
    logger.info("\nRunning simulation steps:")
    for i in range(12):  # Increased to see the final price adjustment
        # Fluctuate prices
        env._fluctuate_prices()
        market_price = env.prices[item]
        logger.info(f"Step {i+1}: Market price for {item}: {market_price}")
        
        # Process orders
        env._fulfill_orders()
        
        # Check order status
        if env.open_orders:
            order = env.open_orders[0]
            logger.info(f"  Order price: {order['price']}")
            
            # Check margin attempts
            if item in env.margin_attempts and 'buy' in env.margin_attempts[item]:
                current_attempt = env.margin_attempts[item]['buy']['current_attempt']
                prices_tried = env.margin_attempts[item]['buy']['prices_tried']
                logger.info(f"  Current attempt: {current_attempt}, Prices tried: {prices_tried}")
                
                # Check wait counter
                order_key = f"{item}_buy_{order['price']}"
                wait_count = env.margin_wait_counters.get(order_key, 0)
                wait_steps = order.get('wait_steps', env.margin_wait_steps)
                logger.info(f"  Wait counter: {wait_count}/{wait_steps} (custom wait steps: {wait_steps})")
        else:
            logger.info("  Order filled!")
            break
    
    # Check if order is still open after all steps
    if env.open_orders:
        order = env.open_orders[0]
        logger.info(f"\nOrder still open after all steps. Final price: {order['price']}")
        logger.info(f"Market price: {env.prices[item]}")
    
    # Log margin success rates
    logger.info(f"\nMargin success rates: {env.margin_success_rates}")
    
    # Log inventory
    logger.info(f"Final inventory: {env.inventory}")
    
    return env

def test_dynamic_wait_steps():
    """Test dynamic wait steps functionality"""
    logger.info("\nTesting dynamic wait steps for margin experimentation...")
    
    # Create environment with margin experimentation enabled
    items = load_test_items()
    
    env = GrandExchangeEnv(
        items=items,
        starting_gp=1000000,
        tick_duration=5,
        max_ticks=1440,
        price_fluctuation_pct=0.05,  # Higher fluctuation for testing
        buy_limit_reset_ticks=240,
        random_seed=42,
        high_vol_items_path="high_vol_items.txt",
        enable_margin_experimentation=True,
        max_margin_attempts=5,
        min_margin_pct=0.02,
        max_margin_pct=0.40,
        margin_wait_steps=3   # Default wait steps
    )
    
    env.reset()
    
    # Test different wait steps for different items
    logger.info("\nTesting different wait steps for different items:")
    
    # Place buy orders with different wait steps
    items_to_test = ["Nature rune", "Law rune", "Death rune"]
    wait_steps = [2, 5, 8]  # Different wait steps for each item
    
    for i, (item, steps) in enumerate(zip(items_to_test, wait_steps)):
        env.prices[item] = items[item]["base_price"]  # Set market price
        
        # Place a buy order with custom wait steps
        action = {
            "type": "buy",
            "item": item,
            "price": int(env.prices[item] * 0.95),  # 5% below market price
            "quantity": 100,
            "wait_steps": steps
        }
        
        # Execute the action
        _, reward, done, info = env.step(action)
        logger.info(f"Buy order placed for {item} with wait_steps={steps}: {info['msg']}")
    
    # Run simulation steps to see different wait steps in action
    logger.info("\nRunning simulation steps with different wait steps:")
    for i in range(15):  # Run enough steps to see different wait behaviors
        # Fluctuate prices
        env._fluctuate_prices()
        
        # Process orders
        env._fulfill_orders()
        
        # Check all orders
        logger.info(f"\nStep {i+1}:")
        for j, order in enumerate(env.open_orders):
            item = order['item']
            market_price = env.prices[item]
            order_key = f"{item}_{order['type']}_{order['price']}"
            wait_count = env.margin_wait_counters.get(order_key, 0)
            wait_steps = order.get('wait_steps', env.margin_wait_steps)
            
            logger.info(f"  Order {j+1}: {item} @ {order['price']} GP (market: {market_price} GP)")
            logger.info(f"    Wait counter: {wait_count}/{wait_steps}")
            
            # Check if margin experimentation is happening
            if item in env.margin_attempts and order['type'] in env.margin_attempts[item]:
                current_attempt = env.margin_attempts[item][order['type']]['current_attempt']
                prices_tried = env.margin_attempts[item][order['type']]['prices_tried']
                logger.info(f"    Attempt: {current_attempt}, Prices tried: {prices_tried}")
    
    # Log margin success rates
    logger.info(f"\nMargin success rates: {env.margin_success_rates}")
    
    return env
    
def test_volume_based_validation_and_rollback():
    """Test volume-based validation and rollback mechanisms"""
    logger.info("\nTesting volume-based validation and rollback mechanisms...")
    
    # Create a mock volume analyzer
    id_to_name_map = {
        "1": "Nature rune",
        "2": "Law rune",
        "3": "Death rune"
    }
    name_to_id_map = {name: id for id, name in id_to_name_map.items()}
    
    volume_analyzer = VolumeAnalyzer(id_to_name_map, name_to_id_map)
    
    # Add some mock volume data
    timestamp = int(time.time())
    
    # Nature rune - high volume
    volume_analyzer.volume_history_1h["1"] = [
        (timestamp - 3600, 5000, 4500),  # (timestamp, high_vol, low_vol)
        (timestamp - 2400, 5200, 4800),
        (timestamp - 1200, 5500, 5000),
        (timestamp, 6000, 5500)
    ]
    
    # Law rune - medium volume
    volume_analyzer.volume_history_1h["2"] = [
        (timestamp - 3600, 2000, 1800),
        (timestamp - 2400, 2100, 1900),
        (timestamp - 1200, 2200, 2000),
        (timestamp, 2300, 2100)
    ]
    
    # Death rune - low volume
    volume_analyzer.volume_history_1h["3"] = [
        (timestamp - 3600, 500, 400),
        (timestamp - 2400, 550, 450),
        (timestamp - 1200, 600, 500),
        (timestamp, 650, 550)
    ]
    
    # Add price data
    volume_analyzer.price_history_1h["1"] = [
        (timestamp - 3600, 200, 190),  # (timestamp, high_price, low_price)
        (timestamp - 2400, 205, 195),
        (timestamp - 1200, 210, 200),
        (timestamp, 215, 205)
    ]
    
    volume_analyzer.price_history_1h["2"] = [
        (timestamp - 3600, 300, 290),
        (timestamp - 2400, 305, 295),
        (timestamp - 1200, 310, 300),
        (timestamp, 315, 305)
    ]
    
    volume_analyzer.price_history_1h["3"] = [
        (timestamp - 3600, 250, 240),
        (timestamp - 2400, 255, 245),
        (timestamp - 1200, 260, 250),
        (timestamp, 265, 255)
    ]
    
    # Create environment with margin experimentation enabled and volume analyzer
    items = load_test_items()
    
    env = GrandExchangeEnv(
        items=items,
        starting_gp=1000000,
        tick_duration=5,
        max_ticks=1440,
        price_fluctuation_pct=0.05,
        buy_limit_reset_ticks=240,
        random_seed=42,
        high_vol_items_path="high_vol_items.txt",
        enable_margin_experimentation=True,
        max_margin_attempts=5,
        min_margin_pct=0.02,
        max_margin_pct=0.40,
        margin_wait_steps=3,
        volume_analyzer=volume_analyzer
    )
    
    env.reset()
    
    # Create a margin comparison analyzer
    margin_analyzer = MarginComparisonAnalyzer(volume_analyzer)
    
    # Test 1: Volume-based validation
    logger.info("\nTest 1: Volume-based validation")
    
    # Test validation for different items with different volumes
    for item_name in ["Nature rune", "Law rune", "Death rune"]:
        # Test with different margin percentages
        for margin_pct in [0.05, 0.15, 0.30]:
            validation_result = margin_analyzer.validate_margin_experiment_with_volume(item_name, margin_pct)
            logger.info(f"Validation for {item_name} with {margin_pct:.2%} margin: {validation_result}")
    
    # Test 2: Simulate failed experiments and test rollback
    logger.info("\nTest 2: Rollback mechanism for failed experiments")
    
    # Create some failed experiments
    env.failed_margin_experiments = {
        "Nature rune": [
            {
                "type": "buy",
                "timestamp": time.time() - 3600,  # 1 hour ago
                "attempts": 5,
                "prices_tried": [190, 180, 170, 160, 150],
                "original_price": 190,
                "final_price": 150,
                "duration": 300  # 5 minutes
            }
        ],
        "Law rune": [
            {
                "type": "sell",
                "timestamp": time.time() - 1800,  # 30 minutes ago
                "attempts": 4,
                "prices_tried": [310, 320, 330, 340],
                "original_price": 310,
                "final_price": 340,
                "duration": 240  # 4 minutes
            }
        ],
        "Death rune": [
            {
                "type": "buy",
                "timestamp": time.time() - 7200,  # 2 hours ago
                "attempts": 8,  # Many attempts
                "prices_tried": [240, 230, 220, 210, 200, 190, 180, 170],
                "original_price": 240,
                "final_price": 170,
                "duration": 600  # 10 minutes (long duration)
            }
        ]
    }
    
    # Run validation and rollback
    rollback_results = validate_and_rollback_failed_experiments(env, margin_analyzer, volume_analyzer)
    logger.info(f"Rollback results: {rollback_results}")
    
    # Check rollback metrics
    rollback_metrics = margin_analyzer.get_rollback_metrics()
    logger.info(f"Rollback metrics: {rollback_metrics}")
    
    return env, margin_analyzer, volume_analyzer

if __name__ == "__main__":
    logger.info("Starting margin experimentation test...")
    
    # Run the basic test
    env = test_margin_experimentation()
    
    # Run the dynamic wait steps test
    env = test_dynamic_wait_steps()
    
    # Run the volume-based validation and rollback test
    try:
        env, margin_analyzer, volume_analyzer = test_volume_based_validation_and_rollback()
        logger.info("\nVolume-based validation and rollback test completed successfully!")
    except Exception as e:
        logger.error(f"Error in volume-based validation and rollback test: {e}")
    
    logger.info("\nAll tests completed!")
    
    # Log the benefits of exploring larger margins
    logger.info("\nBenefits of exploring larger margins (up to 40%):")
    logger.info("1. Potential for significantly better prices in volatile markets")
    logger.info("2. Ability to capture extreme price movements during market events")
    logger.info("3. More comprehensive exploration of the price-volume relationship")
    logger.info("4. Better adaptation to different market conditions for each item")
    logger.info("5. Increased profit potential by finding optimal price points")
    logger.info("6. Dynamic wait steps allow for item-specific patience levels")
    
    # Log the benefits of the new adaptive margin threshold system
    logger.info("\nBenefits of the adaptive margin threshold system:")
    logger.info("1. Automatically adjusts thresholds based on market conditions")
    logger.info("2. Considers volume metrics for more accurate margin expectations")
    logger.info("3. Adapts to market volatility for better risk management")
    logger.info("4. Learns from success/failure rates to optimize thresholds")
    logger.info("5. Provides rollback mechanism for failed margin experiments")
    logger.info("6. Integrates volume-based validation for more reliable margin testing")