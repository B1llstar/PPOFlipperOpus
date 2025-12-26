import numpy as np
import torch
import logging
import json
from ge_env import GrandExchangeEnv
from ppo_agent import PPOAgent
from margin_reward import MarginRewardSystem
from config import ENV_KWARGS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("test_margin_rewards")

def load_test_items():
    """Load a small set of test items for testing"""
    test_items = {
        "Feather": {
            "base_price": 5,
            "buy_limit": 10000,
            "min_price": 4,
            "max_price": 6,
        },
        "Blood rune": {
            "base_price": 300,
            "buy_limit": 8000,
            "min_price": 270,
            "max_price": 330,
        },
        "Chaos rune": {
            "base_price": 100,
            "buy_limit": 7000,
            "min_price": 90,
            "max_price": 110,
        },
        "Death rune": {
            "base_price": 250,
            "buy_limit": 7000,
            "min_price": 225,
            "max_price": 275,
        },
        "Zulrah's scales": {
            "base_price": 200,
            "buy_limit": 10000,
            "min_price": 180,
            "max_price": 220,
        },
        "Cannonball": {
            "base_price": 150,
            "buy_limit": 10000,
            "min_price": 135,
            "max_price": 165,
        },
        "Nature rune": {
            "base_price": 200,
            "buy_limit": 10000,
            "min_price": 180,
            "max_price": 220,
        },
        "Soul rune": {
            "base_price": 180,
            "buy_limit": 10000,
            "min_price": 162,
            "max_price": 198,
        },
        "Mind rune": {
            "base_price": 5,
            "buy_limit": 10000,
            "min_price": 4,
            "max_price": 6,
        },
        "Rune arrow": {
            "base_price": 80,
            "buy_limit": 10000,
            "min_price": 72,
            "max_price": 88,
        }
    }
    return test_items

def test_margin_reward_system():
    """Test the margin reward system"""
    # Create a MarginRewardSystem instance
    margin_system = MarginRewardSystem()
    
    # Test margin calculation
    high_price = 100
    low_price = 90
    margin = margin_system.calculate_margin(high_price, low_price)
    logger.info(f"Margin for item with high price {high_price} and low price {low_price}: {margin:.4f}")
    
    # Test with tax
    high_price = 200
    low_price = 180
    margin = margin_system.calculate_margin(high_price, low_price)
    logger.info(f"Margin for item with high price {high_price} and low price {low_price} (with tax): {margin:.4f}")
    
    # Update margins for some items
    margin_system.update_item_margin("Feather", 5, 4)
    margin_system.update_item_margin("Blood rune", 320, 280)
    margin_system.update_item_margin("Chaos rune", 105, 95)
    
    # Log margin metrics
    margin_system.log_margin_metrics()
    
    # Test reward adjustments
    base_reward = 10.0
    adjusted_reward = margin_system.calculate_reward_adjustment("Feather", base_reward)
    logger.info(f"Adjusted reward for Feather: {adjusted_reward:.4f} (base: {base_reward})")
    
    adjusted_reward = margin_system.calculate_reward_adjustment("Blood rune", base_reward)
    logger.info(f"Adjusted reward for Blood rune: {adjusted_reward:.4f} (base: {base_reward})")
    
    adjusted_reward = margin_system.calculate_reward_adjustment("Chaos rune", base_reward)
    logger.info(f"Adjusted reward for Chaos rune: {adjusted_reward:.4f} (base: {base_reward})")
    
    # Test reward adjustment for item not in high_vol_items
    adjusted_reward = margin_system.calculate_reward_adjustment("Unknown Item", base_reward)
    logger.info(f"Adjusted reward for Unknown Item: {adjusted_reward:.4f} (base: {base_reward})")

def test_environment_with_margin_rewards():
    """Test the environment with margin-based rewards"""
    # Load test items
    test_items = load_test_items()
    
    # Create environment with margin-based rewards
    env_kwargs = ENV_KWARGS.copy()
    env_kwargs["items"] = test_items
    env = GrandExchangeEnv(**env_kwargs)
    
    # Reset environment
    obs = env.reset()
    logger.info(f"Initial observation: {obs.keys()}")
    
    # Check if margin_metrics is in the observation
    if 'margin_metrics' in obs:
        logger.info(f"Margin metrics: {obs['margin_metrics']}")
    else:
        logger.warning("Margin metrics not found in observation")
    
    # Create agent
    item_list = list(test_items.keys())
    price_ranges = {item: (test_items[item]['min_price'], test_items[item]['max_price']) for item in item_list}
    buy_limits = {item: test_items[item]['buy_limit'] for item in item_list}
    
    agent = PPOAgent(
        item_list=item_list,
        price_ranges=price_ranges,
        buy_limits=buy_limits,
        device="cpu"
    )
    
    # Run a few steps
    for i in range(10):
        try:
            # Sample action
            action, _ = agent.sample_action(obs)
            logger.info(f"Step {i}, Action: {action}")
            
            # Take action
            next_obs, reward, done, info = env.step(action)
            logger.info(f"Step {i}, Reward: {reward}")
            
            # Check margin metrics
            if 'margin_metrics' in next_obs:
                logger.info(f"Margin metrics: {next_obs['margin_metrics']}")
            
            obs = next_obs
        except Exception as e:
            logger.error(f"Error in step {i}: {e}")
            break

def main():
    """Main function"""
    logger.info("Testing margin reward system...")
    test_margin_reward_system()
    
    logger.info("\nTesting environment with margin-based rewards...")
    test_environment_with_margin_rewards()

if __name__ == "__main__":
    main()