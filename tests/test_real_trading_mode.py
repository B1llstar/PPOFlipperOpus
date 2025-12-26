import asyncio
import logging
import os
import time
import unittest
import sys
from unittest.mock import MagicMock, patch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_real_trading_mode.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_real_trading_mode")

# Import required modules
from run_real_trading import run_real_trading, read_mapping_file, fetch_marketplace_data, build_items_dict
from real_time_ge_client import RealTimeGrandExchangeClient
from ppo_websocket_integration import PPOWebSocketIntegration
from ge_env import GrandExchangeEnv
from ppo_agent import PPOAgent
from config import ENV_KWARGS, PPO_KWARGS

class TestRealTradingMode(unittest.TestCase):
    """Test cases for the Real Trading Mode."""
    
    async def async_setup(self):
        """Set up the test environment asynchronously."""
        # Create a mock WebSocket integration
        self.websocket_integration = MagicMock(spec=PPOWebSocketIntegration)
        self.websocket_integration.connect = MagicMock(return_value=asyncio.Future())
        self.websocket_integration.connect.return_value.set_result(True)
        self.websocket_integration.disconnect = MagicMock(return_value=asyncio.Future())
        self.websocket_integration.disconnect.return_value.set_result(None)
        self.websocket_integration.system_status = MagicMock(return_value=asyncio.Future())
        self.websocket_integration.system_status.return_value.set_result(True)
        self.websocket_integration.portfolio_update = MagicMock(return_value=asyncio.Future())
        self.websocket_integration.portfolio_update.return_value.set_result(True)
        
        # Create a mock real-time client
        self.real_time_client = MagicMock(spec=RealTimeGrandExchangeClient)
        self.real_time_client.start = MagicMock(return_value=asyncio.Future())
        self.real_time_client.start.return_value.set_result(None)
        self.real_time_client.stop = MagicMock(return_value=asyncio.Future())
        self.real_time_client.stop.return_value.set_result(None)
        
        # Create test data
        self.id_name_map = {"554": "Fire rune", "555": "Water rune", "556": "Air rune"}
        self.buy_limits_map = {"554": 10000, "555": 10000, "556": 10000}
        self.marketplace_data = {
            "554": {"high": 5, "low": 4, "highTime": int(time.time()), "lowTime": int(time.time())},
            "555": {"high": 5, "low": 4, "highTime": int(time.time()), "lowTime": int(time.time())},
            "556": {"high": 5, "low": 4, "highTime": int(time.time()), "lowTime": int(time.time())}
        }
        
        # Create test items
        self.items = {
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
        
        # Create item lists
        self.item_list = list(self.items.keys())
        self.price_ranges = {item: (self.items[item]['min_price'], self.items[item]['max_price']) for item in self.item_list}
        self.buy_limits = {item: self.items[item]['buy_limit'] for item in self.item_list}
        
        # Create a mock environment
        self.env = MagicMock(spec=GrandExchangeEnv)
        self.env.reset.return_value = {
            'gp': 1000000,
            'inventory': {'Fire rune': 0, 'Water rune': 0, 'Air rune': 0},
            'prices': {'Fire rune': 5, 'Water rune': 5, 'Air rune': 5}
        }
        self.env.step.return_value = (
            {
                'gp': 999995,
                'inventory': {'Fire rune': 1, 'Water rune': 0, 'Air rune': 0},
                'prices': {'Fire rune': 5, 'Water rune': 5, 'Air rune': 5}
            },
            0.0,
            False,
            {}
        )
        self.env.update_real_market_data = MagicMock(return_value=asyncio.Future())
        self.env.update_real_market_data.return_value.set_result(None)
        self.env._update_real_orders = MagicMock()
        
        # Create a mock agent
        self.agent = MagicMock(spec=PPOAgent)
        self.agent.sample_action.return_value = (
            {
                'type': 'buy',
                'item': 'Fire rune',
                'price': 5,
                'quantity': 1
            },
            None
        )
        
        # Patch the necessary functions
        self.read_mapping_patch = patch('run_real_trading.read_mapping_file', return_value=(self.id_name_map, self.buy_limits_map))
        self.fetch_marketplace_patch = patch('run_real_trading.fetch_marketplace_data', return_value=self.marketplace_data)
        self.build_items_patch = patch('run_real_trading.build_items_dict', return_value=self.items)
        self.initialize_volume_patch = patch('run_real_trading.initialize_volume_analyzer', return_value=MagicMock())
        self.load_agent_patch = patch('run_real_trading.load_trained_agent', return_value=self.agent)
        self.ge_env_patch = patch('run_real_trading.GrandExchangeEnv', return_value=self.env)
        self.ppo_integration_patch = patch('run_real_trading.PPOWebSocketIntegration', return_value=self.websocket_integration)
        self.real_time_client_patch = patch('run_real_trading.RealTimeGrandExchangeClient', return_value=self.real_time_client)
        
        # Start the patches
        self.read_mapping_patch.start()
        self.fetch_marketplace_patch.start()
        self.build_items_patch.start()
        self.initialize_volume_patch.start()
        self.load_agent_patch.start()
        self.ge_env_patch.start()
        self.ppo_integration_patch.start()
        self.real_time_client_patch.start()
    
    async def async_teardown(self):
        """Tear down the test environment asynchronously."""
        # Stop the patches
        self.read_mapping_patch.stop()
        self.fetch_marketplace_patch.stop()
        self.build_items_patch.stop()
        self.initialize_volume_patch.stop()
        self.load_agent_patch.stop()
        self.ge_env_patch.stop()
        self.ppo_integration_patch.stop()
        self.real_time_client_patch.stop()
    
    async def test_run_real_trading(self):
        """Test the run_real_trading function."""
        logger.info("Testing run_real_trading function")
        
        # Set up the test environment
        await self.async_setup()
        
        # Create a mock for asyncio.sleep to avoid waiting
        with patch('asyncio.sleep', return_value=asyncio.Future()) as mock_sleep:
            mock_sleep.return_value.set_result(None)
            
            # Create a mock for the trading loop to exit after one iteration
            def side_effect(*args, **kwargs):
                # Raise KeyboardInterrupt after one iteration to exit the loop
                self.env.step.side_effect = KeyboardInterrupt()
                return self.env.step.return_value
            
            self.env.step = MagicMock(side_effect=side_effect)
            
            # Run the function
            try:
                await run_real_trading()
            except KeyboardInterrupt:
                pass
            
            # Verify that the WebSocket integration was connected
            self.websocket_integration.connect.assert_called_once()
            
            # Verify that the real-time client was started
            self.real_time_client.start.assert_called_once()
            
            # Verify that the environment was reset
            self.env.reset.assert_called_once()
            
            # Verify that the agent was hooked to the WebSocket integration
            self.websocket_integration.hook_agent.assert_called_once()
            
            # Verify that the environment was hooked to the WebSocket integration
            self.websocket_integration.hook_env.assert_called_once()
            
            # Verify that the agent's sample_action was called
            self.agent.sample_action.assert_called_once()
            
            # Verify that the environment's step was called
            self.env.step.assert_called_once()
            
            # Verify that the WebSocket integration was disconnected
            self.websocket_integration.disconnect.assert_called_once()
            
            # Verify that the real-time client was stopped
            self.real_time_client.stop.assert_called_once()
        
        # Tear down the test environment
        await self.async_teardown()
    
    async def test_read_mapping_file(self):
        """Test the read_mapping_file function."""
        logger.info("Testing read_mapping_file function")
        
        # Create a temporary mapping file
        mapping_data = [
            {"id": 554, "name": "Fire rune", "limit": 10000},
            {"id": 555, "name": "Water rune", "limit": 10000},
            {"id": 556, "name": "Air rune", "limit": 10000}
        ]
        
        # Mock the open function
        with patch('builtins.open', unittest.mock.mock_open(read_data=json.dumps(mapping_data))), \
             patch('os.path.exists', return_value=True), \
             patch('json.load', return_value=mapping_data):
            
            # Call the function
            id_name_map, buy_limits_map = read_mapping_file()
            
            # Verify the results
            self.assertEqual(id_name_map, {"554": "Fire rune", "555": "Water rune", "556": "Air rune"})
            self.assertEqual(buy_limits_map, {"554": 10000, "555": 10000, "556": 10000})
    
    async def test_fetch_marketplace_data(self):
        """Test the fetch_marketplace_data function."""
        logger.info("Testing fetch_marketplace_data function")
        
        # Create a mock client
        client = MagicMock()
        client.get_latest.return_value = self.marketplace_data
        
        # Call the function
        result = fetch_marketplace_data(client)
        
        # Verify the results
        self.assertEqual(result, self.marketplace_data)
        client.get_latest.assert_called_once()
    
    async def test_build_items_dict(self):
        """Test the build_items_dict function."""
        logger.info("Testing build_items_dict function")
        
        # Call the function
        with patch('run_real_trading.ENV_KWARGS', {'strict_mode': False}):
            result = build_items_dict(self.id_name_map, self.buy_limits_map, self.marketplace_data)
            
            # Verify the results
            self.assertEqual(len(result), 3)
            self.assertIn("Fire rune", result)
            self.assertIn("Water rune", result)
            self.assertIn("Air rune", result)
            
            # Check item properties
            for item_name, item_data in result.items():
                self.assertIn("base_price", item_data)
                self.assertIn("buy_limit", item_data)
                self.assertIn("min_price", item_data)
                self.assertIn("max_price", item_data)

def run_tests():
    """Run the tests."""
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add the tests
    test_case = TestRealTradingMode()
    suite.addTest(test_case)
    
    # Run the tests
    runner = unittest.TextTestRunner()
    runner.run(suite)

async def main():
    """Run the tests asynchronously."""
    test_case = TestRealTradingMode()
    
    # Run the tests
    await test_case.test_run_real_trading()
    await test_case.test_read_mapping_file()
    await test_case.test_fetch_marketplace_data()
    await test_case.test_build_items_dict()

if __name__ == "__main__":
    # Run the tests
    asyncio.run(main())