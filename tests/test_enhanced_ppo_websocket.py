import asyncio
import logging
import os
import sys
import time
import json
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import numpy as np
import torch
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_enhanced_ppo_websocket.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_enhanced_ppo_websocket")

# Import the modules to test
from ppo_websocket_integration import PPOWebSocketIntegration, EventTypes
from ge_env import GrandExchangeEnv
from ppo_agent import PPOAgent
from volume_analysis import VolumeAnalyzer
from market_order_manager import MarketOrderManager

class TestEnhancedPPOWebSocketIntegration(unittest.TestCase):
    """Test suite for enhanced features of PPOWebSocketIntegration class"""

    def setUp(self):
        """Set up test environment before each test"""
        # Load actual mappings from file
        try:
            with open("endpoints/mapping.txt", 'r') as f:
                mapping_data = json.load(f)
                
            # Create ID to name mapping
            self.id_to_name_map = {}
            self.name_to_id_map = {}
            for item in mapping_data:
                item_id = str(item.get('id'))
                item_name = item.get('name')
                if item_id and item_name:
                    self.id_to_name_map[item_id] = item_name
                    self.name_to_id_map[item_name] = item_id
            
            logger.info(f"Loaded mapping data for {len(self.id_to_name_map)} items")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load mapping data: {e}")
            # Fallback to mock mappings
            self.id_to_name_map = {
                "1": "Abyssal whip",
                "2": "Dragon bones",
                "3": "Nature rune",
                "4": "Cannonball",
                "5": "Zulrah's scales"
            }
            self.name_to_id_map = {v: k for k, v in self.id_to_name_map.items()}
        
        # Create a volume analyzer
        self.volume_analyzer = VolumeAnalyzer(self.id_to_name_map, self.name_to_id_map)
        
        # Add some test data to volume analyzer
        current_time = int(time.time())
        self.volume_analyzer.volume_history_1h = {
            "1": [(current_time, 100, 50)],  # timestamp, high_vol, low_vol
            "2": [(current_time, 200, 150)],
            "3": [(current_time, 300, 250)],
            "4": [(current_time, 1000, 800)],
            "5": [(current_time, 5000, 4000)]
        }
        self.volume_analyzer.price_history_1h = {
            "1": [(current_time, 1500000, 1450000)],  # timestamp, high_price, low_price
            "2": [(current_time, 3000, 2900)],
            "3": [(current_time, 250, 240)],
            "4": [(current_time, 200, 190)],
            "5": [(current_time, 180, 170)]
        }
        
        # Create test items for GE environment
        self.test_items = {
            "Abyssal whip": {
                "base_price": 1500000,
                "min_price": 1000000,
                "max_price": 2000000,
                "buy_limit": 10
            },
            "Dragon bones": {
                "base_price": 3000,
                "min_price": 2000,
                "max_price": 4000,
                "buy_limit": 100
            },
            "Nature rune": {
                "base_price": 250,
                "min_price": 200,
                "max_price": 300,
                "buy_limit": 1000
            },
            "Cannonball": {
                "base_price": 200,
                "min_price": 150,
                "max_price": 250,
                "buy_limit": 5000
            },
            "Zulrah's scales": {
                "base_price": 180,
                "min_price": 150,
                "max_price": 220,
                "buy_limit": 20000
            }
        }
        
        # Create a PPO WebSocket Integration instance with proper initialization
        self.integration = PPOWebSocketIntegration(
            websocket_url="http://localhost:5178",
            max_slots=8
        )
        
        # Update the id_to_name_map and name_to_id_map in the integration
        self.integration.id_to_name_map = self.id_to_name_map
        self.integration.name_to_id_map = self.name_to_id_map
        
        # Update the volume analyzer mappings
        self.integration.volume_analyzer.id_to_name_map = self.id_to_name_map
        self.integration.volume_analyzer.name_to_id_map = self.name_to_id_map
        
        # Replace the margin_env with our test environment
        self.integration.margin_env = GrandExchangeEnv(
            items=self.test_items,
            enable_margin_experimentation=True,
            max_margin_attempts=10,
            min_margin_pct=0.8,
            max_margin_pct=0.40,
            margin_wait_steps=3,
            volume_analyzer=self.volume_analyzer
        )
        
        # Initialize price history for validation tests
        self.integration.margin_env.price_history = {
            "Abyssal whip": [1500000, 1510000, 1490000, 1505000, 1500000],
            "Dragon bones": [3000, 3100, 2900, 3050, 3000],
            "Nature rune": [250, 260, 240, 255, 250],
            "Cannonball": [200, 205, 195, 200, 200],
            "Zulrah's scales": [180, 185, 175, 180, 180]
        }
        
        # Create a mock PPO agent
        self.agent = MagicMock()
        self.agent.item_list = list(self.test_items.keys())
        self.agent.price_ranges = {item: (data["min_price"], data["max_price"]) 
                                  for item, data in self.test_items.items()}
        self.agent.buy_limits = {item: data["buy_limit"] for item, data in self.test_items.items()}
        
        # Mock the historical client
        self.agent.historical_client = MagicMock()
        self.agent.historical_client._id_to_name_map = self.id_to_name_map
        self.agent.historical_client.get_latest.return_value = {
            "1": {"high": 1500000, "low": 1450000},
            "2": {"high": 3000, "low": 2900},
            "3": {"high": 250, "low": 240},
            "4": {"high": 200, "low": 190},
            "5": {"high": 180, "low": 170}
        }
        
        # Initialize the order manager with test orders
        self.integration.order_manager = MarketOrderManager(
            max_slots=8,
            experiment_timeout=300,
            final_timeout=600
        )
        
        # Mock the session for async methods
        self.mock_session = AsyncMock()
        self.integration.session = self.mock_session
        self.integration.connected = True
        
        # Hook the agent with ID "agent_0" to match what handle_freed_space expects
        self.integration.hook_agent(self.agent, "agent_0")

    async def test_relist_command(self):
        """Test the 'relist' command functionality"""
        logger.info("Testing relist command functionality")
        
        # Add a test order to the order manager
        test_order_id = "test_order_123"
        self.integration.order_manager.add_order(
            order_id=test_order_id,
            item="Abyssal whip",
            order_type="buy",
            is_experiment=False
        )
        
        # Store price and quantity in the order info dictionary manually for testing
        self.integration.order_manager.active_orders[test_order_id]['price'] = 1450000
        self.integration.order_manager.active_orders[test_order_id]['quantity'] = 1
        
        # Mock the cancel_order method to return True
        self.integration.cancel_order = AsyncMock(return_value=True)
        
        # Mock the place_order method to return True
        self.integration.place_order = AsyncMock(return_value=True)
        
        # Test relisting with the same price
        result = await self.integration.relist_order(test_order_id)
        
        # Verify results
        self.assertTrue(result)
        self.integration.cancel_order.assert_called_once_with(test_order_id)
        self.integration.place_order.assert_called_once()
        
        # Reset mocks
        self.integration.cancel_order.reset_mock()
        self.integration.place_order.reset_mock()
        
        # Add another test order
        test_order_id_2 = "test_order_456"
        self.integration.order_manager.add_order(
            order_id=test_order_id_2,
            item="Dragon bones",
            order_type="sell",
            is_experiment=True
        )
        
        # Store price and quantity in the order info dictionary manually for testing
        self.integration.order_manager.active_orders[test_order_id_2]['price'] = 3000
        self.integration.order_manager.active_orders[test_order_id_2]['quantity'] = 50
        
        # Test relisting with a new price
        new_price = 3100
        result = await self.integration.relist_order(test_order_id_2, new_price)
        
        # Verify results
        self.assertTrue(result)
        self.integration.cancel_order.assert_called_once_with(test_order_id_2)
        self.integration.place_order.assert_called_once()
        
        # Verify the new price was used
        call_args = self.integration.place_order.call_args[0][0]
        self.assertEqual(call_args['price'], new_price)
        
        # Test relisting a non-existent order
        result = await self.integration.relist_order("non_existent_order")
        
        # Verify results
        self.assertFalse(result)
        
        # Test relisting when cancel fails
        self.integration.cancel_order.reset_mock()
        self.integration.cancel_order.return_value = False
        
        result = await self.integration.relist_order(test_order_id)
        
        # Verify results
        self.assertFalse(result)
        self.integration.place_order.assert_called_once()  # Should still be from previous call
        
        logger.info("Relist command functionality test completed")

    def test_sell_price_validation(self):
        """Test validation to ensure selling price is never lower than recent data"""
        logger.info("Testing sell price validation")
        
        # Set up the observation
        obs = {
            'prices': {
                "Abyssal whip": 1500000,
                "Dragon bones": 3000,
                "Nature rune": 250,
                "Cannonball": 200,
                "Zulrah's scales": 180
            },
            'inventory': {
                "Abyssal whip": 1,
                "Dragon bones": 50,
                "Nature rune": 500,
                "Cannonball": 1000,
                "Zulrah's scales": 5000
            },
            'gp': 10000000,
            'buy_limits': {
                "Abyssal whip": 0,
                "Dragon bones": 0,
                "Nature rune": 0,
                "Cannonball": 0,
                "Zulrah's scales": 0
            }
        }
        
        # Test valid sell action (price above recent minimum)
        action = {
            'type': 'sell',
            'item': 'Abyssal whip',
            'price': 1495000,  # Above the recent minimum of 1490000
            'quantity': 1
        }
        
        result = self.integration._validate_trade(action, obs)
        self.assertTrue(result)
        
        # Test invalid sell action (price below recent minimum)
        action = {
            'type': 'sell',
            'item': 'Abyssal whip',
            'price': 1480000,  # Below the recent minimum of 1490000
            'quantity': 1
        }
        
        result = self.integration._validate_trade(action, obs)
        self.assertFalse(result)
        
        # Test with another item
        action = {
            'type': 'sell',
            'item': 'Dragon bones',
            'price': 2850,  # Below the recent minimum of 2900
            'quantity': 10
        }
        
        result = self.integration._validate_trade(action, obs)
        self.assertFalse(result)
        
        # Test with a price at exactly the minimum
        action = {
            'type': 'sell',
            'item': 'Nature rune',
            'price': 240,  # Exactly the recent minimum
            'quantity': 100
        }
        
        result = self.integration._validate_trade(action, obs)
        self.assertTrue(result)
        
        # Test that buy actions are not affected by this validation
        action = {
            'type': 'buy',
            'item': 'Abyssal whip',
            'price': 1480000,  # Below the recent minimum, but it's a buy action
            'quantity': 1
        }
        
        result = self.integration._validate_trade(action, obs)
        self.assertTrue(result)
        
        logger.info("Sell price validation test completed")

    def test_simultaneous_decision_making(self):
        """Test the simultaneous decision-making process for all positions"""
        logger.info("Testing simultaneous decision-making process")
        
        # Set up the observation
        obs = {
            'prices': {
                "Abyssal whip": 1550000,  # Profitable position
                "Dragon bones": 2800,     # Loss position
                "Nature rune": 260,       # Profitable position with negative momentum
                "Cannonball": 200,        # Neutral position
                "Zulrah's scales": 180    # Neutral position
            },
            'inventory': {
                "Abyssal whip": 1,
                "Dragon bones": 50,
                "Nature rune": 500,
                "Cannonball": 1000,
                "Zulrah's scales": 5000
            },
            'gp': 10000000,
            'buy_limits': {
                "Abyssal whip": 0,
                "Dragon bones": 0,
                "Nature rune": 0,
                "Cannonball": 0,
                "Zulrah's scales": 0
            }
        }
        
        # Set up buy price history for profit calculation
        self.integration.margin_env.buy_price_history = {
            "Abyssal whip": [1500000],  # Profitable
            "Dragon bones": [3100, 3100, 3100],  # Loss > 5%
            "Nature rune": [240, 240, 240],  # Profitable
            "Cannonball": [200, 200, 200],  # Neutral
            "Zulrah's scales": [180, 180, 180]  # Neutral
        }
        
        # Mock volume metrics for Nature rune to have negative momentum
        self.integration.get_volume_metrics = MagicMock(side_effect=lambda item, analyzer: 
            {"momentum_1h": -0.2} if item == "Nature rune" else 
            {"momentum_1h": 0.1}
        )
        
        # Call the process_all_positions method
        actions = self.integration.process_all_positions(obs)
        
        # Verify results
        self.assertEqual(len(actions), 2)  # Should have 2 sell actions
        
        # Check that we're selling Dragon bones (loss > 5%)
        dragon_bones_action = next((a for a in actions if a['item'] == 'Dragon bones'), None)
        self.assertIsNotNone(dragon_bones_action)
        self.assertEqual(dragon_bones_action['type'], 'sell')
        self.assertEqual(dragon_bones_action['quantity'], 50)
        
        # Check that we're selling Nature rune (profitable but negative momentum)
        nature_rune_action = next((a for a in actions if a['item'] == 'Nature rune'), None)
        self.assertIsNotNone(nature_rune_action)
        self.assertEqual(nature_rune_action['type'], 'sell')
        self.assertEqual(nature_rune_action['quantity'], 500)
        
        # Check that we're NOT selling Abyssal whip (profitable with positive momentum)
        abyssal_whip_action = next((a for a in actions if a['item'] == 'Abyssal whip'), None)
        self.assertIsNone(abyssal_whip_action)
        
        # Check that we're NOT selling Cannonball or Zulrah's scales (neutral positions)
        cannonball_action = next((a for a in actions if a['item'] == 'Cannonball'), None)
        zulrah_scales_action = next((a for a in actions if a['item'] == "Zulrah's scales"), None)
        self.assertIsNone(cannonball_action)
        self.assertIsNone(zulrah_scales_action)
        
        logger.info("Simultaneous decision-making test completed")

    async def test_freed_space_action(self):
        """Test the logic to force an action when a space is freed"""
        logger.info("Testing freed space action logic")
        
        # Mock the order manager to report available slots
        self.integration.order_manager.get_available_slots = MagicMock(return_value=2)
        
        # Set building_portfolio to False to trigger the freed space logic
        self.integration.building_portfolio = False
        
        # Mock the _get_obs method to return a valid observation
        self.integration._get_obs = MagicMock(return_value={
            'prices': {
                "Abyssal whip": 1500000,
                "Dragon bones": 3000,
                "Nature rune": 250,
                "Cannonball": 200,
                "Zulrah's scales": 180
            },
            'inventory': {
                "Abyssal whip": 0,
                "Dragon bones": 0,
                "Nature rune": 0,
                "Cannonball": 0,
                "Zulrah's scales": 0
            },
            'gp': 10000000,
            'buy_limits': {
                "Abyssal whip": 0,
                "Dragon bones": 0,
                "Nature rune": 0,
                "Cannonball": 0,
                "Zulrah's scales": 0
            }
        })
        
        # Mock the agent's neural network outputs
        self.agent.obs_to_tensor = MagicMock(return_value=torch.tensor([0.0]))
        self.agent.model = MagicMock()
        self.agent.model.eval = MagicMock()
        self.agent.model.train = MagicMock()
        
        # Mock the model output
        item_logits = torch.zeros(1, 5)
        item_logits[0, 0] = 1.0  # Select Abyssal whip
        
        price_logits = torch.zeros(1, 10)
        price_logits[0, 5] = 1.0  # Select middle price bin
        
        qty_logits = torch.zeros(1, 10)
        qty_logits[0, 3] = 1.0  # Select quantity bin 3
        
        self.agent.model.return_value = (
            None,  # action_type_logits
            item_logits,
            price_logits,
            qty_logits,
            None,  # wait_steps_logits
            None   # value
        )
        
        # Mock the price_bins and quantity_bins attributes
        self.agent.price_bins = 10
        self.agent.quantity_bins = 10
        
        # Mock the _validate_trade method to return True
        self.integration._validate_trade = MagicMock(return_value=True)
        
        # Mock the place_order method and ensure it's properly set up
        place_order_mock = AsyncMock(return_value=True)
        self.integration.place_order = place_order_mock
        
        # Call the handle_freed_space method
        await self.integration.handle_freed_space()
        
        # Verify results
        place_order_mock.assert_called_once()
        
        # Check that the order has the correct structure
        order_data = self.integration.place_order.call_args[0][0]
        self.assertEqual(order_data['type'], 'buy')
        self.assertEqual(order_data['item'], 'Abyssal whip')
        self.assertGreater(order_data['price'], 0)
        self.assertGreater(order_data['quantity'], 0)
        
        # Test when building_portfolio is True
        self.integration.building_portfolio = True
        self.integration.place_order.reset_mock()
        
        await self.integration.handle_freed_space()
        
        # Verify that place_order was not called - use the mock directly
        place_order_mock.assert_not_called()
        
        # Test when no slots are available
        self.integration.building_portfolio = False
        self.integration.order_manager.get_available_slots = MagicMock(return_value=0)
        
        await self.integration.handle_freed_space()
        
        # Verify that place_order was still not called - use the mock directly
        place_order_mock.assert_not_called()
        
        # Test when validation fails
        self.integration.order_manager.get_available_slots = MagicMock(return_value=2)
        self.integration._validate_trade = MagicMock(return_value=False)
        
        await self.integration.handle_freed_space()
        
        # Verify that place_order was still not called - use the mock directly
        place_order_mock.assert_not_called()
        
        logger.info("Freed space action logic test completed")

    def tearDown(self):
        """Clean up after each test"""
        # Unhook the agent
        if "agent_0" in self.integration.agent_hooks:
            self.integration.unhook_agent("agent_0")

async def run_tests():
    """Run all the tests asynchronously"""
    test = TestEnhancedPPOWebSocketIntegration()
    test.setUp()
    
    # Run the tests
    await test.test_relist_command()
    test.test_sell_price_validation()
    test.test_simultaneous_decision_making()
    await test.test_freed_space_action()
    
    # Clean up
    test.tearDown()
    
    logger.info("All tests completed successfully")

if __name__ == "__main__":
    # Run the synchronous tests
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestEnhancedPPOWebSocketIntegration)
    unittest.TextTestRunner().run(test_suite)
    
    # Run the asynchronous tests
    asyncio.run(run_tests())