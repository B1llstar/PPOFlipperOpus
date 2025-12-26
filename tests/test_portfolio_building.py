import unittest
from unittest.mock import patch, MagicMock, AsyncMock, call
import asyncio
import json
import time
import numpy as np
import torch
from typing import Dict, List, Any

# Import the modules to test
from ppo_websocket_integration import PPOWebSocketIntegration, Order, EventTypes
from enhanced_margin_experimentation import EnhancedMarginExperimentation
from ge_env import GrandExchangeEnv
from ppo_agent import PPOAgent

class TestPortfolioBuilding(unittest.TestCase):
    """
    Comprehensive test suite for portfolio building functionality.
    Tests the fixes made to the portfolio building process.
    """

    def setUp(self):
        """Set up test environment before each test."""
        # Create mock objects
        self.mock_agent = MagicMock(spec=PPOAgent)
        self.mock_env = MagicMock(spec=GrandExchangeEnv)
        
        # Configure mock agent
        self.mock_agent.item_list = ["item1", "item2", "item3", "item4", "item5"]
        self.mock_agent.price_ranges = {
            "item1": [100, 200],
            "item2": [200, 400],
            "item3": [300, 600],
            "item4": [400, 800],
            "item5": [500, 1000]
        }
        self.mock_agent.buy_limits = {
            "item1": 1000,
            "item2": 1000,
            "item3": 1000,
            "item4": 1000,
            "item5": 1000
        }
        
        # Mock sample_action to return a buy action
        self.mock_agent.sample_action.return_value = (
            {
                'type': 'buy',
                'item': 'item1',
                'price': 150,
                'quantity': 10
            },
            None
        )
        
        # Create a PPOWebSocketIntegration instance with mocked websocket
        self.ppo_integration = PPOWebSocketIntegration(websocket_url="ws://localhost:6969")
        
        # Mock the websocket connection
        self.ppo_integration.connected = True
        self.ppo_integration.ws = MagicMock()
        self.ppo_integration.ws.send_json = AsyncMock()
        self.ppo_integration.ws.receive_json = AsyncMock(return_value={"status": "success"})
        self.ppo_integration.session = MagicMock()
        
        # Set up initial state
        self.ppo_integration.margin_env.items = {
            "item1": {"min_price": 100, "max_price": 200, "buy_limit": 1000, "base_price": 100},
            "item2": {"min_price": 200, "max_price": 400, "buy_limit": 1000, "base_price": 200},
            "item3": {"min_price": 300, "max_price": 600, "buy_limit": 1000, "base_price": 300},
            "item4": {"min_price": 400, "max_price": 800, "buy_limit": 1000, "base_price": 400},
            "item5": {"min_price": 500, "max_price": 1000, "buy_limit": 1000, "base_price": 500}
        }
        
        # Mock prices
        self.ppo_integration.margin_env.prices = {
            "item1": 180,
            "item2": 380,
            "item3": 580,
            "item4": 780,
            "item5": 980
        }
        
        # Set up order manager with starting GP
        self.ppo_integration.order_manager.inventory_manager.gp = 1000000
        
        # Create a mock for _get_obs
        self.ppo_integration._get_obs = MagicMock(return_value={
            'gp': 1000000,
            'inventory': {},
            'prices': {
                "item1": 180,
                "item2": 380,
                "item3": 580,
                "item4": 780,
                "item5": 980
            },
            'buy_limits': {
                "item1": 0,
                "item2": 0,
                "item3": 0,
                "item4": 0,
                "item5": 0
            },
            'tick': 0
        })
        
        # Mock submit_event to return True
        self.ppo_integration.submit_event = AsyncMock(return_value=True)
        
        # Mock place_order to return True
        self.ppo_integration.place_order = AsyncMock(return_value=True)

    # Unit Tests for Portfolio Building Flag Management

    def test_portfolio_building_flag_initialization(self):
        """Test that portfolio building flag is initialized correctly."""
        # The flag should be set to True in __init__
        self.assertTrue(self.ppo_integration.building_portfolio)
        self.assertFalse(self.ppo_integration.initial_orders_placed)

    @patch('ppo_websocket_integration.PPOWebSocketIntegration.place_order')
    async def test_portfolio_building_flag_after_initial_orders(self, mock_place_order):
        """Test that portfolio building flag is updated after placing initial orders."""
        mock_place_order.return_value = True
        
        # Place initial orders
        await self.ppo_integration.place_initial_orders(self.mock_agent)
        
        # After successfully placing 8 orders, initial_orders_placed should be True
        self.assertTrue(self.ppo_integration.initial_orders_placed)
        
        # building_portfolio should still be True until all orders are filled
        self.assertTrue(self.ppo_integration.building_portfolio)

    @patch('ppo_websocket_integration.PPOWebSocketIntegration.place_order')
    async def test_portfolio_building_flag_after_partial_orders(self, mock_place_order):
        """Test that portfolio building flag is updated after placing partial initial orders."""
        # Make place_order fail after 4 orders
        mock_place_order.side_effect = [True, True, True, True, False, False, False, False]
        
        # Place initial orders
        await self.ppo_integration.place_initial_orders(self.mock_agent)
        
        # After placing some orders, initial_orders_placed should be True
        self.assertTrue(self.ppo_integration.initial_orders_placed)
        
        # building_portfolio should still be True to allow background completion
        self.assertTrue(self.ppo_integration.building_portfolio)

    @patch('ppo_websocket_integration.PPOWebSocketIntegration.place_order')
    async def test_portfolio_building_flag_after_no_orders(self, mock_place_order):
        """Test that portfolio building flag is updated when no orders can be placed."""
        # Make place_order always fail
        mock_place_order.return_value = False
        
        # Place initial orders
        await self.ppo_integration.place_initial_orders(self.mock_agent)
        
        # After failing to place any orders, both flags should be False
        self.assertFalse(self.ppo_integration.initial_orders_placed)
        self.assertFalse(self.ppo_integration.building_portfolio)

    # Unit Tests for Action Reception Logic

    @patch('ppo_agent.PPOAgent.sample_action')
    async def test_hooked_sample_action_normal_operation(self, mock_sample_action):
        """Test that hooked_sample_action uses normal inferencing during regular operation."""
        # Set up the test
        self.ppo_integration.building_portfolio = False
        
        # Create a mock for the original sample_action
        original_sample_action = MagicMock()
        original_sample_action.return_value = (
            {
                'type': 'buy',
                'item': 'item1',
                'price': 150,
                'quantity': 10
            },
            None
        )
        
        # Create a mock agent
        mock_agent = MagicMock()
        mock_agent.item_list = ["item1", "item2", "item3"]
        
        # Create observation
        obs = {
            'gp': 1000000,
            'inventory': {},
            'prices': {"item1": 180},
            'buy_limits': {"item1": 0},
            'tick': 0
        }
        
        # Hook the agent
        with patch('asyncio.get_event_loop', return_value=MagicMock()):
            with patch('asyncio.run_coroutine_threadsafe'):
                # Create the hooked_sample_action function
                hooked_sample_action = self.ppo_integration._create_hooked_sample_action(
                    mock_agent, 
                    original_sample_action, 
                    "agent_0", 
                    MagicMock()
                )
                
                # Call hooked_sample_action
                action, _ = hooked_sample_action(obs)
                
                # Verify original_sample_action was called
                original_sample_action.assert_called_once_with(obs, False)
                
                # Verify the action was returned unchanged
                self.assertEqual(action['type'], 'buy')
                self.assertEqual(action['item'], 'item1')
                self.assertEqual(action['price'], 150)
                self.assertEqual(action['quantity'], 10)

    # Unit Tests for Profitability Validation

    def test_validate_trade_profitability(self):
        """Test that trades are validated for profitability."""
        # Create an action that would not be profitable
        action = {
            'type': 'buy',
            'item': 'item1',
            'price': 179,  # Just 1 GP below market price
            'quantity': 10
        }
        
        # Create observation
        obs = {
            'gp': 1000000,
            'inventory': {},
            'prices': {"item1": 180},
            'buy_limits': {"item1": 0},
            'tick': 0
        }
        
        # Validate the trade
        result = self.ppo_integration._validate_trade(action, obs)
        
        # The trade should be rejected or adjusted for profitability
        # If adjusted, the price should be lower to ensure MIN_PROFIT_THRESHOLD
        if result:
            self.assertLess(action['price'], 179 - 51)  # 51 is MIN_PROFIT_THRESHOLD
        else:
            self.assertFalse(result)

    def test_validate_trade_profitability_during_portfolio_building(self):
        """Test that profitability validation is more lenient during portfolio building."""
        # Set portfolio building flag
        self.ppo_integration.building_portfolio = True
        
        # Create an action that would not be profitable
        action = {
            'type': 'buy',
            'item': 'item1',
            'price': 179,  # Just 1 GP below market price
            'quantity': 10
        }
        
        # Create observation
        obs = {
            'gp': 1000000,
            'inventory': {},
            'prices': {"item1": 180},
            'buy_limits': {"item1": 0},
            'tick': 0
        }
        
        # Validate the trade
        result = self.ppo_integration._validate_trade(action, obs)
        
        # During portfolio building, the trade should be allowed even if not profitable
        self.assertTrue(result)

    # Unit Tests for Client-Server Coordination

    @patch('ppo_websocket_integration.PPOWebSocketIntegration.submit_event')
    async def test_place_order_client_server_coordination(self, mock_submit_event):
        """Test that place_order coordinates properly with the server."""
        # Set up the test
        mock_submit_event.return_value = True
        
        # Create order data
        order_data = {
            'order_id': 'test_order_1',
            'item': 'item1',
            'type': 'buy',
            'price': 150,
            'quantity': 10
        }
        
        # Call place_order
        result = await self.ppo_integration.place_order(order_data)
        
        # Verify submit_event was called with the right parameters
        mock_submit_event.assert_called_once()
        args, kwargs = mock_submit_event.call_args
        self.assertEqual(args[0], "place_order")
        self.assertEqual(args[1], order_data)
        self.assertEqual(kwargs.get('priority', 0), 0)
        
        # Verify the result
        self.assertTrue(result)

    # Integration Tests for Portfolio Building

    @patch('ppo_websocket_integration.PPOWebSocketIntegration.place_order')
    async def test_place_initial_orders_builds_portfolio(self, mock_place_order):
        """Test that place_initial_orders builds a portfolio with 8 positions."""
        # Set up the test
        mock_place_order.return_value = True
        
        # Place initial orders
        await self.ppo_integration.place_initial_orders(self.mock_agent)
        
        # Verify place_order was called 8 times
        self.assertEqual(mock_place_order.call_count, 8)
        
        # Verify the flags are set correctly
        self.assertTrue(self.ppo_integration.initial_orders_placed)
        self.assertTrue(self.ppo_integration.building_portfolio)

    @patch('ppo_websocket_integration.PPOWebSocketIntegration._place_portfolio_building_order')
    async def test_handle_freed_space_during_portfolio_building(self, mock_place_order):
        """Test that handle_freed_space completes portfolio building when slots free up."""
        # Set up the test
        mock_place_order.return_value = True
        self.ppo_integration.building_portfolio = True
        self.ppo_integration.initial_orders_placed = True
        
        # Mock active orders to simulate 6 filled slots
        self.ppo_integration.order_manager.active_orders = {
            f"order_{i}": MagicMock() for i in range(6)
        }
        
        # Call handle_freed_space
        await self.ppo_integration.handle_freed_space()
        
        # Verify _place_portfolio_building_order was called twice (to fill remaining 2 slots)
        self.assertEqual(mock_place_order.call_count, 2)

    @patch('ppo_websocket_integration.PPOWebSocketIntegration.place_order')
    async def test_handle_freed_space_normal_operation(self, mock_place_order):
        """Test that handle_freed_space uses normal inferencing during regular operation."""
        # Set up the test
        mock_place_order.return_value = True
        self.ppo_integration.building_portfolio = False
        
        # Mock active orders to simulate 7 filled slots
        self.ppo_integration.order_manager.active_orders = {
            f"order_{i}": MagicMock() for i in range(7)
        }
        
        # Mock agent hooks
        self.ppo_integration.agent_hooks = {
            'agent_0': {
                'agent': self.mock_agent,
                'original_sample_action': MagicMock()
            }
        }
        
        # Mock model output
        mock_model = MagicMock()
        mock_model.return_value = (None, torch.tensor([0.1, 0.9]), torch.tensor([0.2, 0.8]), torch.tensor([0.3, 0.7]), None, None)
        self.mock_agent.model = mock_model
        self.mock_agent.obs_to_tensor = MagicMock(return_value=torch.tensor([1.0]))
        
        # Call handle_freed_space
        with patch('torch.no_grad'):
            await self.ppo_integration.handle_freed_space()
        
        # Verify place_order was called
        mock_place_order.assert_called_once()

    # Integration Tests for Margin Experimentation

    @patch('ppo_websocket_integration.PPOWebSocketIntegration.relist_order')
    @patch('ppo_websocket_integration.PPOWebSocketIntegration.cancel_order')
    async def test_check_timeouts_margin_experimentation(self, mock_cancel_order, mock_relist_order):
        """Test that check_timeouts handles margin experimentation correctly."""
        # Set up the test
        # Create a timed out order that is a margin experiment
        order = Order(
            order_id="test_order_1",
            item="item1",
            order_type="buy",
            price=150,
            quantity=10
        )
        order.margin_experiment = True
        
        # Add the order to the order manager
        self.ppo_integration.order_manager.orders = {"test_order_1": order}
        self.ppo_integration.order_manager.active_orders = {"test_order_1": order}
        
        # Mock get_timed_out_orders to return our test order
        self.ppo_integration.order_manager.get_timed_out_orders = MagicMock(return_value=["test_order_1"])
        
        # Mock has_suitable_margin to return False
        self.ppo_integration.order_manager.has_suitable_margin = MagicMock(return_value=False)
        
        # Call check_timeouts
        await self.ppo_integration.check_timeouts()
        
        # Verify relist_order was called with increased price
        mock_relist_order.assert_called_once_with("test_order_1", 157)  # 150 * 1.05 = 157.5, int = 157
        
        # Verify cancel_order was not called
        mock_cancel_order.assert_not_called()

    @patch('ppo_websocket_integration.PPOWebSocketIntegration.relist_order')
    @patch('ppo_websocket_integration.PPOWebSocketIntegration.cancel_order')
    async def test_check_timeouts_normal_timeout(self, mock_cancel_order, mock_relist_order):
        """Test that check_timeouts handles normal timeouts correctly."""
        # Set up the test
        # Create a timed out order that is not a margin experiment
        order = Order(
            order_id="test_order_1",
            item="item1",
            order_type="buy",
            price=150,
            quantity=10
        )
        
        # Add the order to the order manager
        self.ppo_integration.order_manager.orders = {"test_order_1": order}
        self.ppo_integration.order_manager.active_orders = {"test_order_1": order}
        
        # Mock get_timed_out_orders to return our test order
        self.ppo_integration.order_manager.get_timed_out_orders = MagicMock(return_value=["test_order_1"])
        
        # Mock get_order_info to return order info
        self.ppo_integration.order_manager.get_order_info = MagicMock(return_value={
            'order_id': 'test_order_1',
            'item': 'item1',
            'type': 'buy',
            'price': 150,
            'quantity': 10,
            'is_experiment': False
        })
        
        # Call check_timeouts
        await self.ppo_integration.check_timeouts()
        
        # Verify cancel_order was called
        mock_cancel_order.assert_called_once_with("test_order_1")
        
        # Verify relist_order was not called
        mock_relist_order.assert_not_called()

    # End-to-End Tests

    @patch('ppo_websocket_integration.PPOWebSocketIntegration.place_order')
    @patch('ppo_websocket_integration.PPOWebSocketIntegration.submit_event')
    async def test_end_to_end_portfolio_building(self, mock_submit_event, mock_place_order):
        """Test the complete portfolio building process end-to-end."""
        # Set up the test
        mock_place_order.return_value = True
        mock_submit_event.return_value = True
        
        # Hook the agent
        with patch('asyncio.get_event_loop', return_value=MagicMock()):
            with patch('asyncio.run_coroutine_threadsafe'):
                await self.ppo_integration.hook_agent(self.mock_agent, "agent_0")
        
        # Place initial orders
        await self.ppo_integration.place_initial_orders(self.mock_agent)
        
        # Verify place_order was called 8 times
        self.assertEqual(mock_place_order.call_count, 8)
        
        # Verify the flags are set correctly
        self.assertTrue(self.ppo_integration.initial_orders_placed)
        self.assertTrue(self.ppo_integration.building_portfolio)
        
        # Simulate order fulfillment for all 8 orders
        for i in range(8):
            order_id = f"initial_{int(time.time())}_{i}"
            
            # Create a message simulating order fulfillment
            message = {
                "type": "order_filled",
                "data": {
                    "order_id": order_id,
                    "item": f"item{i%5 + 1}",
                    "price": 150,
                    "quantity": 10,
                    "fulfillment_price": 150,
                    "fulfillment_quantity": 10
                }
            }
            
            # Call the handler directly
            await self.ppo_integration._handle_order_filled(message)
        
        # After all orders are fulfilled, building_portfolio should be False
        self.assertFalse(self.ppo_integration.building_portfolio)

    @patch('ppo_websocket_integration.PPOWebSocketIntegration.place_order')
    async def test_end_to_end_action_reception(self, mock_place_order):
        """Test that actions are only received at appropriate times."""
        # Set up the test
        mock_place_order.return_value = True
        
        # Hook the agent
        with patch('asyncio.get_event_loop', return_value=MagicMock()):
            with patch('asyncio.run_coroutine_threadsafe'):
                await self.ppo_integration.hook_agent(self.mock_agent, "agent_0")
        
        # During portfolio building, only buy actions should be placed
        self.ppo_integration.building_portfolio = True
        
        # Mock sample_action to return a sell action
        self.mock_agent.sample_action.return_value = (
            {
                'type': 'sell',
                'item': 'item1',
                'price': 180,
                'quantity': 10
            },
            None
        )
        
        # Place initial orders
        await self.ppo_integration.place_initial_orders(self.mock_agent)
        
        # Verify place_order was called 8 times
        self.assertEqual(mock_place_order.call_count, 8)
        
        # Check that all orders were buy orders
        for call_args in mock_place_order.call_args_list:
            args, kwargs = call_args
            self.assertEqual(args[0]['type'], 'buy')

    @patch('ppo_websocket_integration.PPOWebSocketIntegration.place_order')
    async def test_end_to_end_profitable_trades(self, mock_place_order):
        """Test that only profitable trades are executed."""
        # Set up the test
        mock_place_order.return_value = True
        
        # Hook the agent
        with patch('asyncio.get_event_loop', return_value=MagicMock()):
            with patch('asyncio.run_coroutine_threadsafe'):
                await self.ppo_integration.hook_agent(self.mock_agent, "agent_0")
        
        # After portfolio building, only profitable trades should be executed
        self.ppo_integration.building_portfolio = False
        
        # Mock sample_action to return a potentially unprofitable action
        self.mock_agent.sample_action.return_value = (
            {
                'type': 'buy',
                'item': 'item1',
                'price': 179,  # Just 1 GP below market price
                'quantity': 10
            },
            None
        )
        
        # Mock _validate_trade to check profitability
        original_validate_trade = self.ppo_integration._validate_trade
        validate_trade_calls = []
        
        def mock_validate_trade(action, obs):
            validate_trade_calls.append((action, obs))
            return original_validate_trade(action, obs)
        
        self.ppo_integration._validate_trade = mock_validate_trade
        
        # Simulate freed space
        self.ppo_integration.order_manager.active_orders = {
            f"order_{i}": MagicMock() for i in range(7)
        }
        
        # Mock agent hooks
        self.ppo_integration.agent_hooks = {
            'agent_0': {
                'agent': self.mock_agent,
                'original_sample_action': MagicMock()
            }
        }
        
        # Mock model output
        mock_model = MagicMock()
        mock_model.return_value = (None, torch.tensor([0.1, 0.9]), torch.tensor([0.2, 0.8]), torch.tensor([0.3, 0.7]), None, None)
        self.mock_agent.model = mock_model
        self.mock_agent.obs_to_tensor = MagicMock(return_value=torch.tensor([1.0]))
        
        # Call handle_freed_space
        with patch('torch.no_grad'):
            await self.ppo_integration.handle_freed_space()
        
        # Verify _validate_trade was called
        self.assertTrue(len(validate_trade_calls) > 0)
        
        # If place_order was called, verify the action was adjusted for profitability
        if mock_place_order.call_count > 0:
            args, kwargs = mock_place_order.call_args
            action = args[0]
            # The price should be adjusted to ensure MIN_PROFIT_THRESHOLD
            self.assertLess(action['price'], 179 - 51)  # 51 is MIN_PROFIT_THRESHOLD


# Run the tests
if __name__ == '__main__':
    # Use asyncio to run async tests
    import asyncio
    
    def run_async_test(test_case):
        """Helper function to run async test methods."""
        if asyncio.iscoroutinefunction(test_case):
            return asyncio.run(test_case())
        return test_case()
    
    # Patch unittest.TestCase.run to handle async methods
    original_run = unittest.TestCase.run
    
    def patched_run(self, result=None):
        for name in dir(self):
            if name.startswith('test_') and hasattr(self, name):
                test_method = getattr(self, name)
                if asyncio.iscoroutinefunction(test_method):
                    setattr(self, name, lambda test_method=test_method: run_async_test(test_method))
        return original_run(self, result)
    
    unittest.TestCase.run = patched_run
    
    # Run the tests
    unittest.main()