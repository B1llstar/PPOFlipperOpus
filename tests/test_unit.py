"""
Unit Tests for Order ID-based Management System
-----------------------------------------------
This file contains unit tests for the client-side components of the
Order ID-based management system for RuneScape Grand Exchange trading.
"""

import unittest
import asyncio
import time
from unittest.mock import MagicMock, patch

# Import client components
from ppo_websocket_integration import (
    PPOWebSocketIntegration, 
    Order, 
    InventoryManager, 
    OrderManager
)

# Test configuration
TEST_CONFIG = {
    "websocket_url": "http://localhost:6969",
    "max_slots": 8,
    "initial_gp": 10000000,  # 10M GP
    "test_items": {
        "Abyssal whip": {"buy_price": 1500000, "sell_price": 1550000},
        "Dragon bones": {"buy_price": 2500, "sell_price": 2600},
        "Nature rune": {"buy_price": 250, "sell_price": 270}
    }
}


class TestOrderIDGeneration(unittest.TestCase):
    """Test suite for Order ID generation functionality"""
    
    def setUp(self):
        self.order_manager = OrderManager(max_slots=TEST_CONFIG["max_slots"])
        
    def test_order_id_uniqueness(self):
        """Test that generated order IDs are unique"""
        order_ids = set()
        for _ in range(100):
            order_id = self.order_manager.generate_order_id()
            self.assertNotIn(order_id, order_ids, "Order ID should be unique")
            order_ids.add(order_id)
            
    def test_order_id_format(self):
        """Test that order IDs follow the expected format (prefix + timestamp + counter)"""
        order_id = self.order_manager.generate_order_id("BUY_")
        parts = order_id.split('_')
        
        self.assertEqual(parts[0], "BUY", "Prefix should match")
        self.assertTrue(parts[1].isdigit(), "Second part should be timestamp (digits)")
        self.assertTrue(parts[2].isdigit(), "Third part should be counter (digits)")
        
    def test_prefixed_order_ids(self):
        """Test that order IDs can have different prefixes for different order types"""
        prefixes = ["BUY_", "SELL_", "CANCEL_", "RELIST_", "HOLD_"]
        for prefix in prefixes:
            order_id = self.order_manager.generate_order_id(prefix)
            self.assertTrue(order_id.startswith(prefix), f"Order ID should start with {prefix}")


class TestInventoryManagement(unittest.TestCase):
    """Test suite for inventory management functionality"""
    
    def setUp(self):
        self.inventory_manager = InventoryManager()
        self.inventory_manager.gp = TEST_CONFIG["initial_gp"]
        self.inventory_manager.inventory = {
            "Abyssal whip": 2,
            "Dragon bones": 500
        }
        
    def test_can_sell_validation(self):
        """Test inventory validation for sell orders"""
        # Can sell items we have
        self.assertTrue(self.inventory_manager.can_sell("Abyssal whip", 1))
        self.assertTrue(self.inventory_manager.can_sell("Abyssal whip", 2))
        
        # Cannot sell more than we have
        self.assertFalse(self.inventory_manager.can_sell("Abyssal whip", 3))
        
        # Cannot sell items we don't have
        self.assertFalse(self.inventory_manager.can_sell("Nature rune", 1))
        
    def test_can_buy_validation(self):
        """Test GP validation for buy orders"""
        # Can buy items we can afford
        self.assertTrue(self.inventory_manager.can_buy("Abyssal whip", 1500000, 6))
        
        # Cannot buy more than we can afford
        self.assertFalse(self.inventory_manager.can_buy("Abyssal whip", 1500000, 7))
        
    def test_process_buy_fulfillment(self):
        """Test processing buy order fulfillment"""
        # Process buying 1 whip
        self.inventory_manager.process_buy_fulfillment("Abyssal whip", 1500000, 1)
        
        # Check GP decreased
        self.assertEqual(self.inventory_manager.gp, TEST_CONFIG["initial_gp"] - 1500000)
        
        # Check inventory increased
        self.assertEqual(self.inventory_manager.inventory["Abyssal whip"], 3)
        
    def test_process_sell_fulfillment(self):
        """Test processing sell order fulfillment"""
        # Process selling 1 whip
        self.inventory_manager.process_sell_fulfillment("Abyssal whip", 1550000, 1)
        
        # Check GP increased
        self.assertEqual(self.inventory_manager.gp, TEST_CONFIG["initial_gp"] + 1550000)
        
        # Check inventory decreased
        self.assertEqual(self.inventory_manager.inventory["Abyssal whip"], 1)


class TestOrderManager(unittest.TestCase):
    """Test suite for order management functionality"""
    
    def setUp(self):
        self.order_manager = OrderManager(max_slots=TEST_CONFIG["max_slots"])
        
    def test_active_orders_tracking(self):
        """Test tracking of active orders"""
        # Create an order
        order_id = self.order_manager.generate_order_id("BUY_")
        order = Order(order_id, "Abyssal whip", "buy", 1500000, 1)
        
        # Add order to tracking
        self.order_manager.orders[order_id] = order
        self.order_manager.active_orders[order_id] = order
        self.order_manager.item_to_orders.setdefault("Abyssal whip", []).append(order_id)
        self.order_manager.order_history[order_id] = order
        
        # Verify tracking
        self.assertEqual(len(self.order_manager.active_orders), 1)
        self.assertEqual(len(self.order_manager.orders), 1)
        self.assertEqual(len(self.order_manager.item_to_orders["Abyssal whip"]), 1)
        self.assertEqual(len(self.order_manager.order_history), 1)
        
        # Remove from active orders
        del self.order_manager.active_orders[order_id]
        
        # Verify tracking updated
        self.assertEqual(len(self.order_manager.active_orders), 0)
        self.assertEqual(len(self.order_manager.orders), 1)
        self.assertEqual(len(self.order_manager.order_history), 1)


class TestPPOWebSocketIntegration(unittest.TestCase):
    """Test suite for PPO WebSocket integration functionality"""
    
    def setUp(self):
        self.integration = PPOWebSocketIntegration(
            websocket_url=TEST_CONFIG["websocket_url"],
            max_slots=TEST_CONFIG["max_slots"]
        )
        
    @patch('aiohttp.ClientSession')
    def test_connect(self, mock_session):
        """Test connection to WebSocket server"""
        mock_session.return_value = MagicMock()
        
        # Run the connect method
        result = asyncio.run(self.integration.connect())
        
        # Verify connection successful
        self.assertTrue(result)
        self.assertTrue(self.integration.connected)
        
    @patch('aiohttp.ClientSession.post')
    def test_submit_event(self, mock_post):
        """Test submitting events to the server"""
        # Mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json.return_value = {"event_id": "test_event_1"}
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Set up integration
        self.integration.session = MagicMock()
        self.integration.connected = True
        
        # Submit event
        result = asyncio.run(self.integration.submit_event("test_event", {"test": "data"}))
        
        # Verify event submitted
        self.assertTrue(result)
        mock_post.assert_called_once()


if __name__ == "__main__":
    unittest.main()