import asyncio
import unittest
from unittest.mock import MagicMock, patch
import logging
from ppo_websocket_integration import PPOWebSocketIntegration, EventTypes, Order

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_marketplace_operations.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_marketplace_operations")

class TestMarketplaceOperations(unittest.TestCase):
    """Test suite for marketplace operations"""

    def setUp(self):
        """Set up test environment before each test"""
        # Create a PPO WebSocket Integration instance
        self.integration = PPOWebSocketIntegration(
            websocket_url="http://localhost:6969",
            max_slots=8
        )
        
        # Initialize test item data
        self.test_items = {
            "Feather": {
                "base_price": 5,
                "min_price": 2,
                "max_price": 10,
                "buy_limit": 10000
            }
        }
        
        # Update the margin env with test items
        self.integration.margin_env.items = self.test_items
        
        # Initialize inventory and GP
        self.integration.margin_env.inventory = {"Feather": 0}
        self.integration.margin_env.gp = 1000
        self.integration.margin_env.prices = {"Feather": 5}
        self.integration.margin_env.buy_limits = {"Feather": 0}

        # Initialize order tracking
        self.received_notifications = []

    async def test_marketplace_sequence(self):
        """Test a sequence of marketplace operations"""
        # Mock the session and response
        mock_response = MagicMock()
        mock_response.status = 200
        
        # Track the current operation for response mocking
        self.current_operation = None
        
        async def mock_json():
            if self.current_operation == "place_order":
                return {
                    "success": True,
                    "order_id": "test_order_1",
                    "status": "active",
                    "slot": 1
                }
            elif self.current_operation == "cancel_order":
                return {
                    "success": True,
                    "order_id": "test_order_1",
                    "status": "cancelled"
                }
            elif self.current_operation == "relist_order":
                return {
                    "success": True,
                    "order_id": "test_order_2",
                    "status": "active",
                    "slot": 1
                }
            return {"success": True}
            
        mock_response.json = mock_json
        
        # Create a context manager for post
        class MockContextManager:
            async def __aenter__(self):
                return mock_response
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
        
        # Create a mock session
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=MockContextManager())
        
        # Replace the session
        self.integration.session = mock_session
        self.integration.connected = True

        # 1. Place buy order for 1 Feather at 5 GP
        self.current_operation = "place_order"
        buy_order = {
            'type': 'buy',
            'item': 'Feather',
            'price': 5,
            'quantity': 1
        }
        
        result = await self.integration.place_order(buy_order)
        self.assertTrue(result, "Buy order should be placed successfully")
        
        # Verify order placement notification
        mock_session.post.assert_called_with(
            f"{self.integration.websocket_url}/submit_event",
            json=unittest.mock.ANY
        )
        
        # Simulate order fulfillment
        self.integration.margin_env.open_orders.append(buy_order)
        filled_orders, _ = self.integration._fulfill_orders()
        self.assertEqual(len(filled_orders), 1, "Buy order should be filled")
        self.assertEqual(self.integration.margin_env.inventory["Feather"], 1, "Should have 1 Feather in inventory")

        # 2. Place sell order for 1 Feather at 5 GP
        sell_order = {
            'type': 'sell',
            'item': 'Feather',
            'price': 5,
            'quantity': 1
        }
        
        result = await self.integration.place_order(sell_order)
        self.assertTrue(result, "Sell order should be placed successfully")
        
        # Verify sell order notification
        mock_session.post.assert_called_with(
            f"{self.integration.websocket_url}/submit_event",
            json=unittest.mock.ANY
        )

        # 3. Relist the sell order at 4 GP
        self.current_operation = "relist_order"
        order_id = "test_order_1"  # Use the order ID from our mock response
        result = await self.integration.relist_order(order_id, 4)
        self.assertTrue(result, "Order should be relisted successfully")
        
        # Verify relist notifications (cancel + new order)
        self.assertEqual(
            mock_session.post.call_count >= 2,
            True,
            "Should receive notifications for cancel and new order"
        )

        # 4. Cancel the relisted order
        self.current_operation = "cancel_order"
        order_id = "test_order_2"  # Use the relisted order ID
        result = await self.integration.cancel_order(order_id)
        self.assertTrue(result, "Order should be cancelled successfully")
        
        # Verify cancel notification
        mock_session.post.assert_called_with(
            f"{self.integration.websocket_url}/submit_event",
            json=unittest.mock.ANY
        )
        
        # Verify order status is updated
        self.assertEqual(
            self.integration.order_manager.orders[order_id].status if order_id in self.integration.order_manager.orders else None,
            "canceled",
            "Order status should be updated to canceled"
        )

        # 5. Place final sell order at 2 GP
        self.current_operation = "place_order"
        final_sell_order = {
            'type': 'sell',
            'item': 'Feather',
            'price': 2,
            'quantity': 1
        }
        
        result = await self.integration.place_order(final_sell_order)
        self.assertTrue(result, "Final sell order should be placed successfully")
        
        # Verify final order notification
        mock_session.post.assert_called_with(
            f"{self.integration.websocket_url}/submit_event",
            json=unittest.mock.ANY
        )

def run_tests():
    """Run the test suite"""
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == "__main__":
    # Run the asynchronous tests
    async def run_async_tests():
        test = TestMarketplaceOperations()
        test.setUp()
        await test.test_marketplace_sequence()
        print("Marketplace operation tests completed")
    
    asyncio.run(run_async_tests())