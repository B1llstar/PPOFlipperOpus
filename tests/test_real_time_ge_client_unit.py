import unittest
import asyncio
import time
import json
import logging
from unittest.mock import MagicMock, patch, AsyncMock

# Import the RealTimeGrandExchangeClient
from real_time_ge_client import RealTimeGrandExchangeClient

# Disable logging for tests
logging.disable(logging.CRITICAL)

class TestRealTimeGrandExchangeClient(unittest.TestCase):
    """Unit tests for the RealTimeGrandExchangeClient class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock WebSocket integration
        self.mock_websocket = MagicMock()
        self.mock_websocket.submit_event = AsyncMock()
        self.mock_websocket.system_status = AsyncMock()
        self.mock_websocket.price_update = AsyncMock()
        self.mock_websocket.margin_update = AsyncMock()
        self.mock_websocket.trade_executed = AsyncMock()
        self.mock_websocket.agent_decision = AsyncMock()
        self.mock_websocket.error_event = AsyncMock()
        
        # Create a client with the mock WebSocket integration
        self.client = RealTimeGrandExchangeClient(
            websocket_integration=self.mock_websocket,
            update_interval=1  # 1 second for faster tests
        )
        
        # Mock the get_latest and get_5m methods
        self.client.get_latest = MagicMock(return_value={
            "4151": {  # Abyssal whip
                "high": 1550000,
                "low": 1450000,
                "highTime": int(time.time()),
                "lowTime": int(time.time())
            },
            "536": {  # Dragon bones
                "high": 2600,
                "low": 2400,
                "highTime": int(time.time()),
                "lowTime": int(time.time())
            }
        })
        
        self.client.get_5m = MagicMock(return_value={
            "4151": {  # Abyssal whip
                "avgHighPrice": 1500000,
                "avgLowPrice": 1400000,
                "highPriceVolume": 100,
                "lowPriceVolume": 80
            },
            "536": {  # Dragon bones
                "avgHighPrice": 2500,
                "avgLowPrice": 2300,
                "highPriceVolume": 5000,
                "lowPriceVolume": 4500
            }
        })
        
        # Mock the get_name_for_id and get_id_for_name methods
        self.client.get_name_for_id = MagicMock(side_effect=lambda item_id: {
            "4151": "Abyssal whip",
            "536": "Dragon bones"
        }.get(item_id, f"Item {item_id}"))
        
        self.client.get_id_for_name = MagicMock(side_effect=lambda item_name: {
            "Abyssal whip": "4151",
            "Dragon bones": "536"
        }.get(item_name))
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Ensure the client is stopped
        if self.client._running:
            asyncio.run(self.client.stop())
    
    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.client.update_interval, 1)
        self.assertEqual(self.client.websocket_integration, self.mock_websocket)
        self.assertEqual(self.client.last_update_time, 0)
        self.assertEqual(self.client.market_data, {})
        self.assertEqual(self.client.pending_orders, [])
        self.assertEqual(self.client.active_orders, [])
        self.assertEqual(self.client.order_history, [])
        self.assertFalse(self.client._running)
        self.assertIsNone(self.client._update_task)
    
    async def async_test_start_stop(self):
        """Test start and stop methods."""
        # Start the client
        await self.client.start()
        
        # Check that the client is running
        self.assertTrue(self.client._running)
        self.assertIsNotNone(self.client._update_task)
        
        # Check that market data was updated
        self.assertNotEqual(self.client.market_data, {})
        self.assertNotEqual(self.client.last_update_time, 0)
        
        # Check that system_status was called
        self.mock_websocket.system_status.assert_called_once()
        
        # Stop the client
        await self.client.stop()
        
        # Check that the client is stopped
        self.assertFalse(self.client._running)
        self.assertIsNone(self.client._update_task)
        
        # Check that system_status was called again
        self.assertEqual(self.mock_websocket.system_status.call_count, 2)
    
    def test_start_stop(self):
        """Run the async test for start and stop."""
        asyncio.run(self.async_test_start_stop())
    
    async def async_test_update_market_data(self):
        """Test update_market_data method."""
        # Update market data
        result = await self.client.update_market_data(force=True)
        
        # Check that the method returned True
        self.assertTrue(result)
        
        # Check that market data was updated
        self.assertNotEqual(self.client.market_data, {})
        self.assertNotEqual(self.client.last_update_time, 0)
        
        # Check that get_latest and get_5m were called
        self.client.get_latest.assert_called_once()
        self.client.get_5m.assert_called_once()
        
        # Check that price_update and margin_update were called for each item
        self.assertEqual(self.mock_websocket.price_update.call_count, 2)
        self.assertEqual(self.mock_websocket.margin_update.call_count, 2)
    
    def test_update_market_data(self):
        """Run the async test for update_market_data."""
        asyncio.run(self.async_test_update_market_data())
    
    async def async_test_place_order(self):
        """Test place_order method."""
        # Place an order
        order_id = await self.client.place_order(
            order_type="buy",
            item="Abyssal whip",
            price=1500000,
            quantity=1
        )
        
        # Check that an order ID was returned
        self.assertIsNotNone(order_id)
        
        # Check that the order was added to pending orders
        self.assertEqual(len(self.client.pending_orders), 1)
        self.assertEqual(self.client.pending_orders[0]["type"], "buy")
        self.assertEqual(self.client.pending_orders[0]["item"], "Abyssal whip")
        self.assertEqual(self.client.pending_orders[0]["price"], 1500000)
        self.assertEqual(self.client.pending_orders[0]["quantity"], 1)
        self.assertEqual(self.client.pending_orders[0]["status"], "pending")
        
        # Check that submit_event and agent_decision were called
        self.mock_websocket.submit_event.assert_called_once()
        self.mock_websocket.agent_decision.assert_called_once()
    
    def test_place_order(self):
        """Run the async test for place_order."""
        asyncio.run(self.async_test_place_order())
    
    async def async_test_cancel_order(self):
        """Test cancel_order method."""
        # Place an order
        order_id = await self.client.place_order(
            order_type="buy",
            item="Abyssal whip",
            price=1500000,
            quantity=1
        )
        
        # Cancel the order
        result = await self.client.cancel_order(order_id)
        
        # Check that the method returned True
        self.assertTrue(result)
        
        # Check that the order was removed from pending orders
        self.assertEqual(len(self.client.pending_orders), 0)
        
        # Check that the order was added to order history
        self.assertEqual(len(self.client.order_history), 1)
        self.assertEqual(self.client.order_history[0]["id"], order_id)
        self.assertEqual(self.client.order_history[0]["status"], "cancelled")
        
        # Check that submit_event was called again
        self.assertEqual(self.mock_websocket.submit_event.call_count, 2)
    
    def test_cancel_order(self):
        """Run the async test for cancel_order."""
        asyncio.run(self.async_test_cancel_order())
    
    async def async_test_process_pending_orders(self):
        """Test process_pending_orders method."""
        # Place an order
        order_id = await self.client.place_order(
            order_type="buy",
            item="Abyssal whip",
            price=1500000,
            quantity=1
        )
        
        # Process pending orders
        await self.client.process_pending_orders()
        
        # Check that the order was moved from pending to active
        self.assertEqual(len(self.client.pending_orders), 0)
        self.assertEqual(len(self.client.active_orders), 1)
        self.assertEqual(self.client.active_orders[0]["id"], order_id)
        self.assertEqual(self.client.active_orders[0]["status"], "active")
        
        # Check that submit_event was called again
        self.assertEqual(self.mock_websocket.submit_event.call_count, 2)
    
    def test_process_pending_orders(self):
        """Run the async test for process_pending_orders."""
        asyncio.run(self.async_test_process_pending_orders())
    
    async def async_test_update_active_orders(self):
        """Test update_active_orders method."""
        # Place and process an order
        order_id = await self.client.place_order(
            order_type="buy",
            item="Abyssal whip",
            price=1500000,
            quantity=1
        )
        await self.client.process_pending_orders()
        
        # Update market data
        await self.client.update_market_data(force=True)
        
        # Update active orders
        await self.client.update_active_orders()
        
        # Check that the order was filled
        self.assertEqual(len(self.client.active_orders), 0)
        self.assertEqual(len(self.client.order_history), 1)
        self.assertEqual(self.client.order_history[0]["id"], order_id)
        self.assertEqual(self.client.order_history[0]["status"], "filled")
        
        # Check that trade_executed was called
        self.mock_websocket.trade_executed.assert_called_once()
    
    def test_update_active_orders(self):
        """Run the async test for update_active_orders."""
        asyncio.run(self.async_test_update_active_orders())
    
    async def async_test_update_order(self):
        """Test update_order method."""
        # Place an order
        order_id = await self.client.place_order(
            order_type="buy",
            item="Abyssal whip",
            price=1500000,
            quantity=1
        )
        
        # Update the order
        result = await self.client.update_order(order_id, {"price": 1550000})
        
        # Check that the method returned True
        self.assertTrue(result)
        
        # Check that the order was updated
        self.assertEqual(self.client.pending_orders[0]["price"], 1550000)
        
        # Check that submit_event was called again
        self.assertEqual(self.mock_websocket.submit_event.call_count, 2)
    
    def test_update_order(self):
        """Run the async test for update_order."""
        asyncio.run(self.async_test_update_order())
    
    def test_get_order(self):
        """Test get_order method."""
        # Place an order
        order_id = asyncio.run(self.client.place_order(
            order_type="buy",
            item="Abyssal whip",
            price=1500000,
            quantity=1
        ))
        
        # Get the order
        order = self.client.get_order(order_id)
        
        # Check that the order was returned
        self.assertIsNotNone(order)
        self.assertEqual(order["id"], order_id)
        self.assertEqual(order["type"], "buy")
        self.assertEqual(order["item"], "Abyssal whip")
        self.assertEqual(order["price"], 1500000)
        self.assertEqual(order["quantity"], 1)
        self.assertEqual(order["status"], "pending")
    
    def test_get_orders_by_status(self):
        """Test get_orders_by_status method."""
        # Place an order
        order_id = asyncio.run(self.client.place_order(
            order_type="buy",
            item="Abyssal whip",
            price=1500000,
            quantity=1
        ))
        
        # Get pending orders
        pending_orders = self.client.get_orders_by_status("pending")
        
        # Check that the order was returned
        self.assertEqual(len(pending_orders), 1)
        self.assertEqual(pending_orders[0]["id"], order_id)
        
        # Process the order
        asyncio.run(self.client.process_pending_orders())
        
        # Get active orders
        active_orders = self.client.get_orders_by_status("active")
        
        # Check that the order was returned
        self.assertEqual(len(active_orders), 1)
        self.assertEqual(active_orders[0]["id"], order_id)
        
        # Cancel the order
        asyncio.run(self.client.cancel_order(order_id))
        
        # Get cancelled orders
        cancelled_orders = self.client.get_orders_by_status("cancelled")
        
        # Check that the order was returned
        self.assertEqual(len(cancelled_orders), 1)
        self.assertEqual(cancelled_orders[0]["id"], order_id)

if __name__ == "__main__":
    unittest.main()