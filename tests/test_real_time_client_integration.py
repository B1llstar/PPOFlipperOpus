import asyncio
import logging
import unittest
import time
import json
from unittest.mock import MagicMock, patch, AsyncMock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_real_time_client_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_real_time_client_integration")

# Import required modules
from real_time_ge_client import RealTimeGrandExchangeClient
from ppo_websocket_integration import PPOWebSocketIntegration, EventTypes

class TestRealTimeClientIntegration(unittest.TestCase):
    """Test cases for the Real-Time Client integration with WebSocket."""
    
    async def async_setup(self):
        """Set up the test environment asynchronously."""
        # Create a mock WebSocket integration with AsyncMock for async methods
        self.websocket_integration = MagicMock(spec=PPOWebSocketIntegration)
        self.websocket_integration.connect = AsyncMock(return_value=True)
        self.websocket_integration.disconnect = AsyncMock(return_value=None)
        self.websocket_integration.system_status = AsyncMock(return_value=True)
        self.websocket_integration.submit_event = AsyncMock(return_value=True)
        self.websocket_integration.price_update = AsyncMock(return_value=True)
        self.websocket_integration.margin_update = AsyncMock(return_value=True)
        self.websocket_integration.error_event = AsyncMock(return_value=True)
        self.websocket_integration.trade_executed = AsyncMock(return_value=True)
        self.websocket_integration.agent_decision = AsyncMock(return_value=True)
        
        # Create test market data
        self.latest_data = {
            "554": {"high": 5, "low": 4, "highTime": int(time.time()), "lowTime": int(time.time())},
            "555": {"high": 5, "low": 4, "highTime": int(time.time()), "lowTime": int(time.time())},
            "556": {"high": 5, "low": 4, "highTime": int(time.time()), "lowTime": int(time.time())}
        }
        
        self.five_min_data = {
            "554": {"avgHighPrice": 4, "avgLowPrice": 3, "highPriceVolume": 100, "lowPriceVolume": 50},
            "555": {"avgHighPrice": 4, "avgLowPrice": 3, "highPriceVolume": 100, "lowPriceVolume": 50},
            "556": {"avgHighPrice": 4, "avgLowPrice": 3, "highPriceVolume": 100, "lowPriceVolume": 50}
        }
        
        # Create a mock for the GrandExchangeClient methods
        self.get_latest_patch = patch.object(RealTimeGrandExchangeClient, 'get_latest', return_value=self.latest_data)
        self.get_5m_patch = patch.object(RealTimeGrandExchangeClient, 'get_5m', return_value=self.five_min_data)
        
        # Start the patches
        self.get_latest_patch.start()
        self.get_5m_patch.start()
        
        # Create a real-time client with the mock WebSocket integration
        self.client = RealTimeGrandExchangeClient(
            websocket_integration=self.websocket_integration,
            update_interval=1  # 1 second for faster testing
        )
        
        # Add name mapping methods to the client
        self.client._id_to_name_map = {"554": "Fire rune", "555": "Water rune", "556": "Air rune"}
        self.client._name_to_id_map = {"Fire rune": "554", "Water rune": "555", "Air rune": "556"}
        self.client.get_name_for_id = lambda item_id: self.client._id_to_name_map.get(str(item_id))
        self.client.get_id_for_name = lambda item_name: self.client._name_to_id_map.get(item_name)
    
    async def async_teardown(self):
        """Tear down the test environment asynchronously."""
        # Stop the patches
        self.get_latest_patch.stop()
        self.get_5m_patch.stop()
        
        # Stop the client if it's running
        if hasattr(self, 'client') and self.client._running:
            await self.client.stop()
    
    async def test_client_start_stop(self):
        """Test starting and stopping the real-time client."""
        logger.info("Testing client start/stop")
        
        # Set up the test environment
        await self.async_setup()
        
        # Start the client
        await self.client.start()
        
        # Verify that the client is running
        self.assertTrue(self.client._running)
        
        # Verify that the WebSocket integration was used to send a system status event
        self.websocket_integration.system_status.assert_called_with(
            "client_started", 
            "Real-time Grand Exchange client started"
        )
        
        # Stop the client
        await self.client.stop()
        
        # Verify that the client is not running
        self.assertFalse(self.client._running)
        
        # Verify that the WebSocket integration was used to send a system status event
        self.websocket_integration.system_status.assert_called_with(
            "client_stopped", 
            "Real-time Grand Exchange client stopped"
        )
        
        # Tear down the test environment
        await self.async_teardown()
    
    async def test_market_data_update(self):
        """Test market data updates."""
        logger.info("Testing market data updates")
        
        # Set up the test environment
        await self.async_setup()
        
        # Start the client
        await self.client.start()
        
        # Force a market data update
        result = await self.client.update_market_data(force=True)
        
        # Verify that the update was successful
        self.assertTrue(result)
        
        # Verify that the market data was updated
        self.assertIn("latest", self.client.market_data)
        self.assertIn("5m", self.client.market_data)
        self.assertIn("timestamp", self.client.market_data)
        
        # Verify that the WebSocket integration was used to send price updates
        self.websocket_integration.price_update.assert_called()
        
        # Verify that the WebSocket integration was used to send margin updates
        self.websocket_integration.margin_update.assert_called()
        
        # Stop the client
        await self.client.stop()
        
        # Tear down the test environment
        await self.async_teardown()
    
    async def test_place_order(self):
        """Test placing an order."""
        logger.info("Testing place order")
        
        # Set up the test environment
        await self.async_setup()
        
        # Start the client
        await self.client.start()
        
        # Place an order
        order_id = await self.client.place_order(
            order_type="buy",
            item="Fire rune",
            price=5,
            quantity=1000
        )
        
        # Verify that an order ID was returned
        self.assertIsNotNone(order_id)
        
        # Verify that the order was added to pending orders
        self.assertEqual(len(self.client.pending_orders), 1)
        self.assertEqual(self.client.pending_orders[0]["type"], "buy")
        self.assertEqual(self.client.pending_orders[0]["item"], "Fire rune")
        self.assertEqual(self.client.pending_orders[0]["price"], 5)
        self.assertEqual(self.client.pending_orders[0]["quantity"], 1000)
        
        # Verify that the WebSocket integration was used to submit an event
        self.websocket_integration.submit_event.assert_called_with("place_order", self.client.pending_orders[0])
        
        # Verify that the WebSocket integration was used to send an agent decision event
        self.websocket_integration.agent_decision.assert_called_with(
            agent_id="real_time_agent",
            action_type="buy",
            item="Fire rune",
            price=5,
            quantity=1000,
            reasoning="Real-time trading decision"
        )
        
        # Stop the client
        await self.client.stop()
        
        # Tear down the test environment
        await self.async_teardown()
    
    async def test_cancel_order(self):
        """Test cancelling an order."""
        logger.info("Testing cancel order")
        
        # Set up the test environment
        await self.async_setup()
        
        # Start the client
        await self.client.start()
        
        # Place an order
        order_id = await self.client.place_order(
            order_type="buy",
            item="Fire rune",
            price=5,
            quantity=1000
        )
        
        # Cancel the order
        result = await self.client.cancel_order(order_id)
        
        # Verify that the cancellation was successful
        self.assertTrue(result)
        
        # Verify that the order was removed from pending orders
        self.assertEqual(len(self.client.pending_orders), 0)
        
        # Verify that the order was added to order history
        self.assertEqual(len(self.client.order_history), 1)
        self.assertEqual(self.client.order_history[0]["id"], order_id)
        self.assertEqual(self.client.order_history[0]["status"], "cancelled")
        
        # Verify that the WebSocket integration was used to submit an event
        self.websocket_integration.submit_event.assert_called_with(
            "order_cancelled",
            self.client.order_history[0]
        )
        
        # Stop the client
        await self.client.stop()
        
        # Tear down the test environment
        await self.async_teardown()
    
    async def test_process_pending_orders(self):
        """Test processing pending orders."""
        logger.info("Testing process pending orders")
        
        # Set up the test environment
        await self.async_setup()
        
        # Start the client
        await self.client.start()
        
        # Place an order
        order_id = await self.client.place_order(
            order_type="buy",
            item="Fire rune",
            price=5,
            quantity=1000
        )
        
        # Process pending orders
        await self.client.process_pending_orders()
        
        # Verify that the order was moved from pending to active
        self.assertEqual(len(self.client.pending_orders), 0)
        self.assertEqual(len(self.client.active_orders), 1)
        self.assertEqual(self.client.active_orders[0]["id"], order_id)
        self.assertEqual(self.client.active_orders[0]["status"], "active")
        
        # Verify that the WebSocket integration was used to submit an event
        self.websocket_integration.submit_event.assert_called_with(
            "order_activated",
            self.client.active_orders[0]
        )
        
        # Stop the client
        await self.client.stop()
        
        # Tear down the test environment
        await self.async_teardown()
    
    async def test_update_active_orders(self):
        """Test updating active orders."""
        logger.info("Testing update active orders")
        
        # Set up the test environment
        await self.async_setup()
        
        # Start the client
        await self.client.start()
        
        # Place an order
        order_id = await self.client.place_order(
            order_type="buy",
            item="Fire rune",
            price=5,
            quantity=1000
        )
        
        # Process pending orders
        await self.client.process_pending_orders()
        
        # Update market data
        await self.client.update_market_data(force=True)
        
        # Update active orders
        await self.client.update_active_orders()
        
        # Verify that the order was filled (since market low price <= order price)
        self.assertEqual(len(self.client.active_orders), 0)
        self.assertEqual(len(self.client.order_history), 1)
        self.assertEqual(self.client.order_history[0]["id"], order_id)
        self.assertEqual(self.client.order_history[0]["status"], "filled")
        
        # Verify that the WebSocket integration was used to send a trade executed event
        self.websocket_integration.trade_executed.assert_called_with(
            agent_id="real_time_agent",
            action_type="buy",
            item="Fire rune",
            price=4,  # Market low price
            quantity=1000,
            profit=0.0
        )
        
        # Stop the client
        await self.client.stop()
        
        # Tear down the test environment
        await self.async_teardown()
    
    async def test_update_order(self):
        """Test updating an order."""
        logger.info("Testing update order")
        
        # Set up the test environment
        await self.async_setup()
        
        # Start the client
        await self.client.start()
        
        # Place an order
        order_id = await self.client.place_order(
            order_type="buy",
            item="Fire rune",
            price=5,
            quantity=1000
        )
        
        # Update the order
        result = await self.client.update_order(order_id, {"price": 6})
        
        # Verify that the update was successful
        self.assertTrue(result)
        
        # Verify that the order was updated
        self.assertEqual(self.client.pending_orders[0]["price"], 6)
        
        # Verify that the WebSocket integration was used to submit an event
        self.websocket_integration.submit_event.assert_called_with(
            "order_updated",
            self.client.pending_orders[0]
        )
        
        # Stop the client
        await self.client.stop()
        
        # Tear down the test environment
        await self.async_teardown()
    
    async def test_get_order(self):
        """Test getting an order."""
        logger.info("Testing get order")
        
        # Set up the test environment
        await self.async_setup()
        
        # Start the client
        await self.client.start()
        
        # Place an order
        order_id = await self.client.place_order(
            order_type="buy",
            item="Fire rune",
            price=5,
            quantity=1000
        )
        
        # Get the order
        order = self.client.get_order(order_id)
        
        # Verify that the order was returned
        self.assertIsNotNone(order)
        self.assertEqual(order["id"], order_id)
        self.assertEqual(order["type"], "buy")
        self.assertEqual(order["item"], "Fire rune")
        self.assertEqual(order["price"], 5)
        self.assertEqual(order["quantity"], 1000)
        
        # Stop the client
        await self.client.stop()
        
        # Tear down the test environment
        await self.async_teardown()
    
    async def test_get_orders_by_status(self):
        """Test getting orders by status."""
        logger.info("Testing get orders by status")
        
        # Set up the test environment
        await self.async_setup()
        
        # Start the client
        await self.client.start()
        
        # Place an order
        order_id = await self.client.place_order(
            order_type="buy",
            item="Fire rune",
            price=5,
            quantity=1000
        )
        
        # Get pending orders
        pending_orders = self.client.get_orders_by_status("pending")
        
        # Verify that the order was returned
        self.assertEqual(len(pending_orders), 1)
        self.assertEqual(pending_orders[0]["id"], order_id)
        
        # Process pending orders
        await self.client.process_pending_orders()
        
        # Get active orders
        active_orders = self.client.get_orders_by_status("active")
        
        # Verify that the order was returned
        self.assertEqual(len(active_orders), 1)
        self.assertEqual(active_orders[0]["id"], order_id)
        
        # Cancel the order
        await self.client.cancel_order(order_id)
        
        # Get cancelled orders
        cancelled_orders = self.client.get_orders_by_status("cancelled")
        
        # Verify that the order was returned
        self.assertEqual(len(cancelled_orders), 1)
        self.assertEqual(cancelled_orders[0]["id"], order_id)
        
        # Stop the client
        await self.client.stop()
        
        # Tear down the test environment
        await self.async_teardown()

async def main():
    """Run the tests asynchronously."""
    test_case = TestRealTimeClientIntegration()
    
    # Run the tests
    await test_case.test_client_start_stop()
    await test_case.test_market_data_update()
    await test_case.test_place_order()
    await test_case.test_cancel_order()
    await test_case.test_process_pending_orders()
    await test_case.test_update_active_orders()
    await test_case.test_update_order()
    await test_case.test_get_order()
    await test_case.test_get_orders_by_status()

if __name__ == "__main__":
    # Run the tests
    asyncio.run(main())