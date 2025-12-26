import asyncio
import logging
import time
import json
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_real_time_ge_client.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_real_time_ge_client")

# Import the RealTimeGrandExchangeClient
from real_time_ge_client import RealTimeGrandExchangeClient
from ppo_websocket_integration import PPOWebSocketIntegration

async def test_real_time_client():
    """Test the RealTimeGrandExchangeClient with WebSocket integration."""
    logger.info("Starting real-time client test")
    
    # Create WebSocket integration
    integration = PPOWebSocketIntegration()
    
    # Connect to WebSocket server
    if not await integration.connect():
        logger.error("Failed to connect to WebSocket server")
        return
    
    logger.info("Connected to WebSocket server")
    
    # Create real-time client with a shorter update interval for testing (30 seconds)
    client = RealTimeGrandExchangeClient(
        websocket_integration=integration,
        update_interval=30  # 30 seconds for testing
    )
    
    # Start the client
    await client.start()
    logger.info("Real-time client started")
    
    # Wait for initial market data update
    logger.info("Waiting for initial market data update...")
    await asyncio.sleep(5)
    
    # Place a test buy order
    order_id = await client.place_order(
        order_type="buy",
        item="Abyssal whip",  # Example item
        price=1500000,  # Example price
        quantity=1
    )
    
    if order_id:
        logger.info(f"Placed test buy order: {order_id}")
        
        # Wait for order processing
        logger.info("Waiting for order processing...")
        await asyncio.sleep(5)
        
        # Get order status
        order = client.get_order(order_id)
        logger.info(f"Order status: {order['status']}")
        
        # Update the order
        await client.update_order(order_id, {"price": 1550000})
        logger.info("Updated order price")
        
        # Wait for order update
        await asyncio.sleep(5)
        
        # Get updated order
        updated_order = client.get_order(order_id)
        logger.info(f"Updated order: {updated_order}")
        
        # Cancel the order
        await client.cancel_order(order_id)
        logger.info("Cancelled order")
    else:
        logger.error("Failed to place test order")
    
    # Place a test sell order
    order_id = await client.place_order(
        order_type="sell",
        item="Dragon bones",  # Example item
        price=2500,  # Example price
        quantity=100
    )
    
    if order_id:
        logger.info(f"Placed test sell order: {order_id}")
    
    # Wait for a market data update cycle
    logger.info("Waiting for market data update cycle...")
    await asyncio.sleep(35)  # Wait slightly longer than the update interval
    
    # Get pending and active orders
    pending_orders = client.get_orders_by_status("pending")
    active_orders = client.get_orders_by_status("active")
    
    logger.info(f"Pending orders: {len(pending_orders)}")
    logger.info(f"Active orders: {len(active_orders)}")
    
    # Stop the client
    await client.stop()
    logger.info("Real-time client stopped")
    
    # Disconnect from WebSocket server
    await integration.disconnect()
    logger.info("Disconnected from WebSocket server")

async def test_without_websocket():
    """Test the RealTimeGrandExchangeClient without WebSocket integration."""
    logger.info("Starting real-time client test without WebSocket")
    
    # Create real-time client without WebSocket integration
    client = RealTimeGrandExchangeClient(update_interval=30)
    
    # Start the client
    await client.start()
    logger.info("Real-time client started")
    
    # Wait for initial market data update
    logger.info("Waiting for initial market data update...")
    await asyncio.sleep(5)
    
    # Try to place an order (should fail without WebSocket)
    order_id = await client.place_order(
        order_type="buy",
        item="Abyssal whip",
        price=1500000,
        quantity=1
    )
    
    if order_id:
        logger.info(f"Placed test order: {order_id}")
    else:
        logger.info("Order placement failed as expected (no WebSocket)")
    
    # Wait for a market data update cycle
    logger.info("Waiting for market data update cycle...")
    await asyncio.sleep(35)
    
    # Stop the client
    await client.stop()
    logger.info("Real-time client stopped")

async def main():
    """Run the tests."""
    # Test with WebSocket integration
    try:
        await test_real_time_client()
    except Exception as e:
        logger.error(f"Error in WebSocket test: {e}")
    
    # Test without WebSocket integration
    try:
        await test_without_websocket()
    except Exception as e:
        logger.error(f"Error in non-WebSocket test: {e}")

if __name__ == "__main__":
    asyncio.run(main())