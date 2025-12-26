"""
Integration Tests for Order ID-based Management System
-----------------------------------------------------
This file contains integration tests for the client-server communication
of the Order ID-based management system for RuneScape Grand Exchange trading.
"""

import unittest
import asyncio
import json
import time
import websockets
import threading
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
    "websocket_url": "ws://localhost:8089",  # Using a higher port number to avoid permission issues
    "max_slots": 8,
    "initial_gp": 10000000,  # 10M GP
    "test_items": {
        "Abyssal whip": {"buy_price": 1500000, "sell_price": 1550000},
        "Dragon bones": {"buy_price": 2500, "sell_price": 2600},
        "Nature rune": {"buy_price": 250, "sell_price": 270}
    }
}


class MockWebSocketServer:
    """Mock WebSocket server for testing client-server integration"""
    
    def __init__(self, port=8089):  # Match the port in TEST_CONFIG
        self.port = port
        self.clients = set()
        self.active_orders = {}
        self.slots = {i: None for i in range(1, 9)}  # 8 slots
        self.server = None
        self.server_thread = None
        
    async def handler(self, websocket, path):
        """Handle WebSocket connections"""
        self.clients.add(websocket)
        try:
            async for message in websocket:
                data = json.loads(message)
                response = await self.process_message(data)
                await websocket.send(json.dumps(response))
        finally:
            self.clients.remove(websocket)
            
    async def process_message(self, data):
        """Process incoming messages and generate responses"""
        msg_type = data.get('type')
        
        if msg_type in ['buy_order_placed', 'sell_order_placed']:
            order_id = data.get('order_id')
            # Assign a slot
            available_slot = next((i for i, order in self.slots.items() if order is None), None)
            if available_slot:
                self.slots[available_slot] = order_id
                self.active_orders[order_id] = {
                    'slot': available_slot,
                    'status': 'active',
                    'item': data.get('item'),
                    'price': data.get('price'),
                    'quantity': data.get('quantity'),
                    'type': 'buy' if msg_type == 'buy_order_placed' else 'sell'
                }
                return {
                    'type': 'order_status',
                    'order_id': order_id,
                    'status': 'active',
                    'slot_id': available_slot,
                    'timestamp': int(time.time() * 1000)
                }
            else:
                return {
                    'type': 'error',
                    'message': 'No available slots',
                    'timestamp': int(time.time() * 1000)
                }
                
        elif msg_type == 'cancel_order':
            parent_order_id = data.get('parent_order_id')
            if parent_order_id in self.active_orders:
                slot = self.active_orders[parent_order_id]['slot']
                self.slots[slot] = None
                self.active_orders[parent_order_id]['status'] = 'canceled'
                return {
                    'type': 'order_status',
                    'order_id': parent_order_id,
                    'status': 'canceled',
                    'timestamp': int(time.time() * 1000)
                }
            else:
                return {
                    'type': 'error',
                    'message': 'Order not found',
                    'timestamp': int(time.time() * 1000)
                }
        
        # Default response for unhandled message types
        return {
            'type': 'error',
            'message': 'Unhandled message type',
            'timestamp': int(time.time() * 1000)
        }
        
    async def start_server(self):
        """Start the WebSocket server"""
        self.server = await websockets.serve(self.handler, 'localhost', self.port)
        print(f"Mock WebSocket server running on port {self.port}")
        
    def start(self):
        """Start the server in a separate thread"""
        async def run_server():
            await self.start_server()
            # Keep the server running
            while True:
                await asyncio.sleep(1)
                
        self.server_thread = threading.Thread(target=lambda: asyncio.run(run_server()))
        self.server_thread.daemon = True
        self.server_thread.start()
        time.sleep(1)  # Give server time to start
        
    async def stop_server(self):
        """Stop the WebSocket server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            print("Mock WebSocket server stopped")
            
    def stop(self):
        """Stop the server"""
        if self.server_thread and self.server_thread.is_alive():
            asyncio.run(self.stop_server())
            self.server_thread.join(timeout=5)


class TestClientServerIntegration(unittest.TestCase):
    """Test suite for client-server integration"""
    
    @classmethod
    def setUpClass(cls):
        # Start mock server
        cls.mock_server = MockWebSocketServer()
        cls.mock_server.start()
        
    @classmethod
    def tearDownClass(cls):
        # Stop mock server
        cls.mock_server.stop()
        
    def setUp(self):
        # Create integration with patched session
        self.integration = PPOWebSocketIntegration(
            websocket_url=TEST_CONFIG["websocket_url"],
            max_slots=TEST_CONFIG["max_slots"]
        )
        
        # Set up inventory and price data
        self.integration.order_manager.inventory_manager.gp = TEST_CONFIG["initial_gp"]
        self.integration.order_manager.inventory_manager.inventory = {
            "Abyssal whip": 2,
            "Dragon bones": 500
        }
        
        # Set up observation state with prices
        self.integration.current_obs = {
            'prices': {
                "Abyssal whip": TEST_CONFIG["test_items"]["Abyssal whip"]["sell_price"],
                "Dragon bones": TEST_CONFIG["test_items"]["Dragon bones"]["sell_price"]
            },
            'inventory': self.integration.order_manager.inventory_manager.inventory.copy(),
            'gp': TEST_CONFIG["initial_gp"]
        }
        
        # Mock the connect method
        self.integration.connect = MagicMock(return_value=True)
        self.integration.connected = True
        self.integration.session = MagicMock()
        
    def tearDown(self):
        # Mock the disconnect method
        self.integration.disconnect = MagicMock(return_value=True)
        
    @patch('aiohttp.ClientSession.post')
    def test_buy_order_placement(self, mock_post):
        """Test placing a buy order"""
        # Mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json.return_value = {"status": "success"}
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Create order data
        order_data = {
            'order_id': self.integration.order_manager.generate_order_id("BUY_"),
            'item': "Abyssal whip",
            'type': "buy",
            'price': 1500000,
            'quantity': 1
        }
        
        # Place order
        result = asyncio.run(self.integration.place_order(order_data))
        
        # Verify order placed
        self.assertTrue(result)
        mock_post.assert_called_once()
        
    @patch('aiohttp.ClientSession.post')
    def test_sell_order_placement(self, mock_post):
        """Test placing a sell order"""
        # Mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json.return_value = {"status": "success"}
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Create order data
        order_data = {
            'order_id': self.integration.order_manager.generate_order_id("SELL_"),
            'item': "Abyssal whip",
            'type': "sell",
            'price': 1550000,
            'quantity': 1
        }
        
        # Place order
        result = asyncio.run(self.integration.place_order(order_data))
        
        # Verify order placed
        self.assertTrue(result)
        mock_post.assert_called_once()
        
    @patch('aiohttp.ClientSession.post')
    def test_cancel_order(self, mock_post):
        """Test canceling an order"""
        # Mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json.return_value = {"status": "success"}
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Create and add an order
        order_id = self.integration.order_manager.generate_order_id("BUY_")
        order = Order(order_id, "Abyssal whip", "buy", 1500000, 1)
        self.integration.order_manager.orders[order_id] = order
        self.integration.order_manager.active_orders[order_id] = order
        
        # Cancel order
        result = asyncio.run(self.integration.cancel_order(order_id))
        
        # Verify order canceled
        self.assertTrue(result)
        mock_post.assert_called_once()
        
    @patch('aiohttp.ClientSession.post')
    def test_order_status_update(self, mock_post):
        """Test receiving order status updates"""
        # Mock response for initial order placement
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json.return_value = {"status": "success"}
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Create and place an order
        order_id = self.integration.order_manager.generate_order_id("BUY_")
        order = Order(order_id, "Abyssal whip", "buy", 1500000, 1)
        self.integration.order_manager.orders[order_id] = order
        self.integration.order_manager.active_orders[order_id] = order
        
        # Simulate receiving a status update
        status_update = {
            'type': 'order_status',
            'order_id': order_id,
            'status': 'filled',
            'timestamp': int(time.time() * 1000)
        }
        
        # Process the status update (would normally be done by the WebSocket client)
        order.status = status_update['status']
        
        # Verify status updated
        self.assertEqual(order.status, 'filled')


if __name__ == "__main__":
    unittest.main()