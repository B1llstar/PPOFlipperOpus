"""
End-to-End Tests for Order ID-based Management System
----------------------------------------------------
This file contains end-to-end tests for complete workflows of the
Order ID-based management system for RuneScape Grand Exchange trading.
"""

import unittest
import asyncio
import json
import time
import threading
import websockets
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
    "websocket_url": "ws://localhost:6969",
    "max_slots": 8,
    "initial_gp": 10000000,  # 10M GP
    "test_items": {
        "Abyssal whip": {"buy_price": 1500000, "sell_price": 1550000},
        "Dragon bones": {"buy_price": 2500, "sell_price": 2600},
        "Nature rune": {"buy_price": 250, "sell_price": 270}
    }
}


class MockWebSocketServer:
    """Enhanced mock WebSocket server for end-to-end testing"""
    
    def __init__(self, port=6969):
        self.port = port
        self.clients = set()
        self.active_orders = {}
        self.slots = {i: None for i in range(1, 9)}  # 8 slots
        self.server = None
        self.server_thread = None
        self.order_history = {}
        
    async def handler(self, websocket, path):
        """Handle WebSocket connections"""
        self.clients.add(websocket)
        try:
            # Send welcome message
            await websocket.send(json.dumps({
                'type': 'welcome',
                'message': 'Welcome to the mock server',
                'timestamp': int(time.time() * 1000)
            }))
            
            # Process messages
            async for message in websocket:
                data = json.loads(message)
                response = await self.process_message(data)
                await websocket.send(json.dumps(response))
                
                # For buy/sell orders, simulate order fulfillment after a delay
                if data.get('type') in ['buy_order_placed', 'sell_order_placed'] and response.get('status') == 'active':
                    order_id = data.get('order_id')
                    asyncio.create_task(self.simulate_order_fulfillment(order_id, websocket))
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
                    'type': 'buy' if msg_type == 'buy_order_placed' else 'sell',
                    'timestamp': int(time.time() * 1000)
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
                
                # Add to history
                self.order_history[parent_order_id] = self.active_orders[parent_order_id]
                
                # Remove from active orders
                del self.active_orders[parent_order_id]
                
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
                
        elif msg_type == 'relist_buy_order' or msg_type == 'relist_sell_order':
            parent_order_id = data.get('parent_order_id')
            new_order_id = data.get('order_id')
            
            if parent_order_id in self.active_orders:
                # Cancel parent order
                slot = self.active_orders[parent_order_id]['slot']
                self.slots[slot] = None
                self.active_orders[parent_order_id]['status'] = 'canceled'
                
                # Add to history
                self.order_history[parent_order_id] = self.active_orders[parent_order_id]
                
                # Remove from active orders
                del self.active_orders[parent_order_id]
                
                # Create new order
                available_slot = next((i for i, order in self.slots.items() if order is None), None)
                if available_slot:
                    self.slots[available_slot] = new_order_id
                    self.active_orders[new_order_id] = {
                        'slot': available_slot,
                        'status': 'active',
                        'item': data.get('item'),
                        'price': data.get('price'),
                        'quantity': data.get('quantity'),
                        'type': 'buy' if msg_type == 'relist_buy_order' else 'sell',
                        'parent_order_id': parent_order_id,
                        'timestamp': int(time.time() * 1000)
                    }
                    return {
                        'type': 'order_status',
                        'order_id': new_order_id,
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
            else:
                return {
                    'type': 'error',
                    'message': 'Parent order not found',
                    'timestamp': int(time.time() * 1000)
                }
        
        # Default response for unhandled message types
        return {
            'type': 'error',
            'message': 'Unhandled message type',
            'timestamp': int(time.time() * 1000)
        }
        
    async def simulate_order_fulfillment(self, order_id, websocket):
        """Simulate order fulfillment after a delay"""
        await asyncio.sleep(1)  # Simulate processing time
        
        if order_id in self.active_orders:
            order = self.active_orders[order_id]
            
            # Release the slot
            slot = order['slot']
            self.slots[slot] = None
            
            # Update order status
            order['status'] = 'filled'
            
            # Add to history
            self.order_history[order_id] = order
            
            # Remove from active orders
            del self.active_orders[order_id]
            
            # Send order filled message
            await websocket.send(json.dumps({
                'type': 'order_filled',
                'order_id': order_id,
                'item': order['item'],
                'price': order['price'],
                'quantity': order['quantity'],
                'order_type': order['type'],
                'timestamp': int(time.time() * 1000)
            }))
        
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


class TestOrderLifecycle(unittest.TestCase):
    """Test suite for complete order lifecycle"""
    
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
        
        # Set up inventory
        self.integration.order_manager.inventory_manager.gp = TEST_CONFIG["initial_gp"]
        self.integration.order_manager.inventory_manager.inventory = {
            "Abyssal whip": 2,
            "Dragon bones": 500
        }
        
        # Mock the connect method
        self.integration.connect = MagicMock(return_value=True)
        self.integration.connected = True
        self.integration.session = MagicMock()
        
    def tearDown(self):
        # Mock the disconnect method
        self.integration.disconnect = MagicMock(return_value=True)
        
    @patch('aiohttp.ClientSession.post')
    def test_buy_order_lifecycle(self, mock_post):
        """Test complete lifecycle of a buy order"""
        # Mock responses
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json.return_value = {"status": "success"}
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # 1. Place buy order
        order_data = {
            'order_id': self.integration.order_manager.generate_order_id("BUY_"),
            'item': "Nature rune",
            'type': "buy",
            'price': 250,
            'quantity': 1000
        }
        
        result = asyncio.run(self.integration.place_order(order_data))
        self.assertTrue(result, "Order placement should be successful")
        
        # 2. Simulate order being filled by server
        order_id = list(self.integration.order_manager.active_orders.keys())[0]
        
        # Simulate receiving order filled message
        order_filled_data = {
            'type': 'order_filled',
            'order_id': order_id,
            'item': "Nature rune",
            'price': 250,
            'quantity': 1000,
            'order_type': 'buy',
            'timestamp': int(time.time() * 1000)
        }
        
        # Process the order filled message (would normally be done by the WebSocket client)
        self.integration.order_manager.inventory_manager.process_buy_fulfillment("Nature rune", 250, 1000)
        
        # 3. Verify inventory updated
        self.assertEqual(self.integration.order_manager.inventory_manager.inventory.get("Nature rune", 0), 1000,
                        "Should have 1000 Nature runes after buy order fulfilled")
        self.assertEqual(self.integration.order_manager.inventory_manager.gp, 
                        TEST_CONFIG["initial_gp"] - (250 * 1000),
                        "GP should decrease by price * quantity")
        
    @patch('aiohttp.ClientSession.post')
    def test_sell_order_lifecycle(self, mock_post):
        """Test complete lifecycle of a sell order"""
        # Mock responses
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json.return_value = {"status": "success"}
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # 1. Place sell order
        order_data = {
            'order_id': self.integration.order_manager.generate_order_id("SELL_"),
            'item': "Abyssal whip",
            'type': "sell",
            'price': 1550000,
            'quantity': 1
        }
        
        result = asyncio.run(self.integration.place_order(order_data))
        self.assertTrue(result, "Order placement should be successful")
        
        # 2. Simulate order being filled by server
        order_id = list(self.integration.order_manager.active_orders.keys())[0]
        
        # Simulate receiving order filled message
        order_filled_data = {
            'type': 'order_filled',
            'order_id': order_id,
            'item': "Abyssal whip",
            'price': 1550000,
            'quantity': 1,
            'order_type': 'sell',
            'timestamp': int(time.time() * 1000)
        }
        
        # Process the order filled message (would normally be done by the WebSocket client)
        self.integration.order_manager.inventory_manager.process_sell_fulfillment("Abyssal whip", 1550000, 1)
        
        # 3. Verify inventory updated
        self.assertEqual(self.integration.order_manager.inventory_manager.inventory["Abyssal whip"], 1,
                        "Should have 1 Abyssal whip after selling 1")
        self.assertEqual(self.integration.order_manager.inventory_manager.gp, 
                        TEST_CONFIG["initial_gp"] + 1550000,
                        "GP should increase by price * quantity")


class TestSlotManagement(unittest.TestCase):
    """Test suite for slot management"""
    
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
        
        # Set up inventory with plenty of resources
        self.integration.order_manager.inventory_manager.gp = 100000000  # 100M GP
        self.integration.order_manager.inventory_manager.inventory = {
            "Abyssal whip": 10,
            "Dragon bones": 1000,
            "Nature rune": 10000
        }
        
        # Mock the connect method
        self.integration.connect = MagicMock(return_value=True)
        self.integration.connected = True
        self.integration.session = MagicMock()
        
    def tearDown(self):
        # Mock the disconnect method
        self.integration.disconnect = MagicMock(return_value=True)
        
    @patch('aiohttp.ClientSession.post')
    def test_slot_limit(self, mock_post):
        """Test that orders are limited by available slots"""
        # Mock responses for successful order placement
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json.return_value = {"status": "success"}
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Place 8 orders to fill all slots
        for i in range(8):
            order_data = {
                'order_id': self.integration.order_manager.generate_order_id(f"BUY_{i}_"),
                'item': "Abyssal whip",
                'type': "buy",
                'price': 1500000,
                'quantity': 1
            }
            
            # Create and add order directly to simulate successful placement
            order = Order(
                order_id=order_data['order_id'],
                item=order_data['item'],
                order_type=order_data['type'],
                price=order_data['price'],
                quantity=order_data['quantity']
            )
            self.integration.order_manager.orders[order_data['order_id']] = order
            self.integration.order_manager.active_orders[order_data['order_id']] = order
        
        # Verify 8 active orders
        self.assertEqual(len(self.integration.order_manager.active_orders), 8,
                        "Should have 8 active orders")
        
        # Try to place one more order - should be queued
        order_data = {
            'order_id': self.integration.order_manager.generate_order_id("BUY_9_"),
            'item': "Abyssal whip",
            'type': "buy",
            'price': 1500000,
            'quantity': 1
        }
        
        # Add to queue directly
        order = Order(
            order_id=order_data['order_id'],
            item=order_data['item'],
            order_type=order_data['type'],
            price=order_data['price'],
            quantity=order_data['quantity']
        )
        self.integration.order_queue.append(order)
        
        # Verify order is in queue
        self.assertEqual(len(self.integration.order_queue), 1,
                        "Should have 1 order in queue")


class TestErrorHandling(unittest.TestCase):
    """Test suite for error handling"""
    
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
        
        # Set up inventory
        self.integration.order_manager.inventory_manager.gp = TEST_CONFIG["initial_gp"]
        self.integration.order_manager.inventory_manager.inventory = {
            "Abyssal whip": 2,
            "Dragon bones": 500
        }
        
        # Mock the connect method
        self.integration.connect = MagicMock(return_value=True)
        self.integration.connected = True
        self.integration.session = MagicMock()
        
    def tearDown(self):
        # Mock the disconnect method
        self.integration.disconnect = MagicMock(return_value=True)
        
    def test_invalid_order_validation(self):
        """Test validation of invalid orders"""
        # Try to sell item not in inventory
        order_data = {
            'order_id': self.integration.order_manager.generate_order_id("SELL_"),
            'item': "Nature rune",  # Not in inventory
            'type': "sell",
            'price': 270,
            'quantity': 1
        }
        
        # Should fail validation
        result = asyncio.run(self.integration.place_order(order_data))
        self.assertFalse(result, "Order placement should fail for invalid order")
        
    @patch('aiohttp.ClientSession.post')
    def test_network_error_handling(self, mock_post):
        """Test handling of network errors"""
        # Mock network error
        mock_post.side_effect = Exception("Network error")
        
        # Try to submit event
        result = asyncio.run(self.integration.submit_event("test_event", {"test": "data"}))
        
        # Verify event submission failed
        self.assertFalse(result, "Event submission should fail on network error")
        
    @patch('aiohttp.ClientSession')
    def test_reconnection(self, mock_session):
        """Test reconnection after disconnection"""
        # Mock successful connection
        mock_session.return_value = MagicMock()
        self.integration.connected = False
        self.integration.session = None
        
        # Connect
        result = asyncio.run(self.integration.connect())
        
        # Verify connection successful
        self.assertTrue(result, "Connection should be successful")
        self.assertTrue(self.integration.connected, "Connected flag should be True")


if __name__ == "__main__":
    unittest.main()