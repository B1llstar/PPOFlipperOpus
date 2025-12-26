import asyncio
import time
import logging
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from functools import lru_cache

from ge_rest_client import GrandExchangeClient, GrandExchangeAPIError, RateLimitError

# Set up logger for this module
logger = logging.getLogger("RealTimeGrandExchangeClient")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.FileHandler("real_time_ge_client.log")
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.propagate = False
logger.info("real_time_ge_client logger initialized (imported)")

class RealTimeGrandExchangeClient(GrandExchangeClient):
    """Client for real-time Grand Exchange data with order execution capabilities."""
    
    def __init__(self, websocket_integration=None, update_interval=300, cache_size=128, max_retries=3, backoff_factor=1.0):
        """
        Initialize the Real-Time Grand Exchange client.
        
        Args:
            websocket_integration: WebSocket integration for order execution
            update_interval: Interval in seconds between market data updates (default: 300 seconds / 5 minutes)
            cache_size: Size of the cache for API responses
            max_retries: Maximum number of retries for API requests
            backoff_factor: Backoff factor for retries
        """
        super().__init__(cache_size=cache_size, max_retries=max_retries, backoff_factor=backoff_factor)
        self.websocket_integration = websocket_integration
        self.update_interval = update_interval  # 5 minutes in seconds
        self.last_update_time = 0
        self.market_data = {}
        self.pending_orders = []
        self.active_orders = []
        self.order_history = []
        self._update_task = None
        self._running = False
        logger.info(f"RealTimeGrandExchangeClient initialized with update interval: {update_interval}s")
    
    async def start(self):
        """Start the real-time client and begin periodic updates."""
        if self._running:
            logger.warning("Real-time client already running")
            return
        
        self._running = True
        
        # Initial market data update
        await self.update_market_data(force=True)
        
        # Start background task for periodic updates
        self._update_task = asyncio.create_task(self._periodic_update())
        
        logger.info("Real-time client started")
        
        # Send system status event if WebSocket integration is available
        if self.websocket_integration:
            await self.websocket_integration.system_status(
                "client_started", 
                "Real-time Grand Exchange client started"
            )
    
    async def stop(self):
        """Stop the real-time client and cancel periodic updates."""
        if not self._running:
            logger.warning("Real-time client not running")
            return
        
        self._running = False
        
        # Cancel update task if it exists
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
            self._update_task = None
        
        logger.info("Real-time client stopped")
        
        # Send system status event if WebSocket integration is available
        if self.websocket_integration:
            await self.websocket_integration.system_status(
                "client_stopped", 
                "Real-time Grand Exchange client stopped"
            )
    
    async def _periodic_update(self):
        """Background task for periodic market data updates."""
        try:
            while self._running:
                # Wait for the update interval
                await asyncio.sleep(self.update_interval)
                
                # Update market data
                await self.update_market_data()
                
                # Process pending orders
                await self.process_pending_orders()
                
                # Update active orders
                await self.update_active_orders()
        except asyncio.CancelledError:
            logger.info("Periodic update task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in periodic update task: {e}")
            if self.websocket_integration:
                await self.websocket_integration.error_event(
                    "update_error",
                    f"Error in periodic update task: {e}"
                )
    
    async def update_market_data(self, force=False):
        """
        Fetch latest market data if update interval has passed or if forced.
        
        Args:
            force: Force update regardless of interval
            
        Returns:
            bool: True if data was updated, False otherwise
        """
        current_time = time.time()
        if force or (current_time - self.last_update_time >= self.update_interval):
            try:
                logger.info("Updating market data")
                
                # Fetch latest data
                latest_data = self.get_latest()
                five_min_data = self.get_5m()
                
                # Store data
                self.market_data = {
                    "latest": latest_data,
                    "5m": five_min_data,
                    "timestamp": current_time
                }
                
                self.last_update_time = current_time
                
                # Send price updates via WebSocket if available
                if self.websocket_integration:
                    await self._send_price_updates(latest_data, five_min_data)
                
                logger.info(f"Market data updated with {len(latest_data)} items")
                return True
            except Exception as e:
                logger.error(f"Error updating market data: {e}")
                if self.websocket_integration:
                    await self.websocket_integration.error_event(
                        "market_data_error",
                        f"Error updating market data: {e}"
                    )
                return False
        else:
            logger.debug(f"Skipping market data update, {int(self.update_interval - (current_time - self.last_update_time))}s until next update")
            return False
    
    async def _send_price_updates(self, latest_data, five_min_data):
        """
        Send price updates via WebSocket.
        
        Args:
            latest_data: Latest price data
            five_min_data: 5-minute average data
        """
        for item_id, data in latest_data.items():
            try:
                # Get item name from mapping if available
                item_name = self.get_name_for_id(item_id) if hasattr(self, 'get_name_for_id') else f"Item {item_id}"
                
                # Calculate price change if 5m data is available
                change = 0
                if item_id in five_min_data and 'avgHighPrice' in five_min_data[item_id]:
                    old_price = five_min_data[item_id]['avgHighPrice']
                    new_price = data['high']
                    change = new_price - old_price
                
                # Get volume if available
                volume = None
                if item_id in five_min_data and 'highPriceVolume' in five_min_data[item_id]:
                    volume = five_min_data[item_id]['highPriceVolume']
                
                # Send price update
                await self.websocket_integration.price_update(
                    item=item_name,
                    price=data['high'],
                    change=change,
                    volume=volume
                )
                
                # If margin data is available, send margin update
                if 'high' in data and 'low' in data:
                    margin = data['high'] - data['low']
                    await self.websocket_integration.margin_update(
                        item=item_name,
                        buy_price=data['low'],
                        sell_price=data['high'],
                        margin=margin
                    )
            except Exception as e:
                logger.error(f"Error sending price update for item {item_id}: {e}")
    
    async def place_order(self, order_type, item, price, quantity):
        """
        Place a real order through the WebSocket.
        
        Args:
            order_type: Type of order ('buy' or 'sell')
            item: Item name or ID
            price: Price per item
            quantity: Number of items
            
        Returns:
            str: Order ID if successful, None otherwise
        """
        if self.websocket_integration:
            try:
                # Generate order ID
                order_id = f"order_{int(time.time())}_{hash(str(item))}"
                
                # Create order object
                order = {
                    "id": order_id,
                    "type": order_type,
                    "item": item,
                    "price": price,
                    "quantity": quantity,
                    "status": "pending",
                    "timestamp": int(time.time()),
                    "filled_quantity": 0,
                    "remaining_quantity": quantity
                }
                
                # Add to pending orders
                self.pending_orders.append(order)
                
                logger.info(f"Order placed: {order_type} {quantity}x {item} at {price} gp")
                
                # Submit order through WebSocket
                await self.websocket_integration.submit_event("place_order", order)
                
                # Send agent decision event
                await self.websocket_integration.agent_decision(
                    agent_id="real_time_agent",
                    action_type=order_type,
                    item=item,
                    price=price,
                    quantity=quantity,
                    reasoning="Real-time trading decision"
                )
                
                return order_id
            except Exception as e:
                logger.error(f"Error placing order: {e}")
                if self.websocket_integration:
                    await self.websocket_integration.error_event(
                        "order_error",
                        f"Error placing order: {e}"
                    )
                return None
        else:
            logger.warning("Cannot place order: WebSocket integration not available")
            return None
    
    async def cancel_order(self, order_id):
        """
        Cancel a pending or active order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Find order in pending orders
        for i, order in enumerate(self.pending_orders):
            if order["id"] == order_id:
                # Remove from pending orders
                cancelled_order = self.pending_orders.pop(i)
                cancelled_order["status"] = "cancelled"
                
                # Add to order history
                self.order_history.append(cancelled_order)
                
                logger.info(f"Cancelled pending order: {order_id}")
                
                # Send cancellation event if WebSocket integration is available
                if self.websocket_integration:
                    await self.websocket_integration.submit_event(
                        "order_cancelled",
                        cancelled_order
                    )
                
                return True
        
        # Find order in active orders
        for i, order in enumerate(self.active_orders):
            if order["id"] == order_id:
                # Remove from active orders
                cancelled_order = self.active_orders.pop(i)
                cancelled_order["status"] = "cancelled"
                
                # Add to order history
                self.order_history.append(cancelled_order)
                
                logger.info(f"Cancelled active order: {order_id}")
                
                # Send cancellation event if WebSocket integration is available
                if self.websocket_integration:
                    await self.websocket_integration.submit_event(
                        "order_cancelled",
                        cancelled_order
                    )
                
                return True
        
        logger.warning(f"Order not found for cancellation: {order_id}")
        return False
    
    async def process_pending_orders(self):
        """Process pending orders and move them to active orders."""
        if not self.pending_orders:
            return
        
        logger.info(f"Processing {len(self.pending_orders)} pending orders")
        
        # Process each pending order
        for i in range(len(self.pending_orders) - 1, -1, -1):
            order = self.pending_orders[i]
            
            # Move to active orders
            order["status"] = "active"
            self.active_orders.append(order)
            
            # Remove from pending orders
            self.pending_orders.pop(i)
            
            logger.info(f"Order activated: {order['id']}")
            
            # Send order activation event if WebSocket integration is available
            if self.websocket_integration:
                await self.websocket_integration.submit_event(
                    "order_activated",
                    order
                )
    
    async def update_active_orders(self):
        """Update status of active orders based on market conditions."""
        if not self.active_orders:
            return
        
        logger.info(f"Updating {len(self.active_orders)} active orders")
        
        # Get latest market data
        latest_data = self.market_data.get("latest", {})
        
        # Update each active order
        for i in range(len(self.active_orders) - 1, -1, -1):
            order = self.active_orders[i]
            
            # Check if order can be filled
            filled = await self._check_order_fill(order, latest_data)
            
            if filled:
                # Order fully filled, move to history
                self.order_history.append(order)
                self.active_orders.pop(i)
                
                logger.info(f"Order filled: {order['id']}")
                
                # Send trade executed event if WebSocket integration is available
                if self.websocket_integration:
                    await self.websocket_integration.trade_executed(
                        agent_id="real_time_agent",
                        action_type=order["type"],
                        item=order["item"],
                        price=order["price"],
                        quantity=order["filled_quantity"],
                        profit=0.0  # Profit calculation would be added in a real implementation
                    )
    
    async def _check_order_fill(self, order, latest_data):
        """
        Check if an order can be filled based on market conditions.
        
        Args:
            order: Order to check
            latest_data: Latest market data
            
        Returns:
            bool: True if order was filled, False otherwise
        """
        # Get item ID
        item_id = order["item"]
        if not item_id.isdigit():
            # Convert item name to ID if necessary
            item_id = self.get_id_for_name(item_id) if hasattr(self, 'get_id_for_name') else None
        
        if not item_id or item_id not in latest_data:
            logger.warning(f"Item not found in market data: {order['item']}")
            return False
        
        # Get market price
        market_data = latest_data[item_id]
        
        # Check if order can be filled
        if order["type"] == "buy":
            # Buy order can be filled if market low price is <= order price
            if market_data["low"] <= order["price"]:
                # Fill order
                order["filled_quantity"] = order["quantity"]
                order["remaining_quantity"] = 0
                order["status"] = "filled"
                order["fill_price"] = market_data["low"]
                order["fill_time"] = int(time.time())
                return True
        elif order["type"] == "sell":
            # Sell order can be filled if market high price is >= order price
            if market_data["high"] >= order["price"]:
                # Fill order
                order["filled_quantity"] = order["quantity"]
                order["remaining_quantity"] = 0
                order["status"] = "filled"
                order["fill_price"] = market_data["high"]
                order["fill_time"] = int(time.time())
                return True
        
        return False
    
    async def update_order(self, order_id, updates):
        """
        Update an existing order.
        
        Args:
            order_id: ID of the order to update
            updates: Dictionary of updates to apply
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Find order in pending orders
        for order in self.pending_orders:
            if order["id"] == order_id:
                # Apply updates
                for key, value in updates.items():
                    if key in order and key not in ["id", "timestamp", "status"]:
                        order[key] = value
                
                logger.info(f"Updated pending order: {order_id}")
                
                # Send update event if WebSocket integration is available
                if self.websocket_integration:
                    await self.websocket_integration.submit_event(
                        "order_updated",
                        order
                    )
                
                return True
        
        # Find order in active orders
        for order in self.active_orders:
            if order["id"] == order_id:
                # Apply updates
                for key, value in updates.items():
                    if key in order and key not in ["id", "timestamp", "status"]:
                        order[key] = value
                
                logger.info(f"Updated active order: {order_id}")
                
                # Send update event if WebSocket integration is available
                if self.websocket_integration:
                    await self.websocket_integration.submit_event(
                        "order_updated",
                        order
                    )
                
                return True
        
        logger.warning(f"Order not found for update: {order_id}")
        return False
    
    def get_order(self, order_id):
        """
        Get an order by ID.
        
        Args:
            order_id: ID of the order to get
            
        Returns:
            dict: Order if found, None otherwise
        """
        # Check pending orders
        for order in self.pending_orders:
            if order["id"] == order_id:
                return order
        
        # Check active orders
        for order in self.active_orders:
            if order["id"] == order_id:
                return order
        
        # Check order history
        for order in self.order_history:
            if order["id"] == order_id:
                return order
        
        return None
    
    def get_orders_by_status(self, status):
        """
        Get orders by status.
        
        Args:
            status: Status to filter by ('pending', 'active', 'filled', 'cancelled')
            
        Returns:
            list: List of orders with the specified status
        """
        if status == "pending":
            return self.pending_orders.copy()
        elif status == "active":
            return self.active_orders.copy()
        elif status in ["filled", "cancelled"]:
            return [order for order in self.order_history if order["status"] == status]
        else:
            return []
    
    def get_name_for_id(self, item_id):
        """
        Get the item name for a given item ID.
        
        Args:
            item_id: ID of the item
            
        Returns:
            str: Item name if found, None otherwise
        """
        # Try to use the method from the parent class if it exists
        if hasattr(super(), 'get_name_for_id'):
            return super().get_name_for_id(item_id)
        
        # Otherwise, use the mapping from the parent class if it exists
        if hasattr(self, '_id_to_name_map'):
            return self._id_to_name_map.get(str(item_id))
        
        # If no mapping is available, return a generic name
        return f"Item {item_id}"
    
    def get_id_for_name(self, item_name):
        """
        Get the item ID for a given item name.
        
        Args:
            item_name: Name of the item
            
        Returns:
            str: Item ID if found, None otherwise
        """
        # Try to use the method from the parent class if it exists
        if hasattr(super(), 'get_id_for_name'):
            return super().get_id_for_name(item_name)
        
        # Otherwise, use the mapping from the parent class if it exists
        if hasattr(self, '_name_to_id_map'):
            return self._name_to_id_map.get(item_name)
        
        return None