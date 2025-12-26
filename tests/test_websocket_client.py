import asyncio
import json
import logging
import time
import websockets
from datetime import datetime
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_websocket_client")

# WebSocket server URL
WEBSOCKET_URL = "ws://localhost:5178/ws"

# Event types (matching the server)
class EventTypes:
    TRADE_EXECUTED = "trade_executed"
    PRICE_UPDATE = "price_update"
    MARGIN_UPDATE = "margin_update"
    AGENT_DECISION = "agent_decision"
    PORTFOLIO_UPDATE = "portfolio_update"
    MARKET_ANALYSIS = "market_analysis"
    SYSTEM_STATUS = "system_status"
    ERROR = "error"

# Test client class
class TestWebSocketClient:
    def __init__(self, url: str, client_id: str = "test_client"):
        self.url = url
        self.client_id = client_id
        self.websocket = None
        self.connected = False
        self.received_messages = []
        
    async def connect(self):
        """Connect to the WebSocket server."""
        try:
            self.websocket = await websockets.connect(self.url)
            self.connected = True
            logger.info(f"Connected to {self.url}")
            return True
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            return False
            
    async def register(self, client_type: str = "test", subscriptions: List[str] = None):
        """Register with the server."""
        if not self.connected:
            logger.error("Not connected")
            return False
            
        if subscriptions is None:
            # Subscribe to all event types by default
            subscriptions = [
                EventTypes.TRADE_EXECUTED,
                EventTypes.PRICE_UPDATE,
                EventTypes.MARGIN_UPDATE,
                EventTypes.AGENT_DECISION,
                EventTypes.PORTFOLIO_UPDATE,
                EventTypes.MARKET_ANALYSIS,
                EventTypes.SYSTEM_STATUS,
                EventTypes.ERROR
            ]
            
        try:
            # Send registration message
            await self.websocket.send(json.dumps({
                "action": "register",
                "client_id": self.client_id,
                "client_type": client_type,
                "subscriptions": subscriptions
            }))
            
            # Wait for initial welcome message
            welcome_message = await self.websocket.recv()
            welcome_data = json.loads(welcome_message)
            
            # Send registration message
            await self.websocket.send(json.dumps({
                "action": "register",
                "client_id": self.client_id,
                "client_type": client_type,
                "subscriptions": subscriptions
            }))
            
            # Wait for registration confirmation
            response = await self.websocket.recv()
            response_data = json.loads(response)
            
            if response_data.get("status") == "registered":
                logger.info(f"Registered as {self.client_id}")
                return True
            else:
                logger.error(f"Registration failed: {response_data}")
                return False
                
        except Exception as e:
            logger.error(f"Registration error: {str(e)}")
            return False
            
    async def subscribe(self, subscriptions: List[str]):
        """Update subscriptions."""
        if not self.connected:
            logger.error("Not connected")
            return False
            
        try:
            # Send subscription message
            await self.websocket.send(json.dumps({
                "action": "subscribe",
                "subscriptions": subscriptions
            }))
            
            # Wait for response
            response = await self.websocket.recv()
            response_data = json.loads(response)
            
            if response_data.get("status") == "subscribed":
                logger.info(f"Subscribed to {len(subscriptions)} event types")
                return True
            else:
                logger.error(f"Subscription failed: {response_data}")
                return False
                
        except Exception as e:
            logger.error(f"Subscription error: {str(e)}")
            return False
            
    async def send_command(self, command: str, params: Dict[str, Any] = None):
        """Send a command to the server."""
        if not self.connected:
            logger.error("Not connected")
            return None
            
        if params is None:
            params = {}
            
        try:
            # Send command message
            await self.websocket.send(json.dumps({
                "action": "command",
                "command": command,
                "params": params
            }))
            
            # Wait for response
            response = await self.websocket.recv()
            response_data = json.loads(response)
            
            if response_data.get("status") == "command_result" or response_data.get("type") == "system_status":
                logger.info(f"Command {command} executed")
                return response_data.get("result")
            else:
                logger.error(f"Command failed: {response_data}")
                return None
                
        except Exception as e:
            logger.error(f"Command error: {str(e)}")
            return None
            
    async def listen(self, duration: int = 60):
        """Listen for messages for a specified duration."""
        if not self.connected:
            logger.error("Not connected")
            return
            
        logger.info(f"Listening for messages for {duration} seconds...")
        
        # Set end time
        end_time = time.time() + duration
        
        try:
            while time.time() < end_time:
                # Set timeout to remaining duration
                remaining = max(0.1, end_time - time.time())
                
                # Wait for message with timeout
                try:
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=remaining)
                    
                    # Parse message
                    message_data = json.loads(message)
                    
                    # Store message
                    self.received_messages.append(message_data)
                    
                    # Format timestamp
                    timestamp = message_data.get("timestamp", 0)
                    if timestamp:
                        timestamp_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        timestamp_str = "N/A"
                    
                    # Log message
                    logger.info(f"Received {message_data.get('type', 'unknown')} event at {timestamp_str}")
                    logger.info(f"Data: {json.dumps(message_data, indent=2)}")
                    
                except asyncio.TimeoutError:
                    # Timeout reached, exit loop
                    break
                    
        except Exception as e:
            logger.error(f"Listen error: {str(e)}")
            
        logger.info(f"Received {len(self.received_messages)} messages")
        
    async def disconnect(self):
        """Disconnect from the server."""
        if self.connected and self.websocket:
            await self.websocket.close()
            self.connected = False
            logger.info("Disconnected")


# Main function
async def main():
    # Create client
    client = TestWebSocketClient(WEBSOCKET_URL)
    
    # Connect to server
    if not await client.connect():
        return
    
    # Register with server
    if not await client.register():
        await client.disconnect()
        return
    
    # Send a status command
    status = await client.send_command("get_status")
    if status:
        logger.info(f"Server status: {status}")
    
    # Listen for events for 30 seconds
    await client.listen(30)
    
    # Disconnect
    await client.disconnect()


# Run the test
if __name__ == "__main__":
    asyncio.run(main())