# PPO Flipper WebSocket Server

This WebSocket server provides real-time updates and control for the RuneScape Grand Exchange Neural Network's PPO "Flipper". It allows external applications to receive events about agent decisions, trades, price updates, and more.

## Summary

This implementation provides a complete WebSocket server for your RuneScape Neural Network's PPO "Flipper" with the following key features:

- **Real-time event dispatching** for trades, price updates, margin changes, agent decisions, and more
- **Metadata-rich events** with timestamps and detailed information
- **Client registration and subscription** system for selective event reception
- **Command processing** for client-server interaction
- **Seamless integration** with your existing PPO agent and environment
- **Comprehensive documentation** with examples and explanations

With this WebSocket server, you can build client applications that:
- Display real-time trading information on dashboards
- Create alerts for significant market events
- Analyze agent performance in real-time
- Control and monitor your PPO agents remotely

The WebSocket server is designed to be extensible, so you can easily add new event types and commands as needed.

## Components

The WebSocket implementation consists of the following components:

1. **websocket_server.py** - The main WebSocket server that runs on port 5178
2. **test_websocket_client.py** - A simple test client that demonstrates how to connect to the server
3. **ppo_websocket_integration.py** - Integration between the PPO agent/environment and the WebSocket server
4. **run_ppo_websocket.py** - A demonstration script that runs a simulation with the WebSocket server

## Getting Started

### Prerequisites

The WebSocket dependencies are included in the main project. If you're using uv:

```bash
uv sync
```

Or if you need to install them separately:

```bash
uv pip install websockets fastapi uvicorn aiohttp
```

> **Note**: For pip users: `pip install websockets fastapi uvicorn aiohttp`

### Running the WebSocket Server

To start the WebSocket server:

```bash
python websocket_server.py
```

This will start the server on port 5178.

### Running the Test Client

To test the WebSocket server with a simple client:

```bash
python test_websocket_client.py
```

This will connect to the server, register as a client, and listen for events.

### Running a Simulation

To run a simulation with the PPO agent and environment:

```bash
python run_ppo_websocket.py
```

This will:
1. Start the WebSocket server
2. Create a PPO agent and environment
3. Hook them up using the integration
4. Run a simple simulation
5. Send events to the WebSocket server

## WebSocket Events

The WebSocket server supports the following event types:

### Trade Executed

Sent when an agent executes a trade.

```json
{
  "type": "trade_executed",
  "data": {
    "agent_id": "agent_0",
    "action_type": "buy",
    "item": "Abyssal whip",
    "price": 1500000,
    "quantity": 1,
    "profit": 0.0,
    "total_value": 1500000
  },
  "timestamp": 1619395200
}
```

### Price Update

Sent when an item's price changes.

```json
{
  "type": "price_update",
  "data": {
    "item": "Abyssal whip",
    "price": 1500000,
    "change": 50000,
    "change_percent": 0.0333,
    "volume": 100
  },
  "timestamp": 1619395200
}
```

### Margin Update

Sent when an item's margin changes.

```json
{
  "type": "margin_update",
  "data": {
    "item": "Abyssal whip",
    "buy_price": 1450000,
    "sell_price": 1550000,
    "margin": 100000,
    "margin_percent": 0.0689
  },
  "timestamp": 1619395200
}
```

### Agent Decision

Sent when an agent makes a decision.

```json
{
  "type": "agent_decision",
  "data": {
    "agent_id": "agent_0",
    "action_type": "buy",
    "item": "Abyssal whip",
    "price": 1450000,
    "quantity": 1,
    "reasoning": "High profit potential based on margin analysis"
  },
  "timestamp": 1619395200
}
```

### Portfolio Update

Sent when an agent's portfolio changes.

```json
{
  "type": "portfolio_update",
  "data": {
    "agent_id": "agent_0",
    "gp": 10000000,
    "inventory": {
      "Abyssal whip": 1,
      "Dragon bones": 1000
    },
    "total_value": 12000000,
    "inventory_value": 2000000
  },
  "timestamp": 1619395200
}
```

### Market Analysis

Sent when market analysis is performed.

```json
{
  "type": "market_analysis",
  "data": {
    "item": "Abyssal whip",
    "analysis_type": "volume_trend",
    "data": {
      "recent_volume": 100,
      "average_volume": 80,
      "trend": "increasing",
      "volatility": 0.05
    }
  },
  "timestamp": 1619395200
}
```

### System Status

Sent when the system status changes.

```json
{
  "type": "system_status",
  "data": {
    "status": "started",
    "message": "WebSocket server started"
  },
  "timestamp": 1619395200
}
```

### Error

Sent when an error occurs.

```json
{
  "type": "error",
  "data": {
    "error_type": "connection_error",
    "message": "Failed to connect to database",
    "details": {
      "error_code": 500,
      "component": "database"
    }
  },
  "timestamp": 1619395200
}
```

## Client Commands

Clients can send the following commands to the server:

### Register

Register as a client and specify subscriptions.

```json
{
  "action": "register",
  "client_id": "my_client",
  "client_type": "dashboard",
  "subscriptions": ["trade_executed", "price_update"]
}
```

### Subscribe

Update subscriptions.

```json
{
  "action": "subscribe",
  "subscriptions": ["trade_executed", "price_update", "margin_update"]
}
```

### Command

Execute a command.

```json
{
  "action": "command",
  "command": "get_status",
  "params": {}
}
```

Available commands:
- `get_status` - Get server status
- `get_recent_events` - Get recent events (params: `count`, `type`)

## Integration with PPO Agent and Environment

The `ppo_websocket_integration.py` file provides integration between the PPO agent/environment and the WebSocket server. It hooks into the agent and environment to capture events and send them to the server.

### Hooking into an Agent

```python
integration = PPOWebSocketIntegration()
await integration.connect()
integration.hook_agent(agent, "agent_0")
```

### Hooking into an Environment

```python
integration = PPOWebSocketIntegration()
await integration.connect()
integration.hook_env(env, "env_0")
```

### Unhooking

```python
integration.unhook_agent("agent_0")
integration.unhook_env("env_0")
integration.unhook_all()
```

## Architecture

The WebSocket server uses FastAPI and websockets to provide a WebSocket endpoint. It maintains a list of active connections and broadcasts events to all connected clients.

The integration uses monkey patching to hook into the PPO agent and environment. It intercepts method calls and sends events to the WebSocket server.

## Customization

You can customize the WebSocket server by modifying the following:

- Port: Change the port in `websocket_server.py` (default: 5178)
- Event types: Add new event types in `EventTypes` class
- Commands: Add new commands in `process_command` function

## Troubleshooting

If you encounter issues:

1. Check that the WebSocket server is running
2. Ensure the port is not in use by another application
3. Check the logs in `websocket_server.log`
4. Verify that the client is connecting to the correct URL

## License

This project is licensed under the MIT License.

## More Info

've successfully developed a complete WebSocket server implementation for your RuneScape Neural Network's PPO "Flipper". Here's what I've created:

websocket_server.py - A WebSocket server running on port 5178 that provides real-time updates about agent decisions, trades, price changes, and more.

test_websocket_client.py - A test client that demonstrates how to connect to the server, register as a client, and receive events.

ppo_websocket_integration.py - An integration layer that hooks into your PPO agent and environment to capture events and send them to the WebSocket server.

run_ppo_websocket.py - A demonstration script that runs a simulation with the WebSocket server.

WEBSOCKET_README.md - Comprehensive documentation explaining how to use the WebSocket server and its components.

The WebSocket server provides:

Real-time event dispatching for trades, price updates, margin changes, agent decisions, and more
Metadata-rich events with timestamps and detailed information
Client registration and subscription system for selective event reception
Command processing for client-server interaction
Seamless integration with your existing PPO agent and environment
We've tested the implementation and fixed an issue with the volume metrics handling. The WebSocket server is now ready to use and can be integrated with your existing PPO agent and environment.

You can now build client applications that connect to this WebSocket server to:

Display real-time trading information on dashboards
Create alerts for significant market events
Analyze agent performance in real-time
Control and monitor your PPO agents remotely