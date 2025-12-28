# GEAuto + PPOFlipperOpus Integration Proposal

## Executive Summary

This proposal outlines an integration between the **GEAuto RuneLite plugin** (Java-based game client automation) and **PPOFlipperOpus** (Python-based PPO reinforcement learning trading bot). The integration creates a complete end-to-end autonomous trading system where the PPO agent makes intelligent trading decisions, and GEAuto executes them in the actual game.

---

## Current Architecture Overview

### PPOFlipperOpus (Python)
- **PPO Agent**: Actor-Critic neural network for trading decisions
- **GE Environment**: Gymnasium-compatible simulation environment
- **Real-Time Client**: Market data fetching from OSRS Prices API
- **WebSocket Integration**: Real-time event streaming for dashboard
- **Training Infrastructure**: Multi-agent training with shared knowledge

### GEAuto Plugin (Java)
- **State Machine**: 20+ states for GE interaction (opening GE, typing items, setting prices, confirming, collecting)
- **HTTP API Server**: Endpoints for Buy, Sell, Cancel, Status, Orders, Collect, Health
- **Order Queue Manager**: Priority-based order processing with retry logic
- **Event System**: GEEvent broadcasting for order lifecycle tracking
- **Decision System**: GEDecisionRequest/GEDecision for external decision making

---

## Proposed Integration Architecture

```
+------------------------------------------------------------------+
|                        PPOFlipperOpus                            |
|  +--------------------+     +-----------------------------+      |
|  |   PPO Agent        |<--->|  Inference Engine           |      |
|  |  (Actor-Critic)    |     |  - Real-time decision loop  |      |
|  +--------------------+     |  - State observation        |      |
|                             |  - Action selection         |      |
|  +--------------------+     +-------------+---------------+      |
|  | Market Data Client |                   |                      |
|  | - OSRS Prices API  |                   v                      |
|  | - Volume Analysis  |     +-----------------------------+      |
|  +--------------------+     |  GEAuto Bridge (New)        |      |
|                             |  - HTTP client to GEAuto    |      |
|  +--------------------+     |  - Order translation        |      |
|  | Analytics Engine   |     |  - Event handling           |      |
|  | - P&L Tracking     |     |  - State synchronization    |      |
|  | - Performance      |     +-------------+---------------+      |
|  +--------------------+                   |                      |
+-------------------------------------------|----------------------+
                                            | HTTP/WebSocket
                                            v
+------------------------------------------------------------------+
|                        GEAuto Plugin                             |
|  +--------------------+     +-----------------------------+      |
|  |   HTTP API Server  |<----|  WebSocket Event Stream     |      |
|  |  Port: configurable|     |  - Order updates            |      |
|  +--------------------+     |  - Trade completions        |      |
|           |                 |  - GE state changes         |      |
|           v                 +-----------------------------+      |
|  +--------------------+                                          |
|  | Order Queue Manager|     +-----------------------------+      |
|  | - Priority queue   |<--->|  GE State Machine          |      |
|  | - Retry logic      |     |  - 20+ interaction states   |      |
|  +--------------------+     +-----------------------------+      |
|           |                              |                       |
|           v                              v                       |
|  +--------------------+     +-----------------------------+      |
|  | Game Interface     |     |  GE Interface Utils         |      |
|  | - Inventory        |     |  - Slot detection           |      |
|  | - Bank access      |     |  - Price/qty input          |      |
|  +--------------------+     +-----------------------------+      |
+------------------------------------------------------------------+
                              |
                              v
                    [ OSRS Game Client ]
```

---

## New Components to Build

### 1. GEAuto Bridge (`bridge/geauto_bridge.py`)

A Python client that communicates with GEAuto's HTTP API and translates PPO agent actions into executable orders.

```python
class GEAutoBridge:
    """Bridge between PPO agent and GEAuto plugin."""

    # Core Methods
    async def connect(host: str, port: int) -> bool
    async def disconnect() -> None
    async def health_check() -> HealthStatus

    # Order Execution
    async def submit_buy_order(item: str, quantity: int, price: int) -> OrderResult
    async def submit_sell_order(item: str, quantity: int, price: int) -> OrderResult
    async def cancel_order(order_id: str) -> bool
    async def collect_slot(slot: int) -> CollectionResult

    # State Queries
    async def get_orders() -> List[GEOrder]
    async def get_pending_orders() -> List[GEOrder]
    async def get_active_orders() -> List[GEOrder]
    async def get_free_slots() -> int
    async def get_current_state() -> GEAutoState

    # Event Streaming
    async def subscribe_events() -> AsyncIterator[GEEvent]
```

### 2. Inferencing Interface (`inference/inference_engine.py`)

The brain of the system - takes observations, runs the PPO policy, and translates actions into GEAuto commands.

```python
class InferencingEngine:
    """High-level inferencing interface for trading decisions."""

    def __init__(
        self,
        agent: PPOAgent,
        geauto_bridge: GEAutoBridge,
        market_client: RealTimeGrandExchangeClient,
        analytics: AnalyticsEngine,
        config: InferenceConfig
    ):
        ...

    # Decision Making
    async def make_decision(observation: Observation) -> TradingDecision
    async def execute_decision(decision: TradingDecision) -> ExecutionResult

    # State Management
    async def build_observation() -> Observation
    async def sync_game_state() -> GameState
    async def reconcile_positions() -> ReconciliationReport

    # Decision Strategies
    async def should_enter_position(item: str) -> PositionAdvice
    async def should_exit_position(item: str) -> PositionAdvice
    async def get_optimal_price(item: str, side: str) -> OptimalPrice
    async def get_optimal_quantity(item: str, side: str) -> OptimalQuantity
```

### 3. Inventory Manager (`inventory/inventory_manager.py`)

Tracks in-game inventory state and synchronizes with GEAuto.

```python
class InventoryManager:
    """Manages and tracks inventory state."""

    # State Tracking
    async def sync_with_game() -> InventoryState
    async def get_position(item: str) -> Position
    async def get_all_positions() -> Dict[str, Position]
    async def get_available_gp() -> int

    # Position Analysis
    def get_total_value(prices: Dict[str, int]) -> int
    def get_unrealized_pnl(current_prices: Dict[str, int]) -> float
    def get_position_concentration() -> Dict[str, float]

    # Bank Integration
    async def check_bank_for_item(item: str) -> int
    async def withdraw_from_bank(item: str, quantity: int) -> bool
    async def deposit_to_bank(item: str, quantity: int) -> bool
```

### 4. Analytics Engine (`analytics/analytics_engine.py`)

Real-time performance tracking, P&L calculations, and risk monitoring.

```python
class AnalyticsEngine:
    """Real-time analytics and performance tracking."""

    # Trade Tracking
    def record_trade(trade: CompletedTrade) -> None
    def get_trade_history(limit: int = 100) -> List[CompletedTrade]

    # P&L Calculations
    def get_realized_pnl() -> float
    def get_unrealized_pnl(current_prices: Dict[str, int]) -> float
    def get_total_pnl() -> float
    def get_pnl_by_item() -> Dict[str, float]

    # Performance Metrics
    def get_win_rate() -> float
    def get_avg_profit_per_trade() -> float
    def get_sharpe_ratio(risk_free_rate: float = 0.0) -> float
    def get_max_drawdown() -> float
    def get_roi() -> float

    # Risk Metrics
    def get_concentration_risk() -> float
    def get_volume_at_risk() -> float
    def get_position_sizes() -> Dict[str, float]

    # Reporting
    async def generate_report(period: str) -> PerformanceReport
    async def export_to_firebase(report: PerformanceReport) -> bool
```

### 5. Real-Time Event Stream (`streams/event_stream.py`)

Unified event streaming from all sources (GEAuto, market data, agent decisions).

```python
class UnifiedEventStream:
    """Unified event stream from all sources."""

    # Event Sources
    geauto_events: AsyncQueue[GEEvent]
    market_events: AsyncQueue[MarketEvent]
    agent_events: AsyncQueue[AgentEvent]

    # Subscription
    async def subscribe(event_types: List[EventType]) -> AsyncIterator[Event]
    async def subscribe_all() -> AsyncIterator[Event]

    # Event Handlers
    def on_order_filled(handler: Callable[[OrderFilledEvent], None]) -> None
    def on_price_update(handler: Callable[[PriceUpdateEvent], None]) -> None
    def on_agent_decision(handler: Callable[[AgentDecisionEvent], None]) -> None
    def on_error(handler: Callable[[ErrorEvent], None]) -> None
```

### 6. Trading Controller (`controller/trading_controller.py`)

Orchestrates the entire trading loop with safety controls.

```python
class TradingController:
    """Main trading loop controller with safety controls."""

    def __init__(
        self,
        inference_engine: InferencingEngine,
        analytics: AnalyticsEngine,
        config: TradingConfig
    ):
        ...

    # Lifecycle
    async def start() -> None
    async def stop() -> None
    async def pause() -> None
    async def resume() -> None

    # Safety Controls
    def set_max_position_size(max_gp: int) -> None
    def set_max_loss_limit(max_loss_gp: int) -> None
    def set_trading_hours(start: time, end: time) -> None
    def add_item_to_blacklist(item: str) -> None

    # Manual Overrides
    async def force_close_position(item: str) -> bool
    async def force_cancel_all_orders() -> int
    async def emergency_stop() -> None
```

---

## Data Models

### Observation (for PPO Agent)
```python
@dataclass
class Observation:
    # Portfolio State
    gp: int
    positions: Dict[str, Position]  # item -> Position

    # Market State
    prices: Dict[str, PriceData]  # item -> (high, low, volume)
    spreads: Dict[str, float]     # item -> spread percentage

    # GE State
    active_orders: List[GEOrder]
    pending_orders: List[GEOrder]
    free_slots: int

    # Technical Indicators
    volume_momentum: Dict[str, float]
    price_sma_6h: Dict[str, float]
    price_sma_24h: Dict[str, float]
    volatility: Dict[str, float]

    # Time Features
    hour_of_day: int
    day_of_week: int
```

### Trading Decision
```python
@dataclass
class TradingDecision:
    action_type: ActionType  # HOLD, BUY, SELL, CANCEL, COLLECT
    item: Optional[str]
    quantity: Optional[int]
    price: Optional[int]
    reasoning: str
    confidence: float

    # Metadata
    timestamp: datetime
    agent_id: str
    observation_hash: str
```

### Execution Result
```python
@dataclass
class ExecutionResult:
    success: bool
    order_id: Optional[str]
    actual_price: Optional[int]
    actual_quantity: Optional[int]
    error_message: Optional[str]
    execution_time_ms: int
    ge_slot: Optional[int]
```

---

## GEAuto API Endpoints to Implement/Use

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/status` | GET | Current plugin state |
| `/buy` | POST | Submit buy order |
| `/sell` | POST | Submit sell order |
| `/cancel` | POST | Cancel order by ID |
| `/collect` | POST | Collect from slot |
| `/orders` | GET | List all orders |
| `/order/{id}` | GET | Get specific order |
| `/start` | POST | Start automation |
| `/pause` | POST | Pause automation |
| `/resume` | POST | Resume automation |
| `/stop` | POST | Stop automation |

---

## Event Types

### From GEAuto
- `BUY_PLACED` - Buy order placed in GE slot
- `SELL_PLACED` - Sell order placed in GE slot
- `BUY_PROGRESS` - Partial fill on buy order
- `SELL_PROGRESS` - Partial fill on sell order
- `BUY_FILLED` - Buy order fully filled
- `SELL_FILLED` - Sell order fully filled
- `READY_TO_COLLECT` - Items ready for collection
- `COLLECTED` - Items collected from slot
- `CANCELLED` - Order cancelled
- `ERROR` - Error occurred

### From Market Data
- `PRICE_UPDATE` - Price change detected
- `SPREAD_CHANGE` - Spread changed significantly
- `VOLUME_SPIKE` - Unusual volume detected
- `MARKET_CLOSED` - Market data unavailable

### From Agent
- `DECISION_MADE` - Agent made a decision
- `ACTION_EXECUTED` - Action was executed
- `POSITION_OPENED` - New position opened
- `POSITION_CLOSED` - Position closed
- `RISK_ALERT` - Risk threshold triggered

---

## Configuration

### inference_config.yaml
```yaml
# GEAuto Connection
geauto:
  host: "localhost"
  port: 9696
  timeout_ms: 5000
  retry_attempts: 3

# Agent Settings
agent:
  model_path: "agent_states/actor_best.pth"
  decision_interval_seconds: 60
  min_confidence_threshold: 0.6
  exploration_rate: 0.05

# Risk Management
risk:
  max_position_value_gp: 50_000_000  # 50M max per position
  max_total_exposure_gp: 200_000_000  # 200M total
  max_single_order_gp: 10_000_000  # 10M per order
  stop_loss_percentage: 15.0
  daily_loss_limit_gp: 20_000_000

# Market Data
market:
  update_interval_seconds: 60
  price_staleness_threshold_seconds: 300
  min_volume_threshold: 1000
  max_spread_percentage: 10.0

# Trading Hours (optional)
schedule:
  enabled: false
  start_hour: 8
  end_hour: 22
  timezone: "UTC"

# Analytics
analytics:
  firebase_enabled: true
  firebase_project: "ppoflipperopus"
  report_interval_minutes: 60

# Logging
logging:
  level: "INFO"
  file: "trading.log"
  max_size_mb: 100
  backup_count: 5
```

---

## Implementation Phases

### Phase 1: Core Bridge (Foundation)
- [ ] Implement `GEAutoBridge` HTTP client
- [ ] Add health check and connection management
- [ ] Implement order submission (buy/sell)
- [ ] Implement order status queries
- [ ] Add basic error handling and retries

### Phase 2: State Synchronization
- [ ] Build observation construction from game state
- [ ] Implement inventory synchronization
- [ ] Add GE slot tracking
- [ ] Create state reconciliation logic
- [ ] Handle edge cases (partial fills, timeouts)

### Phase 3: Inferencing Engine
- [ ] Integrate PPO agent for decision making
- [ ] Build action translation layer
- [ ] Implement confidence thresholds
- [ ] Add decision logging and audit trail
- [ ] Create decision explanation system

### Phase 4: Analytics & Monitoring
- [ ] Implement real-time P&L tracking
- [ ] Build performance metrics calculator
- [ ] Create risk monitoring alerts
- [ ] Add Firebase reporting integration
- [ ] Build dashboard data feeds

### Phase 5: Safety & Controls
- [ ] Implement position limits
- [ ] Add loss limits and circuit breakers
- [ ] Create emergency stop functionality
- [ ] Add trading hour restrictions
- [ ] Implement item blacklisting

### Phase 6: Event Streaming & WebSocket
- [ ] Extend GEAuto with WebSocket support for real-time events
- [ ] Build unified event stream in Python
- [ ] Connect to existing dashboard
- [ ] Add event-driven decision triggers
- [ ] Implement event replay for debugging

---

## Directory Structure

```
PPOFlipperOpus/
├── bridge/
│   ├── __init__.py
│   ├── geauto_bridge.py      # HTTP client for GEAuto
│   ├── models.py             # Bridge data models
│   └── exceptions.py         # Bridge-specific exceptions
├── inference/
│   ├── __init__.py
│   ├── inference_engine.py   # Main inferencing logic
│   ├── observation_builder.py # Builds observations
│   ├── action_translator.py  # Translates actions to orders
│   └── decision_explainer.py # Explains agent decisions
├── inventory/
│   ├── __init__.py
│   ├── inventory_manager.py  # Inventory tracking
│   ├── position_tracker.py   # Position management
│   └── bank_interface.py     # Bank operations
├── analytics/
│   ├── __init__.py
│   ├── analytics_engine.py   # Core analytics
│   ├── pnl_calculator.py     # P&L calculations
│   ├── performance_metrics.py # Performance stats
│   ├── risk_monitor.py       # Risk monitoring
│   └── firebase_reporter.py  # Firebase integration
├── streams/
│   ├── __init__.py
│   ├── event_stream.py       # Unified event stream
│   ├── event_types.py        # Event definitions
│   └── handlers.py           # Event handlers
├── controller/
│   ├── __init__.py
│   ├── trading_controller.py # Main controller
│   ├── safety_controls.py    # Safety mechanisms
│   └── scheduler.py          # Trading schedule
├── config/
│   ├── inference_config.yaml # Main config
│   └── risk_limits.yaml      # Risk configuration
└── scripts/
    ├── run_trading.py        # Main entry point
    ├── backtest.py           # Backtesting with GEAuto sim
    └── monitor.py            # Live monitoring dashboard
```

---

## Key Integration Points

### 1. GEAuto Plugin Modifications Needed

**New Endpoints:**
```java
// Already exists - just need to use them:
/buy, /sell, /cancel, /collect, /orders, /status, /health

// New endpoint for decision requests:
POST /decision-request
{
    "context": "NEW_ACTION",
    "pending_action": {...},
    "pending_orders": [...],
    "active_orders": [...],
    "slots_with_items": [0, 2, 5],
    "free_slots": [1, 3, 4, 6, 7]
}

// Response with decision:
{
    "type": "PROCEED" | "SKIP" | "CANCEL" | "COLLECT_SLOT" | "WAIT" | "CUSTOM",
    "target_slot": 2,
    "reason": "Collecting completed buy order",
    "custom_action": {...}  // if type is CUSTOM
}
```

**WebSocket Event Stream (Enhancement):**
```java
// Add WebSocket endpoint for real-time events
ws://localhost:9697/events

// Events pushed in real-time:
{
    "type": "ORDER_PLACED",
    "timestamp": 1703680000000,
    "slot": 0,
    "order": {...},
    "message": "Buy order placed for Cannonball x1000 at 200gp"
}
```

### 2. PPO Agent Observation Space Extension

Current observation space + new fields:
```python
# Add to observation
{
    # Existing fields...

    # New GE-specific fields
    "ge_slots": [slot_state for 8 slots],  # 0=empty, 1=buy_active, 2=sell_active, 3=needs_collect
    "ge_slot_items": [item_id for 8 slots],
    "ge_slot_progress": [fill_percentage for 8 slots],
    "pending_order_count": int,
    "time_since_last_trade": int,  # seconds
}
```

### 3. Action Space Translation

```python
# PPO Action -> GEAuto Command
def translate_action(action: AgentAction) -> GEAutoCommand:
    if action.type == ActionType.BUY:
        return BuyCommand(
            item=action.item,
            quantity=action.quantity,
            price=action.price
        )
    elif action.type == ActionType.SELL:
        return SellCommand(...)
    elif action.type == ActionType.HOLD:
        return NoOpCommand()
    elif action.type == ActionType.COLLECT:
        return CollectCommand(slot=find_collectible_slot())
```

---

## Safety Mechanisms

### Circuit Breakers
1. **Loss Limit**: Stop trading if daily loss exceeds threshold
2. **Rapid Loss**: Pause if 3 consecutive losses in 10 minutes
3. **Connection Loss**: Graceful degradation if GEAuto disconnects
4. **Market Anomaly**: Pause if prices move >20% in 5 minutes

### Validation Checks
1. **Pre-Trade**: Verify sufficient GP, inventory space, GE slots
2. **Post-Trade**: Verify order was placed correctly
3. **Reconciliation**: Periodic sync between PPO state and game state

### Monitoring Alerts
1. **Slack/Discord**: Critical errors, daily summaries
2. **Dashboard**: Real-time performance, active positions
3. **Firebase**: Historical analytics, performance reports

---

## Testing Strategy

### Unit Tests
- Bridge HTTP client mock tests
- Observation builder tests
- Action translator tests
- Analytics calculation tests

### Integration Tests
- GEAuto simulator mode testing
- End-to-end order flow testing
- Event stream handling tests

### Backtesting
- Historical price data replay
- Compare PPO decisions with optimal hindsight
- Risk scenario simulation

---

## Questions for You

1. **GEAuto Source**: Do you have the Java source files for GEAuto, or only the compiled classes? We may need to add the WebSocket event stream.

2. **Real Trading Priority**: Which features are most critical for your first live trading session?

3. **Risk Tolerance**: What are your maximum acceptable losses (per trade, per day, total)?

4. **Dashboard Integration**: Should this integrate with the existing React dashboard, or create a new one?

5. **Firebase Schema**: Do you have an existing Firebase schema for the analytics data?

---

## Next Steps

1. **Approve this proposal** or request modifications
2. **Locate GEAuto source files** if modifications are needed
3. **Begin Phase 1 implementation** of the GEAuto Bridge
4. **Set up testing environment** with GEAuto in simulator mode

---

*Proposal Version: 1.0*
*Date: 2025-12-27*
