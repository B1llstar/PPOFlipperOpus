# PPO Flipper Dashboard - Visual Guide

## System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        USER BROWSER                             │
│                    http://localhost:5173                        │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                    Vue 3 Frontend                         │ │
│  │                                                           │ │
│  │  [Start] [Pause] [Resume] [Stop]  <-- Control Buttons   │ │
│  │                                                           │ │
│  │  Global Metrics:                                         │ │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                  │ │
│  │  │Episodes│Steps │Avg Rwd│Best  │                      │ │
│  │  └──────┘ └──────┘ └──────┘ └──────┘                  │ │
│  │                                                           │ │
│  │  Charts:                                                 │ │
│  │  ┌─────────────────┐ ┌─────────────────┐              │ │
│  │  │ Reward History  │ │ Portfolio Values│              │ │
│  │  │     📈          │ │      📊         │              │ │
│  │  └─────────────────┘ └─────────────────┘              │ │
│  │                                                           │ │
│  │  Agents:                                                 │ │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐         │ │
│  │  │Agent 1 │ │Agent 2 │ │Agent 3 │ │Agent 4 │         │ │
│  │  │💰 Cash │ │💰 Cash │ │💰 Cash │ │💰 Cash │         │ │
│  │  │📊 Stats│ │📊 Stats│ │📊 Stats│ │📊 Stats│         │ │
│  │  └────────┘ └────────┘ └────────┘ └────────┘         │ │
│  │                                                           │ │
│  │  Recent Trades:                                          │ │
│  │  Time    Agent  Type  Item      Price     Profit        │ │
│  │  12:30   #1     BUY   Dragon    2,500 GP  -             │ │
│  │  12:35   #1     SELL  Dragon    2,800 GP  +280 GP      │ │
│  │  ...                                                     │ │
│  └──────────────────────────────────────────────────────────┘ │
└──────────────┬──────────────────────────────┬────────────────┘
               │ WebSocket (updates)          │ REST (control)
               │ ws://localhost:8000/ws       │ POST /api/training/*
               │                              │
┌──────────────┴──────────────────────────────┴────────────────┐
│                    FastAPI Backend                            │
│                  http://localhost:8000                        │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              TrainingManager                             │ │
│  │  - Handles REST API requests                            │ │
│  │  - Broadcasts WebSocket updates                         │ │
│  │  - Manages agent metrics                                │ │
│  │  - Maintains history buffers                            │ │
│  └──────────────────────┬──────────────────────────────────┘ │
│                         │ controls                            │
│  ┌──────────────────────┴──────────────────────────────────┐ │
│  │           TrainingController                             │ │
│  │  - Wraps train_ppo.py logic                             │ │
│  │  - Provides start/stop/pause/resume                     │ │
│  │  - Emits training updates                               │ │
│  │  - Runs in separate thread                              │ │
│  └──────────────────────┬──────────────────────────────────┘ │
│                         │ uses                                │
│  ┌──────────────────────┴──────────────────────────────────┐ │
│  │          GrandExchangeEnv (x4 agents)                    │ │
│  │  - Simulates GE trading environment                     │ │
│  │  - Reads from ge_prices.db                              │ │
│  │  - Tracks positions, cash, trades                       │ │
│  └──────────────────────┬──────────────────────────────────┘ │
│                         │                                     │
│  ┌──────────────────────┴──────────────────────────────────┐ │
│  │              PPOAgent (x4 agents)                        │ │
│  │  - Actor-Critic networks                                │ │
│  │  - Learns trading policy                                │ │
│  │  - Makes buy/sell/hold decisions                        │ │
│  └─────────────────────────────────────────────────────────┘ │
└────────────────────────┬──────────────────────────────────────┘
                         │ reads market data
┌────────────────────────┴──────────────────────────────────────┐
│                      ge_prices.db                              │
│                      (SQLite Database)                         │
│                                                                │
│  Tables:                                                       │
│  - timeseries: Historical price data (5-min intervals)        │
│  - items: Item metadata (names, limits, etc.)                 │
│  - market_snapshots: Market state at each timestamp           │
│                                                                │
│  Sample data:                                                  │
│  item_id  timestamp          high_price  low_price  volume    │
│  2        2024-01-01 12:00   2800        2500        1500     │
│  554      2024-01-01 12:00   150         145         25000    │
│  ...                                                           │
└────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Training Start
```
User clicks "Start" 
  → Frontend: POST /api/training/start
    → Backend: TrainingManager.start_training()
      → TrainingController.start()
        → Creates 4x GrandExchangeEnv (reads ge_prices.db)
        → Creates 4x PPOAgent
        → Spawns training thread
        → Training loop begins
          → For each agent:
            → Get observation from env
            → Agent selects action
            → Env executes trade
            → Calculate reward
            → Update agent
          → Emit updates via callback
            → TrainingManager accumulates metrics
              → WebSocket broadcasts to frontend
                → Frontend updates UI
```

### Real-time Updates
```
Every 500ms:
  Backend → WebSocket message → Frontend
  
Message contains:
  - Training state (idle/running/paused/stopping)
  - Global metrics (episodes, steps, rewards, losses)
  - Agent metrics (cash, portfolio, trades, win rate)
  - History data (reward points, portfolio values)
  - Recent trades (last 500)
```

## Training States

```
    ┌──────┐
    │ IDLE │  <-- Initial state
    └───┬──┘
        │
        │ [Start Training]
        ↓
   ┌─────────┐
   │ RUNNING │  <-- Training active
   └─┬───┬───┘
     │   │
     │   │ [Pause Training]
     │   ↓
     │ ┌────────┐
     │ │ PAUSED │  <-- Training paused (can resume)
     │ └───┬────┘
     │     │
     │     │ [Resume Training]
     │     ↓
     │   back to RUNNING
     │
     │ [Stop Training]
     ↓
  ┌──────────┐
  │ STOPPING │  <-- Cleanup in progress
  └────┬─────┘
       │
       │ (cleanup complete)
       ↓
     back to IDLE
```

## Agent Card Layout

```
┌─────────────────────────────────────────────┐
│ Agent 1                    Ep 42 / Step 156 │ <-- Header
├─────────────────────────────────────────────┤
│ Cash            Portfolio         Total      │
│ 💰 10.5M GP     📊 250K GP       ✅ 10.75M  │ <-- Assets
├─────────────────────────────────────────────┤
│ Episode Reward           Total Reward        │
│ +125.5                   +1,234.8           │ <-- Rewards
├─────────────────────────────────────────────┤
│ Trades  Profitable  Win Rate    Taxes       │
│ 15      12          80%         -2,500 GP   │ <-- Stats
├─────────────────────────────────────────────┤
│ Current Action: Evaluating Dragon bones     │ <-- Action
├─────────────────────────────────────────────┤
│ Holdings:                                    │
│ • Dragon bones    x 150                     │
│ • Nature rune     x 2,500                   │
│ • Cannonball      x 800                     │ <-- Holdings
└─────────────────────────────────────────────┘
```

## Chart Types

### Reward History
```
Y-axis: Reward
X-axis: Episode

   ↑ Reward
400│                              ╱─╲
300│                      ╱──╲ ╱     ╲
200│              ╱──╲ ╱        ╲   ╱
100│      ╱──╲ ╱       ╲          ╲╱
  0│ ─────────────────────────────────→ Episode
     0   10   20   30   40   50   60

Legend:
─ Agent 1 (Green)
─ Agent 2 (Blue)
─ Agent 3 (Orange)
─ Agent 4 (Pink)
```

### Portfolio Values
```
Y-axis: GP Value
X-axis: Episode

   ↑ Value (GP)
11M│                          ╱────
10M│          ╱──────────────
 9M│  ────────
     0   10   20   30   40   50   60 → Episode

All 4 agents shown with different colors
```

## API Flow Diagram

```
┌─────────┐                    ┌─────────┐
│ Browser │                    │ Backend │
└────┬────┘                    └────┬────┘
     │                              │
     │  POST /api/training/start    │
     ├─────────────────────────────>│
     │                              │ Start training thread
     │                              ├─────────────────────┐
     │                              │                     │
     │  {status: "ok"}              │<────────────────────┘
     │<─────────────────────────────┤
     │                              │
     │  WS Connect /ws              │
     ├─────────────────────────────>│
     │                              │
     │  {type: "init", data: {...}} │
     │<─────────────────────────────┤
     │                              │
     │         Every 500ms          │
     │  {type: "update", data: {...}}
     │<─────────────────────────────┤
     │                              │
     │  POST /api/training/pause    │
     ├─────────────────────────────>│
     │                              │ Pause training
     │  {status: "ok"}              │
     │<─────────────────────────────┤
     │                              │
     │  POST /api/training/resume   │
     ├─────────────────────────────>│
     │                              │ Resume training
     │  {status: "ok"}              │
     │<─────────────────────────────┤
     │                              │
     │  POST /api/training/stop     │
     ├─────────────────────────────>│
     │                              │ Stop training
     │  {status: "ok"}              │
     │<─────────────────────────────┤
     │                              │
```

## File Structure

```
PPOFlipperOpus/
├── ge_prices.db                   <-- Market data (SQLite)
├── ppo_config.py                  <-- Training config
├── ppo_agent.py                   <-- PPO agent
│
├── training/
│   ├── ge_environment.py          <-- Trading environment
│   ├── train_ppo.py               <-- Original training script
│   └── training_controller.py     <-- NEW: Control wrapper
│
└── dashboard/
    ├── start.bat                  <-- Quick start (Windows)
    ├── start.sh                   <-- Quick start (Linux/Mac)
    ├── test_backend.py            <-- Backend tests
    ├── DASHBOARD_README.md        <-- Full documentation
    ├── SETUP_SUMMARY.md           <-- This summary
    └── QUICK_REFERENCE.md         <-- Quick ref card
    │
    ├── backend/
    │   ├── server.py              <-- MODIFIED: FastAPI + training
    │   ├── requirements.txt       <-- Python deps
    │   └── backend.log            <-- Runtime logs
    │
    └── frontend/
        ├── package.json           <-- Node deps
        ├── vite.config.js         <-- Vite config
        └── src/
            ├── App.vue            <-- Main dashboard UI
            ├── main.js            <-- Vue entry point
            └── components/
                └── AgentCard.vue  <-- Agent display
```

## Color Scheme

```
Background:     #1a1a2e (Dark blue-grey)
Cards:          #16213e (Darker blue)
Borders:        #2a3f5f (Light blue-grey)
Text:           #eee (Light grey)

Status Colors:
- Idle:         #555 (Grey)
- Running:      #4CAF50 (Green)
- Paused:       #FF9800 (Orange)
- Stopping:     #f44336 (Red)

Value Colors:
- Cash:         #FFD700 (Gold)
- Portfolio:    #fff (White)
- Total Assets: #4CAF50 (Green)
- Profit:       #4CAF50 (Green)
- Loss:         #f44336 (Red)
- Tax:          #f44336 (Red)

Chart Colors:
- Agent 1:      #4CAF50 (Green)
- Agent 2:      #2196F3 (Blue)
- Agent 3:      #FF9800 (Orange)
- Agent 4:      #E91E63 (Pink)
```

## Key Metrics Explained

| Metric | Description | Good Range |
|--------|-------------|------------|
| Episode Reward | Profit/loss in current episode | > 0 |
| Total Reward | Cumulative reward across all episodes | Increasing |
| Win Rate | % of profitable trades | > 50% |
| Policy Loss | How well agent learns actions | Decreasing |
| Value Loss | How well agent estimates values | Decreasing |
| Avg Reward | Average across recent episodes | Increasing |

## Comparison: Before & After

### Before (Command Line)
```
$ python training/train_ppo.py
Agent 0 | Episode 1 | Step 100 | GP: 10500000
Agent 0 | Episode 1 | Step 200 | GP: 10650000
...
^C  <-- Manual interrupt needed
```

### After (Dashboard)
```
Browser: http://localhost:5173

┌────────────────────────────────────────┐
│ PPO Flipper Dashboard        RUNNING   │
│                                         │
│ [Start] [Pause] [Resume] [Stop]        │
│                                         │
│ ┌──────┬──────┬──────┬──────┐         │
│ │ Eps  │Steps │ Avg  │ Best │         │
│ │  42  │7,056 │125.5 │450.2 │         │
│ └──────┴──────┴──────┴──────┘         │
│                                         │
│ [Reward Chart] [Portfolio Chart]       │
│                                         │
│ [Agent 1] [Agent 2] [Agent 3] [Agent 4]│
│                                         │
│ Recent Trades...                        │
└────────────────────────────────────────┘

Click to pause, resume, or stop anytime!
```

## Success! ✨

You now have a **Kohya SS-style dashboard** for PPO training with:
- ✅ Visual training control
- ✅ Real-time monitoring
- ✅ Multi-agent view
- ✅ Interactive charts
- ✅ Trade logging
- ✅ WebSocket updates
- ✅ Database integration (ge_prices.db)
