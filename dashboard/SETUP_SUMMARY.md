# Dashboard Setup Summary

## What Was Done

### ✅ Backend Integration
1. **Created Training Controller** (`training/training_controller.py`)
   - Wraps the PPO training logic from `train_ppo.py`
   - Provides start/stop/pause/resume functionality
   - Emits real-time updates via callbacks
   - Uses `ge_prices.db` for market data

2. **Updated Backend Server** (`dashboard/backend/server.py`)
   - Integrated the real training controller
   - Already had REST API endpoints for training control
   - Already had WebSocket support for real-time updates
   - Added health check endpoint to verify database connection
   - Confirmed database path: reads from `ge_prices.db` in project root

3. **Confirmed Database Usage**
   - Backend correctly reads from `e:\programmingfiles\ppoflipperopus\PPOFlipperOpus\ge_prices.db`
   - Path is calculated relative to server location

### ✅ Frontend Features (Already Implemented)
1. **Training Controls**
   - Start/Stop/Pause/Resume buttons
   - Real-time status badge (idle/running/paused/stopping)

2. **Live Statistics Display**
   - Total episodes and steps
   - Average and best rewards
   - Policy loss and elapsed time

3. **Agent Monitoring**
   - Individual agent cards showing:
     - Cash and portfolio values
     - Episode and total rewards
     - Trade statistics and win rates
     - Current holdings
     - Taxes paid

4. **Interactive Charts**
   - Reward history over episodes (multi-agent)
   - Portfolio values over time

5. **Trade Logging**
   - Recent 500 trades
   - Shows agent, type, item, price, quantity, profit, tax

6. **WebSocket Integration**
   - Real-time updates every 500ms
   - Automatic reconnection on disconnect

## How to Use

### Quick Start (Windows)

```batch
cd dashboard
start.bat
```

This will:
1. Check for `ge_prices.db`
2. Install missing dependencies
3. Start backend on http://localhost:8000
4. Start frontend on http://localhost:5173

### Manual Start

**Terminal 1 - Backend:**
```batch
cd dashboard\backend
python server.py
```

**Terminal 2 - Frontend:**
```batch
cd dashboard\frontend
npm run dev
```

**Terminal 3 - Test (optional):**
```batch
cd dashboard
python test_backend.py
```

### Access Dashboard

Open your browser to: **http://localhost:5173**

## Training Control Flow

1. **Start Training**
   - Click "Start Training" button
   - Backend creates environments from `ge_prices.db`
   - Initializes PPO agents with configured parameters
   - Training runs in background thread
   - Updates stream via WebSocket

2. **Monitor Progress**
   - Watch global metrics update in real-time
   - Track individual agent performance
   - View reward and portfolio charts
   - See recent trades as they execute

3. **Pause/Resume**
   - Pause to inspect current state
   - Resume to continue from same point
   - No data loss during pause

4. **Stop Training**
   - Gracefully stops all agents
   - Saves final state
   - Returns to idle

## Configuration

Edit `ppo_config.py` in project root:

```python
# Environment settings
ENV_KWARGS = {
    'starting_gp': 10_000_000,  # Starting cash
    'max_steps': 168,            # Episode length
    'top_n_items': 50,           # Number of items
}

# PPO settings
PPO_KWARGS = {
    'device': 'cuda',            # or 'cpu'
    'hidden_size': 256,
    'lr': 3e-4,
}

# Training settings
TRAIN_KWARGS = {
    'num_agents': 4,             # Parallel agents
}
```

## API Reference

### REST Endpoints

- `GET /api/health` - Health check with database status
- `GET /api/state` - Complete training state
- `GET /api/config` - Training configuration
- `POST /api/training/start` - Start training
- `POST /api/training/stop` - Stop training
- `POST /api/training/pause` - Pause training
- `POST /api/training/resume` - Resume training

### WebSocket

- `WS /ws` - Real-time updates (500ms interval)

Example message:
```json
{
  "type": "update",
  "data": {
    "training": {
      "state": "running",
      "total_episodes": 42,
      "total_steps": 7056
    },
    "agents": {
      "0": { "cash": 10500000, "trades_executed": 15 }
    },
    "history": {
      "rewards": [...],
      "portfolios": [...]
    }
  }
}
```

## Testing

Run the backend test script:
```batch
python dashboard\test_backend.py
```

This will verify:
- Database connection
- API endpoints
- Training control
- WebSocket functionality

## Troubleshooting

### Database Not Found
```
❌ Error: Database not found at .../ge_prices.db
```

**Solution:** Collect data first:
```batch
cd data
python collect_all.py
```

### Backend Won't Start
Check if port 8000 is already in use:
```batch
netstat -ano | findstr :8000
```

### Frontend Won't Connect
1. Verify backend is running: http://localhost:8000/api/health
2. Check WebSocket URL in `App.vue` matches backend
3. Look for CORS errors in browser console

### Training Not Starting
1. Check backend logs for errors
2. Verify `ge_prices.db` has data
3. Ensure PyTorch is installed correctly
4. Check GPU availability if using CUDA

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   Browser                        │
│  (Vue 3 + Chart.js + WebSocket)                 │
│  http://localhost:5173                           │
└────────────────┬────────────────────────────────┘
                 │ WebSocket (500ms updates)
                 │ REST API (control commands)
┌────────────────┴────────────────────────────────┐
│            FastAPI Backend                       │
│  (server.py + training_controller.py)           │
│  http://localhost:8000                           │
└────────────────┬────────────────────────────────┘
                 │ Reads market data
┌────────────────┴────────────────────────────────┐
│          ge_prices.db (SQLite)                   │
│  (Market history, prices, metadata)              │
└──────────────────────────────────────────────────┘
```

## Key Files Modified/Created

1. **Created:**
   - `training/training_controller.py` - Training control wrapper
   - `dashboard/DASHBOARD_README.md` - Detailed documentation
   - `dashboard/test_backend.py` - Backend testing script
   - `dashboard/start.bat` - Windows quick start

2. **Modified:**
   - `dashboard/backend/server.py` - Integrated real training
     - Added training controller initialization
     - Connected training callbacks
     - Updated control methods
     - Added health check endpoint

3. **Already Working:**
   - `dashboard/frontend/src/App.vue` - Main dashboard UI
   - `dashboard/frontend/src/components/AgentCard.vue` - Agent display
   - `dashboard/backend/server.py` - API and WebSocket server

## Comparison to Kohya SS

Similar features:
- ✅ Start/stop/pause training from UI
- ✅ Real-time metrics display
- ✅ Interactive charts
- ✅ Parameter configuration
- ✅ Multi-process training support

Our additions:
- Multi-agent monitoring (4 agents)
- Individual agent cards
- Trade logging with profit/loss
- Portfolio value tracking
- WebSocket for sub-second updates

## Next Steps (Optional Enhancements)

1. **Model Checkpointing UI**
   - Save/load model buttons
   - Checkpoint management

2. **Advanced Charts**
   - Loss curves (policy, value, entropy)
   - Learning rate schedule
   - Action distribution

3. **Configuration UI**
   - Edit PPO parameters from frontend
   - Dynamic agent count
   - Training presets

4. **Experiment Tracking**
   - Save training runs
   - Compare experiments
   - Export metrics (CSV, JSON)

5. **Live Training Logs**
   - Stream training logs to frontend
   - Filter by agent or severity
   - Download logs

## Support

For issues:
1. Check backend logs: `dashboard/backend/backend.log`
2. Check browser console for frontend errors
3. Run test script: `python dashboard/test_backend.py`
4. Verify database: `ls -la ge_prices.db`

## Success Criteria ✓

- [x] Backend reads from `ge_prices.db`
- [x] Frontend can start training
- [x] Frontend can pause/resume training
- [x] Frontend can stop training
- [x] Real-time statistics display
- [x] Multi-agent monitoring
- [x] Interactive charts
- [x] Trade logging
- [x] WebSocket integration
- [x] Similar to Kohya SS experience

All requirements met! 🎉
