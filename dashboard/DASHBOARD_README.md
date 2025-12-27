# PPO Flipper Training Dashboard

A web-based dashboard for controlling and monitoring PPO agent training in real-time, similar to Kohya SS.

## Features

- **Real-time Training Control**: Start, stop, pause, and resume training from the web interface
- **Live Statistics**: Monitor training metrics including rewards, losses, and portfolio values in real-time
- **Multi-Agent Monitoring**: Track up to 4 agents simultaneously with individual metrics
- **Interactive Charts**: Visualize reward history and portfolio performance over time
- **Trade Logging**: View recent trades with profit/loss information
- **WebSocket Updates**: Sub-second latency updates for smooth monitoring

## Architecture

### Backend (`dashboard/backend/`)
- **FastAPI Server** (`server.py`): REST API and WebSocket server
- **Training Controller** (`training/training_controller.py`): Wraps the PPO training loop with external control
- **Database**: Reads market data from `ge_prices.db` in the project root

### Frontend (`dashboard/frontend/`)
- **Vue 3**: Reactive UI framework
- **Chart.js**: Real-time charting
- **WebSocket Client**: Bidirectional communication with backend

## Setup

### Backend Setup

1. Navigate to the backend directory:
```bash
cd dashboard/backend
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure `ge_prices.db` exists in the project root:
```bash
# From project root
ls ge_prices.db
```

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd dashboard/frontend
```

2. Install Node.js dependencies:
```bash
npm install
```

## Running the Dashboard

### Start the Backend

From the `dashboard/backend/` directory:

```bash
python server.py
```

The backend will start on `http://localhost:8000`

Or use uvicorn directly:
```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

### Start the Frontend

From the `dashboard/frontend/` directory:

```bash
npm run dev
```

The frontend will start on `http://localhost:5173` (or similar Vite port)

### Access the Dashboard

Open your browser and navigate to:
```
http://localhost:5173
```

## Usage

### Training Controls

- **Start Training**: Click the green "Start Training" button to begin training
- **Pause Training**: Click the orange "Pause" button to temporarily halt training
- **Resume Training**: Click the blue "Resume" button to continue paused training
- **Stop Training**: Click the red "Stop" button to completely stop training

### Monitoring

#### Global Metrics
- **Total Episodes**: Number of episodes completed across all agents
- **Total Steps**: Total training steps taken
- **Avg Reward**: Average reward per episode
- **Best Reward**: Highest reward achieved
- **Policy Loss**: Current policy network loss
- **Elapsed Time**: Training duration

#### Agent Cards
Each agent displays:
- Cash balance and portfolio value
- Episode and total rewards
- Trade statistics (total trades, profitable trades, win rate)
- Current holdings
- Taxes paid

#### Charts
- **Reward History**: Track reward progression over episodes
- **Portfolio Values**: Monitor asset values for each agent

#### Recent Trades
View the last 500 trades with:
- Timestamp
- Agent ID
- Trade type (BUY/SELL)
- Item, price, quantity
- Profit/loss and tax

## API Endpoints

### REST API

- `GET /` - Health check
- `GET /api/state` - Get complete training state
- `GET /api/config` - Get training configuration
- `POST /api/training/start` - Start training
- `POST /api/training/stop` - Stop training
- `POST /api/training/pause` - Pause training
- `POST /api/training/resume` - Resume training

### WebSocket

- `WS /ws` - Real-time state updates (500ms intervals)

Example WebSocket message:
```json
{
  "type": "update",
  "data": {
    "training": {
      "state": "running",
      "total_episodes": 42,
      "total_steps": 7056,
      "avg_reward": 125.5,
      "best_reward": 450.2
    },
    "agents": {
      "0": {
        "agent_id": 0,
        "cash": 10500000,
        "portfolio_value": 250000,
        "total_assets": 10750000,
        "trades_executed": 15,
        "profitable_trades": 12
      }
    }
  }
}
```

## Configuration

### Training Parameters

Edit `ppo_config.py` in the project root to adjust:

**Environment Settings (`ENV_KWARGS`)**:
```python
ENV_KWARGS = {
    'starting_gp': 10_000_000,    # Initial cash
    'max_steps': 168,              # Episode length
    'top_n_items': 50,             # Number of tradeable items
}
```

**PPO Settings (`PPO_KWARGS`)**:
```python
PPO_KWARGS = {
    'device': 'cuda',              # 'cuda' or 'cpu'
    'hidden_size': 256,
    'num_layers': 3,
    'lr': 3e-4,
    'gamma': 0.99,
    'clip_epsilon': 0.2,
}
```

**Training Settings (`TRAIN_KWARGS`)**:
```python
TRAIN_KWARGS = {
    'num_agents': 4,               # Number of parallel agents
}
```

### Database Configuration

The backend automatically looks for `ge_prices.db` in the project root. To use a different location, modify `server.py`:

```python
db_path = os.path.join(project_root, 'your_database.db')
```

## Development

### Backend Development

The backend uses FastAPI with automatic reloading:
```bash
uvicorn server:app --reload
```

### Frontend Development

Vite provides hot module replacement:
```bash
npm run dev
```

Changes to Vue components will reflect immediately.

### Testing the Training Controller

Test the training controller independently:
```bash
cd training
python training_controller.py
```

This will run a 20-second test with pause/resume functionality.

## Troubleshooting

### Database Not Found

**Error**: `Database not found at .../ge_prices.db`

**Solution**: Ensure the database exists:
```bash
# From project root
ls -la ge_prices.db
```

If missing, collect data first:
```bash
python data/collect_all.py
```

### WebSocket Connection Failed

**Error**: Frontend can't connect to WebSocket

**Solution**: 
1. Ensure backend is running on port 8000
2. Check CORS settings in `server.py`
3. Verify `WS_URL` in `App.vue` matches backend address

### Training Not Starting

**Error**: Training button clicks do nothing

**Solutions**:
1. Check browser console for errors
2. Verify backend logs for errors
3. Ensure `ge_prices.db` has market data
4. Check that PyTorch is properly installed

### Charts Not Displaying

**Error**: Charts show "No data yet"

**Solution**: 
1. Start training first
2. Wait for at least one episode to complete
3. Check WebSocket connection in browser console

## Architecture Details

### Training Flow

1. User clicks "Start Training" in frontend
2. Frontend sends POST to `/api/training/start`
3. Backend creates `TrainingController` instance
4. Controller initializes environments and agents using `ge_prices.db`
5. Training loop runs in separate thread
6. Controller emits updates via callback
7. Backend accumulates metrics
8. WebSocket broadcasts state to all connected clients
9. Frontend updates UI reactively

### State Management

The backend maintains:
- **TrainingState**: Current state (idle/running/paused/stopping)
- **AgentMetrics**: Per-agent statistics (4 agents by default)
- **TrainingMetrics**: Global metrics
- **History Buffers**: Rolling windows of rewards, portfolios, losses (1000 points)
- **Trade Log**: Recent trades (500 trades)

All updates are thread-safe using locks.

## Performance

- **Update Frequency**: 500ms WebSocket updates
- **Step Performance**: ~100 steps/second (depends on hardware)
- **Memory Usage**: ~2GB RAM for 4 agents
- **Database**: SQLite reads are cached by environment

## Similar Projects

This dashboard is inspired by:
- **Kohya SS**: Stable Diffusion training UI
- **TensorBoard**: ML experiment tracking
- **Weights & Biases**: Experiment monitoring

## License

Same as the parent PPOFlipperOpus project.
