# Quick Reference Card

## 🚀 Start Dashboard

```batch
cd dashboard
start.bat
```

Then open: **http://localhost:5173**

## 📊 Training Controls

| Action | Button | When Available |
|--------|--------|----------------|
| Start  | Green  | When idle |
| Pause  | Orange | When running |
| Resume | Blue   | When paused |
| Stop   | Red    | When running/paused |

## 📈 What You'll See

### Global Metrics
- Total Episodes & Steps
- Average & Best Rewards  
- Policy Loss
- Elapsed Time

### Per Agent (4 agents)
- Cash & Portfolio Value
- Episode & Total Rewards
- Trade Stats & Win Rate
- Current Holdings
- Taxes Paid

### Charts
- Reward History (by episode)
- Portfolio Values (over time)

### Recent Trades
- Last 500 trades
- Type, Item, Price, Quantity
- Profit/Loss, Tax

## ⚙️ Configuration

Edit `ppo_config.py`:
```python
ENV_KWARGS = {'starting_gp': 10_000_000}
PPO_KWARGS = {'lr': 3e-4, 'device': 'cuda'}
TRAIN_KWARGS = {'num_agents': 4}
```

## 🔍 Health Check

```batch
curl http://localhost:8000/api/health
```

## 🧪 Test Backend

```batch
python dashboard\test_backend.py
```

## 📝 Logs

- Backend: `dashboard/backend/backend.log`
- Browser: F12 → Console

## 🆘 Common Issues

**Database not found?**
```batch
cd data
python collect_all.py
```

**Port 8000 in use?**
```batch
netstat -ano | findstr :8000
# Kill process or change port
```

**Frontend won't connect?**
- Check backend is running
- Verify http://localhost:8000/api/health
- Check browser console for errors

## 📦 Dependencies

Backend:
- fastapi, uvicorn, websockets
- torch, numpy

Frontend:
- Vue 3, Chart.js, vue-chartjs

## 🎯 Similar To

**Kohya SS** - Stable Diffusion training UI
- Start/stop/pause controls ✓
- Real-time metrics ✓
- Charts ✓
- Configuration ✓

**Plus our additions:**
- Multi-agent view
- Trade logging
- Portfolio tracking
- Sub-second WebSocket updates
