<template>
  <div class="dashboard">
    <header class="header">
      <h1>PPO Flipper Dashboard</h1>
      <div class="status-badge" :class="trainingState">
        {{ trainingState.toUpperCase() }}
      </div>
    </header>

    <div class="controls">
      <button
        @click="startTraining"
        :disabled="trainingState !== 'idle'"
        class="btn btn-start"
      >
        Start Training
      </button>
      <button
        @click="pauseTraining"
        :disabled="trainingState !== 'running'"
        class="btn btn-pause"
      >
        Pause
      </button>
      <button
        @click="resumeTraining"
        :disabled="trainingState !== 'paused'"
        class="btn btn-resume"
      >
        Resume
      </button>
      <button
        @click="stopTraining"
        :disabled="trainingState === 'idle' || trainingState === 'stopping'"
        class="btn btn-stop"
      >
        Stop
      </button>
    </div>

    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-label">Total Episodes</div>
        <div class="metric-value">{{ training.total_episodes.toLocaleString() }}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Total Steps</div>
        <div class="metric-value">{{ training.total_steps.toLocaleString() }}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Avg Reward</div>
        <div class="metric-value">{{ training.avg_reward.toFixed(2) }}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Best Reward</div>
        <div class="metric-value">{{ training.best_reward.toFixed(2) }}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Policy Loss</div>
        <div class="metric-value">{{ training.policy_loss.toFixed(4) }}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Elapsed Time</div>
        <div class="metric-value">{{ formatTime(training.elapsed_seconds) }}</div>
      </div>
    </div>

    <div class="charts-section">
      <div class="chart-container">
        <h3>Reward History</h3>
        <Line v-if="rewardChartData.labels.length > 0" :data="rewardChartData" :options="chartOptions" />
        <div v-else class="no-data">No data yet</div>
      </div>
      <div class="chart-container">
        <h3>Portfolio Values</h3>
        <Line v-if="portfolioChartData.labels.length > 0" :data="portfolioChartData" :options="chartOptions" />
        <div v-else class="no-data">No data yet</div>
      </div>
    </div>

    <div class="agents-section">
      <h2>Agents</h2>
      <div class="agents-grid">
        <AgentCard
          v-for="(agent, id) in agents"
          :key="id"
          :agent="agent"
        />
      </div>
    </div>

    <div class="trades-section">
      <h2>Recent Trades</h2>
      <div class="trades-table-wrapper">
        <table class="trades-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Agent</th>
              <th>Type</th>
              <th>Item</th>
              <th>Price</th>
              <th>Qty</th>
              <th>Profit</th>
              <th>Tax</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(trade, idx) in recentTrades.slice().reverse()" :key="idx" :class="trade.type">
              <td>{{ formatTimestamp(trade.timestamp) }}</td>
              <td>Agent {{ trade.agent_id + 1 }}</td>
              <td :class="trade.type">{{ trade.type?.toUpperCase() }}</td>
              <td>{{ trade.item }}</td>
              <td>{{ trade.price?.toLocaleString() }} GP</td>
              <td>{{ trade.quantity?.toLocaleString() }}</td>
              <td :class="{ profit: trade.profit > 0, loss: trade.profit < 0 }">
                {{ trade.profit?.toLocaleString() }} GP
              </td>
              <td>{{ trade.tax?.toLocaleString() }} GP</td>
            </tr>
            <tr v-if="recentTrades.length === 0">
              <td colspan="8" class="no-trades">No trades yet</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { Line } from 'vue-chartjs'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js'
import AgentCard from './components/AgentCard.vue'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
)

const API_URL = 'http://localhost:8000'
const WS_URL = 'ws://localhost:8000/ws'

// State
const training = ref({
  state: 'idle',
  total_episodes: 0,
  total_steps: 0,
  avg_reward: 0,
  best_reward: 0,
  policy_loss: 0,
  value_loss: 0,
  entropy: 0,
  elapsed_seconds: 0,
})

const agents = ref({})
const rewardHistory = ref([])
const portfolioHistory = ref([])
const recentTrades = ref([])

let ws = null

const trainingState = computed(() => training.value.state)

// Chart data
const rewardChartData = computed(() => {
  const agentColors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63']
  const datasets = []

  // Group by agent
  const agentData = {}
  rewardHistory.value.forEach(point => {
    if (!agentData[point.agent_id]) {
      agentData[point.agent_id] = []
    }
    agentData[point.agent_id].push(point)
  })

  Object.entries(agentData).forEach(([agentId, data]) => {
    datasets.push({
      label: `Agent ${parseInt(agentId) + 1}`,
      data: data.map(p => p.reward),
      borderColor: agentColors[agentId % agentColors.length],
      backgroundColor: 'transparent',
      tension: 0.4,
    })
  })

  const labels = rewardHistory.value
    .filter(p => p.agent_id === 0)
    .map(p => p.episode)

  return { labels, datasets }
})

const portfolioChartData = computed(() => {
  const agentColors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63']
  const datasets = []
  const numAgents = Object.keys(agents.value).length || 4

  for (let i = 0; i < numAgents; i++) {
    datasets.push({
      label: `Agent ${i + 1}`,
      data: portfolioHistory.value.map(p => p.values?.[i] || 0),
      borderColor: agentColors[i % agentColors.length],
      backgroundColor: 'transparent',
      tension: 0.4,
    })
  }

  const labels = portfolioHistory.value.map(p => p.episode)

  return { labels, datasets }
})

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      position: 'top',
      labels: { color: '#fff' }
    }
  },
  scales: {
    x: {
      ticks: { color: '#aaa' },
      grid: { color: '#333' }
    },
    y: {
      ticks: { color: '#aaa' },
      grid: { color: '#333' }
    }
  }
}

// WebSocket connection
function connectWebSocket() {
  ws = new WebSocket(WS_URL)

  ws.onopen = () => {
    console.log('WebSocket connected')
  }

  ws.onmessage = (event) => {
    const message = JSON.parse(event.data)
    if (message.type === 'init' || message.type === 'update') {
      updateState(message.data)
    }
  }

  ws.onclose = () => {
    console.log('WebSocket disconnected, reconnecting...')
    setTimeout(connectWebSocket, 2000)
  }

  ws.onerror = (error) => {
    console.error('WebSocket error:', error)
  }
}

function updateState(data) {
  if (data.training) {
    training.value = data.training
  }
  if (data.agents) {
    agents.value = data.agents
  }
  if (data.history) {
    rewardHistory.value = data.history.rewards || []
    portfolioHistory.value = data.history.portfolios || []
  }
  if (data.recent_trades) {
    recentTrades.value = data.recent_trades
  }
}

// API calls
async function startTraining() {
  try {
    await fetch(`${API_URL}/api/training/start`, { method: 'POST' })
  } catch (error) {
    console.error('Failed to start training:', error)
  }
}

async function stopTraining() {
  try {
    await fetch(`${API_URL}/api/training/stop`, { method: 'POST' })
  } catch (error) {
    console.error('Failed to stop training:', error)
  }
}

async function pauseTraining() {
  try {
    await fetch(`${API_URL}/api/training/pause`, { method: 'POST' })
  } catch (error) {
    console.error('Failed to pause training:', error)
  }
}

async function resumeTraining() {
  try {
    await fetch(`${API_URL}/api/training/resume`, { method: 'POST' })
  } catch (error) {
    console.error('Failed to resume training:', error)
  }
}

// Formatters
function formatTime(seconds) {
  const hrs = Math.floor(seconds / 3600)
  const mins = Math.floor((seconds % 3600) / 60)
  const secs = Math.floor(seconds % 60)
  return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
}

function formatTimestamp(ts) {
  if (!ts) return '-'
  const date = new Date(ts)
  return date.toLocaleTimeString()
}

onMounted(() => {
  connectWebSocket()
})

onUnmounted(() => {
  if (ws) {
    ws.close()
  }
})
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  background: #1a1a2e;
  color: #eee;
}

.dashboard {
  max-width: 1600px;
  margin: 0 auto;
  padding: 20px;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.header h1 {
  font-size: 28px;
  color: #fff;
}

.status-badge {
  padding: 8px 16px;
  border-radius: 20px;
  font-weight: bold;
  font-size: 14px;
}

.status-badge.idle {
  background: #555;
  color: #ccc;
}

.status-badge.running {
  background: #4CAF50;
  color: #fff;
}

.status-badge.paused {
  background: #FF9800;
  color: #fff;
}

.status-badge.stopping {
  background: #f44336;
  color: #fff;
}

.controls {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
}

.btn {
  padding: 12px 24px;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: bold;
  cursor: pointer;
  transition: all 0.2s;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-start {
  background: #4CAF50;
  color: white;
}

.btn-start:hover:not(:disabled) {
  background: #45a049;
}

.btn-pause {
  background: #FF9800;
  color: white;
}

.btn-pause:hover:not(:disabled) {
  background: #f57c00;
}

.btn-resume {
  background: #2196F3;
  color: white;
}

.btn-resume:hover:not(:disabled) {
  background: #1976D2;
}

.btn-stop {
  background: #f44336;
  color: white;
}

.btn-stop:hover:not(:disabled) {
  background: #d32f2f;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 15px;
  margin-bottom: 30px;
}

.metric-card {
  background: #16213e;
  padding: 20px;
  border-radius: 12px;
  text-align: center;
}

.metric-label {
  font-size: 12px;
  color: #888;
  margin-bottom: 8px;
  text-transform: uppercase;
}

.metric-value {
  font-size: 24px;
  font-weight: bold;
  color: #4CAF50;
}

.charts-section {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  margin-bottom: 30px;
}

.chart-container {
  background: #16213e;
  padding: 20px;
  border-radius: 12px;
  height: 300px;
}

.chart-container h3 {
  margin-bottom: 15px;
  color: #fff;
}

.no-data {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 200px;
  color: #666;
}

.agents-section {
  margin-bottom: 30px;
}

.agents-section h2 {
  margin-bottom: 15px;
  color: #fff;
}

.agents-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
  gap: 15px;
}

.trades-section {
  background: #16213e;
  padding: 20px;
  border-radius: 12px;
}

.trades-section h2 {
  margin-bottom: 15px;
  color: #fff;
}

.trades-table-wrapper {
  overflow-x: auto;
  max-height: 400px;
  overflow-y: auto;
}

.trades-table {
  width: 100%;
  border-collapse: collapse;
}

.trades-table th,
.trades-table td {
  padding: 10px 12px;
  text-align: left;
  border-bottom: 1px solid #333;
}

.trades-table th {
  background: #0f3460;
  color: #fff;
  position: sticky;
  top: 0;
}

.trades-table tr:hover {
  background: #1a3a5c;
}

.trades-table td.buy {
  color: #4CAF50;
}

.trades-table td.sell {
  color: #f44336;
}

.trades-table td.profit {
  color: #4CAF50;
}

.trades-table td.loss {
  color: #f44336;
}

.no-trades {
  text-align: center;
  color: #666;
  padding: 40px !important;
}

@media (max-width: 900px) {
  .charts-section {
    grid-template-columns: 1fr;
  }
}
</style>
