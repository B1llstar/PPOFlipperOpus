<template>
  <div class="agent-card">
    <div class="agent-header">
      <h3>Agent {{ agent.agent_id + 1 }}</h3>
      <span class="episode-badge">Ep {{ agent.episode }} / Step {{ agent.step }}</span>
    </div>

    <div class="agent-stats">
      <div class="stat">
        <span class="stat-label">Cash</span>
        <span class="stat-value cash">{{ formatGP(agent.cash) }}</span>
      </div>
      <div class="stat">
        <span class="stat-label">Portfolio</span>
        <span class="stat-value">{{ formatGP(agent.portfolio_value) }}</span>
      </div>
      <div class="stat">
        <span class="stat-label">Total Assets</span>
        <span class="stat-value total">{{ formatGP(agent.total_assets) }}</span>
      </div>
    </div>

    <div class="agent-stats">
      <div class="stat">
        <span class="stat-label">Episode Reward</span>
        <span class="stat-value" :class="{ positive: agent.episode_reward > 0, negative: agent.episode_reward < 0 }">
          {{ agent.episode_reward?.toFixed(2) }}
        </span>
      </div>
      <div class="stat">
        <span class="stat-label">Total Reward</span>
        <span class="stat-value" :class="{ positive: agent.total_reward > 0, negative: agent.total_reward < 0 }">
          {{ agent.total_reward?.toFixed(2) }}
        </span>
      </div>
    </div>

    <div class="agent-stats">
      <div class="stat">
        <span class="stat-label">Trades</span>
        <span class="stat-value">{{ agent.trades_executed }}</span>
      </div>
      <div class="stat">
        <span class="stat-label">Profitable</span>
        <span class="stat-value">{{ agent.profitable_trades }}</span>
      </div>
      <div class="stat">
        <span class="stat-label">Win Rate</span>
        <span class="stat-value">{{ winRate }}%</span>
      </div>
      <div class="stat">
        <span class="stat-label">Taxes Paid</span>
        <span class="stat-value tax">{{ formatGP(agent.taxes_paid) }}</span>
      </div>
    </div>

    <div class="current-action" v-if="agent.current_action">
      <span class="action-label">Current Action:</span>
      <span class="action-value">{{ agent.current_action }}</span>
    </div>

    <div class="holdings-section" v-if="Object.keys(agent.holdings || {}).length > 0">
      <h4>Holdings</h4>
      <div class="holdings-list">
        <div
          v-for="(qty, item) in agent.holdings"
          :key="item"
          class="holding-item"
        >
          <span class="item-name">{{ item }}</span>
          <span class="item-qty">x{{ qty.toLocaleString() }}</span>
        </div>
      </div>
    </div>
    <div class="holdings-section empty" v-else>
      <h4>Holdings</h4>
      <p class="no-holdings">No items held</p>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  agent: {
    type: Object,
    required: true
  }
})

const winRate = computed(() => {
  if (!props.agent.trades_executed) return 0
  return ((props.agent.profitable_trades / props.agent.trades_executed) * 100).toFixed(1)
})

function formatGP(value) {
  if (!value) return '0 GP'
  if (value >= 1_000_000_000) {
    return (value / 1_000_000_000).toFixed(2) + 'B GP'
  }
  if (value >= 1_000_000) {
    return (value / 1_000_000).toFixed(2) + 'M GP'
  }
  if (value >= 1_000) {
    return (value / 1_000).toFixed(1) + 'K GP'
  }
  return value.toLocaleString() + ' GP'
}
</script>

<style scoped>
.agent-card {
  background: #16213e;
  border-radius: 12px;
  padding: 20px;
  border: 1px solid #2a3f5f;
}

.agent-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
  padding-bottom: 10px;
  border-bottom: 1px solid #2a3f5f;
}

.agent-header h3 {
  color: #fff;
  font-size: 18px;
}

.episode-badge {
  background: #0f3460;
  padding: 4px 10px;
  border-radius: 12px;
  font-size: 12px;
  color: #aaa;
}

.agent-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
  gap: 10px;
  margin-bottom: 12px;
}

.stat {
  text-align: center;
}

.stat-label {
  display: block;
  font-size: 10px;
  color: #888;
  text-transform: uppercase;
  margin-bottom: 4px;
}

.stat-value {
  font-size: 14px;
  font-weight: bold;
  color: #fff;
}

.stat-value.cash {
  color: #FFD700;
}

.stat-value.total {
  color: #4CAF50;
}

.stat-value.tax {
  color: #f44336;
}

.stat-value.positive {
  color: #4CAF50;
}

.stat-value.negative {
  color: #f44336;
}

.current-action {
  background: #0f3460;
  padding: 8px 12px;
  border-radius: 8px;
  margin-bottom: 12px;
  font-size: 13px;
}

.action-label {
  color: #888;
  margin-right: 8px;
}

.action-value {
  color: #4CAF50;
  font-weight: bold;
}

.holdings-section {
  margin-top: 12px;
}

.holdings-section h4 {
  font-size: 12px;
  color: #888;
  text-transform: uppercase;
  margin-bottom: 8px;
}

.holdings-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.holding-item {
  background: #0f3460;
  padding: 6px 10px;
  border-radius: 6px;
  font-size: 12px;
  display: flex;
  gap: 6px;
}

.item-name {
  color: #fff;
}

.item-qty {
  color: #4CAF50;
  font-weight: bold;
}

.no-holdings {
  color: #555;
  font-size: 12px;
  font-style: italic;
}
</style>
