<script setup lang="ts">
import { computed } from 'vue'
import { useTradingStore } from '../stores/trading'

const store = useTradingStore()

const formatGold = (value: number) => {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`
  return value.toLocaleString()
}

const statusClass = computed(() => {
  if (store.status?.inference_online && store.status?.plugin_online) return 'online'
  if (store.status?.inference_online || store.status?.plugin_online) return 'partial'
  return 'offline'
})
</script>

<template>
  <div class="status-bar">
    <div class="status-item">
      <span class="label">Gold</span>
      <span class="value gold">{{ formatGold(store.gold) }}</span>
    </div>

    <div class="status-item">
      <span class="label">Portfolio</span>
      <span class="value">{{ formatGold(store.totalValue) }}</span>
    </div>

    <div class="status-item">
      <span class="label">Holdings</span>
      <span class="value">{{ store.portfolio?.holdings_count || 0 }}</span>
    </div>

    <div class="status-item">
      <span class="label">Active Orders</span>
      <span class="value">{{ store.activeOrders.length }}</span>
    </div>

    <div class="status-item">
      <span class="label">Profit</span>
      <span class="value" :class="{ profit: (store.stats?.total_profit || 0) > 0, loss: (store.stats?.total_profit || 0) < 0 }">
        {{ formatGold(store.stats?.total_profit || 0) }}
      </span>
    </div>

    <div class="status-item status-indicator">
      <span class="label">Status</span>
      <span class="status-dot" :class="statusClass"></span>
      <span class="value">
        {{ store.status?.inference_online ? 'Inference' : '' }}
        {{ store.status?.inference_online && store.status?.plugin_online ? '+' : '' }}
        {{ store.status?.plugin_online ? 'Plugin' : '' }}
        {{ !store.status?.inference_online && !store.status?.plugin_online ? 'Offline' : '' }}
      </span>
    </div>

    <div class="status-item" v-if="store.lastUpdated">
      <span class="label">Updated</span>
      <span class="value small">{{ store.lastUpdated.toLocaleTimeString() }}</span>
    </div>
  </div>
</template>

<style scoped>
.status-bar {
  display: flex;
  gap: 2rem;
  padding: 1rem 1.5rem;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  border-bottom: 2px solid #0f3460;
  flex-wrap: wrap;
}

.status-item {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.label {
  font-size: 0.75rem;
  color: #888;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.value {
  font-size: 1.25rem;
  font-weight: 600;
  color: #e0e0e0;
}

.value.small {
  font-size: 0.9rem;
}

.value.gold {
  color: #ffd700;
}

.value.profit {
  color: #4ade80;
}

.value.loss {
  color: #f87171;
}

.status-indicator {
  flex-direction: row;
  align-items: center;
  gap: 0.5rem;
}

.status-indicator .label {
  margin-right: 0.25rem;
}

.status-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  animation: pulse 2s infinite;
}

.status-dot.online {
  background: #4ade80;
  box-shadow: 0 0 10px #4ade80;
}

.status-dot.partial {
  background: #facc15;
  box-shadow: 0 0 10px #facc15;
}

.status-dot.offline {
  background: #f87171;
  box-shadow: 0 0 10px #f87171;
  animation: none;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}
</style>
