<script setup lang="ts">
import { useTradingStore } from '../stores/trading'

const store = useTradingStore()

const formatGold = (value: number) => {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`
  return value.toLocaleString()
}
</script>

<template>
  <div class="stats-card">
    <h2>Statistics</h2>

    <div class="stats-grid">
      <div class="stat">
        <span class="stat-value">{{ store.stats?.total_trades || 0 }}</span>
        <span class="stat-label">Total Trades</span>
      </div>

      <div class="stat">
        <span class="stat-value buy">{{ store.stats?.total_buys || 0 }}</span>
        <span class="stat-label">Buys</span>
      </div>

      <div class="stat">
        <span class="stat-value sell">{{ store.stats?.total_sells || 0 }}</span>
        <span class="stat-label">Sells</span>
      </div>

      <div class="stat">
        <span class="stat-value" :class="{ profit: (store.stats?.total_profit || 0) > 0, loss: (store.stats?.total_profit || 0) < 0 }">
          {{ formatGold(store.stats?.total_profit || 0) }}
        </span>
        <span class="stat-label">Net Profit</span>
      </div>

      <div class="stat">
        <span class="stat-value">{{ formatGold(store.stats?.total_volume || 0) }}</span>
        <span class="stat-label">Volume</span>
      </div>

      <div class="stat">
        <span class="stat-value">{{ store.portfolio?.holdings_count || 0 }}</span>
        <span class="stat-label">Positions</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.stats-card {
  background: #1a1a2e;
  border-radius: 8px;
  padding: 1.5rem;
  border: 1px solid #0f3460;
}

h2 {
  margin: 0 0 1rem 0;
  font-size: 1.1rem;
  color: #e0e0e0;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
}

.stat {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 1rem;
  background: #16213e;
  border-radius: 6px;
}

.stat-value {
  font-size: 1.5rem;
  font-weight: 700;
  color: #e0e0e0;
}

.stat-value.buy {
  color: #4ade80;
}

.stat-value.sell {
  color: #f87171;
}

.stat-value.profit {
  color: #4ade80;
}

.stat-value.loss {
  color: #f87171;
}

.stat-label {
  font-size: 0.75rem;
  color: #888;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-top: 0.25rem;
}
</style>
