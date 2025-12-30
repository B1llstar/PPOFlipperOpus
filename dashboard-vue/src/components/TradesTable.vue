<script setup lang="ts">
import { computed } from 'vue'
import { useTradingStore } from '../stores/trading'

const store = useTradingStore()

const formatGold = (value: number) => {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`
  return value.toLocaleString()
}

const formatTime = (timestamp: string) => {
  if (!timestamp) return '-'
  const date = new Date(timestamp)
  return date.toLocaleString()
}

const recentTrades = computed(() => store.trades.slice(0, 20))

const totalProfit = computed(() => {
  return store.trades.reduce((sum, t) => sum + (t.profit || 0), 0)
})
</script>

<template>
  <div class="trades-table">
    <div class="table-header">
      <h2>Recent Trades</h2>
      <span class="total" :class="{ profit: totalProfit > 0, loss: totalProfit < 0 }">
        P/L: {{ totalProfit >= 0 ? '+' : '' }}{{ formatGold(totalProfit) }}
      </span>
    </div>

    <div class="table-container" v-if="recentTrades.length > 0">
      <table>
        <thead>
          <tr>
            <th>Type</th>
            <th>Item</th>
            <th class="right">Qty</th>
            <th class="right">Price</th>
            <th class="right">Profit</th>
            <th>Time</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="trade in recentTrades" :key="trade.id">
            <td>
              <span class="trade-type" :class="trade.type === 'buy' ? 'type-buy' : 'type-sell'">
                {{ (trade.type || 'unknown').toUpperCase() }}
              </span>
            </td>
            <td class="item-name">{{ trade.item_name || `Item #${trade.item_id}` }}</td>
            <td class="right">{{ (trade.quantity || 0).toLocaleString() }}</td>
            <td class="right">{{ formatGold(trade.price || 0) }}</td>
            <td class="right" :class="{ profit: (trade.profit || 0) > 0, loss: (trade.profit || 0) < 0 }">
              {{ trade.profit ? ((trade.profit >= 0 ? '+' : '') + formatGold(trade.profit)) : '-' }}
            </td>
            <td class="time">{{ formatTime(trade.timestamp) }}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="empty" v-else>
      <p>No trades yet</p>
    </div>
  </div>
</template>

<style scoped>
.trades-table {
  background: #1a1a2e;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid #0f3460;
}

.table-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 1.5rem;
  background: #16213e;
  border-bottom: 1px solid #0f3460;
}

.table-header h2 {
  margin: 0;
  font-size: 1.1rem;
  color: #e0e0e0;
}

.total {
  font-weight: 600;
}

.total.profit {
  color: #4ade80;
}

.total.loss {
  color: #f87171;
}

.table-container {
  max-height: 350px;
  overflow-y: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
}

th, td {
  padding: 0.75rem 1rem;
  text-align: left;
  border-bottom: 1px solid #0f3460;
}

th {
  background: #16213e;
  color: #888;
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  position: sticky;
  top: 0;
}

td {
  color: #e0e0e0;
}

.right {
  text-align: right;
}

.item-name {
  font-weight: 500;
}

.time {
  color: #666;
  font-size: 0.85rem;
}

.trade-type {
  padding: 0.2rem 0.5rem;
  border-radius: 3px;
  font-size: 0.75rem;
  font-weight: 600;
}

.type-buy {
  background: rgba(74, 222, 128, 0.2);
  color: #4ade80;
}

.type-sell {
  background: rgba(248, 113, 113, 0.2);
  color: #f87171;
}

.profit {
  color: #4ade80;
}

.loss {
  color: #f87171;
}

tr:hover td {
  background: rgba(15, 52, 96, 0.3);
}

.empty {
  padding: 2rem;
  text-align: center;
  color: #666;
}
</style>
