<script setup lang="ts">
import { computed } from 'vue'
import { useTradingStore } from '../stores/trading'

const store = useTradingStore()

const formatGold = (value: number) => {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`
  return value.toLocaleString()
}

const sortedHoldings = computed(() => {
  return [...(store.portfolio?.holdings || store.holdings)].sort((a, b) => {
    const valueA = (a.value || (a.quantity || 0) * (a.avg_price || 0))
    const valueB = (b.value || (b.quantity || 0) * (b.avg_price || 0))
    return valueB - valueA
  })
})

const totalValue = computed(() => {
  return sortedHoldings.value.reduce((sum, h) => {
    return sum + (h.value || (h.quantity || 0) * (h.avg_price || 0))
  }, 0)
})
</script>

<template>
  <div class="holdings-table">
    <div class="table-header">
      <h2>Holdings</h2>
      <span class="total">Total: {{ formatGold(totalValue) }}</span>
    </div>

    <div class="table-container" v-if="sortedHoldings.length > 0">
      <table>
        <thead>
          <tr>
            <th>Item</th>
            <th class="right">Qty</th>
            <th class="right">Avg Price</th>
            <th class="right">Value</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="holding in sortedHoldings" :key="holding.id">
            <td class="item-name">{{ holding.item_name || `Item #${holding.item_id}` }}</td>
            <td class="right">{{ (holding.quantity || 0).toLocaleString() }}</td>
            <td class="right">{{ formatGold(holding.avg_price || 0) }}</td>
            <td class="right value">{{ formatGold(holding.value || (holding.quantity || 0) * (holding.avg_price || 0)) }}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="empty" v-else>
      <p>No holdings</p>
    </div>
  </div>
</template>

<style scoped>
.holdings-table {
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
  color: #ffd700;
  font-weight: 600;
}

.table-container {
  max-height: 400px;
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

.value {
  color: #ffd700;
  font-weight: 600;
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
