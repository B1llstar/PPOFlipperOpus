<script setup lang="ts">
import { computed } from 'vue'
import { useTradingStore } from '../stores/trading'

const store = useTradingStore()

const formatGold = (value: number) => {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`
  return value.toLocaleString()
}

const totalValue = computed(() => {
  return store.positions.reduce((sum, p) => sum + p.total_invested, 0)
})

const positionCount = computed(() => store.positions.length)
</script>

<template>
  <div class="positions-table">
    <div class="table-header">
      <h2>Inventory (Active Positions)</h2>
      <div class="header-stats">
        <span class="count">{{ positionCount }} items</span>
        <span class="total">Value: {{ formatGold(totalValue) }}</span>
      </div>
    </div>

    <div class="table-container" v-if="store.positions.length > 0">
      <table>
        <thead>
          <tr>
            <th>Item</th>
            <th class="right">Qty</th>
            <th class="right">Price</th>
            <th class="right">Value</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="position in store.positions" :key="position.id">
            <td class="item-name">{{ position.item_name }}</td>
            <td class="right">{{ position.quantity.toLocaleString() }}</td>
            <td class="right">{{ formatGold(position.avg_cost) }}</td>
            <td class="right value">{{ formatGold(position.total_invested) }}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="empty" v-else>
      <p>No items in inventory</p>
      <p class="hint">Synced from RuneLite plugin</p>
    </div>
  </div>
</template>

<style scoped>
.positions-table {
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

.header-stats {
  display: flex;
  gap: 1.5rem;
}

.count {
  color: #888;
  font-size: 0.9rem;
}

.total {
  color: #4ade80;
  font-weight: 600;
}

.table-container {
  max-height: 300px;
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
  color: #4ade80;
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

.empty .hint {
  font-size: 0.85rem;
  margin-top: 0.5rem;
  color: #555;
}
</style>
