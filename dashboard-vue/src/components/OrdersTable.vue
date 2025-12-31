<script setup lang="ts">
import { computed, ref } from 'vue'
import { useTradingStore } from '../stores/trading'

const store = useTradingStore()
const showAll = ref(false)
const cancellingOrder = ref<string | null>(null)
const cancellingAll = ref(false)

const formatGold = (value: number) => {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`
  return value.toLocaleString()
}

const formatTime = (timestamp: string) => {
  if (!timestamp) return '-'
  const date = new Date(timestamp)
  return date.toLocaleTimeString()
}

const displayedOrders = computed(() => {
  if (showAll.value) {
    return store.orders.slice(0, 50)
  }
  return store.activeOrders
})

const isActiveStatus = (status: string) => {
  return ['pending', 'received', 'placed', 'active'].includes(status)
}

const getStatusClass = (status: string) => {
  switch (status) {
    case 'completed': return 'status-completed'
    case 'pending':
    case 'received': return 'status-pending'
    case 'placed':
    case 'active': return 'status-active'
    case 'cancelled':
    case 'failed': return 'status-failed'
    default: return ''
  }
}

const getTypeClass = (type: string) => {
  return type === 'buy' ? 'type-buy' : 'type-sell'
}

async function handleCancelOrder(orderId: string) {
  cancellingOrder.value = orderId
  try {
    await store.cancelOrder(orderId)
  } finally {
    cancellingOrder.value = null
  }
}

async function handleCancelAll() {
  if (!confirm(`Cancel all ${store.activeOrders.length} active orders?`)) {
    return
  }
  cancellingAll.value = true
  try {
    const count = await store.cancelAllActiveOrders()
    console.log(`Cancelled ${count} orders`)
  } finally {
    cancellingAll.value = false
  }
}
</script>

<template>
  <div class="orders-table">
    <div class="table-header">
      <h2>Orders</h2>
      <div class="controls">
        <button
          :class="{ active: !showAll }"
          @click="showAll = false"
        >
          Active ({{ store.activeOrders.length }})
        </button>
        <button
          :class="{ active: showAll }"
          @click="showAll = true"
        >
          All ({{ store.orders.length }})
        </button>
        <button
          v-if="store.activeOrders.length > 0"
          class="btn-cancel-all"
          @click="handleCancelAll"
          :disabled="cancellingAll"
        >
          {{ cancellingAll ? 'Cancelling...' : 'Cancel All' }}
        </button>
      </div>
    </div>

    <div class="table-container" v-if="displayedOrders.length > 0">
      <table>
        <thead>
          <tr>
            <th>Type</th>
            <th>Item</th>
            <th class="right">Qty</th>
            <th class="right">Price</th>
            <th class="right">Value</th>
            <th>Status</th>
            <th>Time</th>
            <th class="center">Action</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="order in displayedOrders" :key="order.id">
            <td>
              <span class="order-type" :class="getTypeClass(order.type || '')">
                {{ (order.type || 'unknown').toUpperCase() }}
              </span>
            </td>
            <td class="item-name">{{ order.item_name || `Item #${order.item_id}` }}</td>
            <td class="right">{{ (order.quantity || 0).toLocaleString() }}</td>
            <td class="right">{{ formatGold(order.price || 0) }}</td>
            <td class="right">{{ formatGold((order.quantity || 0) * (order.price || 0)) }}</td>
            <td>
              <span class="order-status" :class="getStatusClass(order.status)">
                {{ order.status }}
              </span>
            </td>
            <td class="time">{{ formatTime(order.created_at) }}</td>
            <td class="center">
              <button
                v-if="isActiveStatus(order.status)"
                class="btn-cancel"
                @click="handleCancelOrder(order.id)"
                :disabled="cancellingOrder === order.id"
              >
                {{ cancellingOrder === order.id ? '...' : 'X' }}
              </button>
              <span v-else class="no-action">-</span>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="empty" v-else>
      <p>{{ showAll ? 'No orders yet' : 'No active orders' }}</p>
    </div>
  </div>
</template>

<style scoped>
.orders-table {
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

.controls {
  display: flex;
  gap: 0.5rem;
}

.controls button {
  padding: 0.4rem 0.8rem;
  border: 1px solid #0f3460;
  background: transparent;
  color: #888;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.85rem;
  transition: all 0.2s;
}

.controls button:hover {
  border-color: #e94560;
  color: #e0e0e0;
}

.controls button.active {
  background: #e94560;
  border-color: #e94560;
  color: white;
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

.time {
  color: #666;
  font-size: 0.85rem;
}

.order-type {
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

.order-status {
  padding: 0.2rem 0.5rem;
  border-radius: 3px;
  font-size: 0.75rem;
  text-transform: capitalize;
}

.status-pending {
  background: rgba(250, 204, 21, 0.2);
  color: #facc15;
}

.status-active {
  background: rgba(59, 130, 246, 0.2);
  color: #3b82f6;
}

.status-completed {
  background: rgba(74, 222, 128, 0.2);
  color: #4ade80;
}

.status-failed {
  background: rgba(248, 113, 113, 0.2);
  color: #f87171;
}

tr:hover td {
  background: rgba(15, 52, 96, 0.3);
}

.center {
  text-align: center;
}

.btn-cancel-all {
  padding: 0.4rem 0.8rem;
  border: 1px solid #ef4444;
  background: rgba(239, 68, 68, 0.2);
  color: #ef4444;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.85rem;
  font-weight: 600;
  transition: all 0.2s;
}

.btn-cancel-all:hover:not(:disabled) {
  background: rgba(239, 68, 68, 0.4);
}

.btn-cancel-all:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-cancel {
  padding: 0.25rem 0.5rem;
  border: none;
  background: rgba(239, 68, 68, 0.2);
  color: #ef4444;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.75rem;
  font-weight: 600;
  transition: all 0.2s;
}

.btn-cancel:hover:not(:disabled) {
  background: rgba(239, 68, 68, 0.4);
}

.btn-cancel:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.no-action {
  color: #444;
}

.empty {
  padding: 2rem;
  text-align: center;
  color: #666;
}
</style>
