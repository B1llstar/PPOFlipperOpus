<script setup lang="ts">
import { ref, computed } from 'vue'
import { useOrders } from '../composables/useOrders'
import type { Order, OrderStatus } from '../types'

const {
  orders,
  loading,
  error,
  pendingOrders,
  activeOrders,
  completedOrders,
  failedOrders,
  cancelOrder,
  deleteOrder
} = useOrders()

const activeTab = ref<'all' | 'pending' | 'active' | 'completed' | 'failed'>('all')

const displayedOrders = computed(() => {
  switch (activeTab.value) {
    case 'pending': return pendingOrders.value
    case 'active': return activeOrders.value
    case 'completed': return completedOrders.value
    case 'failed': return failedOrders.value
    default: return orders.value
  }
})

const formatGold = (amount: number) => {
  if (amount >= 1_000_000) return `${(amount / 1_000_000).toFixed(1)}M`
  if (amount >= 1_000) return `${(amount / 1_000).toFixed(1)}K`
  return amount.toLocaleString()
}

const formatDate = (timestamp: any) => {
  if (!timestamp) return '-'
  const date = timestamp.toDate?.() || new Date(timestamp)
  return date.toLocaleTimeString()
}

const getStatusClass = (status: OrderStatus) => {
  switch (status) {
    case 'pending': return 'bg-yellow-600'
    case 'received': return 'bg-blue-600'
    case 'placed': return 'bg-cyan-600'
    case 'partial': return 'bg-orange-600'
    case 'completed': return 'bg-green-600'
    case 'cancelled': return 'bg-gray-600'
    case 'failed': return 'bg-red-600'
    default: return 'bg-gray-600'
  }
}

const getActionClass = (action: string) => {
  return action === 'buy' ? 'text-green-400' : 'text-orange-400'
}

const handleCancel = async (order: Order) => {
  if (confirm(`Cancel order for ${order.item_name}?`)) {
    await cancelOrder(order.order_id)
  }
}

const handleDelete = async (order: Order) => {
  if (confirm(`Delete order for ${order.item_name}? This cannot be undone.`)) {
    await deleteOrder(order.order_id)
  }
}
</script>

<template>
  <div class="card">
    <h2 class="text-lg font-semibold text-osrs-gold mb-4">Orders</h2>

    <!-- Tabs -->
    <div class="flex gap-2 mb-4 flex-wrap">
      <button
        v-for="tab in ['all', 'pending', 'active', 'completed', 'failed']"
        :key="tab"
        @click="activeTab = tab as any"
        :class="[
          'px-3 py-1 rounded text-sm transition-colors',
          activeTab === tab
            ? 'bg-osrs-gold text-osrs-dark'
            : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
        ]"
      >
        {{ tab.charAt(0).toUpperCase() + tab.slice(1) }}
        <span class="ml-1 text-xs opacity-75">
          ({{
            tab === 'all' ? orders.length :
            tab === 'pending' ? pendingOrders.length :
            tab === 'active' ? activeOrders.length :
            tab === 'completed' ? completedOrders.length :
            failedOrders.length
          }})
        </span>
      </button>
    </div>

    <div v-if="loading" class="text-gray-400">Loading...</div>
    <div v-else-if="error" class="text-red-400">{{ error }}</div>
    <div v-else-if="displayedOrders.length === 0" class="text-gray-400">No orders</div>
    <div v-else class="space-y-2 max-h-96 overflow-y-auto">
      <div
        v-for="order in displayedOrders"
        :key="order.order_id"
        class="bg-gray-800 rounded p-3 border border-gray-700"
      >
        <div class="flex justify-between items-start">
          <div class="flex-1">
            <div class="flex items-center gap-2">
              <span :class="getActionClass(order.action)" class="font-semibold uppercase text-sm">
                {{ order.action }}
              </span>
              <span class="text-white font-medium">{{ order.item_name }}</span>
              <span :class="['px-2 py-0.5 rounded text-xs', getStatusClass(order.status)]">
                {{ order.status }}
              </span>
            </div>
            <div class="text-sm text-gray-400 mt-1">
              <span>{{ order.quantity.toLocaleString() }} @ {{ formatGold(order.price) }}</span>
              <span class="mx-2">|</span>
              <span>Total: {{ formatGold(order.total_cost) }}</span>
            </div>
            <div class="text-xs text-gray-500 mt-1">
              <span v-if="order.filled_quantity > 0">
                Filled: {{ order.filled_quantity }}/{{ order.quantity }}
                ({{ Math.round(order.filled_quantity / order.quantity * 100) }}%)
              </span>
              <span v-if="order.ge_slot !== null" class="ml-2">Slot: {{ order.ge_slot }}</span>
              <span class="ml-2">{{ formatDate(order.created_at) }}</span>
            </div>
            <div v-if="order.error" class="text-xs text-red-400 mt-1">
              Error: {{ order.error }}
            </div>
          </div>
          <div class="flex gap-1">
            <button
              v-if="['pending', 'received', 'placed', 'partial'].includes(order.status)"
              @click="handleCancel(order)"
              class="px-2 py-1 text-xs bg-red-600 hover:bg-red-500 rounded"
              title="Cancel Order"
            >
              Cancel
            </button>
            <button
              @click="handleDelete(order)"
              class="px-2 py-1 text-xs bg-gray-600 hover:bg-gray-500 rounded"
              title="Delete Order"
            >
              Delete
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
