<script setup lang="ts">
import { ref } from 'vue'
import { doc, setDoc, deleteDoc, collection, getDocs, serverTimestamp } from 'firebase/firestore'
import { db, COLLECTIONS, ACCOUNT_DOCS, ORDERS_SUBCOLLECTION } from '../utils/firebase'
import { useAccount } from '../composables/useAccount'
import type { OrderAction, OrderStatus } from '../types'

const { accountId, setAccountId } = useAccount()

const newAccountId = ref(accountId.value)
const isExpanded = ref(false)
const actionLog = ref<string[]>([])

// Test order creation
const testItemId = ref(2)
const testItemName = ref('Cannonball')
const testQuantity = ref(1000)
const testPrice = ref(200)
const testAction = ref<OrderAction>('buy')

const log = (message: string) => {
  const timestamp = new Date().toLocaleTimeString()
  actionLog.value.unshift(`[${timestamp}] ${message}`)
  if (actionLog.value.length > 50) {
    actionLog.value.pop()
  }
}

const updateAccount = () => {
  setAccountId(newAccountId.value)
  log(`Account changed to: ${newAccountId.value}`)
}

const createTestOrder = async () => {
  try {
    const orderId = `test_${Date.now()}`
    const orderRef = doc(db, COLLECTIONS.ACCOUNTS, accountId.value, ORDERS_SUBCOLLECTION, orderId)

    await setDoc(orderRef, {
      order_id: orderId,
      action: testAction.value,
      item_id: testItemId.value,
      item_name: testItemName.value,
      quantity: testQuantity.value,
      price: testPrice.value,
      status: 'pending' as OrderStatus,
      ge_slot: null,
      filled_quantity: 0,
      total_cost: testQuantity.value * testPrice.value,
      created_at: serverTimestamp(),
      received_at: null,
      placed_at: null,
      completed_at: null,
      error: null,
      retry_count: 0,
      confidence: 0.85,
      strategy: 'debug_test'
    })

    log(`Created test order: ${testAction.value} ${testQuantity.value}x ${testItemName.value}`)
  } catch (err: any) {
    log(`Error creating order: ${err.message}`)
  }
}

const clearAllOrders = async () => {
  if (!confirm('Delete ALL orders? This cannot be undone.')) return

  try {
    const ordersRef = collection(db, COLLECTIONS.ACCOUNTS, accountId.value, ORDERS_SUBCOLLECTION)
    const snapshot = await getDocs(ordersRef)

    let count = 0
    for (const docSnap of snapshot.docs) {
      await deleteDoc(docSnap.ref)
      count++
    }

    log(`Deleted ${count} orders`)
  } catch (err: any) {
    log(`Error clearing orders: ${err.message}`)
  }
}

const resetPortfolio = async () => {
  if (!confirm('Reset portfolio to defaults?')) return

  try {
    const portfolioRef = doc(db, COLLECTIONS.ACCOUNTS, accountId.value, ACCOUNT_DOCS.PORTFOLIO, 'current')
    await setDoc(portfolioRef, {
      gold: 10_000_000,
      total_value: 10_000_000,
      holdings_count: 0,
      active_order_count: 0,
      plugin_online: false,
      plugin_version: 'debug',
      last_updated: serverTimestamp()
    })

    log('Portfolio reset to 10M gold')
  } catch (err: any) {
    log(`Error resetting portfolio: ${err.message}`)
  }
}

const clearInventory = async () => {
  if (!confirm('Clear all inventory items?')) return

  try {
    const inventoryRef = doc(db, COLLECTIONS.ACCOUNTS, accountId.value, ACCOUNT_DOCS.INVENTORY, 'current')
    await setDoc(inventoryRef, {
      items: {},
      empty_slots: 28,
      scanned_at: serverTimestamp()
    })

    log('Inventory cleared')
  } catch (err: any) {
    log(`Error clearing inventory: ${err.message}`)
  }
}

const resetGESlots = async () => {
  if (!confirm('Reset all GE slots to empty?')) return

  try {
    const slots: Record<string, { status: string }> = {}
    for (let i = 0; i < 8; i++) {
      slots[i.toString()] = { status: 'empty' }
    }

    const geRef = doc(db, COLLECTIONS.ACCOUNTS, accountId.value, ACCOUNT_DOCS.GE_STATE, 'current')
    await setDoc(geRef, {
      slots,
      free_slots: 8,
      synced_at: serverTimestamp()
    })

    log('GE slots reset to empty')
  } catch (err: any) {
    log(`Error resetting GE slots: ${err.message}`)
  }
}

const simulatePluginHeartbeat = async () => {
  try {
    const portfolioRef = doc(db, COLLECTIONS.ACCOUNTS, accountId.value, ACCOUNT_DOCS.PORTFOLIO, 'current')
    await setDoc(portfolioRef, {
      plugin_online: true,
      last_updated: serverTimestamp()
    }, { merge: true })

    log('Simulated plugin heartbeat')
  } catch (err: any) {
    log(`Error: ${err.message}`)
  }
}
</script>

<template>
  <div class="card border-2 border-red-900">
    <button
      @click="isExpanded = !isExpanded"
      class="flex justify-between items-center w-full"
    >
      <h2 class="text-lg font-semibold text-red-400">Debug Panel</h2>
      <span class="text-gray-400">{{ isExpanded ? '▼' : '▶' }}</span>
    </button>

    <div v-show="isExpanded" class="mt-4 space-y-4">
      <!-- Account Selector -->
      <div class="p-3 bg-gray-800 rounded">
        <label class="text-sm text-gray-400 block mb-2">Account ID</label>
        <div class="flex gap-2">
          <input
            v-model="newAccountId"
            type="text"
            class="flex-1 px-3 py-2 text-sm bg-gray-700 border border-gray-600 rounded text-white"
          />
          <button
            @click="updateAccount"
            class="btn-secondary"
          >
            Switch
          </button>
        </div>
      </div>

      <!-- Test Order Creation -->
      <div class="p-3 bg-gray-800 rounded">
        <label class="text-sm text-gray-400 block mb-2">Create Test Order</label>
        <div class="grid grid-cols-2 gap-2 mb-2">
          <select v-model="testAction" class="px-2 py-1 text-sm bg-gray-700 border border-gray-600 rounded text-white">
            <option value="buy">Buy</option>
            <option value="sell">Sell</option>
          </select>
          <input v-model="testItemName" placeholder="Item name" class="px-2 py-1 text-sm bg-gray-700 border border-gray-600 rounded text-white" />
          <input v-model.number="testItemId" type="number" placeholder="Item ID" class="px-2 py-1 text-sm bg-gray-700 border border-gray-600 rounded text-white" />
          <input v-model.number="testQuantity" type="number" placeholder="Quantity" class="px-2 py-1 text-sm bg-gray-700 border border-gray-600 rounded text-white" />
          <input v-model.number="testPrice" type="number" placeholder="Price" class="px-2 py-1 text-sm bg-gray-700 border border-gray-600 rounded text-white" />
        </div>
        <button @click="createTestOrder" class="btn-primary w-full">
          Create Order
        </button>
      </div>

      <!-- Quick Actions -->
      <div class="p-3 bg-gray-800 rounded">
        <label class="text-sm text-gray-400 block mb-2">Quick Actions</label>
        <div class="grid grid-cols-2 gap-2">
          <button @click="simulatePluginHeartbeat" class="btn-secondary text-sm">
            Simulate Heartbeat
          </button>
          <button @click="resetPortfolio" class="btn-secondary text-sm">
            Reset Portfolio
          </button>
          <button @click="clearInventory" class="btn-danger text-sm">
            Clear Inventory
          </button>
          <button @click="resetGESlots" class="btn-danger text-sm">
            Reset GE Slots
          </button>
          <button @click="clearAllOrders" class="btn-danger text-sm col-span-2">
            Delete All Orders
          </button>
        </div>
      </div>

      <!-- Action Log -->
      <div class="p-3 bg-gray-800 rounded">
        <label class="text-sm text-gray-400 block mb-2">Action Log</label>
        <div class="max-h-32 overflow-y-auto text-xs font-mono space-y-1">
          <div v-for="(entry, i) in actionLog" :key="i" class="text-gray-300">
            {{ entry }}
          </div>
          <div v-if="actionLog.length === 0" class="text-gray-500">
            No actions yet
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
