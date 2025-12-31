import { defineStore } from 'pinia'
import { ref, computed, onUnmounted } from 'vue'
import type { AccountStatus, Holding, Order, Trade, Portfolio, Stats, Position, BankItem, GESlot } from '../types'
import {
  subscribeToAccount,
  subscribeToOrders,
  subscribeToTrades,
  subscribeToHoldings,
  subscribeToPositions,
  subscribeToBank,
  subscribeToGESlots,
  lockPosition as firestoreLockPosition,
  unlockPosition as firestoreUnlockPosition,
  removePosition as firestoreRemovePosition,
  cancelOrder as firestoreCancelOrder,
  cancelAllActiveOrders as firestoreCancelAllActiveOrders
} from '../firebase/firestore'
import type { Unsubscribe } from 'firebase/firestore'

export const useTradingStore = defineStore('trading', () => {
  // State
  const status = ref<AccountStatus | null>(null)
  const holdings = ref<Holding[]>([])
  const orders = ref<Order[]>([])
  const trades = ref<Trade[]>([])
  const positions = ref<Position[]>([])
  const bankItems = ref<BankItem[]>([])
  const bankTotalValue = ref(0)
  const geSlots = ref<Record<string, GESlot>>({})
  const geSlotsAvailable = ref(8)
  const loading = ref(false)
  const error = ref<string | null>(null)
  const connected = ref(false)

  // Subscriptions
  const unsubscribes: Unsubscribe[] = []

  // Computed
  const gold = computed(() => {
    if (status.value?.gold) return status.value.gold
    if (status.value?.current_gold) return status.value.current_gold
    return 0
  })

  const portfolio = computed<Portfolio | null>(() => {
    if (!status.value) return null

    const holdingsValue = holdings.value.reduce((sum, h) =>
      sum + (h.value || (h.quantity || 0) * (h.avg_price || 0)), 0
    )

    const activeOrderCount = orders.value.filter(o =>
      ['pending', 'received', 'placed', 'active'].includes(o.status)
    ).length

    const pendingValue = orders.value
      .filter(o => ['pending', 'received', 'placed', 'active'].includes(o.status))
      .reduce((sum, o) => sum + (o.quantity * o.price), 0)

    return {
      gold: gold.value,
      holdings_value: holdingsValue,
      total_value: gold.value + holdingsValue,
      holdings_count: holdings.value.length,
      active_orders: activeOrderCount,
      pending_value: pendingValue,
      total_profit: status.value.total_profit || 0,
      holdings: holdings.value
    }
  })

  const stats = computed<Stats | null>(() => {
    if (trades.value.length === 0) return null

    let totalBuys = 0
    let totalSells = 0
    let totalProfit = 0
    let totalVolume = 0

    for (const trade of trades.value) {
      if (trade.type === 'buy') totalBuys++
      else if (trade.type === 'sell') totalSells++
      totalProfit += trade.profit || 0
      totalVolume += trade.quantity * trade.price
    }

    return {
      total_trades: trades.value.length,
      total_buys: totalBuys,
      total_sells: totalSells,
      total_profit: totalProfit,
      total_volume: totalVolume
    }
  })

  const totalValue = computed(() => portfolio.value?.total_value || gold.value)

  const activeOrders = computed(() =>
    orders.value.filter(o =>
      ['pending', 'received', 'placed', 'active'].includes(o.status)
    )
  )

  const completedOrders = computed(() =>
    orders.value.filter(o => o.status === 'completed')
  )

  const isOnline = computed(() =>
    status.value?.inference_online || status.value?.plugin_online
  )

  // Actions
  function connect(accountId?: string) {
    if (connected.value) return

    loading.value = true
    error.value = null

    try {
      // Subscribe to account status
      unsubscribes.push(
        subscribeToAccount((data) => {
          status.value = data
          loading.value = false
        }, accountId)
      )

      // Subscribe to orders (real-time)
      unsubscribes.push(
        subscribeToOrders((data) => {
          orders.value = data
        }, accountId)
      )

      // Subscribe to trades (real-time)
      unsubscribes.push(
        subscribeToTrades((data) => {
          trades.value = data
        }, accountId)
      )

      // Subscribe to holdings (real-time)
      unsubscribes.push(
        subscribeToHoldings((data) => {
          holdings.value = data as Holding[]
        }, accountId)
      )

      // Subscribe to positions (real-time)
      unsubscribes.push(
        subscribeToPositions((data) => {
          positions.value = data
        }, accountId)
      )

      // Subscribe to bank (real-time)
      unsubscribes.push(
        subscribeToBank((data) => {
          bankItems.value = data.items
          bankTotalValue.value = data.totalValue
        }, accountId)
      )

      // Subscribe to GE slots (real-time)
      unsubscribes.push(
        subscribeToGESlots((data) => {
          geSlots.value = data.slots
          geSlotsAvailable.value = data.available
        }, accountId)
      )

      connected.value = true
    } catch (e) {
      error.value = 'Failed to connect to Firestore'
      console.error('Firestore connection error:', e)
      loading.value = false
    }
  }

  function disconnect() {
    for (const unsub of unsubscribes) {
      unsub()
    }
    unsubscribes.length = 0
    connected.value = false
  }

  // Position actions
  async function lockPosition(itemId: string) {
    try {
      await firestoreLockPosition(itemId)
    } catch (e) {
      console.error('Failed to lock position:', e)
      throw e
    }
  }

  async function unlockPosition(itemId: string) {
    try {
      await firestoreUnlockPosition(itemId)
    } catch (e) {
      console.error('Failed to unlock position:', e)
      throw e
    }
  }

  async function removePosition(itemId: string) {
    try {
      await firestoreRemovePosition(itemId)
    } catch (e) {
      console.error('Failed to remove position:', e)
      throw e
    }
  }

  // Order actions
  async function cancelOrder(orderId: string) {
    try {
      await firestoreCancelOrder(orderId)
    } catch (e) {
      console.error('Failed to cancel order:', e)
      throw e
    }
  }

  async function cancelAllActiveOrders(): Promise<number> {
    try {
      return await firestoreCancelAllActiveOrders()
    } catch (e) {
      console.error('Failed to cancel all orders:', e)
      throw e
    }
  }

  // Legacy API compatibility - these now just trigger a reconnect
  function startAutoRefresh() {
    connect()
  }

  function stopAutoRefresh() {
    disconnect()
  }

  return {
    // State
    status,
    holdings,
    orders,
    trades,
    positions,
    bankItems,
    bankTotalValue,
    geSlots,
    geSlotsAvailable,
    loading,
    error,
    connected,

    // Computed
    gold,
    portfolio,
    stats,
    totalValue,
    activeOrders,
    completedOrders,
    isOnline,

    // Actions
    connect,
    disconnect,
    lockPosition,
    unlockPosition,
    removePosition,
    cancelOrder,
    cancelAllActiveOrders,

    // Legacy compatibility
    startAutoRefresh,
    stopAutoRefresh
  }
})
