import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { AccountStatus, Holding, Order, Trade, Portfolio, Stats } from '../types'

const API_BASE = 'http://localhost:5001/api'

export const useTradingStore = defineStore('trading', () => {
  // State
  const status = ref<AccountStatus | null>(null)
  const holdings = ref<Holding[]>([])
  const orders = ref<Order[]>([])
  const trades = ref<Trade[]>([])
  const portfolio = ref<Portfolio | null>(null)
  const stats = ref<Stats | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)
  const lastUpdated = ref<Date | null>(null)

  // Computed
  const gold = computed(() => {
    if (portfolio.value?.gold) return portfolio.value.gold
    if (status.value?.gold) return status.value.gold
    if (status.value?.current_gold) return status.value.current_gold
    return 0
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
  async function fetchStatus() {
    try {
      const res = await fetch(`${API_BASE}/status`)
      const data = await res.json()
      if (data.success) {
        status.value = data
      }
    } catch (e) {
      console.error('Failed to fetch status:', e)
    }
  }

  async function fetchHoldings() {
    try {
      const res = await fetch(`${API_BASE}/holdings`)
      const data = await res.json()
      if (data.success) {
        holdings.value = data.holdings
      }
    } catch (e) {
      console.error('Failed to fetch holdings:', e)
    }
  }

  async function fetchOrders() {
    try {
      const res = await fetch(`${API_BASE}/orders`)
      const data = await res.json()
      if (data.success) {
        orders.value = data.orders
      }
    } catch (e) {
      console.error('Failed to fetch orders:', e)
    }
  }

  async function fetchTrades() {
    try {
      const res = await fetch(`${API_BASE}/trades`)
      const data = await res.json()
      if (data.success) {
        trades.value = data.trades
      }
    } catch (e) {
      console.error('Failed to fetch trades:', e)
    }
  }

  async function fetchPortfolio() {
    try {
      const res = await fetch(`${API_BASE}/portfolio`)
      const data = await res.json()
      if (data.success) {
        portfolio.value = data.portfolio
      }
    } catch (e) {
      console.error('Failed to fetch portfolio:', e)
    }
  }

  async function fetchStats() {
    try {
      const res = await fetch(`${API_BASE}/stats`)
      const data = await res.json()
      if (data.success) {
        stats.value = data.stats
      }
    } catch (e) {
      console.error('Failed to fetch stats:', e)
    }
  }

  async function fetchAll() {
    loading.value = true
    error.value = null

    try {
      await Promise.all([
        fetchStatus(),
        fetchHoldings(),
        fetchOrders(),
        fetchTrades(),
        fetchPortfolio(),
        fetchStats()
      ])
      lastUpdated.value = new Date()
    } catch (e) {
      error.value = 'Failed to fetch data'
      console.error('Failed to fetch all:', e)
    } finally {
      loading.value = false
    }
  }

  // Auto-refresh
  let refreshInterval: number | null = null

  function startAutoRefresh(intervalMs = 5000) {
    stopAutoRefresh()
    fetchAll()
    refreshInterval = window.setInterval(fetchAll, intervalMs)
  }

  function stopAutoRefresh() {
    if (refreshInterval) {
      clearInterval(refreshInterval)
      refreshInterval = null
    }
  }

  return {
    // State
    status,
    holdings,
    orders,
    trades,
    portfolio,
    stats,
    loading,
    error,
    lastUpdated,

    // Computed
    gold,
    totalValue,
    activeOrders,
    completedOrders,
    isOnline,

    // Actions
    fetchStatus,
    fetchHoldings,
    fetchOrders,
    fetchTrades,
    fetchPortfolio,
    fetchStats,
    fetchAll,
    startAutoRefresh,
    stopAutoRefresh
  }
})
