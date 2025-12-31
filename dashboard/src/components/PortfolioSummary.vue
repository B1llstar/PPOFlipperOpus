<script setup lang="ts">
import { usePortfolio } from '../composables/usePortfolio'
import { computed } from 'vue'

const { portfolio, loading, error } = usePortfolio()

const formatGold = (amount: number | undefined) => {
  if (!amount) return '0'
  if (amount >= 1_000_000_000) {
    return `${(amount / 1_000_000_000).toFixed(1)}B`
  } else if (amount >= 1_000_000) {
    return `${(amount / 1_000_000).toFixed(1)}M`
  } else if (amount >= 1_000) {
    return `${(amount / 1_000).toFixed(1)}K`
  }
  return amount.toLocaleString()
}

const pluginStatus = computed(() => {
  if (!portfolio.value) return { text: 'Unknown', class: 'text-gray-400' }
  return portfolio.value.plugin_online
    ? { text: 'Online', class: 'text-green-400' }
    : { text: 'Offline', class: 'text-red-400' }
})

const lastUpdated = computed(() => {
  if (!portfolio.value?.last_updated) return 'Never'
  const date = portfolio.value.last_updated.toDate?.() || new Date(portfolio.value.last_updated)
  return date.toLocaleTimeString()
})
</script>

<template>
  <div class="card">
    <h2 class="text-lg font-semibold text-osrs-gold mb-4">Portfolio Summary</h2>

    <div v-if="loading" class="text-gray-400">Loading...</div>
    <div v-else-if="error" class="text-red-400">{{ error }}</div>
    <div v-else-if="!portfolio" class="text-gray-400">No portfolio data</div>
    <div v-else class="space-y-3">
      <!-- Gold -->
      <div class="flex justify-between items-center">
        <span class="text-gray-400">Gold:</span>
        <span class="text-yellow-400 font-mono text-lg">{{ formatGold(portfolio.gold) }}</span>
      </div>

      <!-- Total Value -->
      <div class="flex justify-between items-center">
        <span class="text-gray-400">Total Value:</span>
        <span class="text-green-400 font-mono">{{ formatGold(portfolio.total_value) }}</span>
      </div>

      <!-- Holdings -->
      <div class="flex justify-between items-center">
        <span class="text-gray-400">Holdings:</span>
        <span class="text-blue-400">{{ portfolio.holdings_count }} items</span>
      </div>

      <!-- Active Orders -->
      <div class="flex justify-between items-center">
        <span class="text-gray-400">Active Orders:</span>
        <span class="text-cyan-400">{{ portfolio.active_order_count }}</span>
      </div>

      <hr class="border-gray-700 my-2" />

      <!-- Plugin Status -->
      <div class="flex justify-between items-center">
        <span class="text-gray-400">Plugin:</span>
        <span :class="pluginStatus.class">{{ pluginStatus.text }}</span>
      </div>

      <!-- Version -->
      <div class="flex justify-between items-center">
        <span class="text-gray-400">Version:</span>
        <span class="text-gray-300">{{ portfolio.plugin_version || 'Unknown' }}</span>
      </div>

      <!-- Last Updated -->
      <div class="flex justify-between items-center">
        <span class="text-gray-400">Updated:</span>
        <span class="text-gray-300 text-sm">{{ lastUpdated }}</span>
      </div>
    </div>
  </div>
</template>
