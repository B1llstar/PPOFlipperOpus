<script setup lang="ts">
import { ref, computed } from 'vue'
import { useAccount } from './composables/useAccount'
import PortfolioSummary from './components/PortfolioSummary.vue'
import OrdersPanel from './components/OrdersPanel.vue'
import GESlotsPanel from './components/GESlotsPanel.vue'
import InventoryPanel from './components/InventoryPanel.vue'
import BankPanel from './components/BankPanel.vue'
import DebugPanel from './components/DebugPanel.vue'

const { accountId } = useAccount()
const showDebug = ref(true)
</script>

<template>
  <div class="min-h-screen bg-osrs-dark text-white">
    <!-- Header -->
    <header class="bg-gray-900 border-b border-gray-800 py-4 px-6">
      <div class="max-w-7xl mx-auto flex justify-between items-center">
        <div>
          <h1 class="text-2xl font-bold text-osrs-gold">GE Auto V2 Dashboard</h1>
          <p class="text-sm text-gray-400">Account: {{ accountId }}</p>
        </div>
        <div class="flex items-center gap-4">
          <label class="flex items-center gap-2 text-sm text-gray-400">
            <input v-model="showDebug" type="checkbox" class="rounded" />
            Show Debug
          </label>
        </div>
      </div>
    </header>

    <!-- Main Content -->
    <main class="max-w-7xl mx-auto p-6">
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <!-- Left Column: Portfolio & GE Slots -->
        <div class="space-y-6">
          <PortfolioSummary />
          <GESlotsPanel />
        </div>

        <!-- Middle Column: Orders -->
        <div class="lg:col-span-2">
          <OrdersPanel />
        </div>
      </div>

      <!-- Second Row: Inventory & Bank -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
        <InventoryPanel />
        <BankPanel />
      </div>

      <!-- Debug Panel -->
      <div v-if="showDebug" class="mt-6">
        <DebugPanel />
      </div>
    </main>

    <!-- Footer -->
    <footer class="bg-gray-900 border-t border-gray-800 py-4 px-6 mt-8">
      <div class="max-w-7xl mx-auto text-center text-sm text-gray-500">
        GE Auto V2 - PPO Flipper Dashboard
      </div>
    </footer>
  </div>
</template>
