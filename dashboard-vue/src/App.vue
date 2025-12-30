<script setup lang="ts">
import { onMounted, onUnmounted } from 'vue'
import { useTradingStore } from './stores/trading'
import StatusBar from './components/StatusBar.vue'
import HoldingsTable from './components/HoldingsTable.vue'
import OrdersTable from './components/OrdersTable.vue'
import TradesTable from './components/TradesTable.vue'
import StatsCard from './components/StatsCard.vue'

const store = useTradingStore()

onMounted(() => {
  store.startAutoRefresh(5000) // Refresh every 5 seconds
})

onUnmounted(() => {
  store.stopAutoRefresh()
})
</script>

<template>
  <div class="dashboard">
    <header class="header">
      <h1>PPO Flipper Dashboard</h1>
      <div class="header-info">
        <span v-if="store.loading" class="loading">Updating...</span>
        <span class="account">Account: {{ store.status?.account_id || 'b1llstar' }}</span>
      </div>
    </header>

    <StatusBar />

    <main class="main-content">
      <div class="left-column">
        <HoldingsTable />
        <StatsCard />
      </div>

      <div class="right-column">
        <OrdersTable />
        <TradesTable />
      </div>
    </main>

    <footer class="footer">
      <p>PPO Flipper - OSRS Grand Exchange Trading Bot</p>
    </footer>
  </div>
</template>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #0d1117;
  color: #e0e0e0;
  min-height: 100vh;
}

#app {
  min-height: 100vh;
}
</style>

<style scoped>
.dashboard {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 1.5rem;
  background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
  border-bottom: 2px solid #e94560;
}

.header h1 {
  font-size: 1.5rem;
  font-weight: 700;
  background: linear-gradient(135deg, #e94560, #f97316);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.header-info {
  display: flex;
  gap: 1rem;
  align-items: center;
}

.loading {
  color: #facc15;
  font-size: 0.85rem;
  animation: pulse 1s infinite;
}

.account {
  color: #888;
  font-size: 0.9rem;
}

.main-content {
  flex: 1;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
  padding: 1.5rem;
}

.left-column,
.right-column {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.footer {
  padding: 1rem;
  text-align: center;
  color: #666;
  font-size: 0.85rem;
  border-top: 1px solid #0f3460;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

@media (max-width: 1024px) {
  .main-content {
    grid-template-columns: 1fr;
  }
}
</style>
