<script setup lang="ts">
import { useGEState } from '../composables/useGEState'

const { slots, loading, error, freeSlots, clearSlot } = useGEState()

const formatGold = (amount: number | undefined) => {
  if (!amount) return '-'
  if (amount >= 1_000_000) return `${(amount / 1_000_000).toFixed(1)}M`
  if (amount >= 1_000) return `${(amount / 1_000).toFixed(1)}K`
  return amount.toLocaleString()
}

const getSlotClass = (status: string, type?: string) => {
  if (status === 'empty') return 'ge-slot-empty'
  if (status === 'complete') return 'ge-slot-complete'
  return type === 'buy' ? 'ge-slot-buy' : 'ge-slot-sell'
}

const handleClearSlot = async (slotNumber: number) => {
  if (confirm(`Clear GE slot ${slotNumber}? This is for debugging only.`)) {
    await clearSlot(slotNumber)
  }
}
</script>

<template>
  <div class="card">
    <div class="flex justify-between items-center mb-4">
      <h2 class="text-lg font-semibold text-osrs-gold">GE Slots</h2>
      <span class="text-sm text-gray-400">{{ freeSlots }}/8 free</span>
    </div>

    <div v-if="loading" class="text-gray-400">Loading...</div>
    <div v-else-if="error" class="text-red-400">{{ error }}</div>
    <div v-else class="grid grid-cols-4 gap-2">
      <div
        v-for="slot in slots"
        :key="slot.slotNumber"
        :class="['ge-slot', getSlotClass(slot.status, slot.type)]"
      >
        <div class="text-xs text-gray-500 mb-1">Slot {{ slot.slotNumber }}</div>

        <template v-if="slot.status === 'empty'">
          <div class="text-gray-500 text-sm">Empty</div>
        </template>

        <template v-else>
          <div class="text-xs uppercase font-semibold" :class="slot.type === 'buy' ? 'text-green-400' : 'text-orange-400'">
            {{ slot.type }}
          </div>
          <div class="text-white text-sm truncate" :title="slot.item_name">
            {{ slot.item_name || 'Unknown' }}
          </div>
          <div class="text-xs text-gray-400">
            {{ slot.filled || 0 }}/{{ slot.quantity || 0 }}
          </div>
          <div class="text-xs text-yellow-400">
            {{ formatGold(slot.price) }}
          </div>
          <div v-if="slot.status === 'complete'" class="text-xs text-green-400 mt-1">
            Complete
          </div>
        </template>

        <button
          v-if="slot.status !== 'empty'"
          @click="handleClearSlot(slot.slotNumber)"
          class="mt-2 text-xs text-gray-500 hover:text-red-400 transition-colors"
        >
          Clear
        </button>
      </div>
    </div>
  </div>
</template>
