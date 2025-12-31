<script setup lang="ts">
import { ref, computed } from 'vue'
import { useBank } from '../composables/useBank'

const { bank, loading, error, items, totalItems, removeItem, updateItemQuantity } = useBank()

const searchQuery = ref('')
const editingItem = ref<string | null>(null)
const editQuantity = ref<number>(0)

const filteredItems = computed(() => {
  if (!searchQuery.value) return items.value
  const query = searchQuery.value.toLowerCase()
  return items.value.filter(item =>
    item.name.toLowerCase().includes(query)
  )
})

const startEdit = (itemId: string, currentQuantity: number) => {
  editingItem.value = itemId
  editQuantity.value = currentQuantity
}

const saveEdit = async (itemId: string) => {
  await updateItemQuantity(itemId, editQuantity.value)
  editingItem.value = null
}

const cancelEdit = () => {
  editingItem.value = null
}

const handleRemove = async (itemId: string, itemName: string) => {
  if (confirm(`Remove ${itemName} from bank tracking?`)) {
    await removeItem(itemId)
  }
}

const formatDate = (timestamp: any) => {
  if (!timestamp) return '-'
  const date = timestamp.toDate?.() || new Date(timestamp)
  return date.toLocaleTimeString()
}
</script>

<template>
  <div class="card">
    <div class="flex justify-between items-center mb-4">
      <h2 class="text-lg font-semibold text-osrs-gold">Bank</h2>
      <span class="text-sm text-gray-400">{{ totalItems }} items tracked</span>
    </div>

    <!-- Search -->
    <input
      v-model="searchQuery"
      type="text"
      placeholder="Search items..."
      class="w-full px-3 py-2 mb-3 text-sm bg-gray-800 border border-gray-700 rounded text-white placeholder-gray-500"
    />

    <div v-if="loading" class="text-gray-400">Loading...</div>
    <div v-else-if="error" class="text-red-400">{{ error }}</div>
    <div v-else-if="filteredItems.length === 0" class="text-gray-400">
      {{ searchQuery ? 'No matching items' : 'Bank empty' }}
    </div>
    <div v-else class="space-y-2 max-h-64 overflow-y-auto">
      <div
        v-for="item in filteredItems"
        :key="item.id"
        class="flex items-center justify-between bg-gray-800 rounded p-2 border border-gray-700"
      >
        <div class="flex-1">
          <div class="text-white text-sm">{{ item.name }}</div>
          <div v-if="editingItem === item.id" class="flex items-center gap-2 mt-1">
            <input
              v-model.number="editQuantity"
              type="number"
              min="0"
              class="w-24 px-2 py-1 text-xs bg-gray-700 border border-gray-600 rounded text-white"
            />
            <button @click="saveEdit(item.id)" class="text-xs text-green-400 hover:text-green-300">Save</button>
            <button @click="cancelEdit" class="text-xs text-gray-400 hover:text-gray-300">Cancel</button>
          </div>
          <div v-else class="text-xs text-gray-400">
            Qty: {{ item.quantity.toLocaleString() }}
            <button @click="startEdit(item.id, item.quantity)" class="ml-2 text-blue-400 hover:text-blue-300">Edit</button>
          </div>
        </div>
        <button
          @click="handleRemove(item.id, item.name)"
          class="px-2 py-1 text-xs bg-red-600 hover:bg-red-500 rounded"
        >
          Remove
        </button>
      </div>
    </div>

    <div v-if="bank" class="mt-3 text-xs text-gray-500">
      Last scanned: {{ formatDate(bank.scanned_at) }}
    </div>
  </div>
</template>
