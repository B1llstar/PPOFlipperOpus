import { ref, onUnmounted, watch, computed } from 'vue'
import { doc, onSnapshot, updateDoc, Unsubscribe } from 'firebase/firestore'
import { db, COLLECTIONS, ACCOUNT_DOCS } from '../utils/firebase'
import { useAccount } from './useAccount'
import type { InventoryState, Holding } from '../types'

export function useInventory() {
  const { accountId } = useAccount()

  const inventory = ref<InventoryState | null>(null)
  const loading = ref(true)
  const error = ref<string | null>(null)

  let unsubscribe: Unsubscribe | null = null

  const subscribe = () => {
    if (unsubscribe) {
      unsubscribe()
    }

    loading.value = true
    error.value = null

    const docRef = doc(db, COLLECTIONS.ACCOUNTS, accountId.value, ACCOUNT_DOCS.INVENTORY, 'current')

    unsubscribe = onSnapshot(
      docRef,
      (snapshot) => {
        if (snapshot.exists()) {
          inventory.value = snapshot.data() as InventoryState
        } else {
          inventory.value = null
        }
        loading.value = false
      },
      (err) => {
        console.error('Inventory subscription error:', err)
        error.value = err.message
        loading.value = false
      }
    )
  }

  watch(accountId, () => {
    subscribe()
  }, { immediate: true })

  onUnmounted(() => {
    if (unsubscribe) {
      unsubscribe()
    }
  })

  // Computed
  const items = computed(() => {
    if (!inventory.value?.items) return []
    return Object.entries(inventory.value.items).map(([id, holding]) => ({
      id,
      ...holding
    }))
  })

  const totalItems = computed(() => items.value.length)

  // Actions
  const removeItem = async (itemId: string) => {
    if (!inventory.value) return

    const newItems = { ...inventory.value.items }
    delete newItems[itemId]

    const docRef = doc(db, COLLECTIONS.ACCOUNTS, accountId.value, ACCOUNT_DOCS.INVENTORY, 'current')
    await updateDoc(docRef, { items: newItems })
  }

  const updateItemQuantity = async (itemId: string, quantity: number) => {
    if (!inventory.value) return

    const newItems = { ...inventory.value.items }
    if (newItems[itemId]) {
      newItems[itemId] = { ...newItems[itemId], quantity }
    }

    const docRef = doc(db, COLLECTIONS.ACCOUNTS, accountId.value, ACCOUNT_DOCS.INVENTORY, 'current')
    await updateDoc(docRef, { items: newItems })
  }

  return {
    inventory,
    loading,
    error,
    items,
    totalItems,
    removeItem,
    updateItemQuantity
  }
}
