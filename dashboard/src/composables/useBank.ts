import { ref, onUnmounted, watch, computed } from 'vue'
import { doc, onSnapshot, updateDoc, Unsubscribe } from 'firebase/firestore'
import { db, COLLECTIONS, ACCOUNT_DOCS } from '../utils/firebase'
import { useAccount } from './useAccount'
import type { BankState, Holding } from '../types'

export function useBank() {
  const { accountId } = useAccount()

  const bank = ref<BankState | null>(null)
  const loading = ref(true)
  const error = ref<string | null>(null)

  let unsubscribe: Unsubscribe | null = null

  const subscribe = () => {
    if (unsubscribe) {
      unsubscribe()
    }

    loading.value = true
    error.value = null

    const docRef = doc(db, COLLECTIONS.ACCOUNTS, accountId.value, ACCOUNT_DOCS.BANK, 'current')

    unsubscribe = onSnapshot(
      docRef,
      (snapshot) => {
        if (snapshot.exists()) {
          bank.value = snapshot.data() as BankState
        } else {
          bank.value = null
        }
        loading.value = false
      },
      (err) => {
        console.error('Bank subscription error:', err)
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
    if (!bank.value?.items) return []
    return Object.entries(bank.value.items).map(([id, holding]) => ({
      id,
      ...holding
    }))
  })

  const totalItems = computed(() => bank.value?.total_items || 0)

  // Actions
  const removeItem = async (itemId: string) => {
    if (!bank.value) return

    const newItems = { ...bank.value.items }
    delete newItems[itemId]

    const docRef = doc(db, COLLECTIONS.ACCOUNTS, accountId.value, ACCOUNT_DOCS.BANK, 'current')
    await updateDoc(docRef, {
      items: newItems,
      total_items: Object.keys(newItems).length
    })
  }

  const updateItemQuantity = async (itemId: string, quantity: number) => {
    if (!bank.value) return

    const newItems = { ...bank.value.items }
    if (newItems[itemId]) {
      newItems[itemId] = { ...newItems[itemId], quantity }
    }

    const docRef = doc(db, COLLECTIONS.ACCOUNTS, accountId.value, ACCOUNT_DOCS.BANK, 'current')
    await updateDoc(docRef, { items: newItems })
  }

  return {
    bank,
    loading,
    error,
    items,
    totalItems,
    removeItem,
    updateItemQuantity
  }
}
