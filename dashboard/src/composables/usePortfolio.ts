import { ref, onUnmounted, watch } from 'vue'
import { doc, onSnapshot, Unsubscribe } from 'firebase/firestore'
import { db, COLLECTIONS, ACCOUNT_DOCS } from '../utils/firebase'
import { useAccount } from './useAccount'
import type { Portfolio } from '../types'

export function usePortfolio() {
  const { accountId } = useAccount()

  const portfolio = ref<Portfolio | null>(null)
  const loading = ref(true)
  const error = ref<string | null>(null)

  let unsubscribe: Unsubscribe | null = null

  const subscribe = () => {
    // Clean up previous subscription
    if (unsubscribe) {
      unsubscribe()
    }

    loading.value = true
    error.value = null

    const docRef = doc(db, COLLECTIONS.ACCOUNTS, accountId.value, ACCOUNT_DOCS.PORTFOLIO, 'current')

    unsubscribe = onSnapshot(
      docRef,
      (snapshot) => {
        if (snapshot.exists()) {
          portfolio.value = snapshot.data() as Portfolio
        } else {
          portfolio.value = null
        }
        loading.value = false
      },
      (err) => {
        console.error('Portfolio subscription error:', err)
        error.value = err.message
        loading.value = false
      }
    )
  }

  // Watch for account changes
  watch(accountId, () => {
    subscribe()
  }, { immediate: true })

  onUnmounted(() => {
    if (unsubscribe) {
      unsubscribe()
    }
  })

  return {
    portfolio,
    loading,
    error
  }
}
