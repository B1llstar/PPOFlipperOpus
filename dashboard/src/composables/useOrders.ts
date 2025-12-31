import { ref, onUnmounted, watch, computed } from 'vue'
import {
  collection,
  onSnapshot,
  query,
  orderBy,
  doc,
  updateDoc,
  deleteDoc,
  Unsubscribe,
  where
} from 'firebase/firestore'
import { db, COLLECTIONS, ORDERS_SUBCOLLECTION } from '../utils/firebase'
import { useAccount } from './useAccount'
import type { Order, OrderStatus } from '../types'

export function useOrders() {
  const { accountId } = useAccount()

  const orders = ref<Order[]>([])
  const loading = ref(true)
  const error = ref<string | null>(null)

  let unsubscribe: Unsubscribe | null = null

  const subscribe = () => {
    if (unsubscribe) {
      unsubscribe()
    }

    loading.value = true
    error.value = null

    const ordersRef = collection(db, COLLECTIONS.ACCOUNTS, accountId.value, ORDERS_SUBCOLLECTION)
    const q = query(ordersRef, orderBy('created_at', 'desc'))

    unsubscribe = onSnapshot(
      q,
      (snapshot) => {
        orders.value = snapshot.docs.map(doc => ({
          order_id: doc.id,
          ...doc.data()
        })) as Order[]
        loading.value = false
      },
      (err) => {
        console.error('Orders subscription error:', err)
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

  // Computed filters
  const pendingOrders = computed(() =>
    orders.value.filter(o => o.status === 'pending')
  )

  const activeOrders = computed(() =>
    orders.value.filter(o => ['received', 'placed', 'partial'].includes(o.status))
  )

  const completedOrders = computed(() =>
    orders.value.filter(o => o.status === 'completed')
  )

  const failedOrders = computed(() =>
    orders.value.filter(o => ['cancelled', 'failed'].includes(o.status))
  )

  // Actions
  const cancelOrder = async (orderId: string) => {
    const orderRef = doc(db, COLLECTIONS.ACCOUNTS, accountId.value, ORDERS_SUBCOLLECTION, orderId)
    await updateDoc(orderRef, {
      status: 'cancelled' as OrderStatus,
      error: 'Cancelled by user from dashboard'
    })
  }

  const deleteOrder = async (orderId: string) => {
    const orderRef = doc(db, COLLECTIONS.ACCOUNTS, accountId.value, ORDERS_SUBCOLLECTION, orderId)
    await deleteDoc(orderRef)
  }

  const updateOrderStatus = async (orderId: string, status: OrderStatus) => {
    const orderRef = doc(db, COLLECTIONS.ACCOUNTS, accountId.value, ORDERS_SUBCOLLECTION, orderId)
    await updateDoc(orderRef, { status })
  }

  return {
    orders,
    loading,
    error,
    pendingOrders,
    activeOrders,
    completedOrders,
    failedOrders,
    cancelOrder,
    deleteOrder,
    updateOrderStatus
  }
}
