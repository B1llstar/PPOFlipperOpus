import { ref, onUnmounted, watch, computed } from 'vue'
import { doc, onSnapshot, updateDoc, Unsubscribe } from 'firebase/firestore'
import { db, COLLECTIONS, ACCOUNT_DOCS } from '../utils/firebase'
import { useAccount } from './useAccount'
import type { GEState, GESlot } from '../types'

export function useGEState() {
  const { accountId } = useAccount()

  const geState = ref<GEState | null>(null)
  const loading = ref(true)
  const error = ref<string | null>(null)

  let unsubscribe: Unsubscribe | null = null

  const subscribe = () => {
    if (unsubscribe) {
      unsubscribe()
    }

    loading.value = true
    error.value = null

    const docRef = doc(db, COLLECTIONS.ACCOUNTS, accountId.value, ACCOUNT_DOCS.GE_STATE, 'current')

    unsubscribe = onSnapshot(
      docRef,
      (snapshot) => {
        if (snapshot.exists()) {
          geState.value = snapshot.data() as GEState
        } else {
          geState.value = null
        }
        loading.value = false
      },
      (err) => {
        console.error('GE State subscription error:', err)
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
  const slots = computed(() => {
    if (!geState.value?.slots) return []
    return Object.entries(geState.value.slots).map(([slotNum, slot]) => ({
      slotNumber: parseInt(slotNum),
      ...slot
    }))
  })

  const freeSlots = computed(() => geState.value?.free_slots || 0)

  const activeSlots = computed(() =>
    slots.value.filter(s => s.status === 'active')
  )

  const emptySlots = computed(() =>
    slots.value.filter(s => s.status === 'empty')
  )

  const completeSlots = computed(() =>
    slots.value.filter(s => s.status === 'complete')
  )

  // Actions
  const clearSlot = async (slotNumber: number) => {
    if (!geState.value) return

    const newSlots = { ...geState.value.slots }
    newSlots[slotNumber.toString()] = { status: 'empty' }

    const docRef = doc(db, COLLECTIONS.ACCOUNTS, accountId.value, ACCOUNT_DOCS.GE_STATE, 'current')
    await updateDoc(docRef, {
      slots: newSlots,
      free_slots: Object.values(newSlots).filter(s => s.status === 'empty').length
    })
  }

  return {
    geState,
    loading,
    error,
    slots,
    freeSlots,
    activeSlots,
    emptySlots,
    completeSlots,
    clearSlot
  }
}
