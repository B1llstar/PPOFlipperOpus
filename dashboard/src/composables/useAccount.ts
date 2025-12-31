import { ref, computed } from 'vue'

// Default account ID - can be changed in the UI
const accountId = ref<string>('test_account')

export function useAccount() {
  const setAccountId = (id: string) => {
    accountId.value = id
    // Persist to localStorage
    localStorage.setItem('ge_dashboard_account_id', id)
  }

  const loadAccountId = () => {
    const stored = localStorage.getItem('ge_dashboard_account_id')
    if (stored) {
      accountId.value = stored
    }
  }

  // Load on first use
  loadAccountId()

  return {
    accountId: computed(() => accountId.value),
    setAccountId,
    loadAccountId
  }
}
