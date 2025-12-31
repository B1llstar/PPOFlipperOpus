import {
  collection,
  doc,
  onSnapshot,
  query,
  orderBy,
  limit,
  updateDoc,
  deleteField,
  getDocs,
  Timestamp
} from 'firebase/firestore'
import type { Unsubscribe } from 'firebase/firestore'
import { db, DEFAULT_ACCOUNT_ID } from './config'
import type { AccountStatus, Order, Trade, Position, BankItem, GESlot, Holding } from '../types'

// =============================================================================
// Document References
// =============================================================================

const accountRef = (accountId = DEFAULT_ACCOUNT_ID) =>
  doc(db, 'accounts', accountId)

const positionsRef = (accountId = DEFAULT_ACCOUNT_ID) =>
  doc(db, 'accounts', accountId, 'positions', 'active')

const inventoryRef = (accountId = DEFAULT_ACCOUNT_ID) =>
  doc(db, 'accounts', accountId, 'inventory', 'current')

const bankRef = (accountId = DEFAULT_ACCOUNT_ID) =>
  doc(db, 'accounts', accountId, 'bank', 'current')

const geSlotsRef = (accountId = DEFAULT_ACCOUNT_ID) =>
  doc(db, 'accounts', accountId, 'ge_slots', 'current')

const ordersCollection = (accountId = DEFAULT_ACCOUNT_ID) =>
  collection(db, 'accounts', accountId, 'orders')

const tradesCollection = (accountId = DEFAULT_ACCOUNT_ID) =>
  collection(db, 'accounts', accountId, 'trades')

const holdingsCollection = (accountId = DEFAULT_ACCOUNT_ID) =>
  collection(db, 'accounts', accountId, 'holdings')

// =============================================================================
// Real-time Listeners
// =============================================================================

export function subscribeToAccount(
  callback: (status: AccountStatus) => void,
  accountId = DEFAULT_ACCOUNT_ID
): Unsubscribe {
  return onSnapshot(accountRef(accountId), (snapshot) => {
    if (snapshot.exists()) {
      const data = snapshot.data()

      // Check plugin online status via heartbeat
      let pluginOnline = data.plugin_online || data.pluginOnline || data.online || false
      const lastHeartbeat = data.last_heartbeat || data.lastHeartbeat || data.plugin_heartbeat

      if (lastHeartbeat) {
        const hbTime = lastHeartbeat instanceof Timestamp
          ? lastHeartbeat.toDate()
          : new Date(lastHeartbeat)
        const ageSeconds = (Date.now() - hbTime.getTime()) / 1000
        pluginOnline = ageSeconds < 120
      }

      callback({
        success: true,
        account_id: accountId,
        gold: data.gold || 0,
        current_gold: data.current_gold || 0,
        portfolio_value: data.portfolio_value || 0,
        total_profit: data.total_profit || 0,
        inference_online: data.inference_online || false,
        plugin_online: pluginOnline,
        last_heartbeat: lastHeartbeat ? String(lastHeartbeat) : undefined,
        status: data.status || 'unknown'
      })
    }
  })
}

export function subscribeToOrders(
  callback: (orders: Order[]) => void,
  accountId = DEFAULT_ACCOUNT_ID
): Unsubscribe {
  const q = query(ordersCollection(accountId), orderBy('created_at', 'desc'), limit(100))

  return onSnapshot(q, (snapshot) => {
    const orders: Order[] = []
    snapshot.forEach((doc) => {
      const data = doc.data()
      orders.push({
        id: doc.id,
        item_id: data.item_id,
        item_name: data.item_name,
        type: data.type,
        quantity: data.quantity,
        price: data.price,
        status: data.status,
        confidence: data.confidence,
        created_at: formatTimestamp(data.created_at),
        updated_at: data.updated_at ? formatTimestamp(data.updated_at) : undefined,
        filled_quantity: data.filled_quantity,
        filled_price: data.filled_price
      })
    })
    callback(orders)
  })
}

export function subscribeToTrades(
  callback: (trades: Trade[]) => void,
  accountId = DEFAULT_ACCOUNT_ID
): Unsubscribe {
  const q = query(tradesCollection(accountId), orderBy('timestamp', 'desc'), limit(100))

  return onSnapshot(q, (snapshot) => {
    const trades: Trade[] = []
    snapshot.forEach((doc) => {
      const data = doc.data()
      trades.push({
        id: doc.id,
        item_id: data.item_id,
        item_name: data.item_name,
        type: data.type,
        quantity: data.quantity,
        price: data.price,
        profit: data.profit,
        timestamp: formatTimestamp(data.timestamp)
      })
    })
    callback(trades)
  })
}

export function subscribeToHoldings(
  callback: (holdings: Holding[]) => void,
  accountId = DEFAULT_ACCOUNT_ID
): Unsubscribe {
  return onSnapshot(holdingsCollection(accountId), (snapshot) => {
    const holdings: Holding[] = []
    snapshot.forEach((doc) => {
      const data = doc.data()
      holdings.push({
        id: doc.id,
        item_id: data.item_id,
        item_name: data.item_name,
        quantity: data.quantity || 0,
        avg_price: data.avg_price || 0,
        value: (data.quantity || 0) * (data.avg_price || 0),
        last_updated: data.last_updated
      })
    })
    callback(holdings)
  })
}

export function subscribeToPositions(
  callback: (positions: Position[]) => void,
  accountId = DEFAULT_ACCOUNT_ID
): Unsubscribe {
  // Positions = items in inventory (synced from plugin)
  // These are what PPO can actively trade
  return onSnapshot(inventoryRef(accountId), (snapshot) => {
    if (snapshot.exists()) {
      const data = snapshot.data()
      const items = data.items || {}

      const positions: Position[] = Object.entries(items).map(([itemId, itemData]: [string, any]) => ({
        id: itemId,
        item_id: itemData.item_id ?? parseInt(itemId),
        item_name: itemData.item_name || `Item #${itemId}`,
        quantity: itemData.quantity || 0,
        avg_cost: itemData.price_each || 0,
        total_invested: itemData.total_value || (itemData.quantity || 0) * (itemData.price_each || 0),
        first_acquired: '',
        last_updated: data.updated_at || '',
        source: 'ppo',
        locked: false
      }))

      // Sort by total value descending
      positions.sort((a, b) => b.total_invested - a.total_invested)
      callback(positions)
    } else {
      callback([])
    }
  })
}

export function subscribeToBank(
  callback: (bank: { items: BankItem[], totalValue: number }) => void,
  accountId = DEFAULT_ACCOUNT_ID
): Unsubscribe {
  return onSnapshot(bankRef(accountId), (snapshot) => {
    if (snapshot.exists()) {
      const data = snapshot.data()
      const items = data.items || {}

      const bankItems: BankItem[] = Object.entries(items).map(([itemId, itemData]: [string, any]) => ({
        id: itemId,
        item_id: parseInt(itemId),
        item_name: itemData.item_name || `Item #${itemId}`,
        quantity: itemData.quantity || 0,
        price_each: itemData.price_each || 0,
        total_value: itemData.total_value || 0,
        is_tradeable: itemData.is_tradeable !== false
      }))

      // Sort by total_value descending
      bankItems.sort((a, b) => b.total_value - a.total_value)

      callback({
        items: bankItems,
        totalValue: data.total_value || 0
      })
    } else {
      callback({ items: [], totalValue: 0 })
    }
  })
}

export function subscribeToGESlots(
  callback: (slots: { slots: Record<string, GESlot>, available: number }) => void,
  accountId = DEFAULT_ACCOUNT_ID
): Unsubscribe {
  return onSnapshot(geSlotsRef(accountId), (snapshot) => {
    if (snapshot.exists()) {
      const data = snapshot.data()
      const rawSlots = data.slots || {}

      const slots: Record<string, GESlot> = {}
      for (const [slotNum, slotData] of Object.entries(rawSlots)) {
        // Skip null/empty slot entries
        if (!slotData) continue

        const slot = slotData as any
        // Use 'id' field if item_id is not present (based on your data structure)
        const itemId = slot.item_id ?? slot.id ?? 0
        slots[slotNum] = {
          item_id: itemId,
          item_name: slot.item_name || slot.name || `Item #${itemId}`,
          type: slot.type || 'buy',
          status: slot.status || 'empty',
          quantity: slot.quantity || 0,
          filled_quantity: slot.filled_quantity || slot.filledQuantity || 0,
          price: slot.price || 0
        }
      }

      callback({
        slots,
        available: data.slots_available ?? data.slotsAvailable ?? 8
      })
    } else {
      callback({ slots: {}, available: 8 })
    }
  })
}

// =============================================================================
// Position Management Actions
// =============================================================================

export async function lockPosition(itemId: string, accountId = DEFAULT_ACCOUNT_ID): Promise<void> {
  const ref = positionsRef(accountId)
  await updateDoc(ref, {
    [`items.${itemId}.locked`]: true,
    [`items.${itemId}.last_updated`]: new Date().toISOString(),
    updated_at: new Date().toISOString()
  })
}

export async function unlockPosition(itemId: string, accountId = DEFAULT_ACCOUNT_ID): Promise<void> {
  const ref = positionsRef(accountId)
  await updateDoc(ref, {
    [`items.${itemId}.locked`]: false,
    [`items.${itemId}.last_updated`]: new Date().toISOString(),
    updated_at: new Date().toISOString()
  })
}

export async function removePosition(itemId: string, accountId = DEFAULT_ACCOUNT_ID): Promise<void> {
  const ref = positionsRef(accountId)
  await updateDoc(ref, {
    [`items.${itemId}`]: deleteField(),
    updated_at: new Date().toISOString()
  })
}

// =============================================================================
// Order Management Actions
// =============================================================================

export async function cancelOrder(orderId: string, accountId = DEFAULT_ACCOUNT_ID): Promise<void> {
  const orderRef = doc(db, 'accounts', accountId, 'orders', orderId)
  await updateDoc(orderRef, {
    status: 'cancelled',
    updated_at: new Date().toISOString()
  })
}

export async function cancelAllActiveOrders(accountId = DEFAULT_ACCOUNT_ID): Promise<number> {
  const ordersRef = collection(db, 'accounts', accountId, 'orders')
  const snapshot = await getDocs(ordersRef)

  const activeStatuses = ['pending', 'received', 'placed', 'active']
  const cancelPromises: Promise<void>[] = []

  snapshot.forEach((docSnap) => {
    const data = docSnap.data()
    if (activeStatuses.includes(data.status)) {
      cancelPromises.push(
        updateDoc(docSnap.ref, {
          status: 'cancelled',
          updated_at: new Date().toISOString()
        })
      )
    }
  })

  await Promise.all(cancelPromises)
  return cancelPromises.length
}

// =============================================================================
// Helpers
// =============================================================================

function formatTimestamp(ts: any): string {
  if (!ts) return ''
  if (ts instanceof Timestamp) {
    return ts.toDate().toISOString()
  }
  if (typeof ts === 'string') {
    return ts
  }
  if (ts.toDate) {
    return ts.toDate().toISOString()
  }
  return String(ts)
}
