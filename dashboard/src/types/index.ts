// Types matching the V2 Firestore schema

export type OrderAction = 'buy' | 'sell'
export type OrderStatus = 'pending' | 'received' | 'placed' | 'partial' | 'completed' | 'cancelled' | 'failed'
export type SlotStatus = 'empty' | 'active' | 'complete'

export interface Order {
  order_id: string
  action: OrderAction
  item_id: number
  item_name: string
  quantity: number
  price: number
  status: OrderStatus
  ge_slot: number | null
  filled_quantity: number
  total_cost: number
  created_at: any // Firestore Timestamp
  received_at: any | null
  placed_at: any | null
  completed_at: any | null
  error: string | null
  retry_count: number
  confidence: number
  strategy: string
}

export interface Holding {
  name: string
  quantity: number
}

export interface InventoryState {
  items: Record<string, Holding>
  empty_slots: number
  scanned_at: any
}

export interface BankState {
  items: Record<string, Holding>
  total_items: number
  scanned_at: any
}

export interface GESlot {
  status: SlotStatus
  type?: 'buy' | 'sell'
  item_id?: number
  item_name?: string
  quantity?: number
  price?: number
  filled?: number
  order_id?: string
}

export interface GEState {
  slots: Record<string, GESlot>
  free_slots: number
  synced_at: any
}

export interface Portfolio {
  gold: number
  total_value: number
  holdings_count: number
  active_order_count: number
  plugin_online: boolean
  plugin_version: string
  last_updated: any
}

export interface AccountState {
  portfolio: Portfolio | null
  inventory: InventoryState | null
  bank: BankState | null
  geState: GEState | null
  orders: Order[]
}

// Utility type for formatting
export interface FormattedOrder extends Order {
  formattedCreatedAt: string
  formattedPrice: string
  formattedTotal: string
  fillPercentage: number
}
