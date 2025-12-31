export interface AccountStatus {
  success: boolean
  account_id: string
  gold: number
  current_gold: number
  portfolio_value: number
  total_profit: number
  inference_online: boolean
  plugin_online: boolean
  last_heartbeat?: string
  status: string
}

export interface Holding {
  id: string
  item_id: number
  item_name: string
  quantity: number
  avg_price: number
  value?: number
  last_updated?: string
}

export interface Order {
  id: string
  item_id: number
  item_name: string
  type: 'buy' | 'sell'
  quantity: number
  price: number
  status: 'pending' | 'received' | 'placed' | 'active' | 'completed' | 'cancelled' | 'failed'
  confidence?: number
  created_at: string
  updated_at?: string
  filled_quantity?: number
  filled_price?: number
}

export interface Trade {
  id: string
  item_id: number
  item_name: string
  type: 'buy' | 'sell'
  quantity: number
  price: number
  profit?: number
  timestamp: string
}

export interface Portfolio {
  gold: number
  holdings_value: number
  total_value: number
  holdings_count: number
  active_orders: number
  pending_value: number
  total_profit: number
  holdings: Holding[]
}

export interface Stats {
  total_trades: number
  total_buys: number
  total_sells: number
  total_profit: number
  total_volume: number
}

export interface Inventory {
  gold?: number
  items?: Record<string, number>
  last_updated?: string
}

export interface Position {
  id: string
  item_id: number
  item_name: string
  quantity: number
  avg_cost: number
  total_invested: number
  first_acquired: string
  last_updated: string
  source: 'ppo' | 'manual'
  locked: boolean
}

export interface BankItem {
  id: string
  item_id: number
  item_name: string
  quantity: number
  price_each: number
  total_value: number
  is_tradeable: boolean
}

export interface GESlot {
  item_id: number
  item_name: string
  type: 'buy' | 'sell'
  status: 'active' | 'empty' | 'completed'
  quantity: number
  filled_quantity: number
  price: number
}
