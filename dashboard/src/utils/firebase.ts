import { initializeApp } from 'firebase/app'
import { getFirestore } from 'firebase/firestore'

// Firebase configuration - update with your project details
const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID || 'ppoflipperopus',
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID
}

// Initialize Firebase
const app = initializeApp(firebaseConfig)

// Initialize Firestore
export const db = getFirestore(app)

// Collection paths (matching V2 schema)
export const COLLECTIONS = {
  ACCOUNTS: 'accounts',
  ITEMS: 'items',
  ITEM_NAMES: 'itemNames'
} as const

// Document names within account
export const ACCOUNT_DOCS = {
  PORTFOLIO: 'portfolio',
  INVENTORY: 'inventory',
  BANK: 'bank',
  GE_STATE: 'ge_state'
} as const

// Subcollection for orders
export const ORDERS_SUBCOLLECTION = 'orders'
