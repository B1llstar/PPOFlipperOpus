import { initializeApp } from 'firebase/app'
import { getFirestore } from 'firebase/firestore'

// Firebase configuration for PPOFlipperOpus
// Using client-side access (no service account needed for reading)
const firebaseConfig = {
  projectId: 'ppoflipperopus',
  // For Firestore-only access, we just need the project ID
  // The security rules in Firestore control access
}

// Initialize Firebase
const app = initializeApp(firebaseConfig)

// Initialize Firestore
export const db = getFirestore(app)

// Default account ID
export const DEFAULT_ACCOUNT_ID = 'b1llstar'
