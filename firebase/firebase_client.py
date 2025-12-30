"""
Firebase Client - Core Firestore connection singleton.

Provides centralized Firebase initialization and Firestore access for the PPO inference server.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1 import Client, CollectionReference, DocumentReference

logger = logging.getLogger(__name__)


class FirebaseClient:
    """
    Singleton Firebase client for managing Firestore connections.

    Usage:
        client = FirebaseClient()
        client.initialize("path/to/service_account.json", "account_id")
        db = client.db
    """

    _instance: Optional['FirebaseClient'] = None
    _db: Optional[Client] = None
    _initialized: bool = False
    _account_id: Optional[str] = None

    # Collection names
    COLLECTION_ITEMS = "items"
    COLLECTION_ITEM_NAMES = "itemNames"
    COLLECTION_ACCOUNTS = "accounts"
    COLLECTION_MARKET_DATA = "market_data"

    # Subcollection names
    SUBCOLLECTION_ORDERS = "orders"
    SUBCOLLECTION_TRADES = "trades"
    SUBCOLLECTION_PORTFOLIO = "portfolio"
    SUBCOLLECTION_ACTIONS = "actions"
    SUBCOLLECTION_INVENTORY = "inventory"
    SUBCOLLECTION_BANK = "bank"
    SUBCOLLECTION_GE_SLOTS = "ge_slots"
    SUBCOLLECTION_COMMANDS = "commands"

    # Document names
    DOC_CURRENT = "current"

    def __new__(cls):
        """Singleton pattern - only one Firebase client instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self, service_account_path: str, account_id: str) -> bool:
        """
        Initialize Firebase with service account credentials.

        Args:
            service_account_path: Path to the service account JSON file
            account_id: The account identifier for this session

        Returns:
            True if initialization successful
        """
        if self._initialized:
            logger.warning("Firebase already initialized")
            return True

        self._account_id = account_id

        try:
            path = Path(service_account_path)
            if not path.exists():
                logger.error(f"Service account file not found: {service_account_path}")
                return False

            # Initialize Firebase if not already done
            if not firebase_admin._apps:
                cred = credentials.Certificate(str(path))
                firebase_admin.initialize_app(cred)

            self._db = firestore.client()
            self._initialized = True

            logger.info(f"Firebase initialized successfully for account: {account_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            return False

    @property
    def db(self) -> Client:
        """Get the Firestore client."""
        if not self._initialized or self._db is None:
            raise RuntimeError("Firebase not initialized. Call initialize() first.")
        return self._db

    @property
    def account_id(self) -> str:
        """Get the current account ID."""
        if self._account_id is None:
            raise RuntimeError("Firebase not initialized. Call initialize() first.")
        return self._account_id

    @property
    def initialized(self) -> bool:
        """Check if Firebase is initialized."""
        return self._initialized

    def collection(self, name: str) -> CollectionReference:
        """Get a collection reference."""
        return self.db.collection(name)

    def document(self, path: str) -> DocumentReference:
        """Get a document reference by path."""
        return self.db.document(path)

    def get_account_ref(self) -> DocumentReference:
        """Get the account document reference for the current session."""
        return self.collection(self.COLLECTION_ACCOUNTS).document(self._account_id)

    def get_orders_ref(self) -> CollectionReference:
        """Get the orders subcollection reference."""
        return self.get_account_ref().collection(self.SUBCOLLECTION_ORDERS)

    def get_trades_ref(self) -> CollectionReference:
        """Get the trades subcollection reference."""
        return self.get_account_ref().collection(self.SUBCOLLECTION_TRADES)

    def get_portfolio_ref(self) -> DocumentReference:
        """Get the portfolio document reference."""
        return self.get_account_ref().collection(self.SUBCOLLECTION_PORTFOLIO).document("current")

    def get_actions_ref(self) -> CollectionReference:
        """Get the actions subcollection reference."""
        return self.get_account_ref().collection(self.SUBCOLLECTION_ACTIONS)

    def get_inventory_ref(self) -> DocumentReference:
        """Get the inventory document reference."""
        return self.get_account_ref().collection(self.SUBCOLLECTION_INVENTORY).document(self.DOC_CURRENT)

    def get_bank_ref(self) -> DocumentReference:
        """Get the bank document reference."""
        return self.get_account_ref().collection(self.SUBCOLLECTION_BANK).document(self.DOC_CURRENT)

    def get_ge_slots_ref(self) -> DocumentReference:
        """Get the GE slots document reference."""
        return self.get_account_ref().collection(self.SUBCOLLECTION_GE_SLOTS).document(self.DOC_CURRENT)

    def get_commands_ref(self) -> CollectionReference:
        """Get the commands subcollection reference."""
        return self.get_account_ref().collection(self.SUBCOLLECTION_COMMANDS)

    # =========================================================================
    # Item Lookups
    # =========================================================================

    def get_item_by_id(self, item_id: int) -> Optional[Dict[str, Any]]:
        """Get item data by ID from the items collection."""
        try:
            doc = self.collection(self.COLLECTION_ITEMS).document(str(item_id)).get()
            if doc.exists:
                return doc.to_dict()
        except Exception as e:
            logger.error(f"Failed to get item by ID {item_id}: {e}")
        return None

    def get_item_id_by_name(self, item_name: str) -> Optional[int]:
        """Get item ID by name from the itemNames collection (fast lookup)."""
        try:
            # Sanitize name (same as when stored)
            sanitized_name = item_name.replace("/", "_")
            doc = self.collection(self.COLLECTION_ITEM_NAMES).document(sanitized_name).get()
            if doc.exists:
                data = doc.to_dict()
                return data.get("id") if data else None
        except Exception as e:
            logger.error(f"Failed to get item ID by name '{item_name}': {e}")
        return None

    def get_item_by_name(self, item_name: str) -> Optional[Dict[str, Any]]:
        """Get full item data by name (two lookups: name→ID, then ID→data)."""
        item_id = self.get_item_id_by_name(item_name)
        if item_id is not None:
            return self.get_item_by_id(item_id)
        return None

    # =========================================================================
    # Batch Operations
    # =========================================================================

    def batch_write(self, operations: List[Dict[str, Any]]) -> bool:
        """
        Execute a batch write operation.

        Args:
            operations: List of operations, each with:
                - "type": "set" | "update" | "delete"
                - "ref": DocumentReference
                - "data": dict (for set/update)

        Returns:
            True if successful
        """
        try:
            batch = self.db.batch()

            for op in operations:
                op_type = op.get("type")
                ref = op.get("ref")
                data = op.get("data", {})

                if op_type == "set":
                    batch.set(ref, data)
                elif op_type == "update":
                    batch.update(ref, data)
                elif op_type == "delete":
                    batch.delete(ref)

            batch.commit()
            return True

        except Exception as e:
            logger.error(f"Batch write failed: {e}")
            return False

    # =========================================================================
    # Shutdown
    # =========================================================================

    def shutdown(self):
        """Shutdown Firebase connection."""
        if self._initialized:
            try:
                # Close Firestore client
                if self._db:
                    self._db.close()
            except Exception as e:
                logger.error(f"Error closing Firestore: {e}")

            self._initialized = False
            self._db = None
            logger.info("Firebase shutdown complete")

    @classmethod
    def reset(cls):
        """Reset the singleton instance (useful for testing)."""
        if cls._instance:
            cls._instance.shutdown()
        cls._instance = None
        cls._db = None
        cls._initialized = False
        cls._account_id = None
