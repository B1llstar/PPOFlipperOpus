"""
Firebase Client V2 - Singleton Firestore connection.

Provides access to all V2 collection and document references.
"""

import logging
from pathlib import Path
from typing import Optional

from google.cloud import firestore
from google.oauth2 import service_account

logger = logging.getLogger(__name__)


class FirebaseClientV2:
    """
    Singleton Firebase/Firestore client for V2 architecture.

    Collections:
    - items: Static item database
    - itemNames: Item name -> ID lookup
    - accounts/{accountId}/portfolio: Portfolio summary
    - accounts/{accountId}/inventory: Inventory contents
    - accounts/{accountId}/bank: Bank contents
    - accounts/{accountId}/ge_state: GE slot states
    - accounts/{accountId}/orders: Orders subcollection
    """

    _instance: Optional['FirebaseClientV2'] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        if FirebaseClientV2._initialized:
            return
        self._db: Optional[firestore.Client] = None
        self._account_id: Optional[str] = None
        self._project_id: str = "ppoflipperopus"

    @classmethod
    def get_instance(cls) -> 'FirebaseClientV2':
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(
        self,
        service_account_path: str,
        account_id: str,
        project_id: str = "ppoflipperopus"
    ) -> bool:
        """
        Initialize the Firestore connection.

        Args:
            service_account_path: Path to Firebase service account JSON
            account_id: The account ID (player name, lowercased, spaces -> _)
            project_id: Firebase project ID

        Returns:
            True if initialization successful
        """
        if FirebaseClientV2._initialized and self._db is not None:
            logger.info("Firebase already initialized")
            return True

        try:
            # Verify service account file exists
            sa_path = Path(service_account_path)
            if not sa_path.exists():
                logger.error(f"Service account file not found: {service_account_path}")
                return False

            # Create credentials
            credentials = service_account.Credentials.from_service_account_file(
                str(sa_path)
            )

            # Initialize Firestore client
            self._db = firestore.Client(
                project=project_id,
                credentials=credentials
            )

            self._account_id = account_id
            self._project_id = project_id
            FirebaseClientV2._initialized = True

            logger.info(f"Firebase V2 initialized for account: {account_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            return False

    @property
    def db(self) -> firestore.Client:
        """Get the Firestore client."""
        if self._db is None:
            raise RuntimeError("Firebase not initialized. Call initialize() first.")
        return self._db

    @property
    def account_id(self) -> str:
        """Get the current account ID."""
        if self._account_id is None:
            raise RuntimeError("Firebase not initialized. Call initialize() first.")
        return self._account_id

    @property
    def is_initialized(self) -> bool:
        """Check if Firebase is initialized."""
        return FirebaseClientV2._initialized and self._db is not None

    # =========================================================================
    # Collection References
    # =========================================================================

    @property
    def items_ref(self) -> firestore.CollectionReference:
        """Reference to items collection."""
        return self.db.collection('items')

    @property
    def item_names_ref(self) -> firestore.CollectionReference:
        """Reference to itemNames collection."""
        return self.db.collection('itemNames')

    @property
    def accounts_ref(self) -> firestore.CollectionReference:
        """Reference to accounts collection."""
        return self.db.collection('accounts')

    def account_ref(self, account_id: Optional[str] = None) -> firestore.DocumentReference:
        """Reference to a specific account document."""
        aid = account_id or self._account_id
        return self.accounts_ref.document(aid)

    # =========================================================================
    # Account Document References
    # =========================================================================

    @property
    def portfolio_ref(self) -> firestore.DocumentReference:
        """Reference to portfolio document for current account."""
        return self.account_ref().collection('portfolio').document('current')

    @property
    def inventory_ref(self) -> firestore.DocumentReference:
        """Reference to inventory document for current account."""
        return self.account_ref().collection('inventory').document('current')

    @property
    def bank_ref(self) -> firestore.DocumentReference:
        """Reference to bank document for current account."""
        return self.account_ref().collection('bank').document('current')

    @property
    def ge_state_ref(self) -> firestore.DocumentReference:
        """Reference to ge_state document for current account."""
        return self.account_ref().collection('ge_state').document('current')

    @property
    def orders_ref(self) -> firestore.CollectionReference:
        """Reference to orders subcollection for current account."""
        return self.account_ref().collection('orders')

    # =========================================================================
    # Item Lookups
    # =========================================================================

    def get_item_by_id(self, item_id: int) -> Optional[dict]:
        """
        Get item data by ID.

        Args:
            item_id: The item ID

        Returns:
            Item data dict or None if not found
        """
        try:
            doc = self.items_ref.document(str(item_id)).get()
            if doc.exists:
                return doc.to_dict()
            return None
        except Exception as e:
            logger.error(f"Error getting item {item_id}: {e}")
            return None

    def get_item_by_name(self, item_name: str) -> Optional[dict]:
        """
        Get item data by name.

        Args:
            item_name: The item name (case-insensitive)

        Returns:
            Item data dict or None if not found
        """
        try:
            # Normalize name for lookup
            normalized = item_name.lower().strip()
            doc = self.item_names_ref.document(normalized).get()
            if doc.exists:
                data = doc.to_dict()
                item_id = data.get('id')
                if item_id:
                    return self.get_item_by_id(item_id)
            return None
        except Exception as e:
            logger.error(f"Error getting item by name '{item_name}': {e}")
            return None

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def shutdown(self):
        """Shutdown the Firebase connection."""
        if self._db is not None:
            self._db.close()
            self._db = None
            FirebaseClientV2._initialized = False
            logger.info("Firebase V2 shutdown complete")

    @classmethod
    def reset(cls):
        """Reset the singleton (for testing)."""
        if cls._instance is not None:
            cls._instance.shutdown()
            cls._instance = None
            cls._initialized = False
