"""
Firebase V2 - Simplified Firestore interface for GE Auto V2.

This module provides a clean, streamlined interface between the PPO inference
server and the GE Auto V2 RuneLite plugin via Firestore.

Architecture:
- firebase_client.py: Firestore connection singleton
- state_listener.py: Real-time portfolio/inventory/bank/GE state updates
- order_manager.py: Order creation and status monitoring
- inference_bridge.py: High-level interface for PPO inference
"""

from .firebase_client import FirebaseClientV2
from .state_listener import StateListener
from .order_manager import OrderManagerV2
from .inference_bridge import InferenceBridgeV2

__all__ = [
    'FirebaseClientV2',
    'StateListener',
    'OrderManagerV2',
    'InferenceBridgeV2',
]
