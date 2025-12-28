"""
Firebase Integration Module for PPO Flipper

This module provides Firebase Firestore integration for real-time communication
between the PPO inference server and the GE Auto RuneLite plugin.

Components:
    - FirebaseClient: Core Firestore connection singleton
    - OrderManager: Create and manage trading orders
    - TradeMonitor: Listen for completed trades
    - PortfolioTracker: Track account portfolio state
    - InferenceBridge: Main orchestrator for PPO → Firebase → Plugin
"""

from .firebase_client import FirebaseClient
from .order_manager import OrderManager, OrderStatus, OrderAction
from .trade_monitor import TradeMonitor
from .portfolio_tracker import PortfolioTracker
from .inference_bridge import InferenceBridge

__all__ = [
    'FirebaseClient',
    'OrderManager',
    'OrderStatus',
    'OrderAction',
    'TradeMonitor',
    'PortfolioTracker',
    'InferenceBridge',
]
