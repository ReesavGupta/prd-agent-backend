"""WebSocket package for real-time chat functionality."""

from .manager import WebSocketManager
from .auth import authenticate_websocket

__all__ = ["WebSocketManager", "authenticate_websocket"]