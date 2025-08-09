from __future__ import annotations

from typing import Dict, Any


async def publish_to_chat(chat_id: str, message: Dict[str, Any]) -> None:
    """Broadcast a message to all active websocket connections in a chat.

    Imported lazily inside to avoid import cycles during graph/node imports.
    """
    from app.websocket.manager import websocket_manager  # lazy to avoid circular import
    await websocket_manager._broadcast_to_chat(chat_id, message)


