"""
WebSocket endpoints for real-time chat functionality.
"""

import logging
import json
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from app.websocket.manager import websocket_manager
from app.websocket.auth import authenticate_websocket, handle_auth_error
from app.services.chat_service import chat_service
from app.models.user import PyObjectId

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


@router.websocket("/chats/{chat_id}")
async def websocket_chat_endpoint(
    websocket: WebSocket,
    chat_id: str,
    token: Optional[str] = Query(None, description="JWT authentication token")
):
    """WebSocket endpoint for real-time chat communication."""
    user = None
    
    try:
        # Authenticate user
        if not token:
            await handle_auth_error(websocket, "Authentication token required")
            return
        
        user = await authenticate_websocket(websocket, token)
        if not user:
            await handle_auth_error(websocket, "Invalid authentication token")
            return
        
        # Verify user has access to the chat
        try:
            chat = await chat_service.get_chat(PyObjectId(chat_id), user.id)
            if not chat:
                await handle_auth_error(websocket, "Chat not found or access denied")
                return
        except ValueError:
            await handle_auth_error(websocket, "Invalid chat ID")
            return
        
        # Connect to WebSocket manager
        connected = await websocket_manager.connect(websocket, chat_id, user)
        if not connected:
            logger.error(f"Failed to establish WebSocket connection for chat {chat_id}")
            return
        
        logger.info(f"WebSocket established for user {user.id} in chat {chat_id}")
        
        # Handle messages
        try:
            while True:
                # Receive message
                try:
                    data = await websocket.receive_text()
                    message_data = json.loads(data)
                except json.JSONDecodeError:
                    await websocket_manager._send_error(websocket, "Invalid JSON format")
                    continue
                except WebSocketDisconnect:
                    # Client disconnected; exit receive loop cleanly
                    break
                except Exception as e:
                    # Likely disconnected/closed; avoid tight error loop
                    logger.error(f"Error receiving WebSocket message: {e}")
                    break
                
                # Handle message
                await websocket_manager.handle_message(websocket, message_data)
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for user {user.id} in chat {chat_id}")
        except Exception as e:
            logger.error(f"WebSocket error for user {user.id if user else 'unknown'} in chat {chat_id}: {e}")
        
    except Exception as e:
        logger.error(f"WebSocket endpoint error: {e}")
    
    finally:
        # Clean up connection
        try:
            await websocket_manager.disconnect(websocket)
        except Exception as e:
            logger.error(f"Error cleaning up WebSocket connection: {e}")


@router.websocket("/projects/{project_id}")
async def websocket_project_endpoint(
    websocket: WebSocket,
    project_id: str,
    token: Optional[str] = Query(None, description="JWT authentication token")
):
    """WebSocket endpoint for real-time project updates."""
    # TODO: Implement project WebSocket functionality for Phase 4
    await websocket.close(code=4001, reason="Project WebSocket not implemented yet")


# Health check endpoint for WebSocket service
@router.get("/health")
async def websocket_health():
    """Health check for WebSocket service."""
    active_connections = sum(
        websocket_manager.get_chat_connections_count(chat_id) 
        for chat_id in websocket_manager.active_connections.keys()
    )
    
    return {
        "status": "healthy",
        "active_connections": active_connections,
        "active_chats": len(websocket_manager.active_connections)
    }