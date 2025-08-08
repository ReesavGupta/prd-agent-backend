"""
WebSocket connection manager for real-time chat functionality.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Set
from fastapi import WebSocket, WebSocketDisconnect

from app.models.user import UserInDB, PyObjectId
from app.models.chat import ChatSession
from app.schemas.chat import WebSocketMessageRequest, WebSocketMessageResponse
from app.services.chat_service import chat_service

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manager for WebSocket connections and chat sessions."""
    
    def __init__(self):
        # Active connections: chat_id -> set of websockets
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        
        # Connection to user mapping: websocket -> user_info
        self.connection_users: Dict[WebSocket, Dict[str, str]] = {}
        
        # Connection to session mapping: websocket -> session_id
        self.connection_sessions: Dict[WebSocket, str] = {}
        
        # Typing indicators: chat_id -> set of user_ids
        self.typing_users: Dict[str, Set[str]] = {}
    
    async def connect(
        self, 
        websocket: WebSocket, 
        chat_id: str, 
        user: UserInDB
    ) -> bool:
        """Accept WebSocket connection and create chat session."""
        try:
            await websocket.accept()
            
            # Generate unique websocket ID
            websocket_id = str(uuid.uuid4())
            
            # Create chat session
            session = await chat_service.create_websocket_session(
                chat_id=PyObjectId(chat_id),
                user_id=user.id,
                websocket_id=websocket_id
            )
            
            # Add to active connections
            if chat_id not in self.active_connections:
                self.active_connections[chat_id] = set()
            self.active_connections[chat_id].add(websocket)
            
            # Store connection metadata
            self.connection_users[websocket] = {
                "user_id": str(user.id),
                "chat_id": chat_id,
                "websocket_id": websocket_id
            }
            self.connection_sessions[websocket] = session.session_id
            
            logger.info(f"WebSocket connected for user {user.id} in chat {chat_id}")
            
            # Send connection confirmation
            await self._send_to_websocket(websocket, {
                "type": "connection_confirmed",
                "data": {
                    "session_id": session.session_id,
                    "chat_id": chat_id
                }
            })
            
            return True
        except Exception as e:
            logger.error(f"Error connecting WebSocket: {e}")
            try:
                await websocket.close(code=4000, reason="Connection failed")
            except:
                pass
            return False
    
    async def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection."""
        try:
            if websocket not in self.connection_users:
                return
            
            connection_info = self.connection_users[websocket]
            chat_id = connection_info["chat_id"]
            user_id = connection_info["user_id"]
            session_id = self.connection_sessions.get(websocket)
            
            # Remove from active connections
            if chat_id in self.active_connections:
                self.active_connections[chat_id].discard(websocket)
                if not self.active_connections[chat_id]:
                    del self.active_connections[chat_id]
            
            # Remove from typing users
            if chat_id in self.typing_users:
                self.typing_users[chat_id].discard(user_id)
                if not self.typing_users[chat_id]:
                    del self.typing_users[chat_id]
            
            # Clean up connection metadata
            self.connection_users.pop(websocket, None)
            self.connection_sessions.pop(websocket, None)
            
            # Delete session from database
            if session_id:
                await chat_service.session_repo.delete_session(session_id)
            
            logger.info(f"WebSocket disconnected for user {user_id} in chat {chat_id}")
            
            # Notify other users about disconnection
            await self._broadcast_to_chat(chat_id, {
                "type": "user_disconnected",
                "data": {
                    "user_id": user_id
                }
            }, exclude_websocket=websocket)
            
        except Exception as e:
            logger.error(f"Error handling WebSocket disconnection: {e}")
    
    async def handle_message(self, websocket: WebSocket, data: dict):
        """Handle incoming WebSocket message."""
        try:
            if websocket not in self.connection_users:
                await self._send_error(websocket, "Connection not authenticated")
                return
            
            connection_info = self.connection_users[websocket]
            chat_id = connection_info["chat_id"]
            user_id = connection_info["user_id"]
            session_id = self.connection_sessions.get(websocket)
            
            # Parse message
            try:
                message = WebSocketMessageRequest(**data)
            except Exception as e:
                await self._send_error(websocket, f"Invalid message format: {e}")
                return
            
            # Handle different message types
            if message.type == "send_message":
                await self._handle_send_message(websocket, chat_id, user_id, message.data)
            
            elif message.type == "typing_start":
                await self._handle_typing_start(websocket, chat_id, user_id, session_id)
            
            elif message.type == "typing_stop":
                await self._handle_typing_stop(websocket, chat_id, user_id, session_id)
            
            elif message.type == "ping":
                await self._handle_ping(websocket, session_id)
            
            else:
                await self._send_error(websocket, f"Unknown message type: {message.type}")
            
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            await self._send_error(websocket, "Message processing failed")
    
    async def _handle_send_message(
        self, 
        websocket: WebSocket, 
        chat_id: str, 
        user_id: str, 
        data: dict
    ):
        """Handle send message request."""
        try:
            content = data.get("content")
            if not content:
                await self._send_error(websocket, "Message content is required")
                return
            
            # Create message request
            from app.schemas.chat import MessageCreateRequest
            request = MessageCreateRequest(
                content=content,
                message_type="user"
            )
            
            # Send message through service
            user_message, ai_message = await chat_service.send_message(
                chat_id=PyObjectId(chat_id),
                user_id=PyObjectId(user_id),
                request=request
            )
            
            # Broadcast user message to all connections in chat
            await self._broadcast_to_chat(chat_id, {
                "type": "message_sent",
                "data": {
                    "message_id": str(user_message.id),
                    "user_id": user_id,
                    "content": user_message.content,
                    "timestamp": user_message.timestamp.isoformat(),
                    "message_type": user_message.message_type
                }
            })
            
            # If AI responded, broadcast that too
            if ai_message:
                await self._broadcast_to_chat(chat_id, {
                    "type": "ai_response_complete",
                    "data": {
                        "message_id": str(ai_message.id),
                        "content": ai_message.content,
                        "timestamp": ai_message.timestamp.isoformat(),
                        "metadata": {
                            "model_used": ai_message.metadata.model_used if ai_message.metadata else None,
                            "response_time_ms": ai_message.metadata.response_time_ms if ai_message.metadata else None
                        }
                    }
                })
            
        except Exception as e:
            logger.error(f"Error handling send message: {e}")
            await self._send_error(websocket, "Failed to send message")
    
    async def _handle_typing_start(
        self, 
        websocket: WebSocket, 
        chat_id: str, 
        user_id: str, 
        session_id: str
    ):
        """Handle typing start indicator."""
        try:
            # Update session typing status
            if session_id:
                await chat_service.update_typing_status(session_id, True)
            
            # Add to typing users
            if chat_id not in self.typing_users:
                self.typing_users[chat_id] = set()
            self.typing_users[chat_id].add(user_id)
            
            # Broadcast typing indicator
            await self._broadcast_to_chat(chat_id, {
                "type": "typing_indicator",
                "data": {
                    "user_id": user_id,
                    "is_typing": True
                }
            }, exclude_websocket=websocket)
            
        except Exception as e:
            logger.error(f"Error handling typing start: {e}")
    
    async def _handle_typing_stop(
        self, 
        websocket: WebSocket, 
        chat_id: str, 
        user_id: str, 
        session_id: str
    ):
        """Handle typing stop indicator."""
        try:
            # Update session typing status
            if session_id:
                await chat_service.update_typing_status(session_id, False)
            
            # Remove from typing users
            if chat_id in self.typing_users:
                self.typing_users[chat_id].discard(user_id)
                if not self.typing_users[chat_id]:
                    del self.typing_users[chat_id]
            
            # Broadcast typing stop
            await self._broadcast_to_chat(chat_id, {
                "type": "typing_indicator",
                "data": {
                    "user_id": user_id,
                    "is_typing": False
                }
            }, exclude_websocket=websocket)
            
        except Exception as e:
            logger.error(f"Error handling typing stop: {e}")
    
    async def _handle_ping(self, websocket: WebSocket, session_id: str):
        """Handle ping message for keepalive."""
        try:
            # Update session activity
            if session_id:
                await chat_service.session_repo.update_session_activity(session_id)
            
            # Send pong response
            await self._send_to_websocket(websocket, {
                "type": "pong",
                "data": {"timestamp": datetime.utcnow().isoformat()}
            })
            
        except Exception as e:
            logger.error(f"Error handling ping: {e}")
    
    async def _broadcast_to_chat(
        self, 
        chat_id: str, 
        message: dict, 
        exclude_websocket: Optional[WebSocket] = None
    ):
        """Broadcast message to all connections in a chat."""
        if chat_id not in self.active_connections:
            return
        
        connections = self.active_connections[chat_id].copy()
        for websocket in connections:
            if websocket != exclude_websocket:
                await self._send_to_websocket(websocket, message)
    
    async def _send_to_websocket(self, websocket: WebSocket, message: dict):
        """Send message to a specific WebSocket."""
        try:
            response = WebSocketMessageResponse(**message)
            await websocket.send_text(response.model_dump_json())
        except WebSocketDisconnect:
            # Connection already closed, clean up
            await self.disconnect(websocket)
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
            # Don't disconnect on send errors, connection might still be valid
    
    async def _send_error(self, websocket: WebSocket, error_message: str):
        """Send error message to WebSocket."""
        await self._send_to_websocket(websocket, {
            "type": "error",
            "data": {"message": error_message}
        })
    
    def get_chat_connections_count(self, chat_id: str) -> int:
        """Get number of active connections for a chat."""
        return len(self.active_connections.get(chat_id, set()))
    
    def get_typing_users(self, chat_id: str) -> List[str]:
        """Get list of users currently typing in a chat."""
        return list(self.typing_users.get(chat_id, set()))


# Global WebSocket manager instance
websocket_manager = WebSocketManager()