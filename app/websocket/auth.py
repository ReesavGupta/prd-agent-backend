"""
WebSocket authentication utilities.
"""

import logging
from typing import Optional
from fastapi import WebSocket, WebSocketDisconnect
from jose import JWTError, jwt

from app.core.config import settings
from app.models.user import UserInDB, PyObjectId
from app.db.database import get_database

logger = logging.getLogger(__name__)


async def authenticate_websocket(websocket: WebSocket, token: str) -> Optional[UserInDB]:
    """Authenticate WebSocket connection using JWT token."""
    try:
        # Decode JWT token
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        
        user_id = payload.get("sub")
        if not user_id:
            logger.warning("WebSocket authentication failed: No user ID in token")
            return None
        
        # Get user from database
        db = get_database()
        user_doc = await db.users.find_one({"_id": PyObjectId(user_id)})
        
        if not user_doc:
            logger.warning(f"WebSocket authentication failed: User {user_id} not found")
            return None
        
        user = UserInDB(**user_doc)
        
        # Check if user is active
        if not user.is_active:
            logger.warning(f"WebSocket authentication failed: User {user_id} is inactive")
            return None
        
        # Check token blacklist
        token_doc = await db.token_blacklist.find_one({"token": token})
        if token_doc:
            logger.warning("WebSocket authentication failed: Token is blacklisted")
            return None
        
        logger.info(f"WebSocket authenticated for user {user_id}")
        return user
        
    except JWTError as e:
        logger.warning(f"WebSocket JWT error: {e}")
        return None
    except Exception as e:
        logger.error(f"WebSocket authentication error: {e}")
        return None


async def handle_auth_error(websocket: WebSocket, error_message: str):
    """Handle authentication error by closing WebSocket connection."""
    try:
        await websocket.close(code=4001, reason=error_message)
    except Exception as e:
        logger.error(f"Error closing WebSocket: {e}")