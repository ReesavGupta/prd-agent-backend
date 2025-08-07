"""
Authentication service for handling login, logout, and token management.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple
from bson import ObjectId

from app.db.database import get_database
from app.models.user import UserInDB, TokenBlacklist
from app.schemas.auth import UserRegistration, UserLogin
from app.schemas.token import Token, TokenData
from app.services.security import security_service
from app.services.user_service import user_service
from app.core.config import settings

logger = logging.getLogger(__name__)


class AuthService:
    """Service for authentication operations."""
    
    async def register_user(self, user_data: UserRegistration) -> Tuple[UserInDB, Token]:
        """Register a new user and return user data with tokens."""
        # Create user
        user = await user_service.create_user(user_data)
        
        # Generate tokens
        token_data = {
            "sub": str(user.id),
            "email": user.email
        }
        
        access_token = security_service.create_access_token(token_data)
        refresh_token = security_service.create_refresh_token(token_data)
        
        token = Token(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=security_service.get_token_expiry_seconds("access")
        )
        
        logger.info(f"User registered successfully: {user.email}")
        return user, token
    
    async def login_user(self, login_data: UserLogin) -> Tuple[UserInDB, Token]:
        """Authenticate user and return user data with tokens."""
        # Authenticate user
        user = await user_service.authenticate_user(login_data.email, login_data.password)
        
        if not user:
            raise ValueError("Invalid email or password")
        
        # Generate tokens
        token_data = {
            "sub": str(user.id),
            "email": user.email
        }
        
        access_token = security_service.create_access_token(token_data)
        refresh_token = security_service.create_refresh_token(token_data)
        
        token = Token(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=security_service.get_token_expiry_seconds("access")
        )
        
        logger.info(f"User logged in successfully: {user.email}")
        return user, token
    
    async def refresh_access_token(self, refresh_token: str) -> Token:
        """Generate new access token using refresh token."""
        # Verify refresh token
        token_data = security_service.verify_token(refresh_token)
        
        if not token_data or token_data.token_type != "refresh":
            raise ValueError("Invalid refresh token")
        
        if security_service.is_token_expired(token_data):
            raise ValueError("Refresh token has expired")
        
        # Check if token is blacklisted
        if await self.is_token_blacklisted(refresh_token):
            raise ValueError("Token has been revoked")
        
        # Verify user still exists and is active
        user = await user_service.get_user_by_id(token_data.user_id)
        if not user or not user.is_active:
            raise ValueError("User not found or inactive")
        
        # Generate new tokens
        new_token_data = {
            "sub": str(user.id),
            "email": user.email
        }
        
        access_token = security_service.create_access_token(new_token_data)
        new_refresh_token = security_service.create_refresh_token(new_token_data)
        
        # Blacklist old refresh token
        await self.blacklist_token(refresh_token, token_data.user_id, "Token refreshed")
        
        token = Token(
            access_token=access_token,
            refresh_token=new_refresh_token,
            expires_in=security_service.get_token_expiry_seconds("access")
        )
        
        logger.info(f"Token refreshed for user: {user.email}")
        return token
    
    async def logout_user(self, access_token: str, refresh_token: Optional[str] = None) -> bool:
        """Logout user by blacklisting tokens."""
        # Verify access token
        token_data = security_service.verify_token(access_token)
        
        if not token_data:
            raise ValueError("Invalid access token")
        
        # Blacklist access token
        await self.blacklist_token(access_token, token_data.user_id, "User logout")
        
        # Blacklist refresh token if provided
        if refresh_token:
            refresh_token_data = security_service.verify_token(refresh_token)
            if refresh_token_data:
                await self.blacklist_token(refresh_token, refresh_token_data.user_id, "User logout")
        
        logger.info(f"User logged out: {token_data.email}")
        return True
    
    async def blacklist_token(self, token: str, user_id: str, reason: Optional[str] = None) -> None:
        """Add token to blacklist."""
        db = get_database()
        
        # Get token expiration
        token_data = security_service.verify_token(token)
        if not token_data:
            return
        
        blacklist_entry = TokenBlacklist(
            token=token,
            user_id=ObjectId(user_id),
            expires_at=token_data.exp,
            reason=reason
        )
        
        await db.token_blacklist.insert_one(blacklist_entry.dict(by_alias=True))
    
    async def is_token_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted."""
        db = get_database()
        
        blacklist_entry = await db.token_blacklist.find_one({"token": token})
        return blacklist_entry is not None
    
    # Password reset functionality removed


# Global auth service instance
auth_service = AuthService()
