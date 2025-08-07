"""
Authentication middleware and dependencies for FastAPI.
"""

import logging
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.models.user import UserInDB
from app.services.security import security_service
from app.services.user_service import user_service
from app.services.auth_service import auth_service

logger = logging.getLogger(__name__)

# HTTP Bearer token scheme
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> UserInDB:
    """
    Dependency to get current authenticated user from JWT token.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Extract token from credentials
        token = credentials.credentials
        
        # Check if token is blacklisted
        if await auth_service.is_token_blacklisted(token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Verify token
        token_data = security_service.verify_token(token)
        if token_data is None:
            raise credentials_exception
        
        # Check if token is expired
        if security_service.is_token_expired(token_data):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Check token type
        if token_data.token_type != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Get user from database
        user = await user_service.get_user_by_id(token_data.user_id)
        if user is None:
            raise credentials_exception
        
        # Check if user is active
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is deactivated",
            )
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_current_user: {e}")
        raise credentials_exception


async def get_current_active_user(
    current_user: UserInDB = Depends(get_current_user)
) -> UserInDB:
    """
    Dependency to get current active user.
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


async def get_current_verified_user(
    current_user: UserInDB = Depends(get_current_active_user)
) -> UserInDB:
    """
    Dependency to get current verified user.
    """
    if not current_user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email not verified"
        )
    return current_user


async def get_current_superuser(
    current_user: UserInDB = Depends(get_current_active_user)
) -> UserInDB:
    """
    Dependency to get current superuser.
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user


async def get_optional_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> Optional[UserInDB]:
    """
    Dependency to get current user if token is provided, otherwise return None.
    Useful for endpoints that work for both authenticated and anonymous users.
    """
    if not credentials:
        return None
    
    try:
        # Extract token from credentials
        token = credentials.credentials
        
        # Check if token is blacklisted
        if await auth_service.is_token_blacklisted(token):
            return None
        
        # Verify token
        token_data = security_service.verify_token(token)
        if token_data is None:
            return None
        
        # Check if token is expired
        if security_service.is_token_expired(token_data):
            return None
        
        # Check token type
        if token_data.token_type != "access":
            return None
        
        # Get user from database
        user = await user_service.get_user_by_id(token_data.user_id)
        if user is None or not user.is_active:
            return None
        
        return user
        
    except Exception as e:
        logger.warning(f"Error in get_optional_current_user: {e}")
        return None
