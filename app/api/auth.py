"""
Authentication API endpoints.
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
# Rate limiting removed

from app.schemas.auth import (
    UserRegistration,
    UserLogin,
    ChangePassword
)
from app.schemas.token import Token, RefreshToken
from app.schemas.user import UserResponse, UserProfileResponse
from app.services.auth_service import auth_service
from app.services.user_service import user_service
from app.middleware.auth import get_current_active_user, security
from app.models.user import UserInDB

logger = logging.getLogger(__name__)

# Rate limiting removed

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="Create a new user account with email and password"
)
async def register(user_data: UserRegistration):
    """Register a new user."""
    try:
        user, token = await auth_service.register_user(user_data)
        
        # Convert user to response format
        user_response = UserResponse(
            id=str(user.id),
            email=user.email,
            is_active=user.is_active,
            is_verified=user.is_verified,
            profile=UserProfileResponse(**user.profile.dict()),
            created_at=user.created_at,
            last_login=user.last_login
        )
        
        return {
            "message": "User registered successfully",
            "user": user_response,
            "token": token
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post(
    "/login",
    response_model=dict,
    summary="User login",
    description="Authenticate user with email and password"
)
async def login(login_data: UserLogin):
    """Authenticate user and return access token."""
    try:
        user, token = await auth_service.login_user(login_data)
        
        # Convert user to response format
        user_response = UserResponse(
            id=str(user.id),
            email=user.email,
            is_active=user.is_active,
            is_verified=user.is_verified,
            profile=UserProfileResponse(**user.profile.dict()),
            created_at=user.created_at,
            last_login=user.last_login
        )
        
        return {
            "message": "Login successful",
            "user": user_response,
            "token": token
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post(
    "/refresh",
    response_model=Token,
    summary="Refresh access token",
    description="Generate new access token using refresh token"
)
async def refresh_token(refresh_data: RefreshToken):
    """Refresh access token."""
    try:
        token = await auth_service.refresh_access_token(refresh_data.refresh_token)
        return token
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post(
    "/logout",
    response_model=dict,
    summary="User logout",
    description="Logout user by blacklisting tokens"
)
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    refresh_token: Optional[str] = None
):
    """Logout user by blacklisting tokens."""
    try:
        access_token = credentials.credentials
        await auth_service.logout_user(access_token, refresh_token)
        
        return {"message": "Logout successful"}
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# Password reset endpoints removed


@router.post(
    "/change-password",
    response_model=dict,
    summary="Change password",
    description="Change user password (requires authentication)"
)
async def change_password(
    password_data: ChangePassword,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """Change user password."""
    try:
        success = await user_service.change_password(str(current_user.id), password_data)
        
        if success:
            return {"message": "Password changed successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password change failed"
            )
            
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Password change error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
