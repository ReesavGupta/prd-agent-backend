"""
User management API endpoints.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.user import (
    UserResponse,
    UserProfileResponse,
    UserProfileUpdate
)
from app.services.user_service import user_service
from app.middleware.auth import get_current_active_user
from app.models.user import UserInDB

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["Users"])


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user profile",
    description="Get the profile of the currently authenticated user"
)
async def get_current_user_profile(
    current_user: UserInDB = Depends(get_current_active_user)
):
    """Get current user profile."""
    try:
        user_response = UserResponse(
            id=str(current_user.id),
            email=current_user.email,
            is_active=current_user.is_active,
            is_verified=current_user.is_verified,
            profile=UserProfileResponse(**current_user.profile.dict()),
            created_at=current_user.created_at,
            last_login=current_user.last_login
        )
        
        return user_response
        
    except Exception as e:
        logger.error(f"Get current user profile error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.put(
    "/me",
    response_model=UserResponse,
    summary="Update current user profile",
    description="Update the profile of the currently authenticated user"
)
async def update_current_user_profile(
    profile_data: UserProfileUpdate,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """Update current user profile."""
    try:
        updated_user = await user_service.update_user_profile(
            str(current_user.id),
            profile_data
        )
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        user_response = UserResponse(
            id=str(updated_user.id),
            email=updated_user.email,
            is_active=updated_user.is_active,
            is_verified=updated_user.is_verified,
            profile=UserProfileResponse(**updated_user.profile.dict()),
            created_at=updated_user.created_at,
            last_login=updated_user.last_login
        )
        
        return user_response
        
    except Exception as e:
        logger.error(f"Update user profile error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get(
    "/{user_id}",
    response_model=UserResponse,
    summary="Get user by ID",
    description="Get user profile by user ID (public information only)"
)
async def get_user_by_id(user_id: str):
    """Get user by ID (public information only)."""
    try:
        user = await user_service.get_user_by_id(user_id)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Return only public information
        user_response = UserResponse(
            id=str(user.id),
            email=user.email,
            is_active=user.is_active,
            is_verified=user.is_verified,
            profile=UserProfileResponse(
                first_name=user.profile.first_name,
                last_name=user.profile.last_name,
                bio=user.profile.bio
                # Don't include phone_number and other private info
            ),
            created_at=user.created_at,
            last_login=None  # Don't expose last login to other users
        )
        
        return user_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get user by ID error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
