"""
User-related Pydantic schemas for request/response validation.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, EmailStr, Field

# from app.models.user import PyObjectId  # Not used in this file


class UserProfileUpdate(BaseModel):
    """Schema for updating user profile."""

    first_name: Optional[str] = Field(None, max_length=50, description="First name")
    last_name: Optional[str] = Field(None, max_length=50, description="Last name")
    phone_number: Optional[str] = Field(None, max_length=20, description="Phone number")
    bio: Optional[str] = Field(None, max_length=500, description="User bio")

    class Config:
        json_schema_extra = {
            "example": {
                "first_name": "John",
                "last_name": "Doe",
                "phone_number": "+1234567890",
                "bio": "Software developer passionate about technology"
            }
        }


class UserProfileResponse(BaseModel):
    """Schema for user profile response."""

    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone_number: Optional[str] = None
    bio: Optional[str] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "first_name": "John",
                "last_name": "Doe",
                "phone_number": "+1234567890",
                "bio": "Software developer passionate about technology",
                "preferences": {"theme": "dark", "notifications": True}
            }
        }


class UserResponse(BaseModel):
    """Schema for user response (public information)."""
    
    id: str = Field(..., description="User ID")
    email: EmailStr = Field(..., description="User email")
    is_active: bool = Field(..., description="Whether user is active")
    is_verified: bool = Field(..., description="Whether email is verified")
    profile: UserProfileResponse = Field(..., description="User profile")
    created_at: datetime = Field(..., description="Account creation timestamp")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "email": "user@example.com",
                "is_active": True,
                "is_verified": False,
                "profile": {
                    "first_name": "John",
                    "last_name": "Doe",
                    "bio": "Software developer"
                },
                "created_at": "2023-01-01T00:00:00Z",
                "last_login": "2023-01-02T10:30:00Z"
            }
        }


class UserListResponse(BaseModel):
    """Schema for paginated user list response."""
    
    users: List[UserResponse] = Field(..., description="List of users")
    total: int = Field(..., description="Total number of users")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Number of users per page")
    pages: int = Field(..., description="Total number of pages")

    class Config:
        json_schema_extra = {
            "example": {
                "users": [
                    {
                        "id": "507f1f77bcf86cd799439011",
                        "email": "user1@example.com",
                        "is_active": True,
                        "is_verified": True,
                        "profile": {"first_name": "John", "last_name": "Doe"},
                        "created_at": "2023-01-01T00:00:00Z"
                    }
                ],
                "total": 100,
                "page": 1,
                "per_page": 10,
                "pages": 10
            }
        }


class UserCreate(BaseModel):
    """Schema for creating a user (admin only)."""
    
    email: EmailStr = Field(..., description="User email")
    password: str = Field(..., min_length=8, description="User password")
    is_active: bool = Field(default=True, description="Whether user is active")
    is_verified: bool = Field(default=False, description="Whether email is verified")
    is_superuser: bool = Field(default=False, description="Whether user is superuser")
    first_name: Optional[str] = Field(None, max_length=50, description="First name")
    last_name: Optional[str] = Field(None, max_length=50, description="Last name")

    class Config:
        json_schema_extra = {
            "example": {
                "email": "newuser@example.com",
                "password": "SecurePass123!",
                "is_active": True,
                "is_verified": False,
                "first_name": "Jane",
                "last_name": "Smith"
            }
        }


class UserUpdate(BaseModel):
    """Schema for updating user (admin only)."""
    
    email: Optional[EmailStr] = Field(None, description="User email")
    is_active: Optional[bool] = Field(None, description="Whether user is active")
    is_verified: Optional[bool] = Field(None, description="Whether email is verified")
    is_superuser: Optional[bool] = Field(None, description="Whether user is superuser")

    class Config:
        json_schema_extra = {
            "example": {
                "is_active": False,
                "is_verified": True
            }
        }
