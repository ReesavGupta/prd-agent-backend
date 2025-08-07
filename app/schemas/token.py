"""
Token-related Pydantic schemas for JWT authentication.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class Token(BaseModel):
    """Schema for JWT token response."""
    
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")

    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 1800
            }
        }


class TokenData(BaseModel):
    """Schema for token payload data."""
    
    user_id: str = Field(..., description="User ID")
    email: str = Field(..., description="User email")
    token_type: str = Field(..., description="Token type (access/refresh)")
    exp: datetime = Field(..., description="Token expiration time")
    iat: datetime = Field(..., description="Token issued at time")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "507f1f77bcf86cd799439011",
                "email": "user@example.com",
                "token_type": "access",
                "exp": "2023-01-01T01:00:00Z",
                "iat": "2023-01-01T00:00:00Z"
            }
        }


class RefreshToken(BaseModel):
    """Schema for refresh token request."""
    
    refresh_token: str = Field(..., description="Refresh token")

    class Config:
        json_schema_extra = {
            "example": {
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
            }
        }


class TokenBlacklistRequest(BaseModel):
    """Schema for token blacklist request."""
    
    token: str = Field(..., description="Token to blacklist")
    reason: Optional[str] = Field(None, description="Reason for blacklisting")

    class Config:
        json_schema_extra = {
            "example": {
                "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "reason": "User logout"
            }
        }
