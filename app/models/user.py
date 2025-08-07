"""
User model for MongoDB with proper schema design and extensibility.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, EmailStr
from bson import ObjectId


class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic v2."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema
        return core_schema.no_info_plain_validator_function(cls.validate)

    @classmethod
    def validate(cls, v):
        if isinstance(v, ObjectId):
            return v
        if isinstance(v, str) and ObjectId.is_valid(v):
            return ObjectId(v)
        raise ValueError("Invalid ObjectId")

    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema, handler):
        field_schema.update(type="string", format="objectid")
        return field_schema


class UserRole(BaseModel):
    """User role model for future RBAC implementation."""
    
    name: str = Field(..., description="Role name")
    permissions: List[str] = Field(default_factory=list, description="List of permissions")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class UserProfile(BaseModel):
    """Extended user profile information."""

    first_name: Optional[str] = Field(None, max_length=50)
    last_name: Optional[str] = Field(None, max_length=50)
    phone_number: Optional[str] = Field(None, max_length=20)
    date_of_birth: Optional[datetime] = None
    bio: Optional[str] = Field(None, max_length=500)
    preferences: Dict[str, Any] = Field(default_factory=dict)


class UserInDB(BaseModel):
    """User model as stored in MongoDB."""
    
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    email: EmailStr = Field(..., description="User email address")
    hashed_password: str = Field(..., description="Hashed password")
    
    # Basic user information
    is_active: bool = Field(default=True, description="Whether user account is active")
    is_verified: bool = Field(default=False, description="Whether email is verified")
    is_superuser: bool = Field(default=False, description="Whether user has admin privileges")
    
    # Extended profile
    profile: UserProfile = Field(default_factory=UserProfile)
    
    # Role-based access control (for future use)
    roles: List[UserRole] = Field(default_factory=list, description="User roles")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    
    # Security fields
    failed_login_attempts: int = Field(default=0, description="Number of consecutive failed login attempts")
    locked_until: Optional[datetime] = Field(None, description="Account lock expiration time")
    password_changed_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional user metadata")

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "is_active": True,
                "is_verified": False,
                "profile": {
                    "first_name": "John",
                    "last_name": "Doe"
                },
                "roles": [],
                "created_at": "2023-01-01T00:00:00Z"
            }
        }


class TokenBlacklist(BaseModel):
    """Model for blacklisted JWT tokens."""
    
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    token: str = Field(..., description="JWT token to blacklist")
    user_id: PyObjectId = Field(..., description="User ID who owns the token")
    expires_at: datetime = Field(..., description="Token expiration time")
    blacklisted_at: datetime = Field(default_factory=datetime.utcnow)
    reason: Optional[str] = Field(None, description="Reason for blacklisting")

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# PasswordResetToken model removed
