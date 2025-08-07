"""
Authentication-related Pydantic schemas for request/response validation.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, validator
import re

from app.core.config import settings


class UserRegistration(BaseModel):
    """Schema for user registration request."""
    
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=settings.PASSWORD_MIN_LENGTH, description="User password")
    confirm_password: str = Field(..., description="Password confirmation")
    first_name: Optional[str] = Field(None, max_length=50, description="First name")
    last_name: Optional[str] = Field(None, max_length=50, description="Last name")
    
    @validator("password")
    def validate_password(cls, v):
        """Validate password strength based on settings."""
        errors = []
        
        if settings.PASSWORD_REQUIRE_UPPERCASE and not re.search(r'[A-Z]', v):
            errors.append("Password must contain at least one uppercase letter")
            
        if settings.PASSWORD_REQUIRE_LOWERCASE and not re.search(r'[a-z]', v):
            errors.append("Password must contain at least one lowercase letter")
            
        if settings.PASSWORD_REQUIRE_NUMBERS and not re.search(r'\d', v):
            errors.append("Password must contain at least one number")
            
        if settings.PASSWORD_REQUIRE_SPECIAL and not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            errors.append("Password must contain at least one special character")
            
        if errors:
            raise ValueError("; ".join(errors))
            
        return v
    
    @validator("confirm_password")
    def passwords_match(cls, v, values):
        """Ensure password and confirm_password match."""
        if 'password' in values and v != values['password']:
            raise ValueError("Passwords do not match")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "SecurePass123!",
                "confirm_password": "SecurePass123!",
                "first_name": "John",
                "last_name": "Doe"
            }
        }


class UserLogin(BaseModel):
    """Schema for user login request."""
    
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="User password")

    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "SecurePass123!"
            }
        }


# Password reset schemas removed


class ChangePassword(BaseModel):
    """Schema for password change request."""
    
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=settings.PASSWORD_MIN_LENGTH, description="New password")
    confirm_password: str = Field(..., description="Password confirmation")
    
    @validator("new_password")
    def validate_password(cls, v):
        """Validate password strength based on settings."""
        errors = []
        
        if settings.PASSWORD_REQUIRE_UPPERCASE and not re.search(r'[A-Z]', v):
            errors.append("Password must contain at least one uppercase letter")
            
        if settings.PASSWORD_REQUIRE_LOWERCASE and not re.search(r'[a-z]', v):
            errors.append("Password must contain at least one lowercase letter")
            
        if settings.PASSWORD_REQUIRE_NUMBERS and not re.search(r'\d', v):
            errors.append("Password must contain at least one number")
            
        if settings.PASSWORD_REQUIRE_SPECIAL and not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            errors.append("Password must contain at least one special character")
            
        if errors:
            raise ValueError("; ".join(errors))
            
        return v
    
    @validator("confirm_password")
    def passwords_match(cls, v, values):
        """Ensure new_password and confirm_password match."""
        if 'new_password' in values and v != values['new_password']:
            raise ValueError("Passwords do not match")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "current_password": "OldPassword123!",
                "new_password": "NewSecurePass123!",
                "confirm_password": "NewSecurePass123!"
            }
        }
