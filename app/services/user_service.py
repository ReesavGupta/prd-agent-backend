"""
User management service with CRUD operations and business logic.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.db.database import get_database
from app.models.user import UserInDB, UserProfile, TokenBlacklist
from app.schemas.auth import UserRegistration, ChangePassword
from app.schemas.user import UserProfileUpdate, UserCreate, UserUpdate
from app.services.security import security_service
from app.core.config import settings

logger = logging.getLogger(__name__)


class UserService:
    """Service for user management operations."""
    
    def __init__(self):
        self.db: Optional[AsyncIOMotorDatabase] = None
    
    async def get_db(self) -> AsyncIOMotorDatabase:
        """Get database instance."""
        if self.db is None:
            self.db = get_database()
        return self.db
    
    async def create_user(self, user_data: UserRegistration) -> UserInDB:
        """Create a new user."""
        db = await self.get_db()
        
        # Check if user already exists
        existing_user = await db.users.find_one({"email": user_data.email})
        if existing_user:
            raise ValueError("User with this email already exists")
        
        # Create user profile
        profile = UserProfile(
            first_name=user_data.first_name,
            last_name=user_data.last_name
        )
        
        # Create user document
        user_doc = UserInDB(
            email=user_data.email,
            hashed_password=security_service.hash_password(user_data.password),
            profile=profile
        )
        
        # Insert user into database
        result = await db.users.insert_one(user_doc.dict(by_alias=True))
        user_doc.id = result.inserted_id
        
        logger.info(f"Created new user: {user_data.email}")
        return user_doc
    
    async def get_user_by_email(self, email: str) -> Optional[UserInDB]:
        """Get user by email address."""
        db = await self.get_db()
        user_doc = await db.users.find_one({"email": email})
        
        if user_doc:
            return UserInDB(**user_doc)
        return None
    
    async def get_user_by_id(self, user_id: str) -> Optional[UserInDB]:
        """Get user by ID."""
        db = await self.get_db()
        
        if not ObjectId.is_valid(user_id):
            return None
            
        user_doc = await db.users.find_one({"_id": ObjectId(user_id)})
        
        if user_doc:
            return UserInDB(**user_doc)
        return None
    
    async def authenticate_user(self, email: str, password: str) -> Optional[UserInDB]:
        """Authenticate user with email and password."""
        user = await self.get_user_by_email(email)
        
        if not user:
            return None
            
        if not user.is_active:
            raise ValueError("User account is deactivated")

        if not security_service.verify_password(password, user.hashed_password):
            return None
        
        # Update last login timestamp
        await self.update_last_login(user.id)
        
        return user
    
    # Rate limiting methods removed
    
    async def update_last_login(self, user_id: ObjectId) -> None:
        """Update user's last login timestamp."""
        db = await self.get_db()
        
        await db.users.update_one(
            {"_id": user_id},
            {"$set": {"last_login": datetime.utcnow()}}
        )
    
    async def update_user_profile(self, user_id: str, profile_data: UserProfileUpdate) -> Optional[UserInDB]:
        """Update user profile."""
        db = await self.get_db()
        
        if not ObjectId.is_valid(user_id):
            return None
        
        # Prepare update data
        update_data = {}
        for field, value in profile_data.dict(exclude_unset=True).items():
            if value is not None:
                update_data[f"profile.{field}"] = value
        
        if not update_data:
            return await self.get_user_by_id(user_id)
        
        update_data["updated_at"] = datetime.utcnow()
        
        # Update user
        await db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": update_data}
        )
        
        return await self.get_user_by_id(user_id)
    
    async def change_password(self, user_id: str, password_data: ChangePassword) -> bool:
        """Change user password."""
        user = await self.get_user_by_id(user_id)
        if not user:
            return False
        
        # Verify current password
        if not security_service.verify_password(password_data.current_password, user.hashed_password):
            raise ValueError("Current password is incorrect")
        
        # Hash new password
        new_hashed_password = security_service.hash_password(password_data.new_password)
        
        # Update password
        db = await self.get_db()
        await db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {
                "hashed_password": new_hashed_password,
                "password_changed_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }}
        )
        
        logger.info(f"Password changed for user: {user.email}")
        return True


# Global user service instance
user_service = UserService()
