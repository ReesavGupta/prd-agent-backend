"""
Database connection and configuration for MongoDB.
"""

import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


class Database:
    """MongoDB database connection manager."""
    
    client: Optional[AsyncIOMotorClient] = None
    database: Optional[AsyncIOMotorDatabase] = None


db = Database()


async def connect_to_mongo():
    """Create database connection."""
    try:
        logger.info("Connecting to MongoDB...")
        db.client = AsyncIOMotorClient(settings.MONGODB_URL)
        if db.client is not None:
            db.database = db.client[settings.DATABASE_NAME]
            # Test the connection
            await db.client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")
        else:
            logger.error("MongoDB client is not initialized.")
            raise RuntimeError("MongoDB client is not initialized.")
        
        # Create indexes
        await create_indexes()
        
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise


async def close_mongo_connection():
    """Close database connection."""
    try:
        if db.client:
            logger.info("Closing MongoDB connection...")
            db.client.close()
            logger.info("MongoDB connection closed")
    except Exception as e:
        logger.error(f"Error closing MongoDB connection: {e}")


async def create_indexes():
    """Create database indexes for optimal performance."""
    try:
        if db.database is None:
            return
            
        # Users collection indexes
        users_collection = db.database.users
        
        # Unique index on email
        await users_collection.create_index("email", unique=True)
        
        # Index on created_at for sorting
        await users_collection.create_index("created_at")
        
        # Index on is_active for filtering
        await users_collection.create_index("is_active")
        
        # Compound index for future role-based queries
        await users_collection.create_index([("is_active", 1), ("roles", 1)])
        
        # Token blacklist collection indexes
        token_blacklist_collection = db.database.token_blacklist
        
        # Index on token for fast lookups
        await token_blacklist_collection.create_index("token", unique=True)
        
        # TTL index on expires_at for automatic cleanup
        await token_blacklist_collection.create_index("expires_at", expireAfterSeconds=0)
        
        # Password reset tokens collection indexes removed
        
        logger.info("Database indexes created successfully")
        
    except Exception as e:
        logger.error(f"Error creating database indexes: {e}")
        raise


def get_database() -> AsyncIOMotorDatabase:
    """Get database instance."""
    if db.database is None:
        raise RuntimeError("Database not initialized. Call connect_to_mongo() first.")
    return db.database
