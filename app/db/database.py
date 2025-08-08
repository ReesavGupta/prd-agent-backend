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
        
        # Chat collection indexes
        chats_collection = db.database.chats
        
        # Index on user_id + status for listing user chats
        await chats_collection.create_index([("user_id", 1), ("status", 1)])
        
        # Index on user_id + updated_at for sorting by activity
        await chats_collection.create_index([("user_id", 1), ("updated_at", -1)])
        
        # Index on last_message_at for global chat ordering
        await chats_collection.create_index("last_message_at")
        
        # Messages collection indexes
        messages_collection = db.database.messages
        
        # Index on chat_id + timestamp for message pagination
        await messages_collection.create_index([("chat_id", 1), ("timestamp", 1)])
        
        # Index on user_id + timestamp for user message history
        await messages_collection.create_index([("user_id", 1), ("timestamp", -1)])
        
        # Index on timestamp for global message ordering
        await messages_collection.create_index("timestamp")
        
        # Index on is_deleted for filtering
        await messages_collection.create_index("is_deleted")
        
        # Chat sessions collection indexes
        chat_sessions_collection = db.database.chat_sessions
        
        # Unique index on session_id for WebSocket management
        await chat_sessions_collection.create_index("session_id", unique=True)
        
        # Index on chat_id + user_id for session lookups
        await chat_sessions_collection.create_index([("chat_id", 1), ("user_id", 1)])
        
        # TTL index on expires_at for automatic cleanup
        await chat_sessions_collection.create_index("expires_at", expireAfterSeconds=0)
        
        logger.info("Database indexes created successfully")
        
    except Exception as e:
        logger.error(f"Error creating database indexes: {e}")
        raise


def get_database() -> AsyncIOMotorDatabase:
    """Get database instance."""
    if db.database is None:
        raise RuntimeError("Database not initialized. Call connect_to_mongo() first.")
    return db.database
