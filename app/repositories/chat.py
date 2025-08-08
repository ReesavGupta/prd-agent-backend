"""
Chat and Message repository implementations for MongoDB operations.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.db.database import get_database
from app.models.chat import Chat, Message, ChatSession, ChatMetadata, MessageMetadata
from app.models.user import PyObjectId

logger = logging.getLogger(__name__)


class ChatRepository:
    """Repository for chat operations."""
    
    def __init__(self, db: AsyncIOMotorDatabase = None):
        self.db = db
        self._chats_collection = None
    
    @property
    def chats_collection(self):
        """Get chats collection with lazy loading."""
        if self._chats_collection is None:
            if self.db is None:
                self.db = get_database()
            self._chats_collection = self.db.chats
        return self._chats_collection
    
    async def create_chat(self, chat: Chat) -> Chat:
        """Create a new chat."""
        try:
            chat_dict = chat.model_dump(by_alias=True, exclude={"id"})
            result = await self.chats_collection.insert_one(chat_dict)
            chat.id = PyObjectId(result.inserted_id)
            logger.info(f"Created chat {chat.id} for user {chat.user_id}")
            return chat
        except Exception as e:
            logger.error(f"Error creating chat: {e}")
            raise
    
    async def get_chat_by_id(self, chat_id: PyObjectId, user_id: PyObjectId) -> Optional[Chat]:
        """Get a chat by ID, ensuring user ownership."""
        try:
            chat_doc = await self.chats_collection.find_one({
                "_id": ObjectId(chat_id),
                "user_id": ObjectId(user_id),
                "status": {"$ne": "deleted"}
            })
            
            if chat_doc:
                return Chat(**chat_doc)
            return None
        except Exception as e:
            logger.error(f"Error getting chat {chat_id}: {e}")
            raise
    
    async def list_user_chats(
        self, 
        user_id: PyObjectId, 
        status: str = "active",
        limit: int = 20, 
        offset: int = 0
    ) -> Tuple[List[Chat], int]:
        """List chats for a user with pagination."""
        try:
            query = {"user_id": ObjectId(user_id)}
            if status != "all":
                query["status"] = status
            else:
                query["status"] = {"$ne": "deleted"}
            
            # Get total count
            total = await self.chats_collection.count_documents(query)
            
            # Get paginated results
            cursor = self.chats_collection.find(query).sort([
                ("last_message_at", -1), 
                ("updated_at", -1)
            ]).skip(offset).limit(limit)
            
            chats = []
            async for chat_doc in cursor:
                chats.append(Chat(**chat_doc))
            
            logger.info(f"Listed {len(chats)} chats for user {user_id}")
            return chats, total
        except Exception as e:
            logger.error(f"Error listing chats for user {user_id}: {e}")
            raise
    
    async def update_chat(self, chat_id: PyObjectId, user_id: PyObjectId, updates: Dict[str, Any]) -> bool:
        """Update a chat."""
        try:
            updates["updated_at"] = datetime.utcnow()
            
            result = await self.chats_collection.update_one(
                {"_id": ObjectId(chat_id), "user_id": ObjectId(user_id), "status": {"$ne": "deleted"}},
                {"$set": updates}
            )
            
            success = result.modified_count > 0
            if success:
                logger.info(f"Updated chat {chat_id}")
            return success
        except Exception as e:
            logger.error(f"Error updating chat {chat_id}: {e}")
            raise
    
    async def delete_chat(self, chat_id: PyObjectId, user_id: PyObjectId) -> bool:
        """Soft delete a chat."""
        try:
            result = await self.chats_collection.update_one(
                {"_id": ObjectId(chat_id), "user_id": ObjectId(user_id), "status": {"$ne": "deleted"}},
                {"$set": {"status": "deleted", "updated_at": datetime.utcnow()}}
            )
            
            success = result.modified_count > 0
            if success:
                logger.info(f"Deleted chat {chat_id}")
            return success
        except Exception as e:
            logger.error(f"Error deleting chat {chat_id}: {e}")
            raise
    
    async def update_chat_metadata(
        self, 
        chat_id: PyObjectId, 
        metadata_updates: Dict[str, Any]
    ) -> bool:
        """Update chat metadata."""
        try:
            update_dict = {}
            for key, value in metadata_updates.items():
                update_dict[f"metadata.{key}"] = value
            
            update_dict["updated_at"] = datetime.utcnow()
            
            result = await self.chats_collection.update_one(
                {"_id": ObjectId(chat_id)},
                {"$set": update_dict}
            )
            
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating chat metadata {chat_id}: {e}")
            raise
    
    async def update_last_message_time(self, chat_id: PyObjectId) -> bool:
        """Update the last message timestamp for a chat."""
        try:
            result = await self.chats_collection.update_one(
                {"_id": ObjectId(chat_id)},
                {"$set": {"last_message_at": datetime.utcnow(), "updated_at": datetime.utcnow()}}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating last message time for chat {chat_id}: {e}")
            raise


class MessageRepository:
    """Repository for message operations."""
    
    def __init__(self, db: AsyncIOMotorDatabase = None):
        self.db = db
        self._messages_collection = None
    
    @property
    def messages_collection(self):
        """Get messages collection with lazy loading."""
        if self._messages_collection is None:
            if self.db is None:
                self.db = get_database()
            self._messages_collection = self.db.messages
        return self._messages_collection
    
    async def create_message(self, message: Message) -> Message:
        """Create a new message."""
        try:
            message_dict = message.model_dump(by_alias=True, exclude={"id"})
            result = await self.messages_collection.insert_one(message_dict)
            message.id = PyObjectId(result.inserted_id)
            logger.info(f"Created message {message.id} in chat {message.chat_id}")
            return message
        except Exception as e:
            logger.error(f"Error creating message: {e}")
            raise
    
    async def get_chat_messages(
        self, 
        chat_id: PyObjectId, 
        limit: int = 50, 
        before: Optional[datetime] = None
    ) -> Tuple[List[Message], bool]:
        """Get messages for a chat with pagination."""
        try:
            query = {
                "chat_id": ObjectId(chat_id),
                "is_deleted": False
            }
            
            if before:
                query["timestamp"] = {"$lt": before}
            
            cursor = self.messages_collection.find(query).sort("timestamp", -1).limit(limit + 1)
            
            messages = []
            async for message_doc in cursor:
                messages.append(Message(**message_doc))
            
            # Check if there are more messages
            has_more = len(messages) > limit
            if has_more:
                messages = messages[:-1]  # Remove the extra message
            
            # Reverse to get chronological order
            messages.reverse()
            
            logger.info(f"Retrieved {len(messages)} messages for chat {chat_id}")
            return messages, has_more
        except Exception as e:
            logger.error(f"Error getting messages for chat {chat_id}: {e}")
            raise
    
    async def get_message_by_id(self, message_id: PyObjectId) -> Optional[Message]:
        """Get a message by ID."""
        try:
            message_doc = await self.messages_collection.find_one({
                "_id": ObjectId(message_id),
                "is_deleted": False
            })
            
            if message_doc:
                return Message(**message_doc)
            return None
        except Exception as e:
            logger.error(f"Error getting message {message_id}: {e}")
            raise
    
    async def update_message(self, message_id: PyObjectId, updates: Dict[str, Any]) -> bool:
        """Update a message."""
        try:
            updates["is_edited"] = True
            updates["edited_at"] = datetime.utcnow()
            
            result = await self.messages_collection.update_one(
                {"_id": ObjectId(message_id), "is_deleted": False},
                {"$set": updates}
            )
            
            success = result.modified_count > 0
            if success:
                logger.info(f"Updated message {message_id}")
            return success
        except Exception as e:
            logger.error(f"Error updating message {message_id}: {e}")
            raise
    
    async def delete_message(self, message_id: PyObjectId) -> bool:
        """Soft delete a message."""
        try:
            result = await self.messages_collection.update_one(
                {"_id": ObjectId(message_id), "is_deleted": False},
                {"$set": {"is_deleted": True, "deleted_at": datetime.utcnow()}}
            )
            
            success = result.modified_count > 0
            if success:
                logger.info(f"Deleted message {message_id}")
            return success
        except Exception as e:
            logger.error(f"Error deleting message {message_id}: {e}")
            raise
    
    async def get_chat_message_count(self, chat_id: PyObjectId) -> int:
        """Get total message count for a chat."""
        try:
            count = await self.messages_collection.count_documents({
                "chat_id": ObjectId(chat_id),
                "is_deleted": False
            })
            return count
        except Exception as e:
            logger.error(f"Error counting messages for chat {chat_id}: {e}")
            raise
    
    async def search_messages(
        self, 
        user_id: PyObjectId, 
        query: str, 
        limit: int = 20
    ) -> List[Message]:
        """Search messages by content for a user."""
        try:
            search_query = {
                "user_id": ObjectId(user_id),
                "content": {"$regex": query, "$options": "i"},
                "is_deleted": False
            }
            
            cursor = self.messages_collection.find(search_query).sort("timestamp", -1).limit(limit)
            
            messages = []
            async for message_doc in cursor:
                messages.append(Message(**message_doc))
            
            logger.info(f"Found {len(messages)} messages matching query for user {user_id}")
            return messages
        except Exception as e:
            logger.error(f"Error searching messages: {e}")
            raise


class ChatSessionRepository:
    """Repository for WebSocket chat session operations."""
    
    def __init__(self, db: AsyncIOMotorDatabase = None):
        self.db = db
        self._sessions_collection = None
    
    @property
    def sessions_collection(self):
        """Get sessions collection with lazy loading."""
        if self._sessions_collection is None:
            if self.db is None:
                self.db = get_database()
            self._sessions_collection = self.db.chat_sessions
        return self._sessions_collection
    
    async def create_session(self, session: ChatSession) -> ChatSession:
        """Create a new chat session."""
        try:
            session_dict = session.model_dump(by_alias=True, exclude={"id"})
            result = await self.sessions_collection.insert_one(session_dict)
            session.id = PyObjectId(result.inserted_id)
            logger.info(f"Created chat session {session.session_id}")
            return session
        except Exception as e:
            logger.error(f"Error creating chat session: {e}")
            raise
    
    async def get_session_by_id(self, session_id: str) -> Optional[ChatSession]:
        """Get a session by session ID."""
        try:
            session_doc = await self.sessions_collection.find_one({"session_id": session_id})
            
            if session_doc:
                return ChatSession(**session_doc)
            return None
        except Exception as e:
            logger.error(f"Error getting session {session_id}: {e}")
            raise
    
    async def update_session_activity(self, session_id: str) -> bool:
        """Update session last activity."""
        try:
            result = await self.sessions_collection.update_one(
                {"session_id": session_id},
                {"$set": {"last_activity": datetime.utcnow()}}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating session activity {session_id}: {e}")
            raise
    
    async def update_typing_status(self, session_id: str, is_typing: bool) -> bool:
        """Update typing status for a session."""
        try:
            updates = {
                "is_typing": is_typing,
                "last_activity": datetime.utcnow()
            }
            
            if is_typing:
                updates["typing_started_at"] = datetime.utcnow()
            else:
                updates["typing_started_at"] = None
            
            result = await self.sessions_collection.update_one(
                {"session_id": session_id},
                {"$set": updates}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating typing status {session_id}: {e}")
            raise
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a chat session."""
        try:
            result = await self.sessions_collection.delete_one({"session_id": session_id})
            success = result.deleted_count > 0
            if success:
                logger.info(f"Deleted chat session {session_id}")
            return success
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            raise
    
    async def get_active_sessions_for_chat(self, chat_id: PyObjectId) -> List[ChatSession]:
        """Get all active sessions for a chat."""
        try:
            cursor = self.sessions_collection.find({
                "chat_id": ObjectId(chat_id),
                "connection_state": "connected"
            })
            
            sessions = []
            async for session_doc in cursor:
                sessions.append(ChatSession(**session_doc))
            
            return sessions
        except Exception as e:
            logger.error(f"Error getting active sessions for chat {chat_id}: {e}")
            raise
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        try:
            result = await self.sessions_collection.delete_many({
                "expires_at": {"$lt": datetime.utcnow()}
            })
            
            count = result.deleted_count
            if count > 0:
                logger.info(f"Cleaned up {count} expired chat sessions")
            return count
        except Exception as e:
            logger.error(f"Error cleaning up expired sessions: {e}")
            raise