"""
Chat service layer with business logic for chat and message management.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from bson import ObjectId
from fastapi import UploadFile

from app.models.chat import Chat, Message, ChatSession, ChatMetadata, MessageMetadata, MessageAttachment
from app.models.user import PyObjectId
from app.repositories.chat import ChatRepository, MessageRepository, ChatSessionRepository
from app.schemas.chat import (
    ChatCreateRequest, ChatUpdateRequest, MessageCreateRequest,
    ChatConvertToProjectRequest, SuggestedAction
)
from app.services.ai_service import ai_service
from app.dependencies import get_cloudinary_storage

logger = logging.getLogger(__name__)


class ChatService:
    """Service for chat management operations."""
    
    def __init__(self):
        self.chat_repo = ChatRepository()
        self.message_repo = MessageRepository()
        self.session_repo = ChatSessionRepository()
    
    async def create_chat(self, user_id: PyObjectId, request: ChatCreateRequest) -> Chat:
        """Create a new chat for a user."""
        try:
            # Generate auto chat name if not provided
            chat_name = request.chat_name
            is_auto_named = False
            
            if not chat_name:
                chat_name = await self._generate_chat_name(user_id)
                is_auto_named = True
            
            # Create chat instance
            chat = Chat(
                user_id=user_id,
                chat_name=chat_name,
                is_auto_named=is_auto_named,
                metadata=ChatMetadata()
            )
            
            # Save to database
            created_chat = await self.chat_repo.create_chat(chat)
            logger.info(f"Created chat {created_chat.id} for user {user_id}")
            return created_chat
        except Exception as e:
            logger.error(f"Error creating chat for user {user_id}: {e}")
            raise
    
    async def get_chat(self, chat_id: PyObjectId, user_id: PyObjectId) -> Optional[Chat]:
        """Get a chat by ID, ensuring user ownership."""
        try:
            chat = await self.chat_repo.get_chat_by_id(chat_id, user_id)
            if not chat:
                logger.warning(f"Chat {chat_id} not found or not owned by user {user_id}")
            return chat
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
            chats, total = await self.chat_repo.list_user_chats(user_id, status, limit, offset)
            logger.info(f"Listed {len(chats)} chats for user {user_id}")
            return chats, total
        except Exception as e:
            logger.error(f"Error listing chats for user {user_id}: {e}")
            raise
    
    async def update_chat(
        self, 
        chat_id: PyObjectId, 
        user_id: PyObjectId, 
        request: ChatUpdateRequest
    ) -> bool:
        """Update a chat."""
        try:
            updates = {}
            
            if request.chat_name is not None:
                updates["chat_name"] = request.chat_name
                updates["is_auto_named"] = False
            
            if not updates:
                return True  # Nothing to update
            
            success = await self.chat_repo.update_chat(chat_id, user_id, updates)
            if success:
                logger.info(f"Updated chat {chat_id}")
            else:
                logger.warning(f"Failed to update chat {chat_id} - not found or not owned by user")
            return success
        except Exception as e:
            logger.error(f"Error updating chat {chat_id}: {e}")
            raise
    
    async def delete_chat(self, chat_id: PyObjectId, user_id: PyObjectId) -> bool:
        """Delete a chat (soft delete)."""
        try:
            success = await self.chat_repo.delete_chat(chat_id, user_id)
            if success:
                logger.info(f"Deleted chat {chat_id}")
                # TODO: Clean up associated messages and sessions
            else:
                logger.warning(f"Failed to delete chat {chat_id} - not found or not owned by user")
            return success
        except Exception as e:
            logger.error(f"Error deleting chat {chat_id}: {e}")
            raise
    
    async def send_message(
        self, 
        chat_id: PyObjectId, 
        user_id: PyObjectId, 
        request: MessageCreateRequest
    ) -> Tuple[Message, Optional[Message]]:
        """Send a message in a chat and get AI response."""
        return await self.send_message_with_files(chat_id, user_id, request, [])
    
    async def send_message_with_files(
        self, 
        chat_id: PyObjectId, 
        user_id: PyObjectId, 
        request: MessageCreateRequest,
        files: List[UploadFile]
    ) -> Tuple[Message, Optional[Message]]:
        """Send a message with file attachments in a chat and get AI response."""
        try:
            # Verify chat ownership
            chat = await self.get_chat(chat_id, user_id)
            if not chat:
                raise ValueError("Chat not found or access denied")
            
            # Process file attachments
            attachments = []
            if files:
                attachments = await self._process_file_attachments(chat_id, user_id, files)
            
            # Create user message
            user_message = Message(
                chat_id=chat_id,
                user_id=user_id,
                message_type=request.message_type,
                content=request.content,
                attachments=attachments,
                reply_to=PyObjectId(request.reply_to) if request.reply_to else None,
                thread_id=request.thread_id
            )
            
            # Save user message
            saved_message = await self.message_repo.create_message(user_message)
            
            # Update chat metadata
            await self._update_chat_after_message(chat_id)
            
            # Generate AI response if user message
            ai_response = None
            if request.message_type == "user":
                ai_response = await self._generate_ai_response(chat_id, user_id, saved_message)
            
            logger.info(f"Sent message with {len(attachments)} attachments in chat {chat_id}")
            return saved_message, ai_response
        except Exception as e:
            logger.error(f"Error sending message in chat {chat_id}: {e}")
            raise
    
    async def get_chat_messages(
        self, 
        chat_id: PyObjectId, 
        user_id: PyObjectId,
        limit: int = 50, 
        before: Optional[datetime] = None
    ) -> Tuple[List[Message], bool]:
        """Get messages for a chat with pagination."""
        try:
            # Verify chat ownership
            chat = await self.get_chat(chat_id, user_id)
            if not chat:
                raise ValueError("Chat not found or access denied")
            
            messages, has_more = await self.message_repo.get_chat_messages(chat_id, limit, before)
            logger.info(f"Retrieved {len(messages)} messages for chat {chat_id}")
            return messages, has_more
        except Exception as e:
            logger.error(f"Error getting messages for chat {chat_id}: {e}")
            raise
    
    async def search_messages(
        self, 
        user_id: PyObjectId, 
        query: str, 
        limit: int = 20
    ) -> List[Message]:
        """Search messages for a user."""
        try:
            messages = await self.message_repo.search_messages(user_id, query, limit)
            logger.info(f"Found {len(messages)} messages for query '{query}'")
            return messages
        except Exception as e:
            logger.error(f"Error searching messages: {e}")
            raise
    
    async def convert_chat_to_project(
        self, 
        chat_id: PyObjectId, 
        user_id: PyObjectId, 
        request: ChatConvertToProjectRequest
    ) -> str:
        """Convert a chat to a project."""
        try:
            # Verify chat ownership
            chat = await self.get_chat(chat_id, user_id)
            if not chat:
                raise ValueError("Chat not found or access denied")
            
            # Get chat messages for context
            messages, _ = await self.message_repo.get_chat_messages(chat_id, limit=100)
            
            # Extract project information from chat
            project_data = await self._extract_project_from_chat(chat, messages, request)
            
            # TODO: Create project using project service
            # For now, return placeholder project ID
            project_id = str(PyObjectId())
            
            # Update chat metadata
            await self.chat_repo.update_chat_metadata(chat_id, {
                "has_project": True,
                "project_id": PyObjectId(project_id)
            })
            
            logger.info(f"Converted chat {chat_id} to project {project_id}")
            return project_id
        except Exception as e:
            logger.error(f"Error converting chat {chat_id} to project: {e}")
            raise
    
    async def create_websocket_session(
        self, 
        chat_id: PyObjectId, 
        user_id: PyObjectId, 
        websocket_id: str
    ) -> ChatSession:
        """Create a WebSocket session for a chat."""
        try:
            # Verify chat ownership
            chat = await self.get_chat(chat_id, user_id)
            if not chat:
                raise ValueError("Chat not found or access denied")
            
            # Create session
            session = ChatSession(
                chat_id=chat_id,
                user_id=user_id,
                session_id=str(uuid.uuid4()),
                websocket_id=websocket_id,
                expires_at=datetime.utcnow() + timedelta(hours=24)
            )
            
            created_session = await self.session_repo.create_session(session)
            logger.info(f"Created WebSocket session {created_session.session_id} for chat {chat_id}")
            return created_session
        except Exception as e:
            logger.error(f"Error creating WebSocket session: {e}")
            raise
    
    async def update_typing_status(self, session_id: str, is_typing: bool) -> bool:
        """Update typing status for a session."""
        try:
            success = await self.session_repo.update_typing_status(session_id, is_typing)
            return success
        except Exception as e:
            logger.error(f"Error updating typing status: {e}")
            raise
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired WebSocket sessions."""
        try:
            count = await self.session_repo.cleanup_expired_sessions()
            return count
        except Exception as e:
            logger.error(f"Error cleaning up expired sessions: {e}")
            raise
    
    # Private helper methods
    
    async def _generate_chat_name(self, user_id: PyObjectId) -> str:
        """Generate an auto chat name."""
        try:
            # Get count of user's existing chats
            _, total = await self.chat_repo.list_user_chats(user_id, "all", limit=1, offset=0)
            return f"Chat #{total + 1}"
        except Exception:
            # Fallback to timestamp-based name
            return f"Chat {datetime.utcnow().strftime('%m/%d %H:%M')}"
    
    async def _update_chat_after_message(self, chat_id: PyObjectId) -> None:
        """Update chat metadata after a message is sent."""
        try:
            # Update last message time
            await self.chat_repo.update_last_message_time(chat_id)
            
            # Update message count
            message_count = await self.message_repo.get_chat_message_count(chat_id)
            await self.chat_repo.update_chat_metadata(chat_id, {"message_count": message_count})
        except Exception as e:
            logger.error(f"Error updating chat metadata: {e}")
            # Don't raise - this is non-critical
    
    async def _generate_ai_response(
        self, 
        chat_id: PyObjectId, 
        user_id: PyObjectId, 
        user_message: Message
    ) -> Optional[Message]:
        """Generate AI response to a user message."""
        try:
            # Get recent messages for context
            messages, _ = await self.message_repo.get_chat_messages(chat_id, limit=10)
            
            # Build conversation context
            conversation_history = []
            for msg in messages:
                conversation_history.append({
                    "role": "user" if msg.message_type == "user" else "assistant",
                    "content": msg.content
                })
            
            # Add the new user message
            conversation_history.append({
                "role": "user",
                "content": user_message.content
            })
            
            # Get AI response (using existing AI service)
            start_time = datetime.utcnow()
            
            # Build prompt from conversation history
            prompt = "\n".join([
                f"{msg['role']}: {msg['content']}" for msg in conversation_history
            ])
            
            ai_response = await ai_service.generate_response(
                prompt=prompt,
                user_id=str(user_id),
                max_tokens=1000,
                temperature=0.7
            )
            
            end_time = datetime.utcnow()
            
            response_time = int((end_time - start_time).total_seconds() * 1000)
            
            # Create AI message metadata
            metadata = MessageMetadata(
                tokens_used=ai_response.tokens_used,
                response_time_ms=response_time,
                model_used=ai_response.model,
                provider=ai_response.provider,
                temperature=0.7
            )
            
            # Create AI message
            ai_message = Message(
                chat_id=chat_id,
                user_id=user_id,  # Same user for context
                message_type="assistant",
                content=ai_response.content,
                metadata=metadata,
                reply_to=user_message.id
            )
            
            # Save AI message
            saved_ai_message = await self.message_repo.create_message(ai_message)
            
            # Update chat metadata again
            await self._update_chat_after_message(chat_id)
            
            return saved_ai_message
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return None
    
    async def _extract_project_from_chat(
        self, 
        chat: Chat, 
        messages: List[Message], 
        request: ChatConvertToProjectRequest
    ) -> Dict[str, Any]:
        """Extract project information from chat messages."""
        try:
            # Analyze chat messages to extract project idea
            conversation_text = "\n".join([
                f"{msg.message_type}: {msg.content}" for msg in messages[-20:]  # Last 20 messages
            ])
            
            # Use AI to extract project idea (placeholder for now)
            project_name = request.project_name or f"Project from {chat.chat_name}"
            initial_idea = f"Project idea extracted from chat conversation: {conversation_text[:500]}..."
            
            return {
                "project_name": project_name,
                "initial_idea": initial_idea,
                "source_chat_id": chat.id,
                "created_from_chat": True
            }
        except Exception as e:
            logger.error(f"Error extracting project from chat: {e}")
            raise
    
    async def _generate_suggested_actions(self, message_content: str) -> List[SuggestedAction]:
        """Generate suggested actions based on message content."""
        try:
            actions = []
            
            # Simple keyword-based suggestions (can be enhanced with AI)
            content_lower = message_content.lower()
            
            if any(keyword in content_lower for keyword in ["project", "product", "build", "create", "idea"]):
                actions.append(SuggestedAction(
                    type="create_project",
                    label="Convert to PRD Project",
                    description="Ready to create a full PRD from this conversation?"
                ))
            
            if any(keyword in content_lower for keyword in ["document", "spec", "requirements", "prd"]):
                actions.append(SuggestedAction(
                    type="upload_document",
                    label="Upload Supporting Documents",
                    description="Add documents to provide more context"
                ))
            
            return actions
        except Exception as e:
            logger.error(f"Error generating suggested actions: {e}")
            return []
    
    async def _process_file_attachments(
        self, 
        chat_id: PyObjectId, 
        user_id: PyObjectId, 
        files: List[UploadFile]
    ) -> List[MessageAttachment]:
        """Process file attachments for a message."""
        try:
            attachments = []
            storage = get_cloudinary_storage()
            
            for file in files:
                if not file.filename:
                    continue
                
                # Read file content
                file_content = await file.read()
                if not file_content:
                    continue
                
                # Validate file size (10MB limit)
                max_size = 10 * 1024 * 1024
                if len(file_content) > max_size:
                    logger.warning(f"File {file.filename} too large: {len(file_content)} bytes")
                    continue
                
                # Validate file type
                allowed_types = [
                    "application/pdf", "application/msword", 
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "text/markdown", "text/plain", "image/png", "image/jpeg", "image/jpg"
                ]
                
                content_type = file.content_type or "application/octet-stream"
                if content_type not in allowed_types:
                    logger.warning(f"File {file.filename} has unsupported type: {content_type}")
                    continue
                
                # Generate storage path
                folder_path = f"chats/{user_id}/{chat_id}/attachments"
                
                try:
                    # Upload to Cloudinary
                    upload_result = await storage.upload_file(
                        file_content=file_content,
                        filename=file.filename,
                        content_type=content_type,
                        folder_path=folder_path
                    )
                    
                    # Create attachment object
                    file_id = PyObjectId()  # Generate unique ID for file
                    attachment = MessageAttachment(
                        file_id=file_id,
                        filename=file.filename,
                        storage_key=upload_result["storage_key"],
                        url=upload_result["url"],
                        content_type=content_type,
                        file_size=len(file_content)
                    )
                    
                    attachments.append(attachment)
                    logger.info(f"Processed attachment: {file.filename}")
                    
                except Exception as upload_error:
                    logger.error(f"Error uploading file {file.filename}: {upload_error}")
                    continue
            
            return attachments
        except Exception as e:
            logger.error(f"Error processing file attachments: {e}")
            return []


# Global service instance
chat_service = ChatService()