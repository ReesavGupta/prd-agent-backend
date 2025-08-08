"""
Chat and Message data models for the AI PRD Generator.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from bson import ObjectId
from pydantic import BaseModel, Field, field_validator
from app.models.user import PyObjectId


class MessageAttachment(BaseModel):
    """File attachment for chat messages."""
    file_id: PyObjectId = Field(...)
    filename: str = Field(...)
    storage_key: str = Field(...)  # Cloudinary public_id
    url: str = Field(...)  # File URL
    content_type: str = Field(...)
    file_size: int = Field(...)


class MessageMetadata(BaseModel):
    """Metadata for AI-generated messages."""
    tokens_used: Optional[int] = None
    response_time_ms: Optional[int] = None
    model_used: Optional[str] = None
    context_length: Optional[int] = None
    provider: Optional[str] = None  # openai, gemini, groq
    temperature: Optional[float] = None


class Message(BaseModel):
    """Chat message model."""
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    chat_id: PyObjectId = Field(...)
    user_id: PyObjectId = Field(...)
    message_type: str = Field(...)  # user, assistant, system
    content: str = Field(..., max_length=10000)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[MessageMetadata] = None
    attachments: List[MessageAttachment] = Field(default_factory=list)
    
    # Message threading and context
    reply_to: Optional[PyObjectId] = None
    thread_id: Optional[str] = None
    
    # Message status
    is_edited: bool = False
    edited_at: Optional[datetime] = None
    is_deleted: bool = False
    deleted_at: Optional[datetime] = None

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

    @field_validator("message_type")
    def validate_message_type(cls, v):
        """Validate message type."""
        allowed_types = ["user", "assistant", "system"]
        if v not in allowed_types:
            raise ValueError(f"Message type must be one of: {allowed_types}")
        return v


class ChatMetadata(BaseModel):
    """Chat metadata and statistics."""
    message_count: int = 0
    has_project: bool = False
    project_id: Optional[PyObjectId] = None
    last_ai_model: Optional[str] = None
    total_tokens_used: int = 0
    avg_response_time: Optional[float] = None
    conversation_quality_score: Optional[float] = None


class Chat(BaseModel):
    """Chat session model."""
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    user_id: PyObjectId = Field(...)
    chat_name: str = Field(..., min_length=1, max_length=255)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_message_at: Optional[datetime] = None
    status: str = Field(default="active")  # active, archived, deleted
    metadata: ChatMetadata = Field(default_factory=ChatMetadata)
    
    # Chat context and state
    conversation_context: Dict[str, Any] = Field(default_factory=dict)
    system_prompt: Optional[str] = None
    
    # Auto-generated chat name flag
    is_auto_named: bool = True

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

    @field_validator("status")
    def validate_status(cls, v):
        """Validate chat status."""
        allowed_statuses = ["active", "archived", "deleted"]
        if v not in allowed_statuses:
            raise ValueError(f"Status must be one of: {allowed_statuses}")
        return v


class ChatSession(BaseModel):
    """Active chat session for WebSocket management."""
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    chat_id: PyObjectId = Field(...)
    user_id: PyObjectId = Field(...)
    session_id: str = Field(...)  # WebSocket connection ID
    websocket_id: str = Field(...)
    connection_state: str = Field(default="connected")  # connected, disconnected, typing
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime = Field(...)
    
    # Typing indicators
    is_typing: bool = False
    typing_started_at: Optional[datetime] = None

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

    @field_validator("connection_state")
    def validate_connection_state(cls, v):
        """Validate connection state."""
        allowed_states = ["connected", "disconnected", "typing", "idle"]
        if v not in allowed_states:
            raise ValueError(f"Connection state must be one of: {allowed_states}")
        return v