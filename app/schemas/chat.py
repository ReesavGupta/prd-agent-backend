"""
Chat and Message schemas for API request/response validation.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator
from app.models.user import PyObjectId


class ChatCreateRequest(BaseModel):
    """Request schema for creating a new chat."""
    chat_name: Optional[str] = Field(None, min_length=1, max_length=255, description="Chat name (auto-generated if not provided)")


class ChatUpdateRequest(BaseModel):
    """Request schema for updating a chat."""
    chat_name: Optional[str] = Field(None, min_length=1, max_length=255, description="New chat name")


class ChatMetadataResponse(BaseModel):
    """Response schema for chat metadata."""
    message_count: int
    has_project: bool
    project_id: Optional[str] = None
    last_ai_model: Optional[str] = None
    total_tokens_used: int
    avg_response_time: Optional[float] = None
    conversation_quality_score: Optional[float] = None


class ChatResponse(BaseModel):
    """Response schema for chat information."""
    id: str
    chat_name: str
    created_at: datetime
    updated_at: datetime
    last_message_at: Optional[datetime] = None
    status: str
    metadata: ChatMetadataResponse
    is_auto_named: bool

    class Config:
        from_attributes = True


class ChatListResponse(BaseModel):
    """Response schema for chat list."""
    chats: List[ChatResponse]
    total: int
    limit: int
    offset: int
    has_more: bool


class MessageAttachmentResponse(BaseModel):
    """Response schema for message attachments."""
    file_id: str
    filename: str
    storage_key: str
    url: str
    content_type: str
    file_size: int


class MessageMetadataResponse(BaseModel):
    """Response schema for message metadata."""
    tokens_used: Optional[int] = None
    response_time_ms: Optional[int] = None
    model_used: Optional[str] = None
    context_length: Optional[int] = None
    provider: Optional[str] = None
    temperature: Optional[float] = None


class MessageResponse(BaseModel):
    """Response schema for message information."""
    id: str
    message_type: str
    content: str
    timestamp: datetime
    metadata: Optional[MessageMetadataResponse] = None
    attachments: List[MessageAttachmentResponse] = Field(default_factory=list)
    reply_to: Optional[str] = None
    thread_id: Optional[str] = None
    is_edited: bool = False
    edited_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class MessageCreateRequest(BaseModel):
    """Request schema for creating a message."""
    content: str = Field(..., min_length=1, max_length=10000, description="Message content")
    message_type: str = Field(default="user", description="Message type")
    reply_to: Optional[str] = Field(None, description="ID of message being replied to")
    thread_id: Optional[str] = Field(None, description="Thread identifier")

    @field_validator("message_type")
    def validate_message_type(cls, v):
        """Validate message type."""
        allowed_types = ["user", "assistant", "system"]
        if v not in allowed_types:
            raise ValueError(f"Message type must be one of: {allowed_types}")
        return v


class MessageListResponse(BaseModel):
    """Response schema for message list."""
    messages: List[MessageResponse]
    has_more: bool
    next_before: Optional[datetime] = None


class ChatConvertToProjectRequest(BaseModel):
    """Request schema for converting chat to project."""
    project_name: Optional[str] = Field(None, min_length=1, max_length=255, description="Project name (auto-generated if not provided)")
    include_chat_history: bool = Field(default=True, description="Whether to include chat history in project context")


class ChatConvertToProjectResponse(BaseModel):
    """Response schema for chat to project conversion."""
    project_id: str
    message: str = "Chat converted to project successfully"


class SuggestedAction(BaseModel):
    """Schema for AI suggested actions."""
    type: str = Field(..., description="Action type")
    label: str = Field(..., description="Action label for UI")
    description: Optional[str] = Field(None, description="Action description")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional action metadata")


class AIMessageResponse(BaseModel):
    """Response schema for AI-generated messages."""
    message_id: str
    content: str
    suggested_actions: List[SuggestedAction] = Field(default_factory=list)
    metadata: Optional[MessageMetadataResponse] = None


class WebSocketMessageRequest(BaseModel):
    """Schema for WebSocket message requests."""
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(..., description="Message data")

    @field_validator("type")
    def validate_message_type(cls, v):
        """Validate WebSocket message type."""
        allowed_types = [
            "send_message",
            "typing_start",
            "typing_stop",
            "join_chat",
            "leave_chat",
            "ping",
            # HITL resume message (Phase 2 - will be handled when UI is ready)
            "agent_resume",
        ]
        if v not in allowed_types:
            raise ValueError(f"WebSocket message type must be one of: {allowed_types}")
        return v


class WebSocketMessageResponse(BaseModel):
    """Schema for WebSocket message responses."""
    type: str = Field(..., description="Response type")
    data: Dict[str, Any] = Field(..., description="Response data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Optional schemas for HITL WS payloads (Phase 2)
class AgentInterruptRequest(BaseModel):
    question_id: str
    question: str
    lens: Optional[str] = None
    rationale: Optional[str] = None


class AgentResumeRequest(BaseModel):
    chat_id: str
    answer: Dict[str, str]  # { question_id: str, text: str }


class TypingIndicatorResponse(BaseModel):
    """Response schema for typing indicators."""
    is_typing: bool
    estimated_time: Optional[int] = None  # seconds
    user_info: Optional[Dict[str, str]] = None


class ChatSearchRequest(BaseModel):
    """Request schema for chat search."""
    query: str = Field(..., min_length=1, max_length=100, description="Search query")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum number of results")


class ChatSearchResponse(BaseModel):
    """Response schema for chat search results."""
    messages: List[MessageResponse]
    total_found: int
    query: str