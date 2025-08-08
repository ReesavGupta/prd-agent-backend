"""
AI models for storing AI interactions and configurations.
"""

from datetime import datetime
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field
from bson import ObjectId
from .user import PyObjectId


class AIInteraction(BaseModel):
    """Model for storing AI interactions."""
    
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    user_id: PyObjectId = Field(..., description="User who made the request")
    provider: str = Field(..., description="AI provider used (gemini, groq, openai)")
    model: str = Field(..., description="Model used for generation")
    
    # Request data
    messages: List[Dict[str, str]] = Field(..., description="Conversation messages")
    temperature: Optional[float] = Field(None, description="Temperature used")
    max_tokens: Optional[int] = Field(None, description="Max tokens used")
    
    # Response data
    response_content: str = Field(..., description="AI response content")
    tokens_used: int = Field(0, description="Tokens consumed")
    response_time: float = Field(..., description="Response time in seconds")
    
    # Metadata
    was_cached: bool = Field(False, description="Whether response was from cache")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    project_id: Optional[PyObjectId] = Field(None, description="Associated project ID")
    chat_id: Optional[PyObjectId] = Field(None, description="Associated chat ID")
    
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class AIModelConfig(BaseModel):
    """Model for storing AI provider and model configurations."""
    
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    provider: str = Field(..., description="AI provider (gemini, groq, openai)")
    model_name: str = Field(..., description="Model name")
    
    # Model capabilities
    max_tokens: int = Field(..., description="Maximum tokens supported")
    supports_streaming: bool = Field(True, description="Whether model supports streaming")
    supports_vision: bool = Field(False, description="Whether model supports vision")
    supports_function_calling: bool = Field(False, description="Whether model supports function calling")
    
    # Cost information (optional)
    cost_per_input_token: Optional[float] = Field(None, description="Cost per input token")
    cost_per_output_token: Optional[float] = Field(None, description="Cost per output token")
    
    # Usage statistics
    total_requests: int = Field(0, description="Total requests made")
    total_tokens_used: int = Field(0, description="Total tokens used")
    average_response_time: float = Field(0.0, description="Average response time")
    
    # Configuration
    is_active: bool = Field(True, description="Whether model is active")
    priority: int = Field(1, description="Model priority (1=highest)")
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class AIUsageStats(BaseModel):
    """Model for tracking AI usage statistics."""
    
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    user_id: PyObjectId = Field(..., description="User ID")
    date: datetime = Field(..., description="Date for statistics")
    
    # Daily usage counts
    total_requests: int = Field(0, description="Total requests made")
    successful_requests: int = Field(0, description="Successful requests")
    failed_requests: int = Field(0, description="Failed requests")
    cached_responses: int = Field(0, description="Responses served from cache")
    
    # Token usage
    total_tokens_used: int = Field(0, description="Total tokens consumed")
    
    # Provider breakdown
    provider_usage: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Usage breakdown by provider"
    )
    
    # Response times
    avg_response_time: float = Field(0.0, description="Average response time")
    min_response_time: float = Field(0.0, description="Minimum response time")
    max_response_time: float = Field(0.0, description="Maximum response time")
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}