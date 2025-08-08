"""
AI API endpoints for chat and streaming responses.
Provides both REST endpoints and WebSocket support for real-time streaming.
"""

import json
import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from ..middleware.auth import get_current_user
from ..models.user import UserInDB
from ..services.ai_service import ai_service, RateLimitError, AIProviderError
from ..services.cache_service import cache_service
from ..core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai", tags=["AI"])


class ChatMessage(BaseModel):
    """Chat message structure."""
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat request structure."""
    messages: List[ChatMessage] = Field(..., description="Conversation messages")
    model: Optional[str] = Field(None, description="Specific model to use")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Response creativity (0-2)")
    max_tokens: Optional[int] = Field(None, gt=0, le=8000, description="Maximum response tokens")
    stream: bool = Field(False, description="Enable streaming response")


class ChatResponse(BaseModel):
    """Chat response structure."""
    content: str = Field(..., description="AI response content")
    provider: str = Field(..., description="AI provider used")
    model: str = Field(..., description="Model used")
    tokens_used: int = Field(..., description="Tokens consumed")
    response_time: float = Field(..., description="Response time in seconds")


class StreamingChatRequest(BaseModel):
    """Streaming chat request for WebSocket."""
    messages: List[Dict[str, str]]
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


@router.post("/chat", response_model=ChatResponse)
async def chat_completion(
    request: ChatRequest,
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Generate AI chat completion.
    
    Supports multiple AI providers with automatic failover:
    - Primary: Gemini
    - Secondary: Groq  
    - Tertiary: OpenAI
    """
    try:
        # Convert messages to dict format
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Generate response
        response = await ai_service.generate_response(
            user_id=str(current_user.id),
            messages=messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        return ChatResponse(
            content=response.content,
            provider=response.provider,
            model=response.model,
            tokens_used=response.tokens_used,
            response_time=response.response_time
        )
        
    except RateLimitError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except AIProviderError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Chat completion error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.websocket("/chat/stream")
async def chat_stream_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for streaming chat responses.
    
    Expected message format:
    {
        "user_id": "user_id",
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "optional_model",
        "temperature": 0.7,
        "max_tokens": 4000
    }
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                request_data = json.loads(data)
                
                # Validate required fields
                if "user_id" not in request_data or "messages" not in request_data:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Missing required fields: user_id, messages"
                    }))
                    continue
                
                user_id = request_data["user_id"]
                messages = request_data["messages"]
                model = request_data.get("model")
                temperature = request_data.get("temperature")
                max_tokens = request_data.get("max_tokens")
                
                # Send start streaming indicator
                await websocket.send_text(json.dumps({
                    "type": "stream_start",
                    "message": "Starting AI response generation..."
                }))
                
                # Stream AI response
                full_content = ""
                async for chunk in ai_service.generate_stream(
                    user_id=user_id,
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                ):
                    if chunk.is_complete:
                        # Send completion message
                        await websocket.send_text(json.dumps({
                            "type": "stream_complete",
                            "provider": chunk.provider,
                            "model": chunk.model,
                            "full_content": full_content
                        }))
                    else:
                        # Send chunk
                        full_content += chunk.content
                        await websocket.send_text(json.dumps({
                            "type": "stream_chunk",
                            "content": chunk.content,
                            "provider": chunk.provider,
                            "model": chunk.model
                        }))
                
            except RateLimitError as e:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Rate limit exceeded: {str(e)}"
                }))
            except AIProviderError as e:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"AI provider error: {str(e)}"
                }))
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
            except Exception as e:
                logger.error(f"WebSocket streaming error: {str(e)}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Internal server error"
                }))
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await websocket.close()
        except:
            pass


@router.get("/models")
async def list_available_models():
    """List available AI models and providers."""
    models = {
        "gemini": {
            "available": bool(settings.GEMINI_API_KEY),
            "models": ["gemini-pro", "gemini-pro-vision"],
            "priority": 1
        },
        "groq": {
            "available": bool(settings.GROQ_API_KEY),
            "models": ["mixtral-8x7b-32768", "llama2-70b-4096"],
            "priority": 2
        },
        "openai": {
            "available": bool(settings.OPENAI_API_KEY),
            "models": ["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
            "priority": 3
        }
    }
    
    return {
        "providers": models,
        "default_provider": settings.PRIMARY_AI_PROVIDER,
        "rate_limits": {
            "requests_per_minute": settings.AI_RATE_LIMIT_PER_USER,
            "window_minutes": settings.AI_RATE_LIMIT_WINDOW_MINUTES
        }
    }


@router.get("/health")
async def ai_health_check():
    """Check AI service health and provider availability."""
    health_status = {
        "status": "healthy",
        "providers": {},
        "timestamp": str(logger.info)
    }
    
    # Check each provider
    for provider_name in ["gemini", "groq", "openai"]:
        api_key_setting = f"{provider_name.upper()}_API_KEY"
        has_key = bool(getattr(settings, api_key_setting, ""))
        
        health_status["providers"][provider_name] = {
            "configured": has_key,
            "status": "available" if has_key else "not_configured"
        }
    
    # Overall status
    available_providers = sum(1 for p in health_status["providers"].values() if p["configured"])
    if available_providers == 0:
        health_status["status"] = "unhealthy"
        health_status["message"] = "No AI providers configured"
    elif available_providers < 2:
        health_status["status"] = "degraded"
        health_status["message"] = "Limited AI providers available"
    
    return health_status


@router.get("/cache/stats")
async def get_cache_stats(current_user: UserInDB = Depends(get_current_user)):
    """Get cache statistics."""
    return await cache_service.get_cache_stats()


@router.delete("/cache/user")
async def clear_user_cache(current_user: UserInDB = Depends(get_current_user)):
    """Clear all cached responses for the current user."""
    await cache_service.invalidate_user_cache(str(current_user.id))
    return {"message": "User cache cleared successfully"}


@router.get("/rate-limit/status")
async def get_rate_limit_status(current_user: UserInDB = Depends(get_current_user)):
    """Get current rate limit status for the user."""
    current_count = await ai_service.rate_limiter.get_user_request_count(str(current_user.id))
    
    return {
        "user_id": str(current_user.id),
        "current_requests": current_count,
        "limit": settings.AI_RATE_LIMIT_PER_USER,
        "window_minutes": settings.AI_RATE_LIMIT_WINDOW_MINUTES,
        "remaining": max(0, settings.AI_RATE_LIMIT_PER_USER - current_count)
    }