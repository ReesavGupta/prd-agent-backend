"""
AI Service for managing multiple AI providers with failover support.
Uses LangChain chains with streaming support for real-time responses.
"""

import asyncio
import json
import logging
import time
from enum import Enum
from typing import AsyncIterator, Dict, List, Optional, Union, Any
from datetime import datetime, timedelta

from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from ..core.config import settings
from .cache_service import cache_service, rate_limit_cache

logger = logging.getLogger(__name__)


class AIProvider(str, Enum):
    """Supported AI providers."""
    GEMINI = "gemini"
    GROQ = "groq"
    OPENAI = "openai"


class AIResponse(BaseModel):
    """AI response structure."""
    content: str
    provider: str
    model: str
    tokens_used: int
    response_time: float
    timestamp: datetime


class AIStreamChunk(BaseModel):
    """AI streaming chunk structure."""
    content: str
    is_complete: bool = False
    provider: str
    model: str


class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""
    pass


class AIProviderError(Exception):
    """Raised when AI provider fails."""
    pass


class AIRateLimiter:
    """Per-user rate limiting for AI requests using Redis."""
    
    def __init__(self):
        self.rate_limit_cache = rate_limit_cache
    
    async def check_rate_limit(self, user_id: str) -> bool:
        """Check if user has exceeded rate limit."""
        window_seconds = settings.AI_RATE_LIMIT_WINDOW_MINUTES * 60
        return await self.rate_limit_cache.check_rate_limit(
            user_id,
            settings.AI_RATE_LIMIT_PER_USER,
            window_seconds
        )
    
    async def get_user_request_count(self, user_id: str) -> int:
        """Get current request count for user."""
        window_seconds = settings.AI_RATE_LIMIT_WINDOW_MINUTES * 60
        return await self.rate_limit_cache.get_user_request_count(user_id, window_seconds)


class AIService:
    """Main AI service with provider failover and streaming support."""
    
    def __init__(self):
        self.rate_limiter = AIRateLimiter()
        self.models: Dict[AIProvider, Any] = {}
        self.chains: Dict[AIProvider, Any] = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all available AI providers and chains."""
        # Initialize Gemini
        if settings.GEMINI_API_KEY:
            try:
                self.models[AIProvider.GEMINI] = ChatGoogleGenerativeAI(
                    google_api_key=settings.GEMINI_API_KEY,
                    model=settings.GEMINI_MODEL,
                    temperature=settings.AI_TEMPERATURE,
                    max_output_tokens=settings.AI_MAX_TOKENS,
                    timeout=settings.AI_TIMEOUT_SECONDS
                )
                self._create_chain(AIProvider.GEMINI)
                logger.info("Gemini provider initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini: {e}")
        
        # Initialize Groq
        if settings.GROQ_API_KEY:
            try:
                self.models[AIProvider.GROQ] = ChatGroq(
                    api_key=settings.GROQ_API_KEY,
                    model=settings.GROQ_MODEL,
                    temperature=settings.AI_TEMPERATURE,
                    max_tokens=settings.AI_MAX_TOKENS,
                    timeout=settings.AI_TIMEOUT_SECONDS
                )
                self._create_chain(AIProvider.GROQ)
                logger.info("Groq provider initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq: {e}")
        
        # Initialize OpenAI
        if settings.OPENAI_API_KEY:
            try:
                self.models[AIProvider.OPENAI] = ChatOpenAI(
                    api_key=settings.OPENAI_API_KEY,
                    model=settings.OPENAI_MODEL,
                    temperature=settings.AI_TEMPERATURE,
                    max_completion_tokens=settings.AI_MAX_TOKENS,
                    timeout=settings.AI_TIMEOUT_SECONDS
                )
                self._create_chain(AIProvider.OPENAI)
                logger.info("OpenAI provider initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI: {e}")
        
        if not self.models:
            raise RuntimeError("No AI providers available. Please check your API keys.")
    
    def _create_chain(self, provider: AIProvider):
        """Create LangChain chain for the provider."""
        # Create a flexible prompt template that handles message history
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="messages")
        ])
        
        # Create the chain: prompt -> model -> output parser
        parser = StrOutputParser()
        chain = prompt | self.models[provider] | parser
        
        self.chains[provider] = chain
    
    def _convert_messages_to_langchain(self, messages: List[Dict[str, str]]) -> List:
        """Convert message format to LangChain format."""
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
        return langchain_messages
    
    def _get_provider_order(self) -> List[AIProvider]:
        """Get provider order based on configuration."""
        order = []
        
        # Add providers in priority order
        primary = getattr(AIProvider, settings.PRIMARY_AI_PROVIDER.upper(), None)
        if primary and primary in self.models:
            order.append(primary)
        
        secondary = getattr(AIProvider, settings.SECONDARY_AI_PROVIDER.upper(), None)
        if secondary and secondary in self.models and secondary not in order:
            order.append(secondary)
        
        tertiary = getattr(AIProvider, settings.TERTIARY_AI_PROVIDER.upper(), None)
        if tertiary and tertiary in self.models and tertiary not in order:
            order.append(tertiary)
        
        # Add any remaining providers
        for provider in AIProvider:
            if provider in self.models and provider not in order:
                order.append(provider)
        
        return order
    
    async def generate_response(
        self,
        user_id: str,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
        **kwargs
    ) -> AIResponse:
        """Generate AI response with provider failover and caching."""
        # Check cache first (if enabled)
        if use_cache:
            cached_response = await cache_service.get_cached_response(
                user_id, messages, model=model, temperature=temperature, max_tokens=max_tokens
            )
            if cached_response:
                logger.info("Returning cached AI response")
                return AIResponse(
                    content=cached_response["content"],
                    provider=cached_response["provider"],
                    model=cached_response["model"],
                    tokens_used=cached_response["tokens_used"],
                    response_time=cached_response["response_time"],
                    timestamp=datetime.fromisoformat(cached_response["cached_at"])
                )
        
        # Check rate limit
        if not await self.rate_limiter.check_rate_limit(user_id):
            raise RateLimitError("AI rate limit exceeded. Please try again later.")
        
        # Convert messages to LangChain format
        langchain_messages = self._convert_messages_to_langchain(messages)
        
        # Try providers in order
        provider_order = self._get_provider_order()
        last_error = None
        
        for provider in provider_order:
            try:
                start_time = time.time()
                
                # Use default models per provider (ignore requested model)
                
                if temperature is not None:
                    self.models[provider].temperature = temperature
                
                if max_tokens is not None:
                    if provider == AIProvider.GEMINI:
                        self.models[provider].max_output_tokens = max_tokens
                    else:
                        self.models[provider].max_tokens = max_tokens
                
                logger.info(f"Attempting AI generation with {provider.value}")
                
                # Generate response using the chain
                response_content = await self.chains[provider].ainvoke({
                    "messages": langchain_messages
                })
                
                response_time = time.time() - start_time
                
                response = AIResponse(
                    content=response_content,
                    provider=provider.value,
                    model=getattr(self.models[provider], 'model', getattr(self.models[provider], 'model_name', 'unknown')),
                    tokens_used=0,  # Token counting can be added later
                    response_time=response_time,
                    timestamp=datetime.now()
                )
                
                # Cache the response if caching is enabled
                if use_cache:
                    await cache_service.cache_response(
                        user_id, messages, response.dict(), 
                        model=model, temperature=temperature, max_tokens=max_tokens
                    )
                
                logger.info(f"Successfully generated response with {provider.value}")
                return response
                
            except Exception as e:
                logger.warning(f"Provider {provider.value} failed: {str(e)}")
                last_error = e
                continue
        
        # All providers failed
        raise AIProviderError(f"All AI providers failed. Last error: {str(last_error)}")
    
    async def generate_stream(
        self,
        user_id: str,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[AIStreamChunk]:
        """Generate AI streaming response with provider failover."""
        # Check rate limit
        if not await self.rate_limiter.check_rate_limit(user_id):
            raise RateLimitError("AI rate limit exceeded. Please try again later.")
        
        # Convert messages to LangChain format
        langchain_messages = self._convert_messages_to_langchain(messages)
        
        # Try providers in order
        provider_order = self._get_provider_order()
        last_error = None
        
        for provider in provider_order:
            try:
                # Use default models per provider (ignore requested model)
                
                if temperature is not None:
                    self.models[provider].temperature = temperature
                
                if max_tokens is not None:
                    if provider == AIProvider.GEMINI:
                        self.models[provider].max_output_tokens = max_tokens
                    else:
                        self.models[provider].max_tokens = max_tokens
                
                logger.info(f"Attempting AI streaming with {provider.value}")
                
                # Stream response using the chain
                async for chunk in self.chains[provider].astream({
                    "messages": langchain_messages
                }):
                    yield AIStreamChunk(
                        content=chunk,
                        is_complete=False,
                        provider=provider.value,
                        model=getattr(self.models[provider], 'model', getattr(self.models[provider], 'model_name', 'unknown'))
                    )
                
                # Send completion chunk
                yield AIStreamChunk(
                    content="",
                    is_complete=True,
                    provider=provider.value,
                    model=getattr(self.models[provider], 'model', getattr(self.models[provider], 'model_name', 'unknown'))
                )
                
                logger.info(f"Successfully completed streaming with {provider.value}")
                return
                
            except Exception as e:
                logger.warning(f"Provider {provider.value} streaming failed: {str(e)}")
                last_error = e
                continue
        
        # All providers failed
        raise AIProviderError(f"All AI providers failed for streaming. Last error: {str(last_error)}")
    
    def update_model_config(
        self,
        provider: AIProvider,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """Update model configuration for a specific provider."""
        if provider not in self.models:
            raise ValueError(f"Provider {provider.value} not available")
        
        if model:
            if provider == AIProvider.GEMINI:
                self.models[provider].model = model
            elif provider == AIProvider.GROQ:
                self.models[provider].model_name = model
            elif provider == AIProvider.OPENAI:
                self.models[provider].model = model
        
        if temperature is not None:
            self.models[provider].temperature = temperature
        
        if max_tokens is not None:
            if provider == AIProvider.GEMINI:
                self.models[provider].max_output_tokens = max_tokens
            else:
                self.models[provider].max_tokens = max_tokens


# Global AI service instance
ai_service = AIService()