"""
Redis cache service for AI responses and rate limiting.
Provides caching functionality to reduce AI API calls and improve response times.
"""

import json
import hashlib
import logging
import time
from typing import Optional, Dict, List, Any
from datetime import timedelta

import redis.asyncio as redis
from redis.asyncio import Redis

from ..core.config import settings

logger = logging.getLogger(__name__)


class CacheService:
    """Redis-based cache service for AI responses."""
    
    def __init__(self):
        self.redis: Optional[Redis] = None
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis connection."""
        try:
            self.redis = redis.from_url(
                settings.REDIS_URL,
                db=settings.REDIS_DB,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            logger.info("Redis cache service initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis: {e}")
            self.redis = None
    
    async def is_connected(self) -> bool:
        """Check if Redis is connected."""
        if not self.redis:
            return False
        
        try:
            await self.redis.ping()
            return True
        except Exception:
            return False
    
    def _generate_cache_key(self, user_id: str, messages: List[Dict], **kwargs) -> str:
        """Generate a cache key for AI requests."""
        # Create a hash of the request parameters
        request_data = {
            "messages": messages,
            "model": kwargs.get("model"),
            "temperature": kwargs.get("temperature"),
            "max_tokens": kwargs.get("max_tokens")
        }
        
        request_str = json.dumps(request_data, sort_keys=True)
        request_hash = hashlib.md5(request_str.encode()).hexdigest()
        
        return f"ai_response:{user_id}:{request_hash}"
    
    async def get_cached_response(
        self,
        user_id: str,
        messages: List[Dict],
        **kwargs
    ) -> Optional[Dict]:
        """Get cached AI response."""
        if not await self.is_connected():
            return None
        
        try:
            cache_key = self._generate_cache_key(user_id, messages, **kwargs)
            cached_data = await self.redis.get(cache_key)
            
            if cached_data:
                logger.info(f"Cache hit for key: {cache_key}")
                return json.loads(cached_data)
            
            return None
            
        except Exception as e:
            logger.warning(f"Error getting cached response: {e}")
            return None
    
    async def cache_response(
        self,
        user_id: str,
        messages: List[Dict],
        response: Dict,
        ttl_seconds: int = None,
        **kwargs
    ):
        """Cache AI response."""
        if not await self.is_connected():
            return
        
        try:
            cache_key = self._generate_cache_key(user_id, messages, **kwargs)
            ttl = ttl_seconds or settings.CACHE_TTL
            
            response_data = {
                "content": response.get("content"),
                "provider": response.get("provider"),
                "model": response.get("model"),
                "tokens_used": response.get("tokens_used"),
                "response_time": response.get("response_time"),
                "cached_at": str(response.get("timestamp"))
            }
            
            await self.redis.setex(
                cache_key,
                ttl,
                json.dumps(response_data)
            )
            
            logger.info(f"Cached response for key: {cache_key}")
            
        except Exception as e:
            logger.warning(f"Error caching response: {e}")
    
    async def invalidate_user_cache(self, user_id: str):
        """Invalidate all cached responses for a user."""
        if not await self.is_connected():
            return
        
        try:
            pattern = f"ai_response:{user_id}:*"
            keys = await self.redis.keys(pattern)
            
            if keys:
                await self.redis.delete(*keys)
                logger.info(f"Invalidated {len(keys)} cached responses for user {user_id}")
                
        except Exception as e:
            logger.warning(f"Error invalidating user cache: {e}")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not await self.is_connected():
            return {"status": "disconnected"}
        
        try:
            info = await self.redis.info("memory")
            keyspace = await self.redis.info("keyspace")
            
            # Count AI response keys
            ai_keys = await self.redis.keys("ai_response:*")
            rate_limit_keys = await self.redis.keys("rate_limit:*")
            
            return {
                "status": "connected",
                "memory_used": info.get("used_memory_human", "Unknown"),
                "total_keys": sum(db_info.get("keys", 0) for db_info in keyspace.values()) if keyspace else 0,
                "ai_response_keys": len(ai_keys),
                "rate_limit_keys": len(rate_limit_keys),
                "redis_version": info.get("redis_version", "Unknown")
            }
            
        except Exception as e:
            logger.warning(f"Error getting cache stats: {e}")
            return {"status": "error", "message": str(e)}


class RateLimitCache:
    """Redis-based rate limiting for AI requests."""
    
    def __init__(self, cache_service: CacheService):
        self.cache_service = cache_service
    
    async def check_rate_limit(self, user_id: str, limit: int, window_seconds: int) -> bool:
        """Check if user is within rate limit using sliding window."""
        if not await self.cache_service.is_connected():
            # If Redis is down, allow the request but log it
            logger.warning("Redis unavailable for rate limiting, allowing request")
            return True
        
        try:
            key = f"rate_limit:{user_id}"
            current_time = int(time.time())
            window_start = current_time - window_seconds
            
            # Use Redis pipeline for atomic operations
            async with self.cache_service.redis.pipeline() as pipe:
                # Remove expired entries
                await pipe.zremrangebyscore(key, 0, window_start)
                
                # Count current requests
                await pipe.zcard(key)
                
                # Add current request
                await pipe.zadd(key, {str(current_time): current_time})
                
                # Set expiry
                await pipe.expire(key, window_seconds)
                
                results = await pipe.execute()
            
            current_count = results[1]  # Result of zcard
            
            return current_count < limit
            
        except Exception as e:
            logger.warning(f"Rate limit check error: {e}")
            # If there's an error, allow the request
            return True
    
    async def get_user_request_count(self, user_id: str, window_seconds: int) -> int:
        """Get current request count for user."""
        if not await self.cache_service.is_connected():
            return 0
        
        try:
            key = f"rate_limit:{user_id}"
            current_time = int(time.time())
            window_start = current_time - window_seconds
            
            # Clean expired entries and count
            async with self.cache_service.redis.pipeline() as pipe:
                await pipe.zremrangebyscore(key, 0, window_start)
                await pipe.zcard(key)
                results = await pipe.execute()
            
            return results[1]
            
        except Exception as e:
            logger.warning(f"Error getting request count: {e}")
            return 0


# Global cache service instance
cache_service = CacheService()
rate_limit_cache = RateLimitCache(cache_service)