"""
Redis implementation of the CacheRepository.
"""

import json
import redis.asyncio as redis
from typing import Any, Optional
from app.repositories.base import CacheRepository
from app.core.config import settings


class RedisCacheRepository(CacheRepository):
    """Redis implementation of caching."""

    def __init__(self):
        """Initialize Redis connection."""
        self._redis: Optional[redis.Redis] = None

    async def get_redis(self) -> redis.Redis:
        """Get Redis connection."""
        if self._redis is None:
            self._redis = redis.from_url(
                settings.REDIS_URL,
                db=settings.REDIS_DB,
                decode_responses=True,
                encoding="utf-8"
            )
        return self._redis

    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        try:
            redis_client = await self.get_redis()
            value = await redis_client.get(key)
            if value is None:
                return None
            # Try to deserialize JSON, fallback to string
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        except Exception:
            return None

    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in Redis cache with TTL."""
        try:
            redis_client = await self.get_redis()
            
            # Serialize complex objects to JSON
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value, default=str)
            else:
                serialized_value = str(value)
            
            await redis_client.set(key, serialized_value, ex=ttl)
            return True
        except Exception:
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache."""
        try:
            redis_client = await self.get_redis()
            result = await redis_client.delete(key)
            return result > 0
        except Exception:
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        try:
            redis_client = await self.get_redis()
            result = await redis_client.exists(key)
            return result > 0
        except Exception:
            return False

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern."""
        try:
            redis_client = await self.get_redis()
            keys = await redis_client.keys(pattern)
            if keys:
                return await redis_client.delete(*keys)
            return 0
        except Exception:
            return 0

    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment a numeric value in cache."""
        try:
            redis_client = await self.get_redis()
            return await redis_client.incrby(key, amount)
        except Exception:
            return 0

    async def set_hash(self, key: str, mapping: dict, ttl: int = 3600) -> bool:
        """Set hash values in Redis."""
        try:
            redis_client = await self.get_redis()
            
            # Serialize values in the mapping
            serialized_mapping = {}
            for field, value in mapping.items():
                if isinstance(value, (dict, list)):
                    serialized_mapping[field] = json.dumps(value, default=str)
                else:
                    serialized_mapping[field] = str(value)
            
            await redis_client.hset(key, mapping=serialized_mapping)
            if ttl > 0:
                await redis_client.expire(key, ttl)
            return True
        except Exception:
            return False

    async def get_hash(self, key: str, field: str = None) -> Optional[Any]:
        """Get hash value(s) from Redis."""
        try:
            redis_client = await self.get_redis()
            
            if field:
                value = await redis_client.hget(key, field)
                if value is None:
                    return None
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value
            else:
                values = await redis_client.hgetall(key)
                if not values:
                    return None
                
                # Deserialize all values
                result = {}
                for k, v in values.items():
                    try:
                        result[k] = json.loads(v)
                    except (json.JSONDecodeError, TypeError):
                        result[k] = v
                return result
        except Exception:
            return None

    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()

    # Convenience methods for common cache patterns
    def project_cache_key(self, user_id: str, project_id: str) -> str:
        """Generate cache key for project data."""
        return self.generate_cache_key("project", user_id, project_id)

    def project_list_cache_key(self, user_id: str, status: str, offset: int, limit: int) -> str:
        """Generate cache key for project list."""
        return self.generate_cache_key("project_list", user_id, status, offset, limit)

    def file_cache_key(self, project_id: str, file_id: str) -> str:
        """Generate cache key for file data."""
        return self.generate_cache_key("file", project_id, file_id)

    def user_stats_cache_key(self, user_id: str) -> str:
        """Generate cache key for user statistics."""
        return self.generate_cache_key("user_stats", user_id)