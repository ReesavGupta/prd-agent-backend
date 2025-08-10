"""
Dependency injection setup for the application.
"""

from functools import lru_cache
from app.db.database import get_database
from app.repositories.cloudinary_storage import CloudinaryStorageRepository
from app.repositories.local_storage import LocalStorageRepository
from app.repositories.redis_cache import RedisCacheRepository
from app.repositories.project import ProjectRepository
from app.services.project_service import ProjectService


# Storage and cache instances (singletons)
_storage_repository = None
_cache_repository = None


@lru_cache()
def get_storage_repository():
    """Get file storage repository instance.

    Chooses Cloudinary when credentials are configured; otherwise falls back to
    LocalStorageRepository for development to avoid 500s on upload.
    """
    global _storage_repository
    if _storage_repository is not None:
        return _storage_repository
    from app.core.config import settings
    have_cloudinary = bool(settings.CLOUDINARY_CLOUD_NAME and settings.CLOUDINARY_API_KEY and settings.CLOUDINARY_API_SECRET)
    if have_cloudinary and not settings.USE_LOCAL_STORAGE:
        _storage_repository = CloudinaryStorageRepository()
    else:
        _storage_repository = LocalStorageRepository()
    return _storage_repository


@lru_cache()
def get_cache_repository() -> RedisCacheRepository:
    """Get cache repository instance."""
    global _cache_repository
    if _cache_repository is None:
        _cache_repository = RedisCacheRepository()
    return _cache_repository


def get_cloudinary_storage() -> CloudinaryStorageRepository:
    """Get Cloudinary storage repository instance."""
    return get_storage_repository()


def get_project_repository() -> ProjectRepository:
    """Get project repository instance."""
    database = get_database()
    storage = get_storage_repository()
    cache = get_cache_repository()
    return ProjectRepository(database, storage, cache)


def get_project_service() -> ProjectService:
    """Get project service instance."""
    project_repository = get_project_repository()
    return ProjectService(project_repository)


# Cleanup function for application shutdown
async def cleanup_dependencies():
    """Clean up dependencies on application shutdown."""
    global _cache_repository
    if _cache_repository:
        await _cache_repository.close()