"""
Base repository interfaces and implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import hashlib
import uuid


class FileStorageRepository(ABC):
    """Abstract interface for file storage operations."""

    @abstractmethod
    async def upload_file(
        self,
        file_content: bytes,
        filename: str,
        content_type: str,
        folder_path: str
    ) -> Dict[str, Any]:
        """
        Upload a file to storage.
        
        Args:
            file_content: The file content as bytes
            filename: Original filename
            content_type: MIME type of the file
            folder_path: Storage folder path
            
        Returns:
            Dictionary with upload result including storage_key and url
        """
        pass

    @abstractmethod
    async def delete_file(self, storage_key: str) -> bool:
        """
        Delete a file from storage.
        
        Args:
            storage_key: The storage identifier for the file
            
        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def get_file_url(self, storage_key: str, expires_in: int = 3600) -> str:
        """
        Get a URL for accessing the file.
        
        Args:
            storage_key: The storage identifier for the file
            expires_in: URL expiration time in seconds
            
        Returns:
            Accessible URL for the file
        """
        pass

    @abstractmethod
    async def download_file(self, filename: str, folder_path: str) -> Optional[bytes]:
        """
        Download file content from storage.
        
        Args:
            filename: The filename to download
            folder_path: Storage folder path
            
        Returns:
            File content as bytes, or None if file doesn't exist
        """
        pass

    @abstractmethod
    async def copy_file(self, source_key: str, destination_key: str) -> bool:
        """
        Copy a file within storage.
        
        Args:
            source_key: Source file storage key
            destination_key: Destination file storage key
            
        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def list_files(self, folder_path: str) -> List[Dict[str, Any]]:
        """
        List files in a folder.
        
        Args:
            folder_path: The folder path to list
            
        Returns:
            List of file information dictionaries
        """
        pass

    def generate_file_hash(self, file_content: bytes) -> str:
        """Generate SHA-256 hash of file content."""
        return hashlib.sha256(file_content).hexdigest()

    def generate_storage_key(self, folder_path: str, filename: str, file_hash: str) -> str:
        """Generate a storage key for the file."""
        # Use first 8 characters of hash for uniqueness
        hash_prefix = file_hash[:8]
        # Remove file extension and add hash
        name_without_ext = filename.rsplit('.', 1)[0] if '.' in filename else filename
        extension = filename.rsplit('.', 1)[-1] if '.' in filename else ''
        
        if extension:
            return f"{folder_path}/{name_without_ext}_{hash_prefix}.{extension}"
        return f"{folder_path}/{name_without_ext}_{hash_prefix}"


class CacheRepository(ABC):
    """Abstract interface for caching operations."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass

    @abstractmethod
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern."""
        pass

    def generate_cache_key(self, prefix: str, *args) -> str:
        """Generate a cache key from prefix and arguments."""
        # Convert all args to strings and join with colons
        key_parts = [prefix] + [str(arg) for arg in args]
        return ":".join(key_parts)