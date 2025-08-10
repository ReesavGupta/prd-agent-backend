"""
Local filesystem implementation of the FileStorageRepository for development/testing.

Stores files under a configurable root directory and returns file:// URLs for access.
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict, List

from app.repositories.base import FileStorageRepository
from app.core.config import settings


class LocalStorageRepository(FileStorageRepository):
    """Simple local storage backend. Not suitable for production.

    Files are written under settings.LOCAL_STORAGE_DIR. get_file_url returns a
    file:// URL which callers must handle (e.g., open directly instead of HTTP GET).
    """

    def __init__(self) -> None:
        root = settings.LOCAL_STORAGE_DIR or "local_storage"
        # Normalize to absolute path for stability
        self.root: str = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    def _abs_path_for(self, storage_key: str) -> str:
        # storage_key is expected to be a relative path within the root
        safe = storage_key.lstrip("/\\")
        full = os.path.abspath(os.path.join(self.root, safe))
        # Ensure within root
        if not os.path.commonpath([self.root, full]) == self.root:
            raise ValueError("Invalid storage_key path traversal attempt")
        return full

    async def upload_file(
        self,
        file_content: bytes,
        filename: str,
        content_type: str,
        folder_path: str,
    ) -> Dict[str, Any]:
        # Compute a stable hashed filename like the cloud repository
        file_hash = self.generate_file_hash(file_content)
        storage_key = self.generate_storage_key(folder_path, filename, file_hash)
        abs_dir = os.path.dirname(self._abs_path_for(storage_key))
        os.makedirs(abs_dir, exist_ok=True)
        abs_path = self._abs_path_for(storage_key)
        with open(abs_path, "wb") as f:
            f.write(file_content)
        # Normalize to file:/// URL (POSIX style) for compatibility
        normalized = abs_path.replace("\\", "/")
        file_url = f"file:///{normalized}"
        return {
            "storage_key": storage_key,
            "url": file_url,
            "size": len(file_content),
            "format": (filename.rsplit(".", 1)[-1] if "." in filename else ""),
            "resource_type": content_type or "application/octet-stream",
            "version": "1",
            "file_hash": file_hash,
        }

    async def delete_file(self, storage_key: str) -> bool:
        try:
            abs_path = self._abs_path_for(storage_key)
            if os.path.isfile(abs_path):
                os.remove(abs_path)
                return True
            # If it's a directory (shouldn't be), remove tree
            if os.path.isdir(abs_path):
                shutil.rmtree(abs_path)
                return True
            return False
        except Exception:
            return False

    async def get_file_url(self, storage_key: str, expires_in: int = 3600) -> str:
        # Local files are addressed via file:/// URL
        abs_path = self._abs_path_for(storage_key)
        normalized = abs_path.replace("\\", "/")
        return f"file:///{normalized}"

    async def copy_file(self, source_key: str, destination_key: str) -> bool:
        try:
            src = self._abs_path_for(source_key)
            dst = self._abs_path_for(destination_key)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            return True
        except Exception:
            return False

    async def list_files(self, folder_path: str) -> List[Dict[str, Any]]:
        base = os.path.abspath(os.path.join(self.root, folder_path))
        files: List[Dict[str, Any]] = []
        if not os.path.isdir(base):
            return files
        for root, _dirs, filenames in os.walk(base):
            for name in filenames:
                full = os.path.join(root, name)
                rel = os.path.relpath(full, self.root).replace("\\", "/")
                try:
                    size = os.path.getsize(full)
                except Exception:
                    size = 0
                files.append({
                    "storage_key": rel,
                    "url": f"file://{full}",
                    "size": size,
                    "format": name.rsplit(".", 1)[-1] if "." in name else "",
                    "created_at": "",
                    "resource_type": "file",
                })
        return files

    async def create_folder_structure(self, project_id: str, user_id: str) -> str:
        folder_path = f"projects/{user_id}/{project_id}"
        # Create canonical folders used by the rest of the app
        for sub in ("", "/versions", "/current", "/uploads"):
            os.makedirs(os.path.join(self.root, folder_path + sub), exist_ok=True)
        return folder_path


