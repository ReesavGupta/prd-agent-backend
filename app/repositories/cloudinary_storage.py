"""
Cloudinary implementation of the FileStorageRepository.
"""

import io
from typing import Any, Dict, List, Optional
from cloudinary import uploader, api
from cloudinary.utils import cloudinary_url
import cloudinary
from app.repositories.base import FileStorageRepository
from app.core.config import settings


class CloudinaryStorageRepository(FileStorageRepository):
    """Cloudinary implementation of file storage."""

    def __init__(self):
        """Initialize Cloudinary configuration."""
        cloudinary.config(
            cloud_name=settings.CLOUDINARY_CLOUD_NAME,
            api_key=settings.CLOUDINARY_API_KEY,
            api_secret=settings.CLOUDINARY_API_SECRET,
            secure=settings.CLOUDINARY_SECURE_URL
        )

    async def upload_file(
        self,
        file_content: bytes,
        filename: str,
        content_type: str,
        folder_path: str
    ) -> Dict[str, Any]:
        """Upload file to Cloudinary."""
        try:
            # Generate file hash for uniqueness
            file_hash = self.generate_file_hash(file_content)

            # Build a base public_id WITHOUT folder path and WITHOUT extension.
            # Folder path is passed separately via the 'folder' parameter.
            name_without_ext = filename.rsplit('.', 1)[0] if '.' in filename else filename
            hash_prefix = file_hash[:8]
            public_id_base = f"{name_without_ext}_{hash_prefix}"

            # Prepare folder path with prefix
            full_folder = f"{settings.CLOUDINARY_FOLDER_PREFIX}/{folder_path}"

            # Upload to Cloudinary
            result = uploader.upload(
                io.BytesIO(file_content),
                public_id=public_id_base,
                folder=full_folder,
                resource_type="auto",  # Auto-detect resource type
                use_filename=False,
                unique_filename=False,
                overwrite=True
            )

            return {
                "storage_key": result["public_id"],  # includes folder in the resulting public_id
                "url": result["secure_url"],
                "size": result["bytes"],
                "format": result.get("format", ""),
                "resource_type": result.get("resource_type", ""),
                "version": result.get("version", ""),
                "file_hash": file_hash
            }
            
        except Exception as e:
            raise Exception(f"Failed to upload file to Cloudinary: {str(e)}")

    async def delete_file(self, storage_key: str) -> bool:
        """Delete file from Cloudinary."""
        try:
            result = uploader.destroy(storage_key)
            return result.get("result") == "ok"
        except Exception:
            return False

    async def get_file_url(self, storage_key: str, expires_in: int = 3600) -> str:
        """Get URL for Cloudinary file."""
        try:
            # For Cloudinary, we can generate secure URLs
            url, _ = cloudinary_url(
                storage_key,
                secure=True,
                sign_url=True,  # Generate signed URL for security
                auth_token={
                    "duration": expires_in
                } if expires_in > 0 else None
            )
            return url
        except Exception as e:
            raise Exception(f"Failed to generate file URL: {str(e)}")

    async def download_file(self, filename: str, folder_path: str) -> Optional[bytes]:
        """Download file content from Cloudinary."""
        try:
            # Build the public_id from folder path and filename
            full_folder = f"{settings.CLOUDINARY_FOLDER_PREFIX}/{folder_path}"
            
            # For saved artifacts, the public_id should match what was uploaded
            # Remove extension from filename to match upload pattern
            name_without_ext = filename.rsplit('.', 1)[0] if '.' in filename else filename
            
            # Try to find the file - since we don't have the hash, we need to list files in the folder
            files = await self.list_files(folder_path)
            
            # Find matching file by checking if the filename starts with our expected name
            target_file = None
            for file_info in files:
                public_id = file_info["storage_key"]
                # Extract the filename part from the public_id
                file_basename = public_id.split('/')[-1]
                if file_basename.startswith(name_without_ext) or file_basename == filename:
                    target_file = file_info
                    break
            
            if not target_file:
                return None
                
            # Download the file content using the Cloudinary API
            import requests
            response = requests.get(target_file["url"])
            if response.status_code == 200:
                return response.content
            else:
                return None
                
        except Exception:
            return None

    async def copy_file(self, source_key: str, destination_key: str) -> bool:
        """Copy file within Cloudinary."""
        try:
            # Get the source file
            source_info = api.resource(source_key)
            source_url = source_info["secure_url"]
            
            # Upload to new location
            result = uploader.upload(
                source_url,
                public_id=destination_key,
                overwrite=True
            )
            
            return result.get("public_id") == destination_key
        except Exception:
            return False

    async def list_files(self, folder_path: str) -> List[Dict[str, Any]]:
        """List files in a Cloudinary folder."""
        try:
            full_folder = f"{settings.CLOUDINARY_FOLDER_PREFIX}/{folder_path}"
            
            result = api.resources(
                type="upload",
                prefix=full_folder,
                max_results=500  # Adjust as needed
            )
            
            files = []
            for resource in result.get("resources", []):
                files.append({
                    "storage_key": resource["public_id"],
                    "url": resource["secure_url"],
                    "size": resource["bytes"],
                    "format": resource.get("format", ""),
                    "created_at": resource.get("created_at", ""),
                    "resource_type": resource.get("resource_type", "")
                })
                
            return files
            
        except Exception as e:
            raise Exception(f"Failed to list files: {str(e)}")

    async def create_folder_structure(self, project_id: str, user_id: str) -> str:
        """Create folder structure for a project."""
        folder_path = f"projects/{user_id}/{project_id}"
        
        # Create the basic folder structure by uploading placeholder files
        # and then deleting them (Cloudinary creates folders on upload)
        try:
            placeholder_content = b"placeholder"
            
            # Create main project folder
            await self.upload_file(
                placeholder_content,
                "placeholder.txt",
                "text/plain",
                folder_path
            )
            
            # Create versions folder
            await self.upload_file(
                placeholder_content,
                "placeholder.txt", 
                "text/plain",
                f"{folder_path}/versions"
            )
            
            # Create current folder
            await self.upload_file(
                placeholder_content,
                "placeholder.txt",
                "text/plain",
                f"{folder_path}/current"
            )
            
            # Create uploads folder
            await self.upload_file(
                placeholder_content,
                "placeholder.txt",
                "text/plain", 
                f"{folder_path}/uploads"
            )
            
            # Clean up placeholders
            await self.delete_file(f"{settings.CLOUDINARY_FOLDER_PREFIX}/{folder_path}/placeholder")
            await self.delete_file(f"{settings.CLOUDINARY_FOLDER_PREFIX}/{folder_path}/versions/placeholder")
            await self.delete_file(f"{settings.CLOUDINARY_FOLDER_PREFIX}/{folder_path}/current/placeholder")
            await self.delete_file(f"{settings.CLOUDINARY_FOLDER_PREFIX}/{folder_path}/uploads/placeholder")
            
            return folder_path
            
        except Exception as e:
            # If folder creation fails, return the path anyway
            # Folders will be created on first file upload
            return folder_path