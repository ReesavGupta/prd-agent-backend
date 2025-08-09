"""
Project service layer for business logic.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
from fastapi import HTTPException, UploadFile
import aiofiles
from app.models.project import Project, Upload
from app.schemas.project import (
    ProjectCreateRequest,
    ProjectUpdateRequest,
    ProjectResponse,
    ProjectListResponse,
    ProjectStatsResponse,
    UploadResponse
)
from app.repositories.project import ProjectRepository
from app.schemas.base import BaseResponse


class ProjectService:
    """Service layer for project operations."""

    def __init__(self, project_repository: ProjectRepository):
        self.project_repository = project_repository

    async def create_project(
        self,
        user_id: str,
        project_request: ProjectCreateRequest,
        files: Optional[List[UploadFile]] = None,
        *,
        created_from_chat: bool = False,
        source_chat_id: Optional[str] = None,
    ) -> ProjectResponse:
        """Create a new project with optional file uploads."""
        try:
            # Generate project name if not provided
            project_name = project_request.project_name
            if not project_name:
                # Generate name from first 50 chars of idea
                idea_preview = project_request.initial_idea[:50].strip()
                project_name = f"Project: {idea_preview}..."

            # Create project data
            project_data = {
                "user_id": user_id,
                "project_name": project_name,
                "initial_idea": project_request.initial_idea,
                "created_from_chat": created_from_chat,
                # note: repository will persist as provided; keep source_chat_id optional
                "source_chat_id": source_chat_id,
            }

            # Create project
            project = await self.project_repository.create_project(project_data)

            # Handle file uploads if provided
            if files:
                for file in files:
                    await self._upload_file_to_project(
                        str(project.id),
                        user_id,
                        file
                    )

            return self._project_to_response(project)

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create project: {str(e)}"
            )

    async def get_project(self, project_id: str, user_id: str) -> ProjectResponse:
        """Get project by ID."""
        project = await self.project_repository.get_project(project_id, user_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        return self._project_to_response(project)

    async def update_project(
        self,
        project_id: str,
        user_id: str,
        update_request: ProjectUpdateRequest
    ) -> ProjectResponse:
        """Update project information."""
        # Prepare update data
        update_data = {}
        if update_request.project_name is not None:
            update_data["project_name"] = update_request.project_name
        if update_request.initial_idea is not None:
            update_data["initial_idea"] = update_request.initial_idea
        if update_request.status is not None:
            # Validate status
            allowed_statuses = ["active", "archived", "deleted"]
            if update_request.status not in allowed_statuses:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status. Must be one of: {allowed_statuses}"
                )
            update_data["status"] = update_request.status

        if not update_data:
            raise HTTPException(
                status_code=400,
                detail="No valid fields provided for update"
            )

        project = await self.project_repository.update_project(
            project_id,
            user_id,
            update_data
        )
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        return self._project_to_response(project)

    async def delete_project(self, project_id: str, user_id: str) -> Dict[str, str]:
        """Delete (archive) a project."""
        success = await self.project_repository.delete_project(project_id, user_id)
        if not success:
            raise HTTPException(status_code=404, detail="Project not found")
        
        return {"message": "Project deleted successfully"}

    async def list_projects(
        self,
        user_id: str,
        status: str = "active",
        limit: int = 20,
        offset: int = 0,
        sort_by: str = "updated_at",
        sort_order: str = "desc"
    ) -> Dict[str, any]:
        """List user's projects with pagination."""
        # Validate sort parameters
        allowed_sort_fields = ["created_at", "updated_at", "project_name"]
        if sort_by not in allowed_sort_fields:
            sort_by = "updated_at"
        
        allowed_sort_orders = ["asc", "desc"]
        if sort_order not in allowed_sort_orders:
            sort_order = "desc"

        projects, total = await self.project_repository.list_projects(
            user_id=user_id,
            status=status,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            sort_order=sort_order
        )

        # Convert to list response format
        project_list = [self._project_to_list_response(p) for p in projects]

        return BaseResponse.paginated(
            items=project_list,
            total=total,
            limit=limit,
            offset=offset,
            message="Projects retrieved successfully"
        )

    async def get_project_stats(self, user_id: str) -> ProjectStatsResponse:
        """Get user's project statistics."""
        stats = await self.project_repository.get_project_stats(user_id)
        return ProjectStatsResponse(**stats)

    async def upload_files(
        self,
        project_id: str,
        user_id: str,
        files: List[UploadFile]
    ) -> List[UploadResponse]:
        """Upload files to a project."""
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        uploads = []
        for file in files:
            upload = await self._upload_file_to_project(project_id, user_id, file)
            uploads.append(await self._upload_to_response(upload))

        return uploads

    async def list_project_files(
        self,
        project_id: str,
        user_id: str
    ) -> List[UploadResponse]:
        """List files in a project."""
        uploads = await self.project_repository.list_project_files(project_id, user_id)
        return [await self._upload_to_response(upload) for upload in uploads]

    async def delete_file(self, file_id: str, user_id: str) -> Dict[str, str]:
        """Delete a file from a project."""
        success = await self.project_repository.delete_file(file_id, user_id)
        if not success:
            raise HTTPException(status_code=404, detail="File not found")
        
        return {"message": "File deleted successfully"}

    # Version management methods (placeholder for future AI integration)
    async def create_version(
        self,
        project_id: str,
        user_id: str,
        version_data: Dict
    ) -> Dict[str, str]:
        """Create a new version of the project (placeholder)."""
        # This will be implemented when AI agents are added
        # For now, just return success
        return {
            "message": "Version creation will be available with AI agent integration",
            "version": "v1.0"
        }

    async def rollback_version(
        self,
        project_id: str,
        user_id: str,
        target_version: str
    ) -> Dict[str, str]:
        """Rollback project to a specific version (placeholder)."""
        # This will be implemented when AI agents are added
        return {
            "message": "Version rollback will be available with AI agent integration",
            "current_version": target_version
        }

    # Private helper methods
    async def _upload_file_to_project(
        self,
        project_id: str,
        user_id: str,
        file: UploadFile
    ) -> Upload:
        """Upload a single file to a project."""
        # Validate file size (10MB limit)
        max_size = 10 * 1024 * 1024  # 10MB
        file_content = await file.read()
        if len(file_content) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File {file.filename} exceeds maximum size of 10MB"
            )

        # Validate content type
        allowed_types = [
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/markdown",
            "text/plain",
            "image/png",
            "image/jpeg",
            "image/jpg"
        ]
        
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=415,
                detail=f"File type {file.content_type} not supported"
            )

        # Upload file
        return await self.project_repository.upload_file(
            project_id=project_id,
            user_id=user_id,
            file_content=file_content,
            filename=file.filename or "unknown",
            content_type=file.content_type or "application/octet-stream"
        )

    def _project_to_response(self, project: Project) -> ProjectResponse:
        """Convert Project model to response schema."""
        return ProjectResponse(
            id=str(project.id),
            user_id=str(project.user_id),
            project_name=project.project_name,
            initial_idea=project.initial_idea,
            status=project.status,
            source_chat_id=str(project.source_chat_id) if project.source_chat_id else None,
            created_from_chat=project.created_from_chat,
            current_version=project.current_version,
            storage_path=project.storage_path,
            metadata=project.metadata,
            created_at=project.created_at,
            updated_at=project.updated_at
        )

    def _project_to_list_response(self, project: Project) -> ProjectListResponse:
        """Convert Project model to list response schema."""
        return ProjectListResponse(
            id=str(project.id),
            project_name=project.project_name,
            initial_idea=project.initial_idea,
            status=project.status,
            current_version=project.current_version,
            thinking_lens_status=project.metadata.thinking_lens_status,
            created_at=project.created_at,
            updated_at=project.updated_at
        )

    async def _upload_to_response(self, upload: Upload) -> UploadResponse:
        """Convert Upload model to response schema."""
        # Use stored URL if available, otherwise generate through storage repository
        file_url = upload.url
        if not file_url:
            try:
                file_url = await self.project_repository.file_storage.get_file_url(upload.storage_key)
            except Exception:
                # Fallback to basic Cloudinary URL if get_file_url fails
                from app.core.config import settings
                file_url = f"https://res.cloudinary.com/{settings.CLOUDINARY_CLOUD_NAME}/image/upload/{upload.storage_key}"
        
        return UploadResponse(
            id=str(upload.id),
            filename=upload.filename,
            original_filename=upload.original_filename,
            file_size=upload.file_size,
            content_type=upload.content_type,
            url=file_url,
            uploaded_at=upload.uploaded_at
        )