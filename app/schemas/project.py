"""
Pydantic schemas for Project API requests and responses.
"""

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from app.models.user import PyObjectId
from app.models.project import ThinkingLensStatus, ProjectMetadata


class ProjectCreateRequest(BaseModel):
    """Request schema for creating a new project."""
    project_name: Optional[str] = Field(None, min_length=1, max_length=255)
    initial_idea: str = Field(..., min_length=1, max_length=5000)


class ProjectUpdateRequest(BaseModel):
    """Request schema for updating a project."""
    project_name: Optional[str] = Field(None, min_length=1, max_length=255)
    initial_idea: Optional[str] = Field(None, min_length=1, max_length=5000)
    status: Optional[str] = Field(None)


class ProjectResponse(BaseModel):
    """Response schema for project data."""
    id: str = Field(...)
    user_id: str = Field(...)
    project_name: str = Field(...)
    initial_idea: str = Field(...)
    status: str = Field(...)
    source_chat_id: Optional[str] = None
    created_from_chat: bool = Field(...)
    current_version: str = Field(...)
    storage_path: str = Field(...)
    metadata: ProjectMetadata = Field(...)
    created_at: datetime = Field(...)
    updated_at: datetime = Field(...)


class ProjectListResponse(BaseModel):
    """Response schema for project listings."""
    id: str = Field(...)
    project_name: str = Field(...)
    initial_idea: str = Field(...)
    status: str = Field(...)
    current_version: str = Field(...)
    thinking_lens_status: ThinkingLensStatus = Field(...)
    created_at: datetime = Field(...)
    updated_at: datetime = Field(...)


class ProjectVersionResponse(BaseModel):
    """Response schema for project version data."""
    version: str = Field(...)
    timestamp: datetime = Field(...)
    changes: str = Field(...)
    prd_url: Optional[str] = None
    flowchart_url: Optional[str] = None


class UploadResponse(BaseModel):
    """Response schema for file upload."""
    id: str = Field(...)
    filename: str = Field(...)
    original_filename: str = Field(...)
    file_size: int = Field(...)
    content_type: str = Field(...)
    url: str = Field(...)
    uploaded_at: datetime = Field(...)


class ProjectQueryParams(BaseModel):
    """Query parameters for project listing."""
    status: Optional[str] = Field(default="active")
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    sort_by: str = Field(default="updated_at")
    sort_order: str = Field(default="desc")

    class Config:
        str_strip_whitespace = True


class VersionRollbackRequest(BaseModel):
    """Request schema for version rollback."""
    version: str = Field(..., min_length=1)


class ProjectStatsResponse(BaseModel):
    """Response schema for project statistics."""
    total_projects: int = Field(...)
    active_projects: int = Field(...)
    archived_projects: int = Field(...)
    total_files: int = Field(...)
    total_storage_bytes: int = Field(...)
    recent_activity_count: int = Field(...)