"""
Project data models for the AI PRD Generator.
"""

from datetime import datetime
from typing import Dict, List, Optional
from bson import ObjectId
from pydantic import BaseModel, Field, field_validator
from app.models.user import PyObjectId


class ThinkingLensStatus(BaseModel):
    """Thinking lens completion status."""
    discovery: bool = False
    user_journey: bool = False
    metrics: bool = False
    gtm: bool = False
    risks: bool = False


class ProjectMetadata(BaseModel):
    """Project metadata and status information."""
    thinking_lens_status: ThinkingLensStatus = Field(default_factory=ThinkingLensStatus)
    last_agent_run: Optional[datetime] = None
    total_iterations: int = 0
    file_count: int = 0
    storage_size_bytes: int = 0


class Project(BaseModel):
    """Project data model."""
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    user_id: PyObjectId = Field(...)
    project_name: str = Field(..., min_length=1, max_length=255)
    initial_idea: str = Field(..., min_length=1, max_length=5000)
    status: str = Field(default="active")  # active, archived, deleted
    source_chat_id: Optional[PyObjectId] = None
    created_from_chat: bool = False
    current_version: str = Field(default="v1.0")
    storage_path: str = Field(...)  # Cloudinary folder path
    metadata: ProjectMetadata = Field(default_factory=ProjectMetadata)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

    @field_validator("status")
    def validate_status(cls, v):
        """Validate project status."""
        allowed_statuses = ["active", "archived", "deleted"]
        if v not in allowed_statuses:
            raise ValueError(f"Status must be one of: {allowed_statuses}")
        return v

    @field_validator("current_version")
    def validate_version(cls, v):
        """Validate version format."""
        if not v.startswith("v") or len(v) < 3:
            raise ValueError("Version must be in format 'v1.0', 'v1.1', etc.")
        return v


class ProjectVersion(BaseModel):
    """Project version information."""
    version: str = Field(...)
    timestamp: datetime = Field(...)
    changes: str = Field(...)
    prd_url: Optional[str] = None
    flowchart_url: Optional[str] = None
    metadata_snapshot: Dict = Field(default_factory=dict)


class Upload(BaseModel):
    """File upload model."""
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    user_id: PyObjectId = Field(...)
    project_id: PyObjectId = Field(...)
    filename: str = Field(...)
    original_filename: str = Field(...)
    file_size: int = Field(...)
    content_type: str = Field(...)
    storage_key: str = Field(...)  # Cloudinary public_id
    url: Optional[str] = Field(default=None)  # File URL
    file_hash: str = Field(...)
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

    @field_validator("file_size")
    def validate_file_size(cls, v):
        """Validate file size limits."""
        max_size = 10 * 1024 * 1024  # 10MB
        if v > max_size:
            raise ValueError(f"File size cannot exceed {max_size} bytes")
        return v

    @field_validator("content_type")
    def validate_content_type(cls, v):
        """Validate file content type."""
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
        if v not in allowed_types:
            raise ValueError(f"Content type {v} not allowed")
        return v