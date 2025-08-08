"""
Project API endpoints.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, status, Request
from fastapi.responses import JSONResponse
from app.schemas.project import (
    ProjectCreateRequest,
    ProjectUpdateRequest,
    ProjectResponse,
    ProjectStatsResponse,
    ProjectQueryParams,
    VersionRollbackRequest
)
from app.services.project_service import ProjectService
from app.middleware.auth import get_current_user
from app.models.user import UserInDB as User
from app.schemas.base import BaseResponse


router = APIRouter(prefix="/projects", tags=["projects"])


def get_project_service() -> ProjectService:
    """Dependency to get project service instance."""
    from app.dependencies import get_project_service as _get_project_service
    return _get_project_service()


async def get_optional_files(request: Request) -> Optional[List[UploadFile]]:
    """Dependency to handle optional file uploads safely."""
    try:
        form = await request.form()
        files_data = form.getlist("files")
        
        if not files_data:
            return None
            
        # Filter out non-UploadFile objects and empty files
        valid_files = []
        for file_data in files_data:
            if hasattr(file_data, 'filename') and file_data.filename and file_data.filename.strip():
                valid_files.append(file_data)
        
        return valid_files if valid_files else None
    except Exception:
        return None


@router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    request: Request,
    current_user: User = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service)
):
    """
    Create a new project with optional file uploads.
    
    Args:
        initial_idea: The initial product idea (required)
        project_name: Optional project name (auto-generated if not provided)
        files: Optional file uploads
        current_user: Current authenticated user
        project_service: Project service instance
    
    Returns:
        Created project details
    """
    try:
        # Parse form data manually
        form = await request.form()
        initial_idea = form.get("initial_idea")
        project_name = form.get("project_name")
        
        if not initial_idea:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="initial_idea is required"
            )
        
        # Create project request
        project_request = ProjectCreateRequest(
            project_name=project_name if project_name else None,
            initial_idea=initial_idea
        )
        
        # Get files safely
        files = await get_optional_files(request)
        
        # Create project
        project = await project_service.create_project(
            user_id=str(current_user.id),
            project_request=project_request,
            files=files
        )
        
        return project
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create project: {str(e)}"
        )


@router.get("/", status_code=status.HTTP_200_OK)
async def list_projects(
    status_filter: str = Query("active", alias="status"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("updated_at"),
    sort_order: str = Query("desc"),
    current_user: User = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service)
):
    """
    List user's projects with pagination and filtering.
    
    Query Parameters:
        status: Project status filter (active, archived, deleted)
        limit: Number of projects per page (1-100)
        offset: Number of projects to skip
        sort_by: Field to sort by (created_at, updated_at, project_name)
        sort_order: Sort order (asc, desc)
    """
    try:
        result = await project_service.list_projects(
            user_id=str(current_user.id),
            status=status_filter,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list projects: {str(e)}"
        )


@router.get("/stats", response_model=ProjectStatsResponse)
async def get_project_stats(
    current_user: User = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service)
):
    """Get user's project statistics."""
    try:
        stats = await project_service.get_project_stats(str(current_user.id))
        return stats
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get project stats: {str(e)}"
        )


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    current_user: User = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service)
):
    """Get project details by ID."""
    try:
        project = await project_service.get_project(
            project_id=project_id,
            user_id=str(current_user.id)
        )
        return project
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get project: {str(e)}"
        )


@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    update_request: ProjectUpdateRequest,
    current_user: User = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service)
):
    """Update project information."""
    try:
        project = await project_service.update_project(
            project_id=project_id,
            user_id=str(current_user.id),
            update_request=update_request
        )
        return project
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update project: {str(e)}"
        )


@router.delete("/{project_id}", status_code=status.HTTP_200_OK)
async def delete_project(
    project_id: str,
    current_user: User = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service)
):
    """Delete (soft delete) a project."""
    try:
        result = await project_service.delete_project(
            project_id=project_id,
            user_id=str(current_user.id)
        )
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete project: {str(e)}"
        )


# File management endpoints
@router.post("/{project_id}/uploads", status_code=status.HTTP_201_CREATED)
async def upload_files(
    project_id: str,
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service)
):
    """Upload files to a project."""
    try:
        uploads = await project_service.upload_files(
            project_id=project_id,
            user_id=str(current_user.id),
            files=files
        )
        
        return BaseResponse.success(
            data={"uploaded_files": uploads},
            message=f"Successfully uploaded {len(uploads)} file(s)"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload files: {str(e)}"
        )


@router.get("/{project_id}/uploads", status_code=status.HTTP_200_OK)
async def list_project_files(
    project_id: str,
    current_user: User = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service)
):
    """List files in a project."""
    try:
        files = await project_service.list_project_files(
            project_id=project_id,
            user_id=str(current_user.id)
        )
        
        return BaseResponse.success(
            data={"files": files},
            message="Files retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list project files: {str(e)}"
        )


@router.delete("/uploads/{file_id}", status_code=status.HTTP_200_OK)
async def delete_file(
    file_id: str,
    current_user: User = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service)
):
    """Delete a file from a project."""
    try:
        result = await project_service.delete_file(
            file_id=file_id,
            user_id=str(current_user.id)
        )
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete file: {str(e)}"
        )


# Version control endpoints (placeholder for AI integration)
@router.get("/{project_id}/versions", status_code=status.HTTP_200_OK)
async def list_project_versions(
    project_id: str,
    current_user: User = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service)
):
    """List project versions (placeholder for AI integration)."""
    return BaseResponse.success(
        data={
            "versions": [
                {
                    "version": "v1.0",
                    "timestamp": "2024-01-01T00:00:00Z",
                    "changes": "Initial project creation",
                    "prd_url": None,
                    "flowchart_url": None
                }
            ]
        },
        message="Version management will be available with AI agent integration"
    )


@router.post("/{project_id}/rollback", status_code=status.HTTP_200_OK)
async def rollback_project_version(
    project_id: str,
    rollback_request: VersionRollbackRequest,
    current_user: User = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service)
):
    """Rollback project to a specific version (placeholder for AI integration)."""
    try:
        result = await project_service.rollback_version(
            project_id=project_id,
            user_id=str(current_user.id),
            target_version=rollback_request.version
        )
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to rollback version: {str(e)}"
        )