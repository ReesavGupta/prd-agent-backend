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
from app.core.config import settings
from typing import Any
import hashlib
import logging
import io


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

            # Handle file uploads if provided (index synchronously to avoid races)
            if files:
                for file in files:
                    upload = await self._upload_file_to_project(
                        str(project.id),
                        user_id,
                        file
                    )
                    try:
                        await self._index_upload_blocking(upload)
                    except Exception as e:
                        # Mark the error and fail creation request
                        try:
                            await self.project_repository.uploads_collection.update_one(
                                {"_id": upload.id},
                                {"$set": {"index_error": str(e)}}
                            )
                        except Exception:
                            pass
                        raise HTTPException(status_code=500, detail=f"Indexing failed for file {upload.filename}: {e}")

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

        uploads: List[UploadResponse] = []
        for file in files:
            # Validate size and type (PDF-only, <= 5MB)
            file_content = await file.read()
            if len(file_content) > 5 * 1024 * 1024:
                raise HTTPException(status_code=413, detail=f"File {file.filename} exceeds maximum size of 5MB")
            if (file.content_type or "").lower() != "application/pdf":
                raise HTTPException(status_code=415, detail="Only PDF files are supported")

            # Dedup by file hash per project
            file_hash = self.project_repository.file_storage.generate_file_hash(file_content)
            try:
                logging.getLogger(__name__).info("[UPLOAD] received file: %s size=%d hash=%s", file.filename, len(file_content), file_hash[:12])
            except Exception:
                pass
            # Attempt to find existing upload with same hash for this project
            existing = await self.project_repository.uploads_collection.find_one({
                "project_id": ProjectRepository.ObjectId(project_id) if hasattr(ProjectRepository, 'ObjectId') else None,
            })
            # NOTE: We cannot access ObjectId from repository easily here; fallback to raw query:
            try:
                from bson import ObjectId
                existing = await self.project_repository.uploads_collection.find_one({
                    "project_id": ObjectId(project_id),
                    "file_hash": file_hash,
                })
            except Exception:
                existing = None
            if existing:
                # If existing upload is not indexed, index it now synchronously
                if not bool(existing.get("indexed")):
                    try:
                        await self._index_upload_blocking(Upload(**existing))
                        # refresh existing doc
                        existing = await self.project_repository.uploads_collection.find_one({"_id": existing["_id"]}) or existing
                    except Exception as e:
                        # surface error to caller to avoid returning unusable upload
                        raise HTTPException(status_code=500, detail=f"Indexing failed for existing file {existing.get('filename')}: {e}")
                uploads.append(await self._upload_to_response(Upload(**existing)))
                continue

            # Enforce page cap (30 pages) using PyMuPDF if available
            try:
                import fitz  # PyMuPDF
                with fitz.open(stream=file_content, filetype="pdf") as doc:
                    if doc.page_count > 30:
                        raise HTTPException(status_code=413, detail=f"PDF exceeds maximum page limit (30). Got {doc.page_count}")
            except HTTPException:
                raise
            except Exception:
                # If PyMuPDF not available or fails to read, proceed (Cloudinary upload will still succeed)
                pass

            # Rewind content for upload path
            file.file.seek(0)
            upload = await self.project_repository.upload_file(
                project_id=project_id,
                user_id=user_id,
                file_content=file_content,
                filename=file.filename or "unknown.pdf",
                content_type=file.content_type or "application/pdf",
            )
            # Index synchronously (blocking) to ensure immediate retrievability
            try:
                await self._index_upload_blocking(upload)
                # Re-fetch updated upload doc for accurate response
                try:
                    from bson import ObjectId
                    updated = await self.project_repository.uploads_collection.find_one({"_id": ObjectId(str(upload.id))})
                    if updated:
                        upload = Upload(**updated)
                except Exception:
                    pass
            except Exception as e:
                try:
                    await self.project_repository.uploads_collection.update_one(
                        {"_id": upload.id},
                        {"$set": {"index_error": str(e)}}
                    )
                except Exception:
                    pass
                raise HTTPException(status_code=500, detail=f"Indexing failed for file {upload.filename}: {e}")

            uploads.append(await self._upload_to_response(upload))

        return uploads

    async def _index_upload_blocking(self, upload: Upload) -> None:
        """Index an uploaded file into Pinecone synchronously (blocking).

        Supports PDFs (PyMuPDF), DOCX (docx2txt), Markdown/text (UTF-8 decode). Adds metadata
        including project_id, file_id, filename for single-file retrieval via metadata filter.
        """
        try:
            logger = logging.getLogger(__name__)
            logger.info("[INDEX] start upload_id=%s project_id=%s filename=%s", str(upload.id), str(upload.project_id), upload.filename)
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_core.documents import Document
            import httpx

            # Fetch bytes using a signed URL when available (avoids 401 on protected assets)
            try:
                signed_url = await self.project_repository.file_storage.get_file_url(upload.storage_key, expires_in=300)
            except Exception:
                signed_url = upload.url or ""
            if not signed_url:
                raise RuntimeError("missing_file_url")
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.get(signed_url)
                r.raise_for_status()
                file_bytes = r.content

            # Convert to text
            text: str = ""
            name_lower = (upload.filename or "").lower()
            ctype = (upload.content_type or "").lower()
            if ctype == "application/pdf" or name_lower.endswith(".pdf"):
                try:
                    import fitz  # PyMuPDF
                except Exception:
                    logger.exception("[INDEX] PyMuPDF import failed")
                    raise
                try:
                    doc = fitz.open(stream=file_bytes, filetype="pdf")
                    pages: list[str] = []
                    for pno in range(doc.page_count):
                        page = doc.load_page(pno)
                        pages.append(page.get_text("text") or "")
                    doc.close()
                    text = "\n\n".join(pages)
                except Exception:
                    logger.exception("[INDEX] PDF parse failed upload_id=%s", str(upload.id))
                    raise
            elif name_lower.endswith(".docx") or ctype in (
                "application/msword",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ):
                try:
                    import docx2txt
                    text = docx2txt.process(io.BytesIO(file_bytes)) or ""
                except Exception:
                    logger.exception("[INDEX] DOCX parse failed upload_id=%s", str(upload.id))
                    raise
            else:
                try:
                    text = file_bytes.decode("utf-8", errors="ignore")
                except Exception:
                    logger.exception("[INDEX] text decode failed upload_id=%s", str(upload.id))
                    raise

            text = (text or "").strip()
            if not text:
                raise RuntimeError("empty_text_after_parse")

            # Split
            splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150, add_start_index=True)
            chunks = splitter.split_documents([Document(page_content=text, metadata={"filename": upload.filename})])

            # Enrich metadata
            for d in chunks:
                meta = d.metadata or {}
                meta.update({
                    "project_id": str(upload.project_id),
                    "file_id": str(upload.id),
                    "filename": upload.filename,
                })
                d.metadata = meta

            # Vector store via rag_service (consistent embeddings)
            try:
                from app.services.rag_service import rag_service
                store = rag_service._vector(namespace=str(upload.project_id))
            except Exception:
                logger.exception("[INDEX] vector store init failed")
                raise

            await store.aadd_documents(chunks)
            logger.info("[INDEX] upserted chunks=%d upload_id=%s", len(chunks), str(upload.id))

            # Mark as indexed
            try:
                embed_model = "nomic" if rag_service._use_nomic() else "openai"
            except Exception:
                embed_model = None
            await self.project_repository.uploads_collection.update_one(
                {"_id": upload.id},
                {"$set": {"indexed": True, "num_chunks": len(chunks), "embed_model": embed_model}}
            )
            logger.info("[INDEX] completed upload_id=%s", str(upload.id))
            # Broadcast over chat WS if project has a source_chat_id
            try:
                from bson import ObjectId
                db = self.project_repository.database
                proj = await db.projects.find_one({"_id": ObjectId(str(upload.project_id))})
                chat_id = None
                if proj:
                    scid = proj.get("source_chat_id")
                    if scid is not None:
                        chat_id = str(scid)
                if chat_id:
                    from app.websocket.publisher import publish_to_chat
                    await publish_to_chat(chat_id, {
                        "type": "file_indexed",
                        "data": {
                            "project_id": str(upload.project_id),
                            "file_id": str(upload.id),
                            "filename": upload.filename,
                            "num_chunks": len(chunks),
                        }
                    })
                    logger.info("[INDEX] notified chat_id=%s about file_indexed", chat_id)
                else:
                    logger.info("[INDEX] no source_chat_id found for project_id=%s; skipping WS notify", str(upload.project_id))
            except Exception:
                logger.exception("[INDEX] failed to publish file_indexed event")
        except Exception as e:
            try:
                logging.getLogger(__name__).exception("[INDEX] failed upload_id=%s error=%s", str(getattr(upload, 'id', '')), str(e))
                await self.project_repository.uploads_collection.update_one(
                    {"_id": upload.id},
                    {"$set": {"index_error": str(e)}}
                )
            except Exception:
                pass
            # Propagate error
            raise

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
            indexed=getattr(upload, "indexed", False),
            index_error=getattr(upload, "index_error", None),
            num_chunks=getattr(upload, "num_chunks", 0),
            uploaded_at=upload.uploaded_at,
        )