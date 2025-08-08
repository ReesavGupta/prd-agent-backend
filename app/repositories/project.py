"""
Project repository implementation using MongoDB.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING
from app.models.project import Project, Upload, ProjectVersion
from app.models.user import PyObjectId
from app.repositories.base import FileStorageRepository, CacheRepository


class ProjectRepository:
    """Repository for Project operations."""

    def __init__(
        self, 
        database: AsyncIOMotorDatabase,
        file_storage: FileStorageRepository,
        cache: CacheRepository
    ):
        self.database = database
        self.file_storage = file_storage
        self.cache = cache
        self.collection = database.projects
        self.uploads_collection = database.uploads

    async def create_indexes(self):
        """Create database indexes for optimal query performance."""
        # Project collection indexes
        await self.collection.create_index([("user_id", ASCENDING), ("status", ASCENDING)])
        await self.collection.create_index([("user_id", ASCENDING), ("updated_at", DESCENDING)])
        await self.collection.create_index([("created_at", DESCENDING)])
        await self.collection.create_index([("storage_path", ASCENDING)])
        
        # Upload collection indexes
        await self.uploads_collection.create_index([("user_id", ASCENDING), ("project_id", ASCENDING)])
        await self.uploads_collection.create_index([("file_hash", ASCENDING)])
        await self.uploads_collection.create_index([("uploaded_at", DESCENDING)])

    async def create_project(self, project_data: Dict) -> Project:
        """Create a new project."""
        # Generate storage path
        user_id = str(project_data["user_id"])
        project_id = str(ObjectId())
        storage_path = f"projects/{user_id}/{project_id}"
        
        # Create folder structure in storage
        await self.file_storage.create_folder_structure(project_id, user_id)
        
        # Prepare project document
        project_doc = {
            "_id": ObjectId(project_id),
            "user_id": ObjectId(project_data["user_id"]),
            "project_name": project_data.get("project_name", "Untitled Project"),
            "initial_idea": project_data["initial_idea"],
            "status": "active",
            "source_chat_id": project_data.get("source_chat_id"),
            "created_from_chat": project_data.get("created_from_chat", False),
            "current_version": "v1.0",
            "storage_path": storage_path,
            "metadata": {
                "thinking_lens_status": {
                    "discovery": False,
                    "user_journey": False,
                    "metrics": False,
                    "gtm": False,
                    "risks": False
                },
                "last_agent_run": None,
                "total_iterations": 0,
                "file_count": 0,
                "storage_size_bytes": 0
            },
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # Insert project
        await self.collection.insert_one(project_doc)
        
        # Clear user's project list cache
        await self.cache.invalidate_pattern(f"project_list:{user_id}:*")
        
        # Return project object
        return Project(**project_doc)

    async def get_project(self, project_id: str, user_id: str) -> Optional[Project]:
        """Get project by ID and user ID."""
        # Try cache first
        cache_key = self.cache.project_cache_key(user_id, project_id)
        cached_project = await self.cache.get(cache_key)
        if cached_project:
            return Project(**cached_project)
        
        # Query database
        project_doc = await self.collection.find_one({
            "_id": ObjectId(project_id),
            "user_id": ObjectId(user_id)
        })
        
        if not project_doc:
            return None
        
        project = Project(**project_doc)
        
        # Cache the result
        await self.cache.set(
            cache_key, 
            project.model_dump(mode='json'), 
            ttl=300  # 5 minutes
        )
        
        return project

    async def update_project(
        self, 
        project_id: str, 
        user_id: str, 
        update_data: Dict
    ) -> Optional[Project]:
        """Update project data."""
        update_data["updated_at"] = datetime.utcnow()
        
        result = await self.collection.find_one_and_update(
            {"_id": ObjectId(project_id), "user_id": ObjectId(user_id)},
            {"$set": update_data},
            return_document=True
        )
        
        if not result:
            return None
        
        project = Project(**result)
        
        # Update cache
        cache_key = self.cache.project_cache_key(user_id, project_id)
        await self.cache.set(
            cache_key,
            project.model_dump(mode='json'),
            ttl=300
        )
        
        # Invalidate list cache
        await self.cache.invalidate_pattern(f"project_list:{user_id}:*")
        
        return project

    async def delete_project(self, project_id: str, user_id: str) -> bool:
        """Soft delete project (mark as deleted)."""
        result = await self.collection.update_one(
            {"_id": ObjectId(project_id), "user_id": ObjectId(user_id)},
            {
                "$set": {
                    "status": "deleted",
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        if result.modified_count > 0:
            # Clear caches
            await self.cache.delete(self.cache.project_cache_key(user_id, project_id))
            await self.cache.invalidate_pattern(f"project_list:{user_id}:*")
            return True
        
        return False

    async def list_projects(
        self,
        user_id: str,
        status: str = "active",
        limit: int = 20,
        offset: int = 0,
        sort_by: str = "updated_at",
        sort_order: str = "desc"
    ) -> Tuple[List[Project], int]:
        """List projects with pagination."""
        # Try cache first
        cache_key = self.cache.project_list_cache_key(user_id, status, offset, limit)
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            projects_data, total = cached_result["projects"], cached_result["total"]
            projects = [Project(**p) for p in projects_data]
            return projects, total
        
        # Build query
        query = {
            "user_id": ObjectId(user_id),
            "status": status
        }
        
        # Build sort
        sort_direction = DESCENDING if sort_order == "desc" else ASCENDING
        sort_spec = [(sort_by, sort_direction)]
        
        # Get total count
        total = await self.collection.count_documents(query)
        
        # Get projects
        cursor = self.collection.find(query).sort(sort_spec).skip(offset).limit(limit)
        project_docs = await cursor.to_list(length=limit)
        
        projects = [Project(**doc) for doc in project_docs]
        
        # Cache results for 2 minutes
        cache_data = {
            "projects": [p.model_dump(mode='json') for p in projects],
            "total": total
        }
        await self.cache.set(cache_key, cache_data, ttl=120)
        
        return projects, total

    async def get_project_stats(self, user_id: str) -> Dict:
        """Get user's project statistics."""
        # Try cache first
        cache_key = self.cache.user_stats_cache_key(user_id)
        cached_stats = await self.cache.get(cache_key)
        if cached_stats:
            return cached_stats
        
        # Aggregate statistics
        pipeline = [
            {"$match": {"user_id": ObjectId(user_id)}},
            {
                "$group": {
                    "_id": "$status",
                    "count": {"$sum": 1},
                    "total_files": {"$sum": "$metadata.file_count"},
                    "total_storage": {"$sum": "$metadata.storage_size_bytes"}
                }
            }
        ]
        
        results = await self.collection.aggregate(pipeline).to_list(None)
        
        # Process results
        stats = {
            "total_projects": 0,
            "active_projects": 0,
            "archived_projects": 0,
            "total_files": 0,
            "total_storage_bytes": 0,
            "recent_activity_count": 0
        }
        
        for result in results:
            status = result["_id"]
            count = result["count"]
            
            stats["total_projects"] += count
            if status == "active":
                stats["active_projects"] = count
            elif status == "archived":
                stats["archived_projects"] = count
                
            stats["total_files"] += result["total_files"]
            stats["total_storage_bytes"] += result["total_storage"]
        
        # Get recent activity (projects updated in last 7 days)
        week_ago = datetime.utcnow().replace(day=datetime.utcnow().day - 7)
        stats["recent_activity_count"] = await self.collection.count_documents({
            "user_id": ObjectId(user_id),
            "updated_at": {"$gte": week_ago}
        })
        
        # Cache for 5 minutes
        await self.cache.set(cache_key, stats, ttl=300)
        
        return stats

    # File upload methods
    async def upload_file(
        self,
        project_id: str,
        user_id: str,
        file_content: bytes,
        filename: str,
        content_type: str
    ) -> Upload:
        """Upload file to project."""
        # Get project to verify ownership
        project = await self.get_project(project_id, user_id)
        if not project:
            raise ValueError("Project not found")
        
        # Generate file hash
        file_hash = self.file_storage.generate_file_hash(file_content)
        
        # Upload to storage
        upload_result = await self.file_storage.upload_file(
            file_content,
            filename,
            content_type,
            f"{project.storage_path}/uploads"
        )
        
        # Create upload record
        upload_doc = {
            "user_id": ObjectId(user_id),
            "project_id": ObjectId(project_id),
            "filename": filename.split("/")[-1],  # Remove path if present
            "original_filename": filename,
            "file_size": len(file_content),
            "content_type": content_type,
            "storage_key": upload_result["storage_key"],
            "url": upload_result["url"],
            "file_hash": file_hash,
            "uploaded_at": datetime.utcnow()
        }
        
        insert_result = await self.uploads_collection.insert_one(upload_doc)
        upload_doc["_id"] = insert_result.inserted_id
        
        # Update project metadata
        await self.collection.update_one(
            {"_id": ObjectId(project_id)},
            {
                "$inc": {
                    "metadata.file_count": 1,
                    "metadata.storage_size_bytes": len(file_content)
                },
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        
        # Clear caches
        await self.cache.delete(self.cache.project_cache_key(user_id, project_id))
        await self.cache.invalidate_pattern(f"project_list:{user_id}:*")
        
        return Upload(**upload_doc)

    async def list_project_files(
        self,
        project_id: str,
        user_id: str
    ) -> List[Upload]:
        """List files in a project."""
        # Verify project ownership
        project = await self.get_project(project_id, user_id)
        if not project:
            return []
        
        cursor = self.uploads_collection.find({
            "project_id": ObjectId(project_id),
            "user_id": ObjectId(user_id)
        }).sort("uploaded_at", DESCENDING)
        
        upload_docs = await cursor.to_list(length=None)
        return [Upload(**doc) for doc in upload_docs]

    async def delete_file(self, file_id: str, user_id: str) -> bool:
        """Delete a file."""
        upload_doc = await self.uploads_collection.find_one({
            "_id": ObjectId(file_id),
            "user_id": ObjectId(user_id)
        })
        
        if not upload_doc:
            return False
        
        # Delete from storage
        await self.file_storage.delete_file(upload_doc["storage_key"])
        
        # Delete from database
        await self.uploads_collection.delete_one({"_id": ObjectId(file_id)})
        
        # Update project metadata
        await self.collection.update_one(
            {"_id": upload_doc["project_id"]},
            {
                "$inc": {
                    "metadata.file_count": -1,
                    "metadata.storage_size_bytes": -upload_doc["file_size"]
                },
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        
        # Clear caches
        project_id = str(upload_doc["project_id"])
        await self.cache.delete(self.cache.project_cache_key(user_id, project_id))
        await self.cache.invalidate_pattern(f"project_list:{user_id}:*")
        
        return True