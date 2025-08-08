"""
Chat API endpoints for managing chats and messages.
"""

import logging
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse

from app.schemas.chat import (
    ChatCreateRequest,
    ChatUpdateRequest,
    ChatResponse,
    ChatListResponse,
    ChatConvertToProjectRequest,
    ChatConvertToProjectResponse,
    MessageCreateRequest,
    MessageResponse,
    MessageListResponse,
    MessageAttachmentResponse,
    AIMessageResponse,
    ChatSearchRequest,
    ChatSearchResponse,
    ChatMetadataResponse
)
from app.schemas.base import BaseResponse
from app.services.chat_service import chat_service
from app.middleware.auth import get_current_user
from app.models.user import UserInDB as User, PyObjectId

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chats", tags=["chats"])


def _convert_chat_metadata(metadata):
    """Convert ChatMetadata to ChatMetadataResponse."""
    return ChatMetadataResponse(
        message_count=metadata.message_count,
        has_project=metadata.has_project,
        project_id=str(metadata.project_id) if metadata.project_id else None,
        last_ai_model=metadata.last_ai_model,
        total_tokens_used=metadata.total_tokens_used,
        avg_response_time=metadata.avg_response_time,
        conversation_quality_score=metadata.conversation_quality_score
    )


async def get_optional_files(request: Request) -> List[UploadFile]:
    """Handle optional file uploads."""
    try:
        form = await request.form()
        files = form.getlist("files")
        
        valid_files = []
        for file in files:
            if isinstance(file, UploadFile) and file.filename and file.filename.strip():
                valid_files.append(file)
        
        return valid_files
    except Exception:
        return []


@router.get(
    "",
    response_model=dict,
    summary="List user chats",
    description="Get a paginated list of chats for the authenticated user"
)
async def list_chats(
    status_filter: str = Query("active", alias="status", description="Filter by chat status"),
    limit: int = Query(20, ge=1, le=100, description="Number of chats per page"),
    offset: int = Query(0, ge=0, description="Number of chats to skip"),
    current_user: User = Depends(get_current_user)
):
    """List user chats with pagination."""
    try:
        chats, total = await chat_service.list_user_chats(
            user_id=current_user.id,
            status=status_filter,
            limit=limit,
            offset=offset
        )
        
        # Convert to response format
        chat_responses = []
        for chat in chats:
            chat_response = ChatResponse(
                id=str(chat.id),
                chat_name=chat.chat_name,
                created_at=chat.created_at,
                updated_at=chat.updated_at,
                last_message_at=chat.last_message_at,
                status=chat.status,
                metadata=_convert_chat_metadata(chat.metadata),
                is_auto_named=chat.is_auto_named
            )
            chat_responses.append(chat_response)
        
        return BaseResponse.paginated(
            items=chat_responses,
            total=total,
            limit=limit,
            offset=offset,
            message="Chats retrieved successfully"
        )
    except Exception as e:
        logger.error(f"Error listing chats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve chats"
        )


@router.post(
    "",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new chat",
    description="Create a new chat session"
)
async def create_chat(
    request: ChatCreateRequest,
    current_user: User = Depends(get_current_user)
):
    """Create a new chat."""
    try:
        chat = await chat_service.create_chat(current_user.id, request)
        
        chat_response = ChatResponse(
            id=str(chat.id),
            chat_name=chat.chat_name,
            created_at=chat.created_at,
            updated_at=chat.updated_at,
            last_message_at=chat.last_message_at,
            status=chat.status,
            metadata=_convert_chat_metadata(chat.metadata),
            is_auto_named=chat.is_auto_named
        )
        
        return BaseResponse.success(
            data=chat_response,
            message="Chat created successfully"
        )
    except ValueError as e:
        logger.error(f"Validation error creating chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create chat"
        )


@router.get(
    "/{chat_id}",
    response_model=dict,
    summary="Get chat details",
    description="Get details of a specific chat"
)
async def get_chat(
    chat_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get chat by ID."""
    try:
        chat_obj_id = PyObjectId(chat_id)
        chat = await chat_service.get_chat(chat_obj_id, current_user.id)
        
        if not chat:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat not found"
            )
        
        chat_response = ChatResponse(
            id=str(chat.id),
            chat_name=chat.chat_name,
            created_at=chat.created_at,
            updated_at=chat.updated_at,
            last_message_at=chat.last_message_at,
            status=chat.status,
            metadata=_convert_chat_metadata(chat.metadata),
            is_auto_named=chat.is_auto_named
        )
        
        return BaseResponse.success(
            data=chat_response,
            message="Chat retrieved successfully"
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid chat ID"
        )
    except Exception as e:
        logger.error(f"Error getting chat {chat_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve chat"
        )


@router.put(
    "/{chat_id}",
    response_model=dict,
    summary="Update chat",
    description="Update chat details like name"
)
async def update_chat(
    chat_id: str,
    request: ChatUpdateRequest,
    current_user: User = Depends(get_current_user)
):
    """Update chat."""
    try:
        chat_obj_id = PyObjectId(chat_id)
        success = await chat_service.update_chat(chat_obj_id, current_user.id, request)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat not found"
            )
        
        return BaseResponse.success(
            data={"chat_id": chat_id},
            message="Chat updated successfully"
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid chat ID"
        )
    except Exception as e:
        logger.error(f"Error updating chat {chat_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update chat"
        )


@router.delete(
    "/{chat_id}",
    response_model=dict,
    summary="Delete chat",
    description="Delete a chat (soft delete)"
)
async def delete_chat(
    chat_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete chat."""
    try:
        chat_obj_id = PyObjectId(chat_id)
        success = await chat_service.delete_chat(chat_obj_id, current_user.id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat not found"
            )
        
        return BaseResponse.success(
            data={"chat_id": chat_id},
            message="Chat deleted successfully"
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid chat ID"
        )
    except Exception as e:
        logger.error(f"Error deleting chat {chat_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete chat"
        )


@router.get(
    "/{chat_id}/messages",
    response_model=dict,
    summary="Get chat messages",
    description="Get paginated messages for a chat"
)
async def get_chat_messages(
    chat_id: str,
    limit: int = Query(50, ge=1, le=100, description="Number of messages per page"),
    before: Optional[str] = Query(None, description="ISO timestamp to get messages before"),
    current_user: User = Depends(get_current_user)
):
    """Get messages for a chat."""
    try:
        chat_obj_id = PyObjectId(chat_id)
        
        # Parse before timestamp if provided
        before_dt = None
        if before:
            try:
                before_dt = datetime.fromisoformat(before.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid before timestamp format"
                )
        
        messages, has_more = await chat_service.get_chat_messages(
            chat_id=chat_obj_id,
            user_id=current_user.id,
            limit=limit,
            before=before_dt
        )
        
        # Convert to response format
        message_responses = []
        for message in messages:
            # Convert attachments
            attachment_responses = []
            for attachment in message.attachments:
                attachment_response = MessageAttachmentResponse(
                    file_id=str(attachment.file_id),
                    filename=attachment.filename,
                    storage_key=attachment.storage_key,
                    url=attachment.url,
                    content_type=attachment.content_type,
                    file_size=attachment.file_size
                )
                attachment_responses.append(attachment_response)
            
            message_response = MessageResponse(
                id=str(message.id),
                message_type=message.message_type,
                content=message.content,
                timestamp=message.timestamp,
                metadata=message.metadata,
                attachments=attachment_responses,
                reply_to=str(message.reply_to) if message.reply_to else None,
                thread_id=message.thread_id,
                is_edited=message.is_edited,
                edited_at=message.edited_at
            )
            message_responses.append(message_response)
        
        # Calculate next_before for pagination
        next_before = None
        if has_more and messages:
            next_before = messages[0].timestamp  # First message timestamp
        
        response_data = {
            "messages": message_responses,
            "has_more": has_more,
            "next_before": next_before.isoformat() if next_before else None
        }
        
        return BaseResponse.success(
            data=response_data,
            message="Messages retrieved successfully"
        )
    except ValueError as e:
        if "Chat not found" in str(e):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat not found or access denied"
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid chat ID"
        )
    except Exception as e:
        logger.error(f"Error getting messages for chat {chat_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve messages"
        )


@router.post(
    "/{chat_id}/messages",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
    summary="Send message",
    description="Send a message in a chat with optional file attachments and get AI response"
)
async def send_message(
    request: Request,
    chat_id: str,
    current_user: User = Depends(get_current_user)
):
    """Send a message in a chat."""
    try:
        chat_obj_id = PyObjectId(chat_id)
        
        # Get parameters from query string or form data
        query_params = request.query_params
        content = query_params.get("content")
        if not content:
            # Try to get from form data
            try:
                form = await request.form()
                content = form.get("content")
            except:
                pass
        
        if not content:
            raise ValueError("Content is required")
        
        message_type = query_params.get("message_type", "user")
        reply_to = query_params.get("reply_to")
        thread_id = query_params.get("thread_id")
        
        # Get files
        files = await get_optional_files(request)
        
        # Create message request
        message_request = MessageCreateRequest(
            content=content,
            message_type=message_type,
            reply_to=reply_to,
            thread_id=thread_id
        )
        
        user_message, ai_message = await chat_service.send_message_with_files(
            chat_id=chat_obj_id,
            user_id=current_user.id,
            request=message_request,
            files=files
        )
        
        # Convert attachments to response format
        attachment_responses = []
        for attachment in user_message.attachments:
            attachment_response = MessageAttachmentResponse(
                file_id=str(attachment.file_id),
                filename=attachment.filename,
                storage_key=attachment.storage_key,
                url=attachment.url,
                content_type=attachment.content_type,
                file_size=attachment.file_size
            )
            attachment_responses.append(attachment_response)
        
        # Convert user message to response format
        user_message_response = MessageResponse(
            id=str(user_message.id),
            message_type=user_message.message_type,
            content=user_message.content,
            timestamp=user_message.timestamp,
            metadata=user_message.metadata,
            attachments=attachment_responses,
            reply_to=str(user_message.reply_to) if user_message.reply_to else None,
            thread_id=user_message.thread_id,
            is_edited=user_message.is_edited,
            edited_at=user_message.edited_at
        )
        
        response_data = {
            "user_message": user_message_response,
            "ai_response": None
        }
        
        # Add AI response if available
        if ai_message:
            ai_message_response = MessageResponse(
                id=str(ai_message.id),
                message_type=ai_message.message_type,
                content=ai_message.content,
                timestamp=ai_message.timestamp,
                metadata=ai_message.metadata,
                attachments=[],
                reply_to=str(ai_message.reply_to) if ai_message.reply_to else None,
                thread_id=ai_message.thread_id,
                is_edited=ai_message.is_edited,
                edited_at=ai_message.edited_at
            )
            
            # TODO: Generate suggested actions
            suggested_actions = []
            
            response_data["ai_response"] = {
                "message": ai_message_response,
                "suggested_actions": suggested_actions
            }
        
        return BaseResponse.success(
            data=response_data,
            message="Message sent successfully"
        )
    except ValueError as e:
        if "Chat not found" in str(e):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat not found or access denied"
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error sending message in chat {chat_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send message"
        )


@router.post(
    "/{chat_id}/convert-to-project",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
    summary="Convert chat to project",
    description="Convert a chat conversation into a PRD project"
)
async def convert_chat_to_project(
    chat_id: str,
    request: ChatConvertToProjectRequest,
    current_user: User = Depends(get_current_user)
):
    """Convert chat to project."""
    try:
        chat_obj_id = PyObjectId(chat_id)
        
        project_id = await chat_service.convert_chat_to_project(
            chat_id=chat_obj_id,
            user_id=current_user.id,
            request=request
        )
        
        response_data = ChatConvertToProjectResponse(
            project_id=project_id,
            message="Chat converted to project successfully"
        )
        
        return BaseResponse.success(
            data=response_data,
            message="Chat converted to project successfully"
        )
    except ValueError as e:
        if "Chat not found" in str(e):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat not found or access denied"
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error converting chat {chat_id} to project: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to convert chat to project"
        )


@router.post(
    "/search",
    response_model=dict,
    summary="Search messages",
    description="Search messages across all user chats"
)
async def search_messages(
    request: ChatSearchRequest,
    current_user: User = Depends(get_current_user)
):
    """Search messages for a user."""
    try:
        messages = await chat_service.search_messages(
            user_id=current_user.id,
            query=request.query,
            limit=request.limit
        )
        
        # Convert to response format
        message_responses = []
        for message in messages:
            message_response = MessageResponse(
                id=str(message.id),
                message_type=message.message_type,
                content=message.content,
                timestamp=message.timestamp,
                metadata=message.metadata,
                attachments=[],
                reply_to=str(message.reply_to) if message.reply_to else None,
                thread_id=message.thread_id,
                is_edited=message.is_edited,
                edited_at=message.edited_at
            )
            message_responses.append(message_response)
        
        response_data = ChatSearchResponse(
            messages=message_responses,
            total_found=len(message_responses),
            query=request.query
        )
        
        return BaseResponse.success(
            data=response_data,
            message="Search completed successfully"
        )
    except Exception as e:
        logger.error(f"Error searching messages: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search messages"
        )