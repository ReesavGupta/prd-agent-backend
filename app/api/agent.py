"""
Agent-specific API endpoints as defined in backend/docs/agent.md.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import json

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status

from app.middleware.auth import get_current_user
from app.models.user import UserInDB as User
from app.schemas.base import BaseResponse
from app.schemas.project import ProjectCreateRequest
from app.schemas.chat import ChatCreateRequest
from app.services.project_service import ProjectService
from app.services.chat_service import chat_service
from app.services.ai_service import ai_service
from app.dependencies import get_project_service
from app.agent.runtime import resume_run
from app.core.config import settings
from app.services.cache_service import cache_service


router = APIRouter(prefix="/agent", tags=["agent"])


def _normalize_files(form_files: List[UploadFile]) -> List[UploadFile]:
    valid: List[UploadFile] = []
    for f in form_files:
        if isinstance(f, UploadFile) and f.filename and f.filename.strip():
            valid.append(f)
    return valid


# Simple in-memory idempotency and rate-limit store for HTTP resume (process-local)
_last_resume_key_by_chat: Dict[str, str] = {}
_last_resume_time_by_chat: Dict[str, datetime] = {}


@router.post("/chats/{chat_id}/resume", status_code=status.HTTP_202_ACCEPTED)
async def http_resume_agent(
    chat_id: str,
    payload: Dict[str, Any],
    current_user: User = Depends(get_current_user),
):
    """Resume an interrupted agent thread over HTTP (alternative to WS).

    Accepts two shapes (one required):
    - { "answer": { "question_id": str, "text": str } }
    - { "accept": { "question_id": str, "finish": bool } }
    Provides simple process-local idempotency and rate-limiting per chat.
    """
    try:
        # Basic auth: ensure user owns the chat (reuse get_chat)
        await chat_service.get_chat(PyObjectId(chat_id), current_user.id)

        body = payload or {}
        if not ("answer" in body or "accept" in body):
            raise HTTPException(status_code=422, detail="Expected 'answer' or 'accept'")

        # Simple idempotency: build a resume key
        if "answer" in body:
            ans = body.get("answer") or {}
            qid = str(ans.get("question_id") or "")
            if not qid:
                raise HTTPException(status_code=422, detail="answer.question_id is required")
            resume_key = f"answer:{qid}:{str(ans.get('text') or '')[:24]}"
        else:
            acc = body.get("accept") or {}
            qid = str(acc.get("question_id") or "")
            if not qid:
                raise HTTPException(status_code=422, detail="accept.question_id is required")
            resume_key = f"accept:{qid}:{'finish' if bool(acc.get('finish')) else 'skip'}"

        # Rate-limit and idempotency window: 5 seconds
        now = datetime.utcnow()
        last_key = _last_resume_key_by_chat.get(chat_id)
        last_time = _last_resume_time_by_chat.get(chat_id)
        if last_key == resume_key and last_time and (now - last_time) < timedelta(seconds=5):
            return BaseResponse.success(message="Duplicate resume suppressed")

        # Perform resume
        if "answer" in body:
            await resume_run(chat_id, {"type": "answer", "question_id": qid, "text": str(ans.get("text") or "")})
        else:
            await resume_run(chat_id, {"type": "accept", "question_id": qid, "finish": bool(acc.get("finish"))})

        _last_resume_key_by_chat[chat_id] = resume_key
        _last_resume_time_by_chat[chat_id] = now

        return BaseResponse.success(message="Resume accepted")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to resume agent: {str(e)}")


@router.post("/ingest-idea", status_code=status.HTTP_201_CREATED)
async def ingest_idea(
    request: Request,
    current_user: User = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service),
):
    """
    Create a new project and a chat from a one-liner idea. Accepts JSON or multipart.

    Returns: { project_id, chat_id, message }
    """
    try:
        idea: Optional[str] = None
        files: Optional[List[UploadFile]] = None
        metadata: Optional[Dict[str, Any]] = None

        content_type = request.headers.get("content-type", "").lower()
        if content_type.startswith("application/json"):
            body = await request.json()
            idea = (body or {}).get("idea")
            metadata = (body or {}).get("metadata")
        else:
            # multipart/form-data
            form = await request.form()
            idea = form.get("idea") or form.get("initial_idea")
            files = _normalize_files(form.getlist("files")) if "files" in form else None

        if not idea or not str(idea).strip():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="idea is required")

        # Create a chat first for workspace
        chat = await chat_service.create_chat(current_user.id, ChatCreateRequest(chat_name=None))

        # Create the project, linking to the chat
        project_req = ProjectCreateRequest(project_name=None, initial_idea=idea.strip())
        project = await project_service.create_project(
            user_id=str(current_user.id),
            project_request=project_req,
            files=files,
            created_from_chat=True,
            source_chat_id=str(chat.id),
        )

        return BaseResponse.success(
            data={
                "project_id": project.id,
                "chat_id": str(chat.id),
            },
            message="Idea ingested; chat and project created",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest idea: {str(e)}")


@router.post("/projects/{project_id}/clarifications", status_code=status.HTTP_200_OK)
async def get_clarifications(
    project_id: str,
    payload: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service),
):
    """
    Generate clarifying questions based on the initial idea.

    Expects: { "initial_idea": str, "num_questions": int }
    Returns: { "questions": [str, ...] }
    """
    try:
        # Verify project access
        await project_service.get_project(project_id, str(current_user.id))

        initial_idea = (payload or {}).get("initial_idea")
        num_questions = int((payload or {}).get("num_questions") or 5)
        if not initial_idea:
            raise HTTPException(status_code=422, detail="initial_idea is required")

        # Instruct the model to return strict JSON with 3 questions per lens
        system_msg = {
            "role": "system",
            "content": (
                "You are an expert product manager. Given an idea, generate concise clarifying questions "
                "for each thinking lens: discovery, user_journey, metrics, gtm, risks. "
                "Constraints: exactly 3 questions per lens; one sentence each; no numbering or bullet markers; "
                "return ONLY valid minified JSON with this shape: "
                "{\"discovery\":[\"...\",\"...\",\"...\"],\"user_journey\":[\"...\",\"...\",\"...\"],\"metrics\":[\"...\",\"...\",\"...\"],\"gtm\":[\"...\",\"...\",\"...\"],\"risks\":[\"...\",\"...\",\"...\"]}."
            ),
        }
        user_msg = {
            "role": "user",
            "content": (
                f"Idea: {initial_idea}\n"
                "Output only the JSON object with arrays for each lens as specified."
            ),
        }

        try:
            response = await ai_service.generate_response(
                user_id=str(current_user.id),
                messages=[system_msg, user_msg],
                temperature=0.1,
                use_cache=False,
            )
            raw = response.content.strip()
            parsed: Dict[str, List[str]] = json.loads(raw)
            # Normalize and flatten into desired order
            order = ["discovery", "user_journey", "metrics", "gtm", "risks"]
            by_lens: Dict[str, List[str]] = {}
            questions: List[str] = []
            for key in order:
                arr = parsed.get(key, []) if isinstance(parsed, dict) else []
                # Clean and cap to 3
                cleaned = [str(q).strip().rstrip("? ") + ("?" if not str(q).strip().endswith("?") else "") for q in arr]
                cleaned = [q for q in cleaned if q]
                by_lens[key] = cleaned[:3]
                questions.extend(by_lens[key])
            # Ensure exactly 15 items if possible
            questions = [q for q in questions if q]
            if len(questions) < 15:
                # Fill with safe defaults if the model under-delivered
                defaults = {
                    "discovery": [
                        "Who is the primary target user?",
                        "What core problem are we solving?",
                        "Which alternatives do users have now?",
                    ],
                    "user_journey": [
                        "What is the main happy-path user flow?",
                        "What entry points lead users to the product?",
                        "What key edge cases should we handle?",
                    ],
                    "metrics": [
                        "What success metrics define value delivery?",
                        "What guardrail metrics must not degrade?",
                        "What timeframe is expected to see impact?",
                    ],
                    "gtm": [
                        "Who is the buyer persona and decision-maker?",
                        "Which channels will we use to reach users?",
                        "What initial pricing and packaging do we expect?",
                    ],
                    "risks": [
                        "What are the top risks or assumptions?",
                        "What dependencies could block progress?",
                        "What mitigations will we prepare for key risks?",
                    ],
                }
                fallback: List[str] = []
                for k in order:
                    need = 3 - len(by_lens.get(k, []))
                    if need > 0:
                        by_lens[k] = (by_lens.get(k, []) + defaults[k])[:3]
                    fallback.extend(defaults[k])
                # Only append until we reach 15
                needed = 15 - len(questions)
                questions.extend(fallback[:needed])
        except Exception:
            # On any error, return defaults
            questions = []
            by_lens = {}
            defaults = {
                "discovery": [
                    "Who is the primary target user?",
                    "What core problem are we solving?",
                    "Which alternatives do users have now?",
                ],
                "user_journey": [
                    "What is the main happy-path user flow?",
                    "What entry points lead users to the product?",
                    "What key edge cases should we handle?",
                ],
                "metrics": [
                    "What success metrics define value delivery?",
                    "What guardrail metrics must not degrade?",
                    "What timeframe is expected to see impact?",
                ],
                "gtm": [
                    "Who is the buyer persona and decision-maker?",
                    "Which channels will we use to reach users?",
                    "What initial pricing and packaging do we expect?",
                ],
                "risks": [
                    "What are the top risks or assumptions?",
                    "What dependencies could block progress?",
                    "What mitigations will we prepare for key risks?",
                ],
            }
            for k in ["discovery", "user_journey", "metrics", "gtm", "risks"]:
                by_lens[k] = defaults[k]
                questions.extend(defaults[k])

        return BaseResponse.success(data={"questions": questions, "by_lens": by_lens})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate clarifications: {str(e)}")


@router.post("/projects/{project_id}/save-artifacts", status_code=status.HTTP_200_OK)
async def save_artifacts(
    project_id: str,
    payload: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service),
):
    """
    Persist current PRD and Mermaid drafts to storage and create a version checkpoint.
    Returns URLs, version, checkpoint_id, and etag.
    """
    try:
        prd_markdown = (payload or {}).get("prd_markdown") or ""
        mermaid = (payload or {}).get("mermaid") or ""
        _etag = (payload or {}).get("etag")

        # Verify project ownership
        project = await project_service.get_project(project_id, str(current_user.id))

        # Upload current artifacts
        storage = project_service.project_repository.file_storage
        folder_current = f"{project.storage_path}/current"
        folder_versions = f"{project.storage_path}/versions"

        prd_bytes = prd_markdown.encode("utf-8")
        mmd_bytes = mermaid.encode("utf-8")

        prd_upload = await storage.upload_file(
            file_content=prd_bytes,
            filename="prd.md",
            content_type="text/markdown",
            folder_path=folder_current,
        )
        mmd_upload = await storage.upload_file(
            file_content=mmd_bytes,
            filename="flowchart.mmd",
            content_type="text/plain",
            folder_path=folder_current,
        )

        # Create versioned checkpoint with timestamp id
        from datetime import datetime
        checkpoint_id = datetime.utcnow().isoformat() + "Z"
        version_folder = f"{folder_versions}/{checkpoint_id}"
        _ = await storage.upload_file(prd_bytes, "prd.md", "text/markdown", version_folder)
        _ = await storage.upload_file(mmd_bytes, "flowchart.mmd", "text/plain", version_folder)

        # Update project version metadata (simple bump: v1.x -> v1.(x+1))
        try:
            base, minor = project.current_version.split(".")
            minor_int = int(minor)
            new_version = f"{base}.{minor_int + 1}"
        except Exception:
            new_version = "v1.1"

        await project_service.project_repository.update_project(
            project_id=project_id,
            user_id=str(current_user.id),
            update_data={
                "current_version": new_version,
                "updated_at": datetime.utcnow(),
                "metadata.last_agent_run": datetime.utcnow(),
            },
        )

        # Compose response
        etag = prd_upload.get("file_hash")  # simple etag from file hash
        # Compute and cache PRD summary (persisted)
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            # Simple summarization prompt:
            sys = {"role": "system", "content": "Summarize the PRD concisely in 8-12 bullet points. No code fences."}
            usr = {"role": "user", "content": prd_markdown[:18000]}
            resp = await ai_service.generate_response(
                user_id=str(current_user.id),
                messages=[sys, usr],
                temperature=0.2,
                max_tokens=800,
                use_cache=False,
            )
            if await cache_service.is_connected():
                key = f"prd_summary:{project_id}:{etag}"
                await cache_service.redis.setex(key, settings.PRD_SUMMARY_TTL_SECONDS, resp.content or "")
        except Exception:
            pass

        return BaseResponse.success(
            data={
                "prd_url": prd_upload["url"],
                "flowchart_url": mmd_upload["url"],
                "version": new_version,
                "checkpoint_id": checkpoint_id,
                "etag": etag,
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save artifacts: {str(e)}")

