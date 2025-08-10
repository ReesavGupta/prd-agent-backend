"""
WebSocket connection manager for real-time chat functionality.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Set, Any
import re
import asyncio
from fastapi import WebSocket, WebSocketDisconnect

from app.models.user import UserInDB, PyObjectId
from app.models.chat import ChatSession
from app.schemas.chat import WebSocketMessageRequest, WebSocketMessageResponse
from app.services.chat_service import chat_service
from app.agent.state import AgentState
from app.agent.runtime import run_iteration, start_run, resume_run, start_flowchart_run
from app.agent.commands import detect_rename_title, apply_prd_title_rename, is_pure_rename_title_command
from app.services.ai_service import ai_service
from app.services.cache_service import cache_service
from app.services.rag_service import get_prd_summary, select_sections, retrieve_attachment
from app.core.config import settings
from app.db.database import get_database

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manager for WebSocket connections and chat sessions."""
    
    def __init__(self):
        # Active connections: chat_id -> set of websockets
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        
        # Connection to user mapping: websocket -> user_info
        self.connection_users: Dict[WebSocket, Dict[str, str]] = {}
        
        # Connection to session mapping: websocket -> session_id
        self.connection_sessions: Dict[WebSocket, str] = {}
        
        # Typing indicators: chat_id -> set of user_ids
        self.typing_users: Dict[str, Set[str]] = {}
        
        # Pending agent interrupt per chat (question awaiting answer)
        self.pending_interrupts: Dict[str, Dict[str, Any]] = {}
        # Run-in-flight indicator per chat (guards concurrent runs / races)
        self.run_in_flight: Dict[str, bool] = {}
        # Flowchart-only run in-flight indicator per chat (separate lifecycle)
        self.flow_run_in_flight: Dict[str, bool] = {}

    # --- Edit command helpers -------------------------------------------------
    _RENAME_PATTERNS = [
        re.compile(r"^\s*rename\s+(?:the\s+)?(?:prd\s+)?(?:title\s+)?to\s+(.+)$", re.I),
        re.compile(r"^\s*change\s+(?:the\s+)?(?:prd\s+)?title\s+(?:to|as)\s+(.+)$", re.I),
        re.compile(r"^\s*set\s+(?:the\s+)?(?:prd\s+)?title\s+to\s+(.+)$", re.I),
    ]

    def _extract_rename_title(self, content: str) -> Optional[str]:
        if not content:
            return None
        text = content.strip().strip('"\'')
        for pat in self._RENAME_PATTERNS:
            m = pat.match(text)
            if m:
                title = m.group(1).strip().strip('"\'')
                # guard against empty or too short titles
                if title:
                    return title
        return None

    def _rename_prd_title(self, markdown: str, new_title: str) -> str:
        """Rename or insert top-level PRD title (# Heading) deterministically."""
        if not new_title:
            return markdown
        lines = (markdown or "").splitlines()
        # Find first non-empty line
        idx = None
        for i, line in enumerate(lines):
            if line.strip():
                idx = i
                break
        if idx is None:
            return f"# {new_title}\n\n"
        if lines[idx].lstrip().startswith("#"):
            # Replace the entire heading line with new title
            lines[idx] = f"# {new_title}"
            return "\n".join(lines) + ("\n" if markdown.endswith("\n") else "")
        # No heading at top; insert new title before first non-empty line
        prefix = lines[:idx]
        rest = lines[idx:]
        return "\n".join(prefix + [f"# {new_title}", ""] + rest)
    
    async def connect(
        self, 
        websocket: WebSocket, 
        chat_id: str, 
        user: UserInDB
    ) -> bool:
        """Accept WebSocket connection and create chat session."""
        try:
            await websocket.accept()
            
            # Generate unique websocket ID
            websocket_id = str(uuid.uuid4())
            
            # Create chat session
            session = await chat_service.create_websocket_session(
                chat_id=PyObjectId(chat_id),
                user_id=user.id,
                websocket_id=websocket_id
            )
            
            # Add to active connections
            if chat_id not in self.active_connections:
                self.active_connections[chat_id] = set()
            self.active_connections[chat_id].add(websocket)
            
            # Store connection metadata
            self.connection_users[websocket] = {
                "user_id": str(user.id),
                "chat_id": chat_id,
                "websocket_id": websocket_id
            }
            self.connection_sessions[websocket] = session.session_id
            
            logger.info(f"WebSocket connected for user {user.id} in chat {chat_id}")
            
            # Send connection confirmation
            await self._send_to_websocket(websocket, {
                "type": "connection_confirmed",
                "data": {
                    "session_id": session.session_id,
                    "chat_id": chat_id
                }
            })
            
            return True
        except Exception as e:
            logger.error(f"Error connecting WebSocket: {e}")
            try:
                await websocket.close(code=4000, reason="Connection failed")
            except:
                pass
            return False
    
    async def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection."""
        try:
            if websocket not in self.connection_users:
                return
            
            connection_info = self.connection_users[websocket]
            chat_id = connection_info["chat_id"]
            user_id = connection_info["user_id"]
            session_id = self.connection_sessions.get(websocket)
            
            # Remove from active connections
            if chat_id in self.active_connections:
                self.active_connections[chat_id].discard(websocket)
                if not self.active_connections[chat_id]:
                    del self.active_connections[chat_id]
            
            # Remove from typing users
            if chat_id in self.typing_users:
                self.typing_users[chat_id].discard(user_id)
                if not self.typing_users[chat_id]:
                    del self.typing_users[chat_id]
            
            # Clean up connection metadata
            self.connection_users.pop(websocket, None)
            self.connection_sessions.pop(websocket, None)
            
            # Delete session from database
            if session_id:
                await chat_service.session_repo.delete_session(session_id)
            
            logger.info(f"WebSocket disconnected for user {user_id} in chat {chat_id}")
            
            # Notify other users about disconnection
            await self._broadcast_to_chat(chat_id, {
                "type": "user_disconnected",
                "data": {
                    "user_id": user_id
                }
            }, exclude_websocket=websocket)
            
        except Exception as e:
            logger.error(f"Error handling WebSocket disconnection: {e}")
    
    async def handle_message(self, websocket: WebSocket, data: dict):
        """Handle incoming WebSocket message."""
        try:
            if websocket not in self.connection_users:
                await self._send_error(websocket, "Connection not authenticated")
                return
            
            connection_info = self.connection_users[websocket]
            chat_id = connection_info["chat_id"]
            user_id = connection_info["user_id"]
            session_id = self.connection_sessions.get(websocket)
            
            # Parse message
            try:
                message = WebSocketMessageRequest(**data)
            except Exception as e:
                await self._send_error(websocket, f"Invalid message format: {e}")
                return
            
            # Handle different message types
            if message.type == "send_message":
                await self._handle_send_message(websocket, chat_id, user_id, message.data)
            
            elif message.type == "typing_start":
                await self._handle_typing_start(websocket, chat_id, user_id, session_id)
            
            elif message.type == "typing_stop":
                await self._handle_typing_stop(websocket, chat_id, user_id, session_id)
            
            elif message.type == "ping":
                await self._handle_ping(websocket, session_id)
            
            elif message.type == "agent_resume":
                await self._handle_agent_resume(websocket, chat_id, user_id, message.data)
            
            else:
                await self._send_error(websocket, f"Unknown message type: {message.type}")
            
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            await self._send_error(websocket, "Message processing failed")
    
    async def _handle_send_message(
        self, 
        websocket: WebSocket, 
        chat_id: str, 
        user_id: str, 
        data: dict
    ):
        """Handle send message request."""
        try:
            # Flowchart-only mode does not require textual message 'content'
            mode = data.get("mode")
            if mode == "flowchart":
                if self.flow_run_in_flight.get(chat_id):
                    await self._send_error(websocket, "Flowchart generation is already in progress for this chat")
                    return
                base_prd_markdown = (data.get("base_prd_markdown") or "").strip()
                if not base_prd_markdown:
                    await self._send_error(websocket, "base_prd_markdown is required for flowchart mode")
                    return
                state = AgentState(
                    project_id=str(data.get("project_id", "")),
                    chat_id=chat_id,
                    user_id=user_id,
                    base_prd_markdown=base_prd_markdown,
                    base_mermaid=(data.get("base_mermaid") or ""),
                    prd_markdown=base_prd_markdown,
                    ws_chat_id=chat_id,
                )
                # Tag outgoing events as flowchart-kind
                async def _send_event(evt: Dict[str, Any]) -> None:
                    payload = dict(evt)
                    try:
                        d = payload.get("data") or {}
                        if isinstance(d, dict) and d.get("kind") != "flowchart":
                            d = {**d, "kind": "flowchart"}
                            payload["data"] = d
                    except Exception:
                        pass
                    await self._broadcast_to_chat(chat_id, payload)
                state.send_event = _send_event  # type: ignore
                # Kick off with a stream_start for UX
                await self._broadcast_to_chat(chat_id, {"type": "stream_start", "data": {"project_id": state.project_id, "kind": "flowchart"}})
                # Run minimal graph on separate thread namespace
                self.flow_run_in_flight[chat_id] = True
                try:
                    await start_flowchart_run(state, thread_id=f"fc:{chat_id}")
                finally:
                    self.flow_run_in_flight[chat_id] = False
                return

            # Default conversational/agent modes require content
            content = data.get("content")
            if not content:
                await self._send_error(websocket, "Message content is required")
                return
            
            # Agent mode: run LangGraph per-message iteration and stream back to sender
            mode = data.get("mode")
            if mode == "agent":
                # If a question is pending for this chat, forward raw message into the agent thread for classification
                pending = self.pending_interrupts.get(chat_id)
                if pending:
                    try:
                        # Echo user's message into chat stream
                        await self._broadcast_to_chat(chat_id, {
                            "type": "message_sent",
                            "data": {
                                "message_id": str(uuid.uuid4()),
                                "user_id": user_id,
                                "content": content,
                                "timestamp": datetime.utcnow().isoformat(),
                                "message_type": "user"
                            }
                        })
                        # Do NOT clear interrupt here. Let the graph classify whether this is an answer or a generic query
                        await resume_run(chat_id, {"type": "message", "text": content})
                        return
                    except Exception as e:
                        logger.error(f"Error forwarding message to pending interrupt: {e}")
                        # If resume fails, fall back to starting a fresh run below

                # If a run is in-flight but pending hasn't been recorded yet, wait briefly for interrupt
                if self.run_in_flight.get(chat_id):
                    try:
                        await asyncio.sleep(0.15)
                    except Exception:
                        pass
                    pending = self.pending_interrupts.get(chat_id)
                    if pending:
                        try:
                            await self._broadcast_to_chat(chat_id, {
                                "type": "message_sent",
                                "data": {
                                    "message_id": str(uuid.uuid4()),
                                    "user_id": user_id,
                                    "content": content,
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "message_type": "user"
                                }
                            })
                            await resume_run(chat_id, {"type": "message", "text": content})
                            return
                        except Exception as e:
                            logger.error(f"Error resuming after grace wait (message forward): {e}")
                    # Still busy: reject starting a new run to avoid duplicate questions
                    await self._send_error(websocket, "Agent is busy processing. Please wait for the question to appear, then reply.")
                    return

                # Build state from client-provided context and stream via this websocket
                # Apply deterministic edit commands (e.g., rename PRD title) before starting a run
                base_prd_markdown = (data.get("base_prd_markdown", "") or "")
                rename_to = detect_rename_title(content or "")
                if rename_to:
                    try:
                        # Apply deterministic rename locally
                        base_prd_markdown = apply_prd_title_rename(base_prd_markdown, rename_to)
                        # If the message is a pure rename command, reflect immediately and do not start a run
                        if is_pure_rename_title_command(content or ""):
                            await self._broadcast_to_chat(chat_id, {
                                "type": "artifacts_preview",
                                "data": {
                                    "prd_markdown": base_prd_markdown,
                                    "mermaid": data.get("base_mermaid", "") or None,
                                    "thinking_lens_status": data.get("ui_overrides", {}).get("thinking_lens_status") if isinstance(data.get("ui_overrides"), dict) else None,
                                },
                            })
                            # Echo a short assistant confirmation
                            await self._broadcast_to_chat(chat_id, {
                                "type": "ai_response_complete",
                                "data": {
                                    "message": f"Renamed PRD title to '{rename_to}'",
                                },
                            })
                            return
                    except Exception as e:
                        logger.error(f"Failed to apply rename command: {e}")

                state = AgentState(
                    project_id=str(data.get("project_id", "")),
                    chat_id=chat_id,
                    user_id=user_id,
                    idea=data.get("initial_idea"),
                    qa=data.get("clarifications") or [],
                    last_messages=data.get("last_messages") or [],
                    base_prd_markdown=base_prd_markdown,
                    base_mermaid=data.get("base_mermaid", "") or "",
                    attachments=data.get("attachments") or [],
                    ui_overrides=data.get("ui_overrides") or None,
                    ws_chat_id=chat_id,
                    generate_flowchart=bool(data.get("generate_flowchart", False)),
                )
                # Use HITL-capable start_run bound to thread_id=chat_id
                await start_run(state, thread_id=chat_id)
                return
            if mode == "chat":
                await self._handle_chat_mode(websocket, chat_id, user_id, data)
                return
            
            # Create message request
            from app.schemas.chat import MessageCreateRequest
            request = MessageCreateRequest(
                content=content,
                message_type="user"
            )
            
            # Send message through service
            user_message, ai_message = await chat_service.send_message(
                chat_id=PyObjectId(chat_id),
                user_id=PyObjectId(user_id),
                request=request
            )
            
            # Broadcast user message to all connections in chat
            await self._broadcast_to_chat(chat_id, {
                "type": "message_sent",
                "data": {
                    "message_id": str(user_message.id),
                    "user_id": user_id,
                    "content": user_message.content,
                    "timestamp": user_message.timestamp.isoformat(),
                    "message_type": user_message.message_type
                }
            })
            
            # If AI responded, broadcast that too
            if ai_message:
                await self._broadcast_to_chat(chat_id, {
                    "type": "ai_response_complete",
                    "data": {
                        "message_id": str(ai_message.id),
                        "content": ai_message.content,
                        "timestamp": ai_message.timestamp.isoformat(),
                        "metadata": {
                            "model_used": ai_message.metadata.model_used if ai_message.metadata else None,
                            "response_time_ms": ai_message.metadata.response_time_ms if ai_message.metadata else None
                        }
                    }
                })
            
        except Exception as e:
            logger.error(f"Error handling send message: {e}")
            await self._send_error(websocket, "Failed to send message")

    async def _handle_chat_mode(self, websocket: WebSocket, chat_id: str, user_id: str, data: dict) -> None:
        """Handle chat mode with PRD-summary + optional single-file RAG over indexed PDF."""
        try:
            project_id = str(data.get("project_id") or "").strip()
            content = str(data.get("content") or "").strip()
            if not project_id:
                await self._send_error(websocket, "project_id is required for chat mode")
                return
            # Echo user's message to room
            logger.info("[CHAT] chat_mode recv chat_id=%s user_id=%s content_len=%d proj=%s attach=%s", chat_id, user_id, len(content or ""), project_id, bool(data.get("attachment_file_id")))
            await self._broadcast_to_chat(chat_id, {
                "type": "message_sent",
                "data": {
                    "message_id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "content": content,
                    "timestamp": datetime.utcnow().isoformat(),
                    "message_type": "user",
                },
            })

            # Gather PRD summary (from cache or ephemeral from provided prd_markdown)
            prd_markdown = (data.get("prd_markdown") or "").strip()
            summary_text = await get_prd_summary(project_id=project_id, user_id=user_id, prd_markdown=prd_markdown)

            # Optional single-file RAG
            attachment_file_id = (data.get("attachment_file_id") or "").strip()
            retrieved_text = ""
            indexing_in_progress = False
            if attachment_file_id:
                try:
                    from bson import ObjectId
                    db = get_database()
                    upload_doc = await db.uploads.find_one({"_id": ObjectId(attachment_file_id), "project_id": ObjectId(project_id)})
                    if upload_doc and upload_doc.get("indexed"):
                        logger.info("[CHAT] retrieval using file_id=%s", attachment_file_id)
                        retrieved_text = await retrieve_attachment(project_id=project_id, file_id=attachment_file_id, query=content)
                    else:
                        logger.info("[CHAT] attachment not indexed yet file_id=%s", attachment_file_id)
                        indexing_in_progress = True
                except Exception:
                    pass

            # Optional assistant preface if indexing not ready
            if indexing_in_progress:
                try:
                    await self._broadcast_to_chat(chat_id, {
                        "type": "message_sent",
                        "data": {
                            "message_id": f"p:{int(datetime.utcnow().timestamp()*1000)}",
                            "user_id": user_id,
                            "content": "Indexing in progress for the attached document. Answering from PRD only.",
                            "timestamp": datetime.utcnow().isoformat(),
                            "message_type": "assistant",
                        },
                    })
                except Exception:
                    pass

            # Section spotlights (lightweight classifier using headings from PRD markdown if provided)
            prd_section_spotlights = ""
            try:
                if prd_markdown:
                    prd_section_spotlights = await select_sections(prd_markdown=prd_markdown, question=content, user_id=user_id)
            except Exception:
                prd_section_spotlights = ""

            # Build prompt
            context_blocks = []
            if summary_text:
                context_blocks.append(f"PRD Summary:\n{summary_text}")
            if prd_section_spotlights:
                context_blocks.append(f"PRD Sections:\n{prd_section_spotlights}")
            if retrieved_text:
                context_blocks.append(f"Document Snippets:\n{retrieved_text}")
            context_str = "\n\n".join(context_blocks) if context_blocks else "(no context)"

            system_msg = {
                "role": "system",
                "content": (
                    "You are a concise product assistant. Use ONLY the provided context to answer the user's question. "
                    "If the answer is not present, say you don't have enough information. Keep answers brief and specific."
                ),
            }
            user_msg = {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion:\n{content}"}

            await self._broadcast_to_chat(chat_id, {"type": "stream_start", "data": {"project_id": project_id}})
            final_text = []
            last_provider = None
            last_model = None
            try:
                async for chunk in ai_service.generate_stream(
                    user_id=user_id,
                    messages=[system_msg, user_msg],
                    temperature=0.2,
                    max_tokens=800,
                ):
                    if chunk.is_complete:
                        await self._broadcast_to_chat(chat_id, {
                            "type": "ai_response_streaming",
                            "data": {"delta": "", "is_complete": True, "provider": chunk.provider, "model": chunk.model},
                        })
                        last_provider = chunk.provider
                        last_model = chunk.model
                        break
                    delta = chunk.content or ""
                    final_text.append(delta)
                    last_provider = chunk.provider
                    last_model = chunk.model
                    await self._broadcast_to_chat(chat_id, {
                        "type": "ai_response_streaming",
                        "data": {"delta": delta, "is_complete": False, "provider": chunk.provider, "model": chunk.model},
                    })
            except Exception as e:
                logger.error("[CHAT] generation failed: %s", e)
                await self._send_error(websocket, f"Generation failed: {e}")
                return

            await self._broadcast_to_chat(chat_id, {
                "type": "ai_response_complete",
                "data": {"message": "", "provider": last_provider, "model": last_model},
            })
        except Exception as e:
            logger.error(f"chat_mode error: {e}")
            await self._send_error(websocket, "Chat mode processing failed")
    
    async def _handle_agent_resume(
        self,
        websocket: WebSocket,
        chat_id: str,
        user_id: str,
        data: dict,
    ) -> None:
        """Resume an interrupted agent thread with human input."""
        try:
            payload = data or {}
            if "answer" in payload:
                answer = payload["answer"] or {}
                if not isinstance(answer, dict) or not answer.get("question_id"):
                    await self._send_error(websocket, "Invalid resume payload: missing answer.question_id")
                    return
                await self._broadcast_to_chat(chat_id, {
                    "type": "agent_interrupt_cleared",
                    "data": {"question_id": answer.get("question_id")}
                })
                await resume_run(chat_id, {"type": "answer", "question_id": answer.get("question_id"), "text": answer.get("text", "")})
            elif "accept" in payload:
                accept = payload["accept"] or {}
                if not isinstance(accept, dict) or not accept.get("question_id"):
                    await self._send_error(websocket, "Invalid resume payload: missing accept.question_id")
                    return
                await self._broadcast_to_chat(chat_id, {
                    "type": "agent_interrupt_cleared",
                    "data": {"question_id": accept.get("question_id")}
                })
                await resume_run(chat_id, {"type": "accept", "question_id": accept.get("question_id"), "finish": bool(accept.get("finish"))})
            else:
                await self._send_error(websocket, "Invalid resume payload: expected 'answer' or 'accept'")
        except Exception as e:
            logger.error(f"Error handling agent_resume: {e}")
            await self._send_error(websocket, "Failed to resume agent")
    
    async def _handle_typing_start(
        self, 
        websocket: WebSocket, 
        chat_id: str, 
        user_id: str, 
        session_id: str
    ):
        """Handle typing start indicator."""
        try:
            # Update session typing status
            if session_id:
                await chat_service.update_typing_status(session_id, True)
            
            # Add to typing users
            if chat_id not in self.typing_users:
                self.typing_users[chat_id] = set()
            self.typing_users[chat_id].add(user_id)
            
            # Broadcast typing indicator
            await self._broadcast_to_chat(chat_id, {
                "type": "typing_indicator",
                "data": {
                    "user_id": user_id,
                    "is_typing": True
                }
            }, exclude_websocket=websocket)
            
        except Exception as e:
            logger.error(f"Error handling typing start: {e}")
    
    async def _handle_typing_stop(
        self, 
        websocket: WebSocket, 
        chat_id: str, 
        user_id: str, 
        session_id: str
    ):
        """Handle typing stop indicator."""
        try:
            # Update session typing status
            if session_id:
                await chat_service.update_typing_status(session_id, False)
            
            # Remove from typing users
            if chat_id in self.typing_users:
                self.typing_users[chat_id].discard(user_id)
                if not self.typing_users[chat_id]:
                    del self.typing_users[chat_id]
            
            # Broadcast typing stop
            await self._broadcast_to_chat(chat_id, {
                "type": "typing_indicator",
                "data": {
                    "user_id": user_id,
                    "is_typing": False
                }
            }, exclude_websocket=websocket)
            
        except Exception as e:
            logger.error(f"Error handling typing stop: {e}")
    
    async def _handle_ping(self, websocket: WebSocket, session_id: str):
        """Handle ping message for keepalive."""
        try:
            # Update session activity
            if session_id:
                await chat_service.session_repo.update_session_activity(session_id)
            
            # Send pong response
            await self._send_to_websocket(websocket, {
                "type": "pong",
                "data": {"timestamp": datetime.utcnow().isoformat()}
            })
            
        except Exception as e:
            logger.error(f"Error handling ping: {e}")
    
    async def _broadcast_to_chat(
        self, 
        chat_id: str, 
        message: dict, 
        exclude_websocket: Optional[WebSocket] = None
    ):
        """Broadcast message to all connections in a chat."""
        # Track pending interrupt lifecycle for auto-routing
        try:
            mtype = str(message.get("type"))
            data_obj = message.get("data") or {}
            kind = None
            if isinstance(data_obj, dict):
                kind = data_obj.get("kind")
            if mtype == "agent_interrupt_request":
                # Ignore interrupts emitted by flowchart runs (there shouldn't be any)
                if kind != "flowchart":
                    if isinstance(data_obj, dict) and data_obj.get("question_id"):
                        self.pending_interrupts[chat_id] = data_obj
                    self.run_in_flight[chat_id] = True
            elif mtype == "agent_interrupt_cleared":
                if kind != "flowchart":
                    self.pending_interrupts.pop(chat_id, None)
            elif mtype == "stream_start":
                if kind == "flowchart":
                    self.flow_run_in_flight[chat_id] = True
                else:
                    self.run_in_flight[chat_id] = True
            elif mtype == "ai_response_complete":
                if kind == "flowchart":
                    self.flow_run_in_flight[chat_id] = False
                else:
                    if chat_id not in self.pending_interrupts:
                        self.run_in_flight[chat_id] = False
            elif mtype == "error":
                if kind == "flowchart":
                    self.flow_run_in_flight[chat_id] = False
                else:
                    self.run_in_flight[chat_id] = False
        except Exception:
            pass

        if chat_id not in self.active_connections:
            return
        
        connections = self.active_connections[chat_id].copy()
        for websocket in connections:
            if websocket != exclude_websocket:
                await self._send_to_websocket(websocket, message)
    
    async def _send_to_websocket(self, websocket: WebSocket, message: dict):
        """Send message to a specific WebSocket."""
        try:
            def _sanitize(obj):
                if isinstance(obj, (str, int, float, bool)) or obj is None:
                    return obj
                if isinstance(obj, dict):
                    return {str(k): _sanitize(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_sanitize(v) for v in obj]
                # Fallback: stringify anything else (e.g., functions)
                try:
                    return str(obj)
                except Exception:
                    return "<non-serializable>"

            safe_message = {
                "type": str(message.get("type")),
                "data": _sanitize(message.get("data", {})),
            }
            response = WebSocketMessageResponse(**safe_message)
            await websocket.send_text(response.model_dump_json())
        except WebSocketDisconnect:
            # Connection already closed, clean up
            await self.disconnect(websocket)
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
            # Don't disconnect on send errors, connection might still be valid
    
    async def _send_error(self, websocket: WebSocket, error_message: str):
        """Send error message to WebSocket."""
        await self._send_to_websocket(websocket, {
            "type": "error",
            "data": {"message": error_message}
        })
    
    def get_chat_connections_count(self, chat_id: str) -> int:
        """Get number of active connections for a chat."""
        return len(self.active_connections.get(chat_id, set()))
    
    def get_typing_users(self, chat_id: str) -> List[str]:
        """Get list of users currently typing in a chat."""
        return list(self.typing_users.get(chat_id, set()))


# Global WebSocket manager instance
websocket_manager = WebSocketManager()