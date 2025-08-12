from __future__ import annotations

from typing import Any, Dict, List, Optional, Callable
from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """State passed through the LangGraph per-message run.

    This state is ephemeral (no checkpointer). Frontend provides last_messages and drafts.
    A non-serializable send_event callback may be included for streaming.
    """

    # Identity
    project_id: str
    chat_id: str
    user_id: str

    # Inputs
    idea: Optional[str] = None
    qa: Optional[List[Dict[str, str]]] = None
    last_messages: List[Dict[str, str]] = Field(default_factory=list)
    base_prd_markdown: str = ""
    base_mermaid: str = ""
    attachments: Optional[List[Dict[str, Any]]] = None

    # Outputs
    prd_markdown: str = ""
    mermaid: str = ""
    thinking_lens_status: Dict[str, bool] = Field(default_factory=dict)
    sections_status: Dict[str, bool] = Field(default_factory=dict)
    all_sections_status: Dict[str, bool] = Field(default_factory=dict)
    telemetry: Dict[str, Any] = Field(default_factory=dict)
    # Whether to generate a flowchart in this run (initial runs should keep this False)
    generate_flowchart: bool = False

    # HITL loop state (Phase 2)
    iteration_index: int = 0
    answered_qa: List[Dict[str, str]] = Field(default_factory=list)
    pending_question: Optional[Dict[str, Any]] = None
    # Distinguish what kind of pending question is active (e.g., 'section' or 'confirm')
    pending_question_kind: Optional[str] = None
    asked_questions: List[str] = Field(default_factory=list)
    # Store the last human message received while waiting at an interrupt
    last_user_message: Optional[str] = None
    # Track last asked section/question so we can classify later messages against it
    last_section_target: Optional[str] = None
    last_section_question: Optional[str] = None
    last_section_qid: Optional[str] = None
    completion_targets: Dict[str, Any] = Field(
        default_factory=lambda: {
            # Section-based completion targets aligned to PRD template
            "sections": [
                "1. Product Overview / Purpose",
                "2. Objectives & KPIs",
                "3. User Personas",
                "4. User Needs",
                "5. Functional Requirements",
                "6. Non-Functional Requirements",
                "7. Technical Architecture & Integrations",
                "8. Risks & Open Questions",
            ],
            # Soft guardrail: do not finish until we've incorporated at least this many answers
            "min_questions_before_finish": 4,
            "max_questions": 8,
        }
    )
    # Alias to thinking_lens_status for clarity in loop control
    coverage_status: Dict[str, bool] = Field(default_factory=dict)
    # UI overrides (e.g., lens coverage hints) provided by frontend
    ui_overrides: Optional[Dict[str, Any]] = None

    # Initial question plan: one question per template section (Option 2)
    initial_question_plan: List[Dict[str, str]] = Field(default_factory=list)
    plan_cursor: int = 0

    # Runtime only (not persisted)
    send_event: Optional[Callable[[Dict[str, Any]], Any]] = Field(default=None, exclude=True)
    # Minimal identifiers for nodes to publish via manager when callback is absent
    ws_chat_id: Optional[str] = Field(default=None, exclude=True)
    # Optional single-file attachment to bias RAG in agent mode (carried over a single resume/incorporation cycle)
    pending_attachment_file_id: Optional[str] = None

