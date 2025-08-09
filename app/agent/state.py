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
    telemetry: Dict[str, Any] = Field(default_factory=dict)
    # Whether to generate a flowchart in this run (initial runs should keep this False)
    generate_flowchart: bool = False

    # HITL loop state (Phase 2)
    iteration_index: int = 0
    answered_qa: List[Dict[str, str]] = Field(default_factory=list)
    pending_question: Optional[Dict[str, Any]] = None
    asked_questions: List[str] = Field(default_factory=list)
    completion_targets: Dict[str, Any] = Field(
        default_factory=lambda: {
            # Section-based completion targets aligned to PRD template
            "sections": [
                "1. Product Overview / Purpose",
                "2. Objective and Goals",
                "3. User Personas",
                "4. User Needs & Requirements",
                "5. Features & Functional Requirements",
                "6. Non-Functional Requirements",
                "7. User Stories / UX Flow / Design Notes",
                "8. Technical Specifications",
                "9. Assumptions, Constraints & Dependencies",
                "10. Timeline & Milestones",
                "11. Success Metrics / KPIs",
                "12. Release Criteria",
                "13. Open Questions / Issues",
                "14. Budget and Resources",
            ],
            # Soft guardrail: do not finish until we've incorporated at least this many answers
            "min_questions_before_finish": 4,
            "max_questions": 14,
        }
    )
    # Alias to thinking_lens_status for clarity in loop control
    coverage_status: Dict[str, bool] = Field(default_factory=dict)
    # UI overrides (e.g., lens coverage hints) provided by frontend
    ui_overrides: Optional[Dict[str, Any]] = None

    # Runtime only (not persisted)
    send_event: Optional[Callable[[Dict[str, Any]], Any]] = Field(default=None, exclude=True)
    # Minimal identifiers for nodes to publish via manager when callback is absent
    ws_chat_id: Optional[str] = Field(default=None, exclude=True)

