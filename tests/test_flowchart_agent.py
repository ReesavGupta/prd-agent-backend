import asyncio
import pytest

from app.agent.state import AgentState
from app.agent.runtime import start_flowchart_run


@pytest.mark.asyncio
async def test_flowchart_only_run_generates_mermaid(monkeypatch):
    # Arrange: minimal PRD outline
    prd = """# Product Requirement Document: Demo

### 1. Product Overview / Purpose

Short overview.
"""
    # A dummy send_event collector
    events = []

    async def send(evt):
        events.append(evt)

    state = AgentState(
        project_id="p1",
        chat_id="c1",
        user_id="u1",
        base_prd_markdown=prd,
        prd_markdown=prd,
        base_mermaid="",
    )
    state.send_event = send  # type: ignore

    # Act
    await start_flowchart_run(state, thread_id="fc:c1")

    # Assert
    assert isinstance(state.mermaid, str)
    assert len(state.mermaid.strip()) > 0
    # Must look like Mermaid (graph/flowchart prefix)
    head = state.mermaid.strip().lower()
    assert head.startswith("graph ") or head.startswith("flowchart ")
    # Must have emitted artifacts_preview once with mermaid
    assert any(e.get("type") == "artifacts_preview" and (e.get("data") or {}).get("mermaid") for e in events)



