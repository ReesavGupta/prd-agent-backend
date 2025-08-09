from __future__ import annotations

from typing import Any, Dict

from app.agent.state import AgentState
from app.agent.graph import compile_graph
from langgraph.types import Command


async def run_iteration(state: AgentState) -> Dict[str, Any]:
    """Run the compiled graph once with streaming via state's send_event callback."""
    graph = compile_graph()
    # Execute graph; nodes themselves stream events via state.send_event
    await graph.ainvoke(state)
    return {"prd_markdown": state.prd_markdown, "mermaid": state.mermaid}


async def start_run(state: AgentState, thread_id: str) -> Dict[str, Any]:
    """Start or continue a HITL-capable run bound to a thread id (chat_id)."""
    graph = compile_graph()
    config = {"configurable": {"thread_id": thread_id}}
    await graph.ainvoke(state, config=config)
    return {"prd_markdown": state.prd_markdown, "mermaid": state.mermaid}


async def resume_run(thread_id: str, resume_payload: Dict[str, Any]) -> None:
    """Resume an interrupted run with provided payload."""
    graph = compile_graph()
    config = {"configurable": {"thread_id": thread_id}}
    await graph.ainvoke(Command(resume=resume_payload), config=config)

