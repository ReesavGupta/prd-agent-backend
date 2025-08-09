from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from app.agent.state import AgentState
from app.agent.nodes import (
    prepare_context,
    generate_prd,
    generate_mermaid,
    postprocess,
    analyze_gaps,
    propose_next_question,
    await_human_answer,
    incorporate_answer,
)
from app.agent.checkpoint import get_checkpointer
from typing import Any


def compile_graph() -> StateGraph:
    builder = StateGraph(AgentState)
    builder.add_node("prepare_context", prepare_context)
    builder.add_node("generate_prd", generate_prd)
    builder.add_node("analyze_gaps", analyze_gaps)
    builder.add_node("propose_next_question", propose_next_question)
    builder.add_node("await_human_answer", await_human_answer)
    builder.add_node("incorporate_answer", incorporate_answer)
    builder.add_node("generate_mermaid", generate_mermaid)
    builder.add_node("postprocess", postprocess)

    builder.add_edge(START, "prepare_context")
    builder.add_edge("prepare_context", "generate_prd")
    # HITL loop: PRD -> analyze -> propose/finish -> [await -> incorporate -> analyze]* -> finalize
    builder.add_edge("generate_prd", "analyze_gaps")

    # Conditional edges after analyze_gaps
    def route_after_analyze(state: Any) -> str:
        try:
            tele = getattr(state, "telemetry", {}) or {}
            should_finish = bool(tele.get("should_finish"))
            gen_flow = bool(getattr(state, "generate_flowchart", False))
            if should_finish and gen_flow:
                return "finish_flow"
            if should_finish:
                return "finish"
            return "continue"
        except Exception:
            return "continue"

    try:
        builder.add_conditional_edges(
            "analyze_gaps",
            route_after_analyze,
            {
                "finish": "postprocess",            # finish without flowchart
                "finish_flow": "generate_mermaid",  # gated flowchart generation
                "continue": "propose_next_question",
            },
        )
    except Exception:
        # Fallback if conditional edges API unavailable
        builder.add_edge("analyze_gaps", "propose_next_question")
        builder.add_edge("analyze_gaps", "postprocess")

    # Conditional edges after propose_next_question
    def route_after_propose(state: Any) -> str:
        try:
            pq = getattr(state, "pending_question", None)
            if pq:
                return "ask"
            # No pending question → finalize path; optionally with flowchart
            gen_flow = bool(getattr(state, "generate_flowchart", False))
            return "finalize_flow" if gen_flow else "finalize"
        except Exception:
            return "finalize"

    try:
        builder.add_conditional_edges(
            "propose_next_question",
            route_after_propose,
            {
                "ask": "await_human_answer",
                "finalize": "postprocess",
                "finalize_flow": "generate_mermaid",
            },
        )
    except Exception:
        builder.add_edge("propose_next_question", "await_human_answer")
        builder.add_edge("propose_next_question", "generate_mermaid")

    # Loop back after answer incorporation
    builder.add_edge("await_human_answer", "incorporate_answer")
    builder.add_edge("incorporate_answer", "analyze_gaps")

    # Finalize: flowchart generation is optional/gated by runtime logic; keep edge if invoked
    builder.add_edge("generate_mermaid", "postprocess")
    builder.add_edge("postprocess", END)

    # Phase 1: compile with in-memory checkpointer to enable future interrupts
    return builder.compile(checkpointer=get_checkpointer())

