from __future__ import annotations
from typing import Any, Dict, List, Optional
import logging
from datetime import datetime
import time

from app.agent.state import AgentState
from app.services.ai_service import ai_service
from app.agent.commands import apply_prd_title_rename

try:
    from langgraph.types import interrupt
except Exception:
    def interrupt(payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"type": "accept"}

logger = logging.getLogger(__name__)

# --- New Workflow Nodes (Gather-then-Generate) ---

async def detect_domain_and_plan(state: AgentState) -> AgentState:
    from app.websocket.publisher import publish_to_chat
    async def _emit(event: Dict[str, Any]) -> None:
        if callable(state.send_event):
            await state.send_event(event)
        elif state.ws_chat_id:
            await publish_to_chat(state.ws_chat_id, event)

    await _emit({"type": "stream_start", "data": {"project_id": state.project_id}})
    
    idea_text = state.idea or ""
    
    domain_slug, domain_label = _detect_domain_from_prompt(idea_text)
    state.detected_domain = domain_slug
    
    state.current_phase = state.interview_plan.get("phases", [])[0]
    
    summary_prompt = {
        "role": "system",
        "content": "You are a product analyst. Summarize the following product idea into a concise one-sentence description. Output only the summary."
    }
    idea_prompt = {"role": "user", "content": f"Product Idea: {idea_text}"}
    try:
        response = await ai_service.generate_response(
            user_id=state.user_id,
            messages=[summary_prompt, idea_prompt],
            temperature=0.1,
            max_tokens=150
        )
        state.conversation_summary = response.content or f"A product based on the idea: {idea_text[:100]}..."
    except Exception as e:
        logger.error(f"Failed to generate initial summary: {e}")
        state.conversation_summary = f"A product idea was provided: {idea_text[:100]}..."

    state.telemetry = {
        "detected_domain": domain_slug,
        "initial_summary_generated": True,
    }
    
    await _emit({
        "type": "message_sent",
        "data": {
            "message_id": f"agent_start:{int(time.time())}",
            "user_id": state.user_id,
            "content": f"Thanks for the idea! I've categorized it as a '{domain_label}' product. Let's start by exploring the problem context.",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "message_type": "assistant",
        },
    })
    
    return state

async def propose_next_question(state: AgentState) -> AgentState:
    """
    Proposes the next question based on the current interview phase and history.
    """
    if not state.current_phase:
        state.pending_question = None
        return state

    phase_questions = {
        "1. Problem Context": "What is the core problem you are trying to solve for your users?",
        "2. Strategic Foundation": "What are the key business goals or success metrics for this product?",
        "3. User Experience": "Can you describe the ideal user journey, from start to finish?",
        "4. Technical & Operational": "Are there any known technical constraints or integration requirements?",
        "5. Implementation Planning": "What would a Minimum Viable Product (MVP) look like for the first release?",
        "6. Risk & Compliance": "What are the biggest risks or compliance hurdles you foresee?",
    }
    
    question_text = phase_questions.get(state.current_phase, "What else should we consider?")
    
    qid = f"q_{state.iteration_index + 1}"
    state.pending_question = {
        "id": qid,
        "question": question_text,
        "phase": state.current_phase,
        "rationale": f"Gathering info for the '{state.current_phase}' phase.",
    }
    state.asked_questions.append(question_text)
    
    return state

async def store_answer_and_manage_context(state: AgentState) -> AgentState:
    """
    Stores the user's answer and decides whether to move to the next phase
    or end the interview.
    """
    if not state.last_user_message or not state.pending_question:
        return state

    state.interview_history.append({
        "question": state.pending_question.get("question", ""),
        "answer": state.last_user_message,
        "phase": state.current_phase,
    })
    state.iteration_index += 1

    state.conversation_summary += f"\n- Q: {state.pending_question.get('question', '')}\n- A: {state.last_user_message}"
    
    current_phase_index = state.interview_plan["phases"].index(state.current_phase)
    
    if current_phase_index + 1 < len(state.interview_plan["phases"]):
        state.current_phase = state.interview_plan["phases"][current_phase_index + 1]
        state.telemetry["interview_complete"] = False
    else:
        state.current_phase = None
        state.telemetry["interview_complete"] = True

    state.pending_question = None
    state.last_user_message = None

    return state

def _detect_domain_from_prompt(idea_text: str) -> tuple[str, str]:
    """Heuristic domain detection based on keywords from the system prompt."""
    t = (idea_text or "").lower()
    if any(k in t for k in ["enterprise", "workflow", "admin", "integration"]):
        return ("B2B", "B2B/Enterprise")
    if any(k in t for k in ["user acquisition", "retention", "monetization", "onboarding"]):
        return ("Consumer", "Consumer Product")
    if any(k in t for k in ["compliance", "safety", "regulatory", "hipaa", "clinical"]):
        return ("Healthcare", "Healthcare/Regulated")
    if any(k in t for k in ["developer", "api", "sdk", "rate limit", "scalability"]):
        return ("Technical", "Technical/API Product")
    if any(k in t for k in ["marketplace", "network effect", "supply", "demand"]):
        return ("Marketplace", "Marketplace/Platform")
    return ("General", "General Product")


async def await_human_answer(state: AgentState) -> AgentState:
    if not state.pending_question:
        return state
    
    payload = {
        "type": "question",
        "question": state.pending_question.get("question"),
        "phase": state.pending_question.get("phase"),
        "question_id": state.pending_question.get("id"),
    }

    try:
        if callable(state.send_event):
            await state.send_event({"type": "agent_interrupt_request", "data": payload})
        else:
            from app.websocket.publisher import publish_to_chat
            if state.ws_chat_id:
                await publish_to_chat(state.ws_chat_id, {"type": "agent_interrupt_request", "data": payload})
                await publish_to_chat(state.ws_chat_id, {
                    "type": "message_sent",
                    "data": {
                        "message_id": payload["question_id"],
                        "user_id": state.user_id,
                        "content": payload["question"],
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "message_type": "assistant",
                    },
                })
    except Exception as e:
        logger.error(f"Error emitting interrupt request: {e}")

    response = interrupt(payload)
    
    if isinstance(response, dict) and "text" in response:
        state.last_user_message = response["text"]

    if isinstance(response, dict) and response.get('action') == 'proceed_to_generate':
        state.telemetry['user_confirmed_generation'] = True

    return state

async def confirm_final_generation(state: AgentState) -> AgentState:
    question = "I believe I have enough information to generate the full PRD. Would you like me to proceed?"
    qid = "confirm_generate_prd"
    state.pending_question = {
        "id": qid,
        "question": question,
        "phase": "finalization",
        "rationale": "Confirm before generating the final document.",
    }
    return state

async def generate_final_prd(state: AgentState) -> AgentState:
    from app.websocket.publisher import publish_to_chat
    async def _emit(event: Dict[str, Any]) -> None:
        if callable(state.send_event):
            await state.send_event(event)
        elif state.ws_chat_id:
            await publish_to_chat(state.ws_chat_id, event)

    await _emit({"type": "stream_start", "data": {"project_id": state.project_id}})
    await _emit({"type": "ai_response_streaming", "data": {"delta": "Generating the final PRD based on our conversation...", "is_complete": False}})

    system_prompt = (
        "You are a Senior Product Manager AI. Your task is to generate a comprehensive "
        "Product Requirements Document (PRD) from the provided interview history. "
        "Structure the output with clear markdown headings (H3 for sections). "
        "Ensure the document is professional, well-written, and directly reflects the information provided."
    )
    
    conversation = f"Initial Idea: {state.idea}\n\nInterview Q&A:\n"
    for qa in state.interview_history:
        conversation += f"- Phase: {qa.get('phase', '')}\n"
        conversation += f"  - Q: {qa.get('question', '')}\n"
        conversation += f"  - A: {qa.get('answer', '')}\n"

    user_prompt = {"role": "user", "content": conversation}
    system_message = {"role": "system", "content": system_prompt}

    try:
        response = await ai_service.generate_response(
            user_id=state.user_id,
            messages=[system_message, user_prompt],
            temperature=0.2,
            max_tokens=4000
        )
        final_prd = response.content or "# PRD Generation Failed"
    except Exception as e:
        logger.error(f"Final PRD generation failed: {e}")
        final_prd = f"# PRD Generation Error\n\nAn error occurred: {e}"

    if state.base_prd_markdown and state.base_prd_markdown.startswith("#"):
        title = state.base_prd_markdown.splitlines()[0].lstrip("# ").strip()
        final_prd = apply_prd_title_rename(final_prd, title)

    state.prd_markdown = final_prd
    
    await _emit({"type": "artifacts_preview", "data": {"prd_markdown": state.prd_markdown}})
    await _emit({"type": "ai_response_complete", "data": {"message": "PRD generated."}})

    return state

async def generate_mermaid(state: AgentState) -> AgentState:
    logger.info("Mermaid generation would occur here if enabled.")
    return state

async def postprocess(state: AgentState) -> AgentState:
    from app.websocket.publisher import publish_to_chat
    async def _emit(event: Dict[str, Any]) -> None:
        if callable(state.send_event):
            await state.send_event(event)
        elif state.ws_chat_id:
            await publish_to_chat(state.ws_chat_id, event)
            
    await _emit({
        "type": "ai_response_complete",
        "data": {
            "message": "Workflow complete.",
            "project_id": state.project_id,
        },
    })
    return state
