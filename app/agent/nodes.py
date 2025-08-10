from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.agent.state import AgentState
import logging
import re
import json
from app.services.ai_service import ai_service
from app.agent.commands import detect_rename_title, apply_prd_title_rename
from app.agent.intent import classify_intent, verify_is_answer
from functools import lru_cache
from pathlib import Path
import time
from datetime import datetime
try:
    from langgraph.types import interrupt  # type: ignore
except Exception:  # pragma: no cover - local dev fallback if langgraph not installed
    def interrupt(payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"type": "accept"}


def prepare_context(state: AgentState) -> AgentState:
    """Trim and normalize inputs to keep within token/size budgets."""
    # Keep at most 10 messages
    state.last_messages = (state.last_messages or [])[-10:]

    # Do not truncate PRD or Mermaid; send full artifacts downstream

    # Limit attachments and prefer preview text
    if state.attachments:
        trimmed: List[Dict[str, Any]] = []
        for att in state.attachments[:3]:
            trimmed.append({
                "name": att.get("name"),
                "type": att.get("type"),
                "text": att.get("text") or "",
                "url": att.get("url") if not att.get("text") else None,
            })
        state.attachments = trimmed

    return state


@lru_cache(maxsize=1)
def _load_prd_template_text() -> str:
    """Load the canonical PRD template Markdown from backend/docs/PRD_template.md."""
    try:
        docs_path = Path(__file__).resolve().parents[2] / "docs" / "PRD_template.md"
        return docs_path.read_text(encoding="utf-8")
    except Exception as e:
        # Fail-safe minimal template if file missing
        return (
            "# Product Requirement Document: {TITLE}\n\n"
            "### 1. Product Overview / Purpose\n\n"
            "### 2. Objective and Goals\n\n"
            "### 3. User Personas\n\n"
            "### 4. User Needs & Requirements\n\n"
            "### 5. Features & Functional Requirements\n\n"
            "### 6. Non-Functional Requirements\n\n"
            "### 7. User Stories / UX Flow / Design Notes\n\n"
            "### 8. Technical Specifications\n\n"
            "### 9. Assumptions, Constraints & Dependencies\n\n"
            "### 10. Timeline & Milestones\n\n"
            "### 11. Success Metrics / KPIs\n\n"
            "### 12. Release Criteria\n\n"
            "### 13. Open Questions / Issues\n\n"
            "### 14. Budget and Resources\n"
        )


def _normalize_section_name(name: str) -> str:
    t = (name or "").strip().lower()
    # collapse whitespace and punctuation variants
    t = re.sub(r"\s+", " ", t)
    t = t.replace("&", "and").replace("—", "-")
    return t


def _parse_sections_from_markdown(md: str) -> List[Tuple[str, int, int]]:
    """Parse sections as (heading_text, start_idx, end_idx). Excludes top-level H1 title."""
    lines = (md or "").splitlines()
    sections: List[Tuple[str, int, int]] = []
    current_head: Optional[str] = None
    current_start: Optional[int] = None
    for i, line in enumerate(lines):
        if line.strip().startswith("###"):
            # close previous
            if current_head is not None and current_start is not None:
                sections.append((current_head, current_start, i))
            current_head = line.strip().lstrip("# ")
            current_start = i + 1
    if current_head is not None and current_start is not None:
        sections.append((current_head, current_start, len(lines)))
    return sections


def _canonicalize_heading_key(text: str) -> str:
    """Canonicalize a section heading to improve matching against template entries.

    Rules:
    - Strip leading '#' and whitespace
    - Drop any inline description after ':'
    - Collapse consecutive whitespace
    - Replace '&' with 'and'; normalize dashes
    - Lowercase result
    Example:
      "1.Product Overview / Purpose: Summary ..." -> "1. product overview / purpose"
      "1. Product Overview / Purpose" -> "1. product overview / purpose"
    """
    if not text:
        return ""
    # Remove leading hashes/spaces
    t = text.lstrip('# ').strip()
    # Keep only part before ':' (inline description removed)
    if ':' in t:
        t = t.split(':', 1)[0]
    # Normalize punctuation variants
    t = t.replace('—', '-').replace('&', 'and')
    # Ensure there is a space after a leading number like "1.Product" -> "1. Product"
    t = re.sub(r'^(\d+)\.(\S)', r'\1. \2', t)
    # Collapse whitespace
    t = re.sub(r'\s+', ' ', t)
    return t.strip().lower()


def _compute_sections_status(prd_markdown: str, target_order: List[str]) -> Dict[str, bool]:
    sections = _parse_sections_from_markdown(prd_markdown)
    # Build map: normalized section key -> content length
    content_by_key: Dict[str, int] = {}
    lines = (prd_markdown or "").splitlines()
    for head, s, e in sections:
        # Remove the heading line content from body length
        body = "\n".join(lines[s:e]).strip()
        key = _canonicalize_heading_key(head)
        content_by_key[key] = len(body)
    result: Dict[str, bool] = {}
    for sec in target_order:
        key = _canonicalize_heading_key(sec)
        length = content_by_key.get(key, 0)
        # Consider a section complete if it has >= 60 non-whitespace chars
        result[sec] = bool(length >= 60)
    return result


def _find_section_range(prd_markdown: str, target_section: str) -> Tuple[int, int, List[str], str]:
    """Find the start/end (line indices) for the BODY of a target section.

    Returns (start_idx, end_idx, lines, found_heading)
    - start_idx/end_idx are indices into lines[] where body is located (exclusive of heading)
    - If not found, returns (-1, -1, lines, "")
    """
    lines = (prd_markdown or "").splitlines()
    target_key = _canonicalize_heading_key(target_section)
    for head, s, e in _parse_sections_from_markdown(prd_markdown):
        if _canonicalize_heading_key(head) == target_key:
            return s, e, lines, head
    return -1, -1, lines, ""


def _sanitize_section_body(markdown_body: str) -> str:
    """Normalize model output that should be ONLY the body of a section.

    - Strip surrounding fences/backticks
    - Remove any accidental headings
    - Trim excessive blank lines
    """
    body = (markdown_body or "").strip()
    if body.startswith("```"):
        body = body.strip("`\n ")
        # common case of ```markdown
        if body.lower().startswith("markdown"):
            body = body[len("markdown"):].lstrip()
    # Drop any leading markdown headings accidentally emitted
    cleaned_lines: List[str] = []
    for raw in body.splitlines():
        line = raw.rstrip()
        if line.lstrip().startswith("### ") or line.lstrip().startswith("## ") or line.lstrip().startswith("# "):
            continue
        cleaned_lines.append(line)
    # Collapse excessive blank lines
    out: List[str] = []
    prev_blank = False
    for line in cleaned_lines:
        is_blank = (line.strip() == "")
        if is_blank and prev_blank:
            continue
        out.append(line)
        prev_blank = is_blank
    return "\n".join(out).strip()


def _replace_section_body(prd_markdown: str, target_section: str, new_body: str) -> str:
    """Replace ONLY the body of the target section, preserving all other text.

    If the section is not found, append it to the end using the exact template heading.
    """
    start, end, lines, found_heading = _find_section_range(prd_markdown, target_section)
    body = _sanitize_section_body(new_body)
    if start == -1:
        # Append new section at the end
        block = [f"### {target_section}", "", body, ""]
        out = ("\n".join(lines) + ("\n" if not prd_markdown.endswith("\n") else "")) + "\n".join(block)
        return out
    # Replace content between start and end with sanitized body
    before = lines[:start]
    after = lines[end:]
    new_block: List[str] = [body, ""] if body else [""]
    updated = before + new_block + after
    # Preserve final newline if original had it
    result = "\n".join(updated)
    if prd_markdown.endswith("\n") and not result.endswith("\n"):
        result += "\n"
    return result


def _section_to_lens(section: str) -> str:
    key = _canonicalize_heading_key(section)
    if key.startswith("1. "):
        return "discovery"
    if key.startswith("2. "):
        return "metrics"
    if key.startswith("3. "):
        return "discovery"
    if key.startswith("4. "):
        return "discovery"
    if key.startswith("5. "):
        return "user_journey"
    if key.startswith("6. "):
        return "risks"
    if key.startswith("7. "):
        return "user_journey"
    if key.startswith("8. "):
        return "risks"
    if key.startswith("9. "):
        return "risks"
    if key.startswith("10. "):
        return "gtm"
    if key.startswith("11. "):
        return "metrics"
    if key.startswith("12. "):
        return "metrics"
    if key.startswith("13. "):
        return "risks"
    if key.startswith("14. "):
        return "gtm"
    return "discovery"


def _detect_domain(idea_text: str) -> tuple[str, str]:
    t = (idea_text or "").lower()
    # Heuristic domain detection
    dev_markers = ("cursor", "code", "coding", "ide", "editor", "repo", "git", "developer", "copilot")
    fitness_markers = ("fitness", "workout", "health", "steps", "calorie", "weight")
    if any(m in t for m in dev_markers):
        return ("ai_coding_assistant", "an AI coding assistant for developers")
    if any(m in t for m in fitness_markers):
        return ("fitness_tracker", "a fitness tracker")
    return ("general", "a product")


def _domain_hints(domain_slug: str) -> str:
    if domain_slug == "ai_coding_assistant":
        return "languages, editors, latency targets, repo indexing, privacy, model choice, context sources, on-device vs cloud"
    if domain_slug == "fitness_tracker":
        return "primary goals, devices, sensors, integrations, data privacy, battery life, sync reliability"
    return "key decisions, scope, constraints relevant to the product domain"


def _domain_specific_fallback_question(section: str, domain_slug: str) -> str:
    s = _canonicalize_heading_key(section)
    if domain_slug == "ai_coding_assistant":
        if s.startswith("1. "):
            return "Which developer workflows will v1 focus on (e.g., inline completions, refactor, chat in editor)?"
        if s.startswith("2. "):
            return "What 2–3 measurable v1 targets will we hit (e.g., p95 latency ≤ 200ms, ≥ 60% suggestion acceptance)?"
        if s.startswith("3. "):
            return "Which roles/editors are in scope first (e.g., JS full‑stack in VS Code, Python ML in JetBrains)?"
        if s.startswith("4. "):
            return "What top pain points in the coding flow must v1 solve (e.g., context hopping, test scaffolding)?"
        if s.startswith("5. "):
            return "Which must‑have v1 features and language/editor scope are required to deliver value?"
        if s.startswith("6. "):
            return "What latency, privacy, and enterprise constraints must v1 meet (p95 targets, data boundaries)?"
        if s.startswith("7. "):
            return "What is the happy‑path from writing code to applying suggestions and verifying changes?"
        if s.startswith("8. "):
            return "Which context sources, model family, and indexing approach will we use (repo size, monorepos)?"
        if s.startswith("11. "):
            return "Which success metrics matter most (suggestion acceptance, task time saved, p95 latency, crash‑free %)?"
        if s.startswith("12. "):
            return "What release criteria must v1 pass (supported languages/editors, p95 latency, reliability)?"
    if domain_slug == "fitness_tracker":
        if s.startswith("1. "):
            return "Which primary use case will v1 target (weight loss, cardio, strength) and on which platforms?"
        if s.startswith("2. "):
            return "What 2–3 measurable v1 goals will we track (DAU, day‑7 retention, goal adherence)?"
        if s.startswith("3. "):
            return "Who are the first personas (e.g., beginners, busy professionals) and their key constraints?"
        if s.startswith("5. "):
            return "Which must‑have v1 features deliver value (activity tracking, goals, reminders, social)?"
        if s.startswith("6. "):
            return "What NFRs matter most (battery life, data privacy, sync reliability)?"
        if s.startswith("8. "):
            return "Which sensors and integrations are needed (Apple Health/Google Fit), and what tech stack?"
        if s.startswith("11. "):
            return "Which metrics define success (active days/week, goal adherence, retention)?"
    # General fallback by lens
    lens = _section_to_lens(section)
    if lens == "metrics" and s.startswith("2. "):
        return "What 2–3 measurable v1 targets will define success (include numeric thresholds and timeframe)?"
    if lens == "user_journey" and s.startswith("5. "):
        return "Which 5–8 core features are essential for v1 and why?"
    if lens == "discovery" and s.startswith("1. "):
        return "What core value does the product deliver and who is the primary user?"
    return f"{section}: what specific details should we capture here?"

def _system_prompt_initial_outline(template_text: str, title: str) -> str:
    """Prompt the model to output ONLY the outline based on the canonical template."""
    return (
        "You are a product manager generating an initial PRD outline.\n"
        "Rules:\n"
        "- Begin with exactly one H1 line: '# Product Requirement Document: "
        + title.replace("\n", " ").strip()
        + "'\n"
        "- Then output the sections exactly as in the template below.\n"
        "- Do NOT add any commentary before or after.\n"
        "- Do NOT include code fences.\n\n"
        "Template (copy sections and headings verbatim; it's okay if bodies are empty initially):\n\n"
        + template_text
    )


def _system_prompt_refine() -> str:
    return (
        "You will update only ONE section of the PRD based on the user's answer.\n"
        "- Preserve the existing structure and all headings exactly.\n"
        "- Replace ONLY the body of the target section with detailed, concrete content.\n"
        "- Output the FULL updated PRD Markdown.\n"
        "- Do NOT add any extra commentary."
    )


async def generate_prd(state: AgentState) -> AgentState:
    """Generate PRD OUTLINE ONLY and stream preface (if any) to chat, PRD to editor."""
    from app.websocket.publisher import publish_to_chat

    async def _emit(event: Dict[str, Any]) -> None:
        if callable(state.send_event):
            await state.send_event(event)  # type: ignore
        elif state.ws_chat_id:
            await publish_to_chat(state.ws_chat_id, event)

    await _emit({"type": "stream_start", "data": {"project_id": state.project_id}})

    # Build outline-only prompt
    template_text = _load_prd_template_text()
    derived_title = (state.idea or "Untitled").strip()[:120]
    system_msg = {
        "role": "system",
        "content": _system_prompt_initial_outline(template_text, derived_title),
    }
    user_msg = {"role": "user", "content": f"Idea: {state.idea or ''}"}
    messages = [system_msg, user_msg]

    prd_accum = ""
    seen_prd_heading = False
    last_provider: str | None = None
    last_model: str | None = None
    import time
    start_time = time.time()
    # Simple pacing for artifacts_preview
    last_emit_len = 0
    async for chunk in ai_service.generate_stream(
        user_id=state.user_id,
        messages=messages,
        temperature=0.2,
        max_tokens=1200,
    ):
        if chunk.is_complete:
            # Close any open chat stream
            await _emit({
                "type": "ai_response_streaming",
                "data": {"delta": "", "is_complete": True, "provider": chunk.provider, "model": chunk.model},
            })
            break
        delta = chunk.content
        last_provider = chunk.provider
        last_model = chunk.model
        if not seen_prd_heading:
            # Detect the PRD heading boundary
            if "# Product Requirement Document:" in (prd_accum + delta):
                seen_prd_heading = True
                # Compute from first heading occurrence
                combined = prd_accum + delta
                idx = combined.find("# Product Requirement Document:")
                prd_accum = combined[idx:]
                # Stop chat streaming from now on; start editor updates immediately
                await _emit({
                    "type": "artifacts_preview",
                    "data": {"prd_markdown": prd_accum, "mermaid": None},
                })
                last_emit_len = len(prd_accum)
            else:
                # Forward preface delta to chat
                await _emit({
                    "type": "ai_response_streaming",
                    "data": {"delta": delta, "is_complete": False, "provider": chunk.provider, "model": chunk.model},
                })
                prd_accum += delta
        else:
            prd_accum += delta
            # Throttle: emit only if significant growth
            if len(prd_accum) - last_emit_len >= 80:
                await _emit({
                    "type": "artifacts_preview",
                    "data": {"prd_markdown": prd_accum, "mermaid": None},
                })
                last_emit_len = len(prd_accum)

    # Finalize
    state.prd_markdown = prd_accum
    # Emit final artifacts + sections status
    sections_order = state.completion_targets.get("sections", []) if isinstance(state.completion_targets, dict) else []
    sec_status = _compute_sections_status(state.prd_markdown, sections_order)
    state.sections_status = sec_status
    await _emit({
        "type": "artifacts_preview",
        "data": {
            "prd_markdown": state.prd_markdown,
            "mermaid": None,
            "sections_status": sec_status,
        },
    })
    # Mark end of this streaming segment so UI can re-enable input promptly
    await _emit({
        "type": "ai_response_complete",
        "data": {"message": "Outline generated"},
    })
    try:
        elapsed_ms = int((time.time() - start_time) * 1000)
        state.telemetry = {
            "provider": last_provider,
            "model": last_model,
            "response_time_ms": elapsed_ms,
        }
    except Exception:
        pass
    return state


async def generate_mermaid(state: AgentState) -> AgentState:
    """Generate Mermaid code (non-stream for simplicity).

    Supports two entry modes:
    - PRD-run path: invoked after PRD work; uses state.prd_markdown
    - Flowchart-only path: invoked via flowchart agent; uses state.base_prd_markdown/state.prd_markdown
      and may include state.base_mermaid as a prior diagram to update.
    """
    from app.websocket.publisher import publish_to_chat

    async def _emit(event: Dict[str, Any]) -> None:
        if callable(state.send_event):
            await state.send_event(event)  # type: ignore
        elif state.ws_chat_id:
            await publish_to_chat(state.ws_chat_id, event)
    logger = logging.getLogger(__name__)

    owner_prompt = (
        "You are an experienced Product Manager and Technical Writer skilled at translating Product "
        "Requirements Documents (PRDs) into clear, accurate, and professional Mermaid flowcharts. "
        "Your goal is to take a PRD.md file and produce a valid Mermaid `flowchart TD` diagram that faithfully "
        "represents the process, systems, and interactions described.\n\n"
        "Follow these rules and guidelines:\n\n"
        "1. Purpose – read the PRD, identify actors/systems/agents/processes/states/data flows; produce a clear top-down diagram.\n"
        "2. Structure – use Mermaid `flowchart TD`; group nodes with subgraphs; include user actions, system processes, and data persistence nodes; include loops/branches if described; reflect happy path and key secondary flows.\n"
        "3. Filling Gaps – make minimal, obvious assumptions only to maintain continuity (e.g., input validation).\n"
        "4. Accuracy – reflect actual structure/order; use PRD terminology; keep component names consistent with the PRD.\n"
        "5. Readability – concise node names; directional arrows; use labeled edges where helpful; split into subgraphs for clarity.\n"
        "6. Output Requirements – ONLY output Mermaid code enclosed in triple backticks as ```mermaid ... ```; ensure valid syntax that renders.\n"
        "7. Tone – choose clarity over complexity; resolve minor ambiguities pragmatically."
    )

    prior_mermaid = (state.base_mermaid or "").strip()
    prd_text = state.prd_markdown or state.base_prd_markdown or ""

    system_msg = {"role": "system", "content": owner_prompt}
    if prior_mermaid:
        user_content = (
            "PRD markdown (full):\n" + prd_text + "\n\n"
            "Previous Mermaid (update/refine to match the PRD; preserve structure/IDs where sensible):\n"
            + prior_mermaid
        )
    else:
        user_content = "PRD markdown (full):\n" + prd_text
    user_msg = {"role": "user", "content": user_content}

    logger.info("[mermaid] generating diagram: prd_len=%s user_id=%s", len(state.prd_markdown or ""), state.user_id)
    # Helper: try to generate mermaid given a PRD text
    async def _gen_from_prd(prd: str) -> str:
        sm = {"role": "system", "content": owner_prompt}
        um = {"role": "user", "content": user_content.replace(prd_text, prd, 1)}
        resp = await ai_service.generate_response(
            user_id=state.user_id,
            messages=[sm, um],
            temperature=0.2,
            max_tokens=1200,
            use_cache=False,
        )
        return (resp.content or "").strip()

    # Try direct generation; on provider error (e.g., length), summarize PRD and retry once
    try:
        response = await ai_service.generate_response(
            user_id=state.user_id,
            messages=[system_msg, user_msg],
            temperature=0.2,
            max_tokens=1200,
            use_cache=False,
        )
        raw = response.content or ""
        summarized_used = False
    except Exception as e:
        # Summarization fallback (Phase 3): reduce PRD, then regenerate
        logger.warning("[mermaid] initial generation failed, attempting summarization fallback: %s", e)
        try:
            sum_system = {
                "role": "system",
                "content": (
                    "Summarize the following PRD into concise plain text focusing on key actors, "
                    "systems, interactions, and flows. Keep structure/order. No code fences, no markdown, <= 1200 tokens."
                ),
            }
            sum_user = {"role": "user", "content": prd_text[:24000]}
            sum_resp = await ai_service.generate_response(
                user_id=state.user_id,
                messages=[sum_system, sum_user],
                temperature=0.0,
                max_tokens=1000,
                use_cache=False,
            )
            summarized = (sum_resp.content or prd_text).strip()
            raw = await _gen_from_prd(summarized)
            summarized_used = True
            try:
                # annotate telemetry for postprocess message
                state.telemetry = {**(state.telemetry or {}), "flowchart_note": "Flowchart derived from summarized PRD due to size"}
            except Exception:
                pass
        except Exception as ee:
            logger.error("[mermaid] summarization fallback failed: %s", ee)
            raw = ""
            summarized_used = False

    # Continue with normalization pipeline on 'raw'
    logger.info("[mermaid] raw_len=%s prefix=%r", len(raw), raw[:60])
    mermaid = raw.strip()
    if mermaid.startswith("```"):
        mermaid = mermaid.strip("`\n ")
        if mermaid.startswith("mermaid"):
            mermaid = mermaid[len("mermaid"):].lstrip()
    logger.info("[mermaid] normalized_len=%s prefix=%r", len(mermaid), mermaid[:60])

    def looks_like_mermaid(text: str) -> bool:
        if not text:
            return False
        head = text.strip().lower()
        if head.startswith("graph ") or head.startswith("flowchart "):
            return True
        # other valid diagram types
        for kw in ("sequencediagram", "classdiagram", "gantt", "stateDiagram-v2", "erDiagram", "journey", "pie", "mindmap"):
            if head.startswith(kw.lower()):
                return True
        return False

    def looks_like_c4(text: str) -> bool:
        c4_markers = ("c4context", "c4container", "c4component", "person(", "container(", "system_boundary", "rel(")
        t = text.strip().lower()
        return any(m in t for m in c4_markers)

    # Retry strategy if empty or not valid mermaid (e.g., C4 DSL)
    if not mermaid or not looks_like_mermaid(mermaid) or looks_like_c4(mermaid):
      if looks_like_c4(mermaid):
          logger.warning("[mermaid] detected C4/Structurizr-like output; retrying for raw Mermaid flowchart")
      logger.warning("[mermaid] empty output on first attempt; retrying with stricter prompt and no cache")
      retry_system = {
          "role": "system",
          "content": (
              "You are a helpful systems architect. Output ONLY raw Mermaid flowchart code. "
              "Start with 'graph TD' (or 'graph LR'). Do NOT use C4, PlantUML, Structurizr DSL, titles, or backticks. "
              "No prose. Only the diagram."
          ),
      }
      retry_user = {
          "role": "user",
          "content": (
              "Derive a Mermaid flowchart from this PRD. Use 'flowchart TD' and no backticks. "
              "Do not use C4/PlantUML/Structurizr. Reflect the PRD accurately.\n\n" 
              f"PRD:\n{prd_text[:12000]}" + (f"\n\nPrevious Mermaid (optional):\n{prior_mermaid}" if prior_mermaid else "")
          ),
      }
      try:
        retry_resp = await ai_service.generate_response(
            user_id=state.user_id,
            messages=[retry_system, retry_user],
            temperature=0.1,
            max_tokens=1200,
            use_cache=False,
        )
        raw2 = retry_resp.content or ""
        logger.info("[mermaid] retry raw_len=%s prefix=%r", len(raw2), raw2[:60])
        m2 = raw2.strip()
        if m2.startswith("```"):
            m2 = m2.strip("`\n ")
            if m2.startswith("mermaid"):
                m2 = m2[len("mermaid"):].lstrip()
        logger.info("[mermaid] retry normalized_len=%s prefix=%r", len(m2), m2[:60])
        mermaid = m2
      except Exception as e:
        logger.error("[mermaid] retry failed: %s", e)

    # Final fallback: ensure we don't overwrite with empty; add minimal but valid default
    if mermaid and looks_like_mermaid(mermaid) and not looks_like_c4(mermaid):
        state.mermaid = mermaid
    else:
        logger.warning("[mermaid] still empty/invalid after retry; using minimal default")
        state.mermaid = "flowchart TD; U[User]-->F[Frontend]; F-->B[Backend]; B-->D[(Database)]"

    await _emit({
        "type": "artifacts_preview",
        "data": {
            "prd_markdown": state.prd_markdown,
            "mermaid": state.mermaid,
            "thinking_lens_status": state.coverage_status or {
                "discovery": True,
                "user_journey": True,
                "metrics": True,
                "gtm": True,
                "risks": True,
            },
            "sections_status": state.sections_status or {},
        },
    })
    return state


async def postprocess(state: AgentState) -> AgentState:
    from app.websocket.publisher import publish_to_chat

    async def _emit(event: Dict[str, Any]) -> None:
        if callable(state.send_event):
            await state.send_event(event)  # type: ignore
        elif state.ws_chat_id:
            await publish_to_chat(state.ws_chat_id, event)
    # Best-effort enrichment from telemetry if present
    provider = state.telemetry.get("provider") if isinstance(state.telemetry, dict) else None
    model = state.telemetry.get("model") if isinstance(state.telemetry, dict) else None
    response_time_ms = state.telemetry.get("response_time_ms") if isinstance(state.telemetry, dict) else None

    await _emit({
        "type": "ai_response_complete",
        "data": {
            "message": "Artifacts generated",
            "project_id": state.project_id,
            "provider": provider,
            "model": model,
            "response_time_ms": response_time_ms,
        },
    })
    return state


# ----------------------
# HITL loop nodes (Phase 2)
# ----------------------

def analyze_gaps(state: AgentState) -> AgentState:
    """Compute template-section completeness and determine whether to finish or continue."""
    full_text = state.prd_markdown or ""
    sections_order: List[str] = state.completion_targets.get("sections", []) if isinstance(state.completion_targets, dict) else []
    max_q = int(state.completion_targets.get("max_questions", 14)) if isinstance(state.completion_targets, dict) else 14

    sec_status = _compute_sections_status(full_text, sections_order)
    state.sections_status = sec_status
    sections_ok = all(sec_status.values()) if sec_status else False

    forced_finish = bool((state.telemetry or {}).get("force_finish"))
    min_q = int(state.completion_targets.get("min_questions_before_finish", 0)) if isinstance(state.completion_targets, dict) else 0
    have_min = state.iteration_index >= max(min_q, 0)
    hit_limit = state.iteration_index >= max_q
    # Require at least min_q Q/A incorporations unless explicitly forced to finish
    should_finish = forced_finish or (sections_ok and have_min) or hit_limit

    state.telemetry = {**(state.telemetry or {}), "sections_ok": sections_ok, "should_finish": should_finish}
    return state


async def propose_next_question(state: AgentState) -> AgentState:
    """Select the earliest incomplete PRD section and ask a guided question with required phrasing."""
    # If there is already a pending question (including a confirmation), do not propose a new one
    if state.pending_question:
        return state
    if (state.telemetry or {}).get("should_finish"):
        state.pending_question = None
        return state

    order: List[str] = state.completion_targets.get("sections", []) if isinstance(state.completion_targets, dict) else []
    status = state.sections_status or {}
    target: Optional[str] = None
    for sec in order:
        if not bool(status.get(sec)):
            target = sec
            break
    if not target:
        state.pending_question = None
        return state

    # Deterministic base as fallback
    qmap: Dict[str, str] = {
        "1. Product Overview / Purpose": "what is the core product and primary value proposition, and who is it primarily for?",
        "2. Objective and Goals": "what are the 2–3 concrete goals for the first release, and how will we know we've succeeded?",
        "3. User Personas": "which user personas will use this and what defining traits or roles do they have?",
        "4. User Needs & Requirements": "what specific user problems or needs must be addressed in v1?",
        "5. Features & Functional Requirements": "which 5–8 core features must be included, with a brief description each?",
        "6. Non-Functional Requirements": "what performance, reliability, and security expectations should we set?",
        "7. User Stories / UX Flow / Design Notes": "what is the happy-path user flow from entry to success, step-by-step?",
        "8. Technical Specifications": "what is the preferred tech stack or key integrations we should plan for?",
        "9. Assumptions, Constraints & Dependencies": "are there known constraints, dependencies, or assumptions we must work within?",
        "10. Timeline & Milestones": "what is the desired timeline and high-level milestones?",
        "11. Success Metrics / KPIs": "what primary success metrics and guardrails should we track?",
        "12. Release Criteria": "what functionality, performance, and reliability criteria are required for launch?",
        "13. Open Questions / Issues": "what outstanding decisions or risks should we call out now?",
        "14. Budget and Resources": "what budget or resource constraints should we assume?",
    }
    base = qmap.get(target, f"what details do you want to include in the {target} section?")

    # Try generating a tailored, idea-specific question using current context
    idea_text = (state.idea or "").strip()
    try:
        s_idx, e_idx, lines, _h = _find_section_range(state.prd_markdown, target)
        existing_body = "\n".join(lines[s_idx:e_idx]).strip() if s_idx != -1 else ""
    except Exception:
        existing_body = ""
    # Include last 2 Q/As as context
    recent_qa = state.answered_qa[-2:] if isinstance(state.answered_qa, list) else []

    def _is_valid_question(text: str) -> bool:
        t = (text or "").strip()
        if not t or t == "?":
            return False
        # must include at least one alphabetic character
        if not any(ch.isalpha() for ch in t):
            return False
        # minimal length to avoid trivial outputs
        if len(t) < 12:
            return False
        return True

    def _tailored_fallback(idea: str, sec: str, generic: str) -> str:
        idea_short = (idea or "this product").strip()
        if len(idea_short) > 60:
            idea_short = idea_short[:60].rstrip() + "…"
        # Domain- and lens-aware fallback
        domain_slug, domain_label = _detect_domain(idea)
        domain_q = _domain_specific_fallback_question(sec, domain_slug)
        if domain_q and not domain_q.startswith(sec):
            return domain_q
        # Default tailored wrapper
        return f"{sec}: For '{idea_short}', {generic}"

    specialized_q: Optional[str] = None
    try:
        sys = {
            "role": "system",
            "content": (
                "Write ONE specific, concise question to elicit the most useful information to complete the given PRD section. "
                "Tailor it to the product idea and current draft. Constraints: 1 question only; ≤ 160 chars; no preface, no numbering, no quotes; end with '?'."
            ),
        }
        ctx_parts: List[str] = []
        if idea_text:
            ctx_parts.append(f"Idea: {idea_text}")
        ctx_parts.append(f"Section: {target}")
        if existing_body:
            body_trim = existing_body.strip()
            if len(body_trim) > 700:
                body_trim = body_trim[:700]
            ctx_parts.append(f"Existing section draft:\n{body_trim}")
        if recent_qa:
            # format the last QAs compactly
            qa_lines = []
            for qa in recent_qa:
                q = (qa.get("question") or "").strip()
                a = (qa.get("answer") or "").strip()
                if q and a:
                    qa_lines.append(f"- Q: {q} | A: {a}")
            if qa_lines:
                ctx_parts.append("Recent answers:\n" + "\n".join(qa_lines))
        usr = {"role": "user", "content": "\n\n".join(ctx_parts)}
        resp = await ai_service.generate_response(
            user_id=state.user_id,
            messages=[sys, usr],
            temperature=0.2,
            max_tokens=80,
            use_cache=False,
        )
        candidate = (resp.content or "").strip()
        # sanitize
        candidate = candidate.strip().strip('\'"')  # remove stray single/double quotes at both ends
        # If multiple lines, take first non-empty line
        if "\n" in candidate:
            for line in candidate.splitlines():
                lt = line.strip()
                if lt:
                    candidate = lt
                    break
        if not candidate.endswith("?"):
            candidate = candidate.rstrip(". ") + "?"
        if len(candidate) > 200:
            candidate = candidate[:200].rstrip() + "?"
        # Validate; reject trivial outputs like '?' or empty
        specialized_q = candidate if _is_valid_question(candidate) else None
    except Exception:
        specialized_q = None

    if specialized_q:
        phrased = specialized_q
    else:
        phrased = _tailored_fallback(idea_text, target, base) if idea_text else f"{target}: {base}"
    qid = f"sec_{order.index(target)+1}"
    state.pending_question = {
        "id": qid,
        "question": phrased,
        "section": target,
        "rationale": f"Fill out the '{target}' section with concrete details.",
    }
    state.pending_question_kind = "section"
    state.last_section_target = target
    state.last_section_question = phrased
    state.last_section_qid = qid
    return state


async def await_human_answer(state: AgentState) -> AgentState:
    """Pause the graph and request human input using LangGraph interrupt().

    Enhancement: classify the next user message before proceeding. If the user
    sends a generic query instead of a direct answer, address it first and ask
    permission to proceed with the pending section question.
    """
    if not state.pending_question:
        return state
    payload = {
        "type": "question",
        "question": state.pending_question.get("question"),
        "section": state.pending_question.get("section"),
        "question_id": state.pending_question.get("id"),
        "rationale": state.pending_question.get("rationale"),
    }
    # Notify client over WebSocket before pausing execution
    try:
        if callable(state.send_event):
            await state.send_event({  # type: ignore
                "type": "agent_interrupt_request",
                "data": payload,
            })
        else:
            from app.websocket.publisher import publish_to_chat
            if state.ws_chat_id:
                await publish_to_chat(state.ws_chat_id, {"type": "agent_interrupt_request", "data": payload})
        # Also emit the assistant question as a chat message exactly once, with stable id
        try:
            qid = str(state.pending_question.get("id") or "")
            qtext = str(state.pending_question.get("question") or "").strip()
            if qid and qtext and state.ws_chat_id:
                from app.websocket.publisher import publish_to_chat
                await publish_to_chat(state.ws_chat_id, {
                    "type": "message_sent",
                    "data": {
                        "message_id": f"q:{qid}",
                        "user_id": state.user_id,
                        "content": qtext,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "message_type": "assistant",
                    },
                })
        except Exception:
            pass
    except Exception:
        pass
    # Track asked question for de-duplication before interrupting
    try:
        qtext = state.pending_question.get("question", "") if state.pending_question else ""
        if qtext:
            if not isinstance(state.asked_questions, list):
                state.asked_questions = []
            state.asked_questions.append(qtext)
    except Exception:
        pass

    response = interrupt(payload)
    # Expected resume shapes (extended):
    # - {"type": "answer", "question_id": str, "text": str}
    # - {"type": "accept", "finish"?: bool}
    # - {"type": "message", "text": str}  # generic/free-form message
    state.telemetry = {**(state.telemetry or {}), "last_interrupt": payload}

    def _looks_like_direct_answer(text: str, section: str, question: str) -> bool:
        t = (text or "").strip()
        tl = t.lower()
        # must have enough content and not be a question
        if len(tl) < 12:
            return False
        if tl.endswith("?"):
            return False
        # filter common command/generic prefixes
        command_starts = (
            "rename ", "change ", "set ", "summarize ", "make ", "can you ", "please ", "could you ", "what is ", "how ",
        )
        if tl.startswith(command_starts):
            return False
        # avoid pure confirmations
        if tl in {"ok", "okay", "yes", "no", "maybe", "sure", "fine", "later"}:
            return False
        # don't treat a pure echo of the question as an answer
        if question:
            ql = question.strip().lower().rstrip("? .")
            if ql and ql in tl and len(t.split()) < 8:
                return False
        # light section keyword check for metrics
        key = _canonicalize_heading_key(section)
        if key.startswith("11. "):
            if ("kpi" not in tl) and ("metric" not in tl) and ("target" not in tl):
                return False
        return True

    if isinstance(response, dict) and response.get("type") == "answer":
        # Direct answer path (explicit)
        try:
            from app.websocket.publisher import publish_to_chat
            if state.ws_chat_id:
                qid_clear = (state.pending_question.get("id") if state.pending_question else None) or state.last_section_qid or ""
                if qid_clear:
                    await publish_to_chat(state.ws_chat_id, {"type": "agent_interrupt_cleared", "data": {"question_id": qid_clear}})
        except Exception:
            pass
        state.answered_qa.append({
            "question": state.pending_question.get("question", ""),
            "answer": response.get("text", ""),
            "section": state.pending_question.get("section"),
        })
        state.pending_question = None
        state.pending_question_kind = None
        return state

    if isinstance(response, dict) and response.get("type") == "accept":
        # User chose to skip/finish
        if response.get("finish"):
            state.telemetry = {**(state.telemetry or {}), "force_finish": True}
        try:
            from app.websocket.publisher import publish_to_chat
            if state.ws_chat_id:
                qid_clear = (state.pending_question.get("id") if state.pending_question else None) or state.last_section_qid or ""
                if qid_clear:
                    await publish_to_chat(state.ws_chat_id, {"type": "agent_interrupt_cleared", "data": {"question_id": qid_clear}})
        except Exception:
            pass
        state.pending_question = None
        state.pending_question_kind = None
        return state

    # Fallback: treat as a generic message requiring classification/handling
    if isinstance(response, dict) and response.get("type") == "message":
        from app.websocket.publisher import publish_to_chat  # local import to avoid cycles

        def _looks_like_yes(text: str) -> bool:
            t = (text or "").strip().lower()
            return t in {"yes", "y", "ok", "okay", "sure", "yep", "proceed", "continue", "go ahead", "do it"} or t.startswith("yes") or t.startswith("ok")

        def _looks_like_no(text: str) -> bool:
            t = (text or "").strip().lower()
            return t in {"no", "n", "skip", "later", "not now", "stop"} or t.startswith("no ") or "not now" in t

        # Keep looping interrupts until we either collect an answer or the user declines
        current = response
        while isinstance(current, dict) and current.get("type") == "message":
            user_text = str(current.get("text") or "").strip()
            state.last_user_message = user_text
            sec = state.pending_question.get("section") if state.pending_question else (state.last_section_target or "")
            qtxt = state.pending_question.get("question") if state.pending_question else (state.last_section_question or "")
            qid = state.pending_question.get("id") if state.pending_question else (state.last_section_qid or "")

            # Use LLM intent classification first
            try:
                intent, args = await classify_intent(
                    user_id=state.user_id,
                    user_text=user_text,
                    section=sec,
                    question=qtxt,
                    idea=state.idea,
                    recent_qa=state.answered_qa,
                )
            except Exception:
                intent, args = ("generic", {})

            # Handle rename_title intent
            if intent == "rename_title":
                new_title = (args or {}).get("new_title") or detect_rename_title(user_text)
                if not new_title:
                    # Nothing to do; treat as generic
                    intent = "generic"
                else:
                    try:
                        state.prd_markdown = apply_prd_title_rename(state.prd_markdown, str(new_title))
                        if state.ws_chat_id:
                            await publish_to_chat(state.ws_chat_id, {
                                "type": "artifacts_preview",
                                "data": {
                                    "prd_markdown": state.prd_markdown,
                                    "mermaid": None,
                                    "sections_status": state.sections_status or {},
                                },
                            })
                            await publish_to_chat(state.ws_chat_id, {
                                "type": "message_sent",
                                "data": {
                                    "message_id": f"t:{int(time.time()*1000)}",
                                    "user_id": state.user_id,
                                    "content": f"Renamed PRD title to '{new_title}'.",
                                    "timestamp": datetime.utcnow().isoformat() + "Z",
                                    "message_type": "assistant",
                                },
                            })
                        # Ask to proceed with section question
                        proceed_text = f"Would you like to proceed with the PRD question for '{sec}' now?"
                        if state.ws_chat_id:
                            await publish_to_chat(state.ws_chat_id, {
                                "type": "message_sent",
                                "data": {
                                    "message_id": f"p:{qid}",
                                    "user_id": state.user_id,
                                    "content": proceed_text,
                                    "timestamp": datetime.utcnow().isoformat() + "Z",
                                    "message_type": "assistant",
                                },
                            })
                        state.pending_question_kind = "confirm"
                        state.pending_question = {
                            "id": qid,
                            "question": f"Proceed with section '{sec}'?",
                            "section": sec,
                            "rationale": "Confirm before resuming PRD updates.",
                        }
                        current = interrupt({"type": "confirm", "question_id": qid, "section": sec})
                        continue
                    except Exception:
                        # If rename fails, treat as generic below
                        intent = "generic"

            # Confirm intents
            if intent == "confirm_yes" and (state.pending_question_kind or "") == "confirm":
                # Re-ask the original section question and pause
                question_payload = {
                    "type": "question",
                    "question": state.last_section_question or qtxt,
                    "section": state.last_section_target or sec,
                    "question_id": state.last_section_qid or qid,
                    "rationale": state.pending_question.get("rationale") if state.pending_question else None,
                }
                if state.ws_chat_id and (state.last_section_qid or qid):
                    await publish_to_chat(state.ws_chat_id, {"type": "agent_interrupt_request", "data": question_payload})
                    msg_text = (state.last_section_question or qtxt or "").strip()
                    if msg_text:
                        await publish_to_chat(state.ws_chat_id, {
                            "type": "message_sent",
                            "data": {
                                "message_id": f"q:{state.last_section_qid or qid}",
                                "user_id": state.user_id,
                                "content": msg_text,
                                "timestamp": datetime.utcnow().isoformat() + "Z",
                                "message_type": "assistant",
                            },
                        })
                state.pending_question_kind = "section"
                state.pending_question = {
                    "id": state.last_section_qid or qid,
                    "question": state.last_section_question or qtxt,
                    "section": state.last_section_target or sec,
                    "rationale": state.pending_question.get("rationale") if state.pending_question else None,
                }
                current = interrupt(question_payload)
                continue

            if intent == "confirm_no" and (state.pending_question_kind or "") == "confirm":
                if state.ws_chat_id:
                    await publish_to_chat(state.ws_chat_id, {
                        "type": "message_sent",
                        "data": {
                            "message_id": f"a:{int(time.time()*1000)}",
                            "user_id": state.user_id,
                            "content": "No problem — we can continue when you're ready.",
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                            "message_type": "assistant",
                        },
                    })
                state.pending_question = None
                state.pending_question_kind = None
                return state
                try:
                    # Apply rename to current PRD in state and emit artifacts update
                    state.prd_markdown = apply_prd_title_rename(state.prd_markdown, new_title)
                    if state.ws_chat_id:
                        await publish_to_chat(state.ws_chat_id, {
                            "type": "artifacts_preview",
                            "data": {
                                "prd_markdown": state.prd_markdown,
                                "mermaid": None,
                                "sections_status": state.sections_status or {},
                            },
                        })
                        # Also inform chat with a short assistant message
                        await publish_to_chat(state.ws_chat_id, {
                            "type": "message_sent",
                            "data": {
                                "message_id": f"t:{int(time.time()*1000)}",
                                "user_id": state.user_id,
                                "content": f"Renamed PRD title to '{new_title}'.",
                                "timestamp": datetime.utcnow().isoformat() + "Z",
                                "message_type": "assistant",
                            },
                        })
                        # Ask permission to proceed with section question
                        proceed_text = f"Would you like to proceed with the PRD question for '{sec}' now?"
                        await publish_to_chat(state.ws_chat_id, {
                            "type": "message_sent",
                            "data": {
                                "message_id": f"p:{qid}",
                                "user_id": state.user_id,
                                "content": proceed_text,
                                "timestamp": datetime.utcnow().isoformat() + "Z",
                                "message_type": "assistant",
                            },
                        })
                    state.pending_question_kind = "confirm"
                    state.pending_question = {
                        "id": qid,
                        "question": f"Proceed with section '{sec}'?",
                        "section": sec,
                        "rationale": "Confirm before resuming PRD updates.",
                    }
                    current = interrupt({"type": "confirm", "question_id": qid, "section": sec})
                    continue
                except Exception:
                    # On error, continue with generic handling below
                    pass

            if (state.pending_question_kind or "") == "confirm":
                if _looks_like_yes(user_text):
                    # Re-ask the original section question and pause for an explicit answer
                    question_payload = {
                        "type": "question",
                        "question": state.last_section_question or qtxt,
                        "section": state.last_section_target or sec,
                        "question_id": state.last_section_qid or qid,
                        "rationale": state.pending_question.get("rationale") if state.pending_question else None,
                    }
                    if state.ws_chat_id and (state.last_section_qid or qid):
                        await publish_to_chat(state.ws_chat_id, {"type": "agent_interrupt_request", "data": question_payload})
                        msg_text = (state.last_section_question or qtxt or "").strip()
                        if msg_text:
                            await publish_to_chat(state.ws_chat_id, {
                                "type": "message_sent",
                                "data": {
                                    "message_id": f"q:{state.last_section_qid or qid}",
                                    "user_id": state.user_id,
                                    "content": msg_text,
                                    "timestamp": datetime.utcnow().isoformat() + "Z",
                                    "message_type": "assistant",
                                },
                            })
                    # Reset pending to the original section question and pause again
                    state.pending_question_kind = "section"
                    state.pending_question = {
                        "id": state.last_section_qid or qid,
                        "question": state.last_section_question or qtxt,
                        "section": state.last_section_target or sec,
                        "rationale": state.pending_question.get("rationale") if state.pending_question else None,
                    }
                    current = interrupt(question_payload)
                    # Loop back to process the new "current" response
                    continue
                if _looks_like_no(user_text):
                    # Respect user's choice; clear pending and acknowledge
                    if state.ws_chat_id:
                        await publish_to_chat(state.ws_chat_id, {
                            "type": "message_sent",
                            "data": {
                                "message_id": f"a:{int(time.time()*1000)}",
                                "user_id": state.user_id,
                                "content": "No problem — we can continue when you're ready.",
                                "timestamp": datetime.utcnow().isoformat() + "Z",
                                "message_type": "assistant",
                            },
                        })
                        try:
                            qid_clear = state.last_section_qid or qid
                            if qid_clear:
                                await publish_to_chat(state.ws_chat_id, {"type": "agent_interrupt_cleared", "data": {"question_id": qid_clear}})
                        except Exception:
                            pass
                    state.pending_question = None
                    state.pending_question_kind = None
                    return state
                # Ask for explicit confirmation again
                if state.ws_chat_id and (state.last_section_target or sec):
                    tip = f"If you'd like to proceed with the PRD question for '{state.last_section_target or sec}', reply 'yes'. Otherwise ask me anything."
                    await publish_to_chat(state.ws_chat_id, {
                        "type": "message_sent",
                        "data": {
                            "message_id": f"h:{int(time.time()*1000)}",
                            "user_id": state.user_id,
                            "content": tip,
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                            "message_type": "assistant",
                        },
                    })
                current = interrupt({"type": "confirm", "question_id": state.last_section_qid or qid, "section": state.last_section_target or sec})
                continue

            # Section rename intent (update heading text) – robustly locate by number or exact heading match
            if intent == "rename_section_heading":
                new_heading = (args or {}).get("new_heading")
                section_ref = (args or {}).get("section_ref")
                try:
                    # Use the existing parser to find the section range reliably
                    if not new_heading:
                        raise ValueError("new_heading missing")
                    # Determine target section by reference or current 'sec'
                    target_section = None
                    if isinstance(section_ref, str) and section_ref.strip():
                        # Find by ordinal prefix "1"/"1." or exact heading match
                        ref = section_ref.strip().rstrip('.')
                        sections = _parse_sections_from_markdown(state.prd_markdown)
                        for head, s_idx, e_idx in sections:
                            key = _canonicalize_heading_key(head)
                            if key.startswith(f"{ref}. ") or key == _canonicalize_heading_key(head):
                                target_section = head
                                break
                    if not target_section and sec:
                        target_section = sec
                    if not target_section:
                        # fallback: use the current question's section
                        target_section = sec or "1. Product Overview / Purpose"

                    # Replace the heading line text of the found section
                    start, end, lines, found = _find_section_range(state.prd_markdown, target_section)
                    if start != -1:
                        # heading is the line just before start
                        heading_idx = start - 1
                        if heading_idx >= 0 and lines[heading_idx].strip().startswith("###"):
                            lines[heading_idx] = f"### {new_heading}"
                            state.prd_markdown = "\n".join(lines)
                        else:
                            # If unable to find heading line, prepend a new one before body
                            updated = lines[:start] + [f"### {new_heading}", ""] + lines[start:]
                            state.prd_markdown = "\n".join(updated)
                    else:
                        # If section not found, append a new one at end
                        state.prd_markdown = _replace_section_body(state.prd_markdown, new_heading, "")

                        if state.ws_chat_id:
                            await publish_to_chat(state.ws_chat_id, {
                                "type": "artifacts_preview",
                                "data": {"prd_markdown": state.prd_markdown, "mermaid": None, "sections_status": state.sections_status or {}},
                            })
                            await publish_to_chat(state.ws_chat_id, {
                                "type": "message_sent",
                                "data": {"message_id": f"h:{int(time.time()*1000)}", "user_id": state.user_id, "content": f"Section heading updated to '{new_heading}'.", "timestamp": datetime.utcnow().isoformat() + "Z", "message_type": "assistant"},
                            })
                        # Ask to proceed
                        if state.ws_chat_id:
                            await publish_to_chat(state.ws_chat_id, {
                                "type": "message_sent",
                                "data": {"message_id": f"p:{qid}", "user_id": state.user_id, "content": f"Proceed with the PRD question for '{sec}' now?", "timestamp": datetime.utcnow().isoformat() + "Z", "message_type": "assistant"},
                            })
                        state.pending_question_kind = "confirm"
                        state.pending_question = {"id": qid, "question": f"Proceed with section '{sec}'?", "section": sec, "rationale": "Confirm before resuming PRD updates."}
                        current = interrupt({"type": "confirm", "question_id": qid, "section": sec})
                        continue
                except Exception:
                    pass

            # Not in confirm mode yet → classify as answer vs generic using LLM outcome first and a verifier
            is_answer = (intent == "answer")
            if not is_answer:
                # Ask the LLM to verify strictly if it's an answer
                is_answer = await verify_is_answer(state.user_id, user_text, sec or "", qtxt or "")
            if is_answer or _looks_like_direct_answer(user_text, sec or "", qtxt or ""):
                state.answered_qa.append({
                    "question": qtxt or (state.pending_question.get("question", "") if state.pending_question else ""),
                    "answer": user_text,
                    "section": sec or (state.pending_question.get("section") if state.pending_question else ""),
                })
                # Clear interrupt
                try:
                    if state.ws_chat_id:
                        await publish_to_chat(state.ws_chat_id, {"type": "agent_interrupt_cleared", "data": {"question_id": state.last_section_qid or qid}})
                except Exception:
                    pass
                state.pending_question = None
                state.pending_question_kind = None
                return state

            # Generic: answer briefly, then request permission to proceed
            if state.ws_chat_id and user_text:
                sys = {"role": "system", "content": "Answer the user's question briefly and helpfully in one sentence."}
                usr = {"role": "user", "content": user_text}
                try:
                    resp = await ai_service.generate_response(user_id=state.user_id, messages=[sys, usr], temperature=0.2, max_tokens=120)
                    brief = (resp.content or "").strip()
                except Exception:
                    brief = "Got it."
                await publish_to_chat(state.ws_chat_id, {
                    "type": "message_sent",
                    "data": {
                        "message_id": f"g:{int(time.time()*1000)}",
                        "user_id": state.user_id,
                        "content": brief or "Got it.",
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "message_type": "assistant",
                    },
                })
                proceed_text = f"Would you like to proceed with the PRD question for '{sec}' now?"
                await publish_to_chat(state.ws_chat_id, {
                    "type": "message_sent",
                    "data": {
                        "message_id": f"p:{qid}",
                        "user_id": state.user_id,
                        "content": proceed_text,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "message_type": "assistant",
                    },
                })
            state.pending_question_kind = "confirm"
            state.pending_question = {
                "id": qid,
                "question": f"Proceed with section '{sec}'?",
                "section": sec,
                "rationale": "Confirm before resuming PRD updates.",
            }
            current = interrupt({"type": "confirm", "question_id": qid, "section": sec})
            # Loop will continue and handle the confirmation path

        # If we exit the loop, handle any non-message current
        if isinstance(current, dict) and current.get("type") == "answer":
            state.answered_qa.append({
                "question": state.last_section_question or (state.pending_question.get("question", "") if state.pending_question else ""),
                "answer": current.get("text", ""),
                "section": state.last_section_target or (state.pending_question.get("section") if state.pending_question else ""),
            })
            # Clear interrupt
            try:
                if state.ws_chat_id:
                    await publish_to_chat(state.ws_chat_id, {"type": "agent_interrupt_cleared", "data": {"question_id": state.last_section_qid or (state.pending_question.get("id") if state.pending_question else "")}})
            except Exception:
                pass
            state.pending_question = None
            state.pending_question_kind = None
            return state
        if isinstance(current, dict) and current.get("type") == "accept":
            if current.get("finish"):
                state.telemetry = {**(state.telemetry or {}), "force_finish": True}
            try:
                if state.ws_chat_id:
                    await publish_to_chat(state.ws_chat_id, {"type": "agent_interrupt_cleared", "data": {"question_id": state.last_section_qid or (state.pending_question.get("id") if state.pending_question else "")}})
            except Exception:
                pass
            state.pending_question = None
            state.pending_question_kind = None
            return state
        # Unknown shape → keep waiting (no state change)
        return state

    # Unknown shape: keep waiting (no state change)
    return state


async def incorporate_answer(state: AgentState) -> AgentState:
    """Regenerate/expand PRD for the targeted section given latest answer; emit preview."""
    if not state.answered_qa:
        return state
    # Emit helper (streams via callback if present, else broadcast to chat)
    from app.websocket.publisher import publish_to_chat

    async def _emit(event: Dict[str, Any]) -> None:
        if callable(state.send_event):
            await state.send_event(event)  # type: ignore
        elif getattr(state, "ws_chat_id", None):
            await publish_to_chat(state.ws_chat_id, event)  # type: ignore

    send = state.send_event or (lambda *_args, **_kwargs: None)
    # Build messages to refine PRD using newest Q/A with retry
    latest_one = state.answered_qa[-1]
    target_section = latest_one.get("section") or ""
    system_msg = {
        "role": "system",
        "content": (
            "You will update only ONE section of the PRD based on the user's answer.\n"
            "- Preserve the existing structure and all headings exactly.\n"
            "- Replace ONLY the body of the target section with detailed, concrete content.\n"
            "- Output the FULL updated PRD Markdown.\n"
            "- Do NOT add any extra commentary."
        ),
    }
    user_msg = {
        "role": "user",
        "content": (
            f"Current PRD (full):\n{state.prd_markdown}\n\n"
            f"Target section: {target_section}\n"
            f"Answer: {latest_one.get('answer','')}\n"
        ),
    }
    # 1) Update PRD deterministically: ask model for BODY ONLY, then splice into PRD
    # Prepare existing body to give the model focused context
    try:
        s_idx, e_idx, lines, _head = _find_section_range(state.prd_markdown, target_section)
        existing_body = "\n".join(lines[s_idx:e_idx]).strip() if s_idx != -1 else ""
    except Exception:
        existing_body = ""

    style_hints: Dict[str, str] = {
        "1. Product Overview / Purpose": "Write 1–2 short paragraphs (<= 120 words total). State product, primary value, and primary audience.",
        "2. Objective and Goals": "Write 2–4 bullet points. Each goal must be concrete and measurable (include a target and timeframe where possible).",
        "3. User Personas": "Write 1–2 concise personas with bullets: key traits, goals, pain points, context. Use respectful, neutral language.",
        "4. User Needs & Requirements": "Write 5–8 bullet points of user needs/problems (not solutions). Keep each bullet crisp.",
        "5. Features & Functional Requirements": "Write 5–8 bullets, each naming a feature and its outcome. Keep neutral, no fluff.",
        "6. Non-Functional Requirements": "Write bullets for performance, reliability, security, privacy, accessibility. Add sensible targets where possible.",
        "7. User Stories / UX Flow / Design Notes": "Write a short happy-path flow as numbered steps (5–8) and any key design notes.",
        "8. Technical Specifications": "List stack/integrations/constraints as bullets. Keep practical and specific.",
        "9. Assumptions, Constraints & Dependencies": "List assumptions and constraints as bullets. Mention external dependencies.",
        "10. Timeline & Milestones": "Write 3–5 milestones with rough timing (e.g., Week 2, Week 6, Launch).",
        "11. Success Metrics / KPIs": "List 3–5 primary metrics plus any guardrails.",
        "12. Release Criteria": "Bullet list of must-pass checks for launch: functionality, performance, reliability, support.",
        "13. Open Questions / Issues": "List 3–7 decisions/risks to resolve.",
        "14. Budget and Resources": "Brief bullets for team roles and budget bands if known.",
    }

    refine_system = {
        "role": "system",
        "content": (
            "You update PRDs. Output ONLY the Markdown BODY for the specified section. "
            "NO headings, NO backticks, NO preface/postface. "
            "Do not copy the user's words verbatim; rewrite professionally and concretely. "
            "Keep it concise."
        ),
    }
    refine_user = {
        "role": "user",
        "content": (
            f"Section: {target_section}\n"
            f"Style rules: {style_hints.get(target_section, 'Keep it concise and structured with short paragraphs or bullets.')}\n\n"
            f"Existing body (may be empty):\n{existing_body}\n\n"
            f"User answer (source to interpret/rewrite, not to copy):\n{latest_one.get('answer','')}\n"
        ),
    }

    try:
        # First attempt: concise rewrite
        resp = await ai_service.generate_response(
            user_id=state.user_id,
            messages=[refine_system, refine_user],
            temperature=0.3,
            max_tokens=900,
            use_cache=False,
        )
        section_body = _sanitize_section_body(resp.content)

        # If under-sized, try one expansion pass before fallback
        try:
            body_text = section_body.strip() if section_body else ""
            needs_expand = False
            if target_section.startswith('1.') and len(body_text) < 180:
                needs_expand = True
            if target_section.startswith('2.'):
                bullets = [ln for ln in body_text.splitlines() if ln.strip().startswith(('-', '*'))]
                if len(bullets) < 3:
                    needs_expand = True
            if not body_text:
                needs_expand = True
            if needs_expand:
                expand_system = {
                    "role": "system",
                    "content": (
                        "Expand the section BODY with concrete, useful detail. Maintain concision and structure. "
                        "No headings, no backticks."
                    ),
                }
                expand_user = {
                    "role": "user",
                    "content": (
                        f"Section: {target_section}\n"
                        f"Style rules: {style_hints.get(target_section, '')}\n\n"
                        f"Draft body to improve:\n{body_text or existing_body}\n"
                    ),
                }
                resp2 = await ai_service.generate_response(
                    user_id=state.user_id,
                    messages=[expand_system, expand_user],
                    temperature=0.3,
                    max_tokens=900,
                    use_cache=False,
                )
                improved = _sanitize_section_body(resp2.content)
                if improved and len(improved.strip()) > len(body_text):
                    section_body = improved
        except Exception:
            pass

        # If the model still under-delivered, synthesize a minimal useful body
        if not section_body or len(section_body.strip()) < 60:
            # Deterministic guardrails per section
            ans = (latest_one.get('answer','') or '').strip()
            if target_section.startswith('2.'):
                # Objectives: ensure at least 3 measurable bullets
                fallback = [
                    "- Deliver a polished, responsive UI rated ≥ 4.5/5 in usability testing (N ≥ 10).",
                    "- Achieve time-to-interactive ≤ 2s on mid‑range devices for core flows.",
                    "- Reach CSAT ≥ 4.2/5 across the first 50 user sessions.",
                ]
                # If the answer mentions UI/feel, bias bullets toward UX
                if ans:
                    lower = ans.lower()
                    if 'ui' in lower or 'feel' in lower or 'design' in lower:
                        fallback[0] = "- Ship a clean, accessible UI (WCAG AA) with ≥ 4.5/5 SUS score (N ≥ 10)."
                section_body = "\n".join(fallback)
            elif target_section.startswith('1.'):
                section_body = (
                    "This product serves its primary audience with a clear value proposition and focused problem–solution fit. "
                    "It explains what the product is, who it is for, and why it matters, in practical terms that guide scope and priorities."
                )
            else:
                # Generic fallback
                section_body = ans[:300] if ans else "(to be detailed)"
        state.prd_markdown = _replace_section_body(state.prd_markdown, target_section, section_body)
    except Exception:
        # On error, leave PRD unchanged
        pass
    # Recompute section status
    order = state.completion_targets.get("sections", []) if isinstance(state.completion_targets, dict) else []
    state.sections_status = _compute_sections_status(state.prd_markdown, order)
    await _emit({
        "type": "artifacts_preview",
        "data": {
            "prd_markdown": state.prd_markdown,
            "mermaid": None,
            "sections_status": state.sections_status or {},
        },
    })
    # 3) Briefly acknowledge AFTER updating PRD
    try:
        await _emit({"type": "stream_start", "data": {"project_id": state.project_id}})
        ack_system = {
            "role": "system",
            "content": (
                "You are a concise product partner. Acknowledge in 1 short sentence and note the section updated. "
                "Do NOT ask a new question."
            ),
        }
        ack_user = {
            "role": "user",
            "content": (
                f"Question: {latest_one.get('question','')}\n"
                f"Answer: {latest_one.get('answer','')}\n"
                f"Section: {target_section}"
            ),
        }
        last_provider: str | None = None
        last_model: str | None = None
        async for chunk in ai_service.generate_stream(
            user_id=state.user_id,
            messages=[ack_system, ack_user],
            temperature=0.2,
            max_tokens=80,
        ):
            if chunk.is_complete:
                await _emit({
                    "type": "ai_response_streaming",
                    "data": {"delta": "", "is_complete": True, "provider": chunk.provider, "model": chunk.model},
                })
            else:
                last_provider = chunk.provider
                last_model = chunk.model
                await _emit({
                    "type": "ai_response_streaming",
                    "data": {"delta": chunk.content, "is_complete": False, "provider": chunk.provider, "model": chunk.model},
                })
        await _emit({
            "type": "ai_response_complete",
            "data": {"message": "Acknowledged user answer", "provider": last_provider, "model": last_model},
        })
    except Exception:
        pass
    state.iteration_index += 1
    return state

