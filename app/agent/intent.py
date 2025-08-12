from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

def _get_ai():
    # Lazy import to avoid heavy provider init during module import time
    from app.services.ai_service import ai_service  # type: ignore
    return ai_service


INTENTS = {
    "answer",
    "followup_question",
    "update_section_content",
    "request_section_insert",
    "request_section_delete",
    "rename_title",
    "rename_section_heading",
    "confirm_yes",
    "confirm_no",
    "generic",
}


def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`\n ")
        if t.lower().startswith("json"):
            t = t[len("json"):].lstrip()
    return t


async def classify_intent(
    user_id: str,
    user_text: str,
    section: Optional[str] = None,
    question: Optional[str] = None,
    idea: Optional[str] = None,
    recent_qa: Optional[list] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Use the LLM to classify a user's message intent.

    Returns a tuple: (intent, args)
    intent ∈ {
        'answer',
        'followup_question',
        'update_section_content',
        'request_section_insert',
        'request_section_delete',
        'rename_title',
        'rename_section_heading',
        'confirm_yes', 'confirm_no', 'generic'
    }
    args: additional info, e.g., { new_title, new_heading, section_ref, instructions, question, target_section_guess }
    """
    try:
        sys = {
            "role": "system",
            "content": (
                "You are an intent classifier for a PRD authoring assistant.\n"
                "The user message can be: a direct answer to the current PRD section question, a follow-up question, a request to update section content, a request to insert/delete a section, a rename command (title or section), a yes/no confirm, or generic.\n"
                "Output ONLY minified JSON {\"intent\": string, \"args\": object}.\n"
                "Allowed intents: answer, followup_question, update_section_content, request_section_insert, request_section_delete, rename_title, rename_section_heading, confirm_yes, confirm_no, generic.\n"
                "Rules:\n"
                "- If the user clearly requests changing the PRD title, intent=rename_title; args.new_title = string.\n"
                "- If the user requests changing a section heading, intent=rename_section_heading; args.new_heading = string; args.section_ref = ordinal like '11' or the existing heading text.\n"
                "- If the user requests updating a section body (add/edit details), intent=update_section_content; args.section_ref = ordinal or heading; args.instructions = the instruction text.\n"
                "- If the user asks a question unrelated to the pending PRD question, intent=followup_question; args.question = normalized question; args.target_section_guess optional.\n"
                "- If the user asks to insert a new section, intent=request_section_insert; args.section_ref (where) and args.new_heading recommended.\n"
                "- If the user asks to delete a section, intent=request_section_delete; args.section_ref required.\n"
                "- If the message explicitly confirms to proceed (yes/ok/proceed), intent=confirm_yes.\n"
                "- If the message declines to proceed (no/skip/later), intent=confirm_no.\n"
                "- If the message is a direct answer to the provided PRD section question, intent=answer.\n"
                "- Otherwise intent=generic.\n"
                "Important: Do NOT classify requests to modify the document (rename/change/update) as 'answer'.\n"
                "Never include explanations or extra keys."
            ),
        }
        # Provide context to help intent judgement
        parts = []
        if idea:
            parts.append(f"Idea: {idea}")
        if section:
            parts.append(f"Section: {section}")
        if question:
            parts.append(f"Question: {question}")
        if recent_qa:
            try:
                compact = []
                for qa in recent_qa[-2:]:
                    q = (qa.get("question") or "").strip()
                    a = (qa.get("answer") or "").strip()
                    if q and a:
                        compact.append(f"- Q: {q} | A: {a}")
                if compact:
                    parts.append("Recent:\n" + "\n".join(compact))
            except Exception:
                pass
        parts.append(f"User: {user_text}")
        usr = {"role": "user", "content": "\n\n".join(parts)}
        resp = await _get_ai().generate_response(
            user_id=user_id,
            messages=[sys, usr],
            temperature=0.0,
            max_tokens=200,
            use_cache=False,
        )
        raw = _strip_code_fences(resp.content or "{}")
        data = json.loads(raw)
        intent = str(data.get("intent", "")).strip().lower()
        if intent not in INTENTS:
            return ("generic", {})
        args = data.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        return (intent, args)
    except Exception:
        # Fallback to generic when LLM or JSON parsing fails
        return ("generic", {})


async def verify_is_answer(user_id: str, user_text: str, section: Optional[str], question: Optional[str]) -> bool:
    """Ask the LLM to strictly verify whether the user_text is a direct answer to the question.

    Returns True only if it's a clear answer (not a command, not a generic query).
    """
    try:
        sys = {
            "role": "system",
            "content": (
                "Judge if the user text is a direct answer to the provided question. "
                "If it is an instruction/command (e.g., rename/change/set), or a generic smalltalk, or a different request, return false.\n"
                "Output ONLY minified JSON {\"is_answer\": true|false}."
            ),
        }
        usr = {
            "role": "user",
            "content": f"Section: {section or ''}\nQuestion: {question or ''}\nUser: {user_text}",
        }
        resp = await _get_ai().generate_response(
            user_id=user_id,
            messages=[sys, usr],
            temperature=0.0,
            max_tokens=50,
            use_cache=False,
        )
        raw = _strip_code_fences(resp.content or "{}")
        data = json.loads(raw)
        return bool(data.get("is_answer") is True)
    except Exception:
        return False


async def detect_is_question(user_id: str, user_text: str) -> bool:
    """Use the LLM to decide if a message is a question (follow-up question path)."""
    try:
        sys = {
            "role": "system",
            "content": (
                "Determine if the user text is a question. Output ONLY minified JSON {\"is_question\": true|false}."
            ),
        }
        usr = {"role": "user", "content": user_text or ""}
        resp = await _get_ai().generate_response(
            user_id=user_id,
            messages=[sys, usr],
            temperature=0.0,
            max_tokens=20,
            use_cache=False,
        )
        raw = _strip_code_fences(resp.content or "{}")
        data = json.loads(raw)
        return bool(data.get("is_question") is True)
    except Exception:
        return False


async def map_text_to_section(
    user_id: str,
    prd_markdown: str,
    text: str,
) -> Dict[str, Any]:
    """Map free-form text to a PRD section reference using the LLM.

    Returns: { "section_ref": str, "confidence": float }
    """
    try:
        sys = {
            "role": "system",
            "content": (
                "You are mapping user text to a PRD section.\n"
                "Given PRD headings and user text, return ONLY minified JSON {\"section_ref\": string, \"confidence\": number}.\n"
                "section_ref should be either the exact heading text or an ordinal like '11' matching a section."
            ),
        }
        # Provide just the headings to keep context compact
        # Headings are lines that begin with '### '
        try:
            import re
            headings = re.findall(r"^###\s+(.+)$", prd_markdown or "", flags=re.M)
            headings_text = "\n".join([f"- {h}" for h in headings]) or "(no headings)"
        except Exception:
            headings_text = "(no headings)"
        usr = {
            "role": "user",
            "content": f"Headings:\n{headings_text}\n\nUser text:\n{text}",
        }
        resp = await _get_ai().generate_response(
            user_id=user_id,
            messages=[sys, usr],
            temperature=0.0,
            max_tokens=120,
            use_cache=False,
        )
        raw = _strip_code_fences(resp.content or "{}")
        data = json.loads(raw)
        out = {
            "section_ref": str(data.get("section_ref") or "").strip(),
            "confidence": float(data.get("confidence") or 0.0),
        }
        return out
    except Exception:
        return {"section_ref": "", "confidence": 0.0}

