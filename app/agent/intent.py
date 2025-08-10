from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

from app.services.ai_service import ai_service


INTENTS = {
    "answer",
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
    intent ∈ { 'answer', 'rename_title', 'confirm_yes', 'confirm_no', 'generic' }
    args: any additional info, e.g., { new_title: str }
    """
    try:
        sys = {
            "role": "system",
            "content": (
                "You are an intent classifier for a PRD authoring assistant. "
                "The user may either answer a PRD section question, ask a generic question, confirm yes/no, or issue a command like renaming the PRD title.\n"
                "Output ONLY minified JSON of shape {\"intent\": string, \"args\": object}.\n"
                "Allowed intent values: answer, rename_title, rename_section_heading, confirm_yes, confirm_no, generic.\n"
                "Rules:\n"
                "- If the user clearly requests changing the PRD title, use intent=rename_title and set args.new_title to the title string.\n"
                "- If the user requests changing a section heading, use intent=rename_section_heading and set args.new_heading and optionally args.section_ref (e.g., '1', '1.', or the current heading text).\n"
                "- If the user explicitly confirms (yes/ok/proceed) to continue, use intent=confirm_yes.\n"
                "- If the user declines (no/skip/later), use intent=confirm_no.\n"
                "- If the message looks like a direct answer to the provided PRD section question, use intent=answer.\n"
                "- Otherwise, use intent=generic.\n"
                "Important: Do NOT classify a request to modify the document (rename, change, set) as 'answer'.\n"
                "- Never include explanations or extra keys."
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
        resp = await ai_service.generate_response(
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
        resp = await ai_service.generate_response(
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


