from __future__ import annotations

import re
from typing import Optional


_RENAME_PATTERNS = [
    # rename/change/set title
    re.compile(r"^\s*rename\s+(?:the\s+)?(?:prd\s+)?(?:title\s+)?to\s+(.+)$", re.I),
    re.compile(r"^\s*change\s+(?:the\s+)?(?:prd\s+)?title\s+(?:to|as)\s+(.+)$", re.I),
    re.compile(r"^\s*set\s+(?:the\s+)?(?:prd\s+)?title\s+to\s+(.+)$", re.I),
    # natural variants
    re.compile(r"^\s*can\s+you\s+(?:please\s+)?(?:rename|change|set)\s+(?:the\s+)?(?:prd\s+)?title\s+(?:to|as)\s+(.+)$", re.I),
    re.compile(r"^\s*please\s+(?:rename|change|set)\s+(?:the\s+)?(?:prd\s+)?title\s+(?:to|as)\s+(.+)$", re.I),
    # common misspellings of change
    re.compile(r"^\s*cahnge\s+(?:the\s+)?(?:prd\s+)?(?:title|document\s+title|doc(?:ument)?\s+title|name)\s+(?:to|as)\s+(.+)$", re.I),
    re.compile(r"^\s*chnage\s+(?:the\s+)?(?:prd\s+)?(?:title|document\s+title|doc(?:ument)?\s+title|name)\s+(?:to|as)\s+(.+)$", re.I),
    re.compile(r"^\s*chagne\s+(?:the\s+)?(?:prd\s+)?(?:title|document\s+title|doc(?:ument)?\s+title|name)\s+(?:to|as)\s+(.+)$", re.I),
    # "to be changed to" phrasing
    re.compile(r"^\s*(?:i\s+want\s+)?(?:the\s+)?(?:prd\s+)?(?:document\s+)?(?:title|name)[^\n]{0,40}?to\s+be\s+changed\s+to\s+(.+)$", re.I),
    # "call it" / "name it"
    re.compile(r"^\s*(?:call|name)\s+(?:it|the\s+(?:doc|document|prd))\s+(.+)$", re.I),
]


def detect_rename_title(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.strip().strip("'\"")
    for pat in _RENAME_PATTERNS:
        m = pat.match(t)
        if m:
            title = m.group(1).strip().strip("'\"")
            if title:
                return title
    # Heuristic fallback: look for verbs + objects and extract after to/as/named/call
    tl = t.lower()
    verbs = ("rename", "re-name", "change", "cahnge", "chnage", "chagne", "set", "update", "name", "call")
    objs = ("title", "name", "document", "doc", "prd")
    if any(v in tl for v in verbs) and any(o in tl for o in objs):
        # common delimiters
        for delim in (" to be changed to ", " be changed to ", " to ", " as ", " named ", " call it ", " call the ", " call "):
            if delim in tl:
                idx = tl.find(delim) + len(delim)
                candidate = t[idx:].strip().strip("'\". ")
                if candidate:
                    return candidate
    return None


def is_pure_rename_title_command(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    for pat in _RENAME_PATTERNS:
        if pat.fullmatch(t):
            return True
    return False


def apply_prd_title_rename(markdown: str, new_title: str) -> str:
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


