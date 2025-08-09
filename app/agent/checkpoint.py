from __future__ import annotations

from typing import Optional
from pathlib import Path
import os

from langgraph.checkpoint.memory import InMemorySaver

try:
    from langgraph.checkpoint.mongodb import MongodbSaver  # type: ignore
except Exception:
    MongodbSaver = None  # type: ignore

_CHECKPOINTER: Optional[object] = None


def _build_mongo_checkpointer() -> Optional[object]:
    """Create a MongoDB-backed checkpointer if possible.

    Uses env HITL_MONGO_URI (fallback to app settings MONGODB_URL) and HITL_MONGO_DB (fallback to DATABASE_NAME).
    Collection name defaults to 'langgraph_checkpoints'.
    """
    if MongodbSaver is None:
        return None
    try:
        from app.core.config import settings  # lazy import to avoid circulars at module import
        mongo_uri = os.getenv("HITL_MONGO_URI", settings.MONGODB_URL)
        db_name = os.getenv("HITL_MONGO_DB", settings.DATABASE_NAME)
        collection = os.getenv("HITL_MONGO_COLLECTION", "langgraph_checkpoints")
        return MongodbSaver(connection_string=mongo_uri, db_name=db_name, collection_name=collection)  # type: ignore
    except Exception:
        return None


def get_checkpointer() -> object:
    """Return a process-local checkpointer instance.

    Preference order: SQLite (if available) → InMemory.
    """
    global _CHECKPOINTER
    if _CHECKPOINTER is not None:
        return _CHECKPOINTER
    mongo_cp = _build_mongo_checkpointer()
    if mongo_cp is not None:
        _CHECKPOINTER = mongo_cp
        return _CHECKPOINTER
    _CHECKPOINTER = InMemorySaver()
    return _CHECKPOINTER


