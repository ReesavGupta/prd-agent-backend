"""Middleware package."""

from .auth import (
    get_current_user,
    get_current_active_user,
    get_current_verified_user,
    get_current_superuser,
    get_optional_current_user
)

__all__ = [
    "get_current_user",
    "get_current_active_user", 
    "get_current_verified_user",
    "get_current_superuser",
    "get_optional_current_user"
]

# Optional: simple request timing middleware hook for analytics (Phase 3 placeholder)
try:
    from fastapi import Request
    import time

    async def timing_middleware(request: Request, call_next):  # type: ignore
        start = time.time()
        response = await call_next(request)
        try:
            duration_ms = int((time.time() - start) * 1000)
            response.headers["X-Process-Time-ms"] = str(duration_ms)
        except Exception:
            pass
        return response
except Exception:
    pass
