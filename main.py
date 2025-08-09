"""
FastAPI Authentication API Application.
"""

import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from app.middleware import timing_middleware
from fastapi.responses import JSONResponse
# Rate limiting removed

from app.core.config import settings
from app.db.database import connect_to_mongo, close_mongo_connection
from app.api import auth_router, users_router, projects_router, ai_router, chats_router, websocket_router, agent_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Rate limiting removed

@asynccontextmanager
async def lifespan(app):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting up...")
    await connect_to_mongo()
    
    # Initialize database indexes
    from app.dependencies import get_project_repository, cleanup_dependencies
    project_repo = get_project_repository()
    await project_repo.create_indexes()
    logger.info("Database indexes created")
    
    logger.info("Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down...")
    await cleanup_dependencies()
    await close_mongo_connection()
    logger.info("Application shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Comprehensive user authentication system with JWT tokens and MongoDB",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Rate limiting middleware removed

# Add CORS middleware (dev-friendly)
cors_kwargs = dict(
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=settings.ALLOWED_METHODS,
    allow_headers=settings.ALLOWED_HEADERS,
)
# In DEBUG, allow any origin via regex to ease local dev across ports
if settings.DEBUG:
    cors_kwargs["allow_origin_regex"] = ".*"

app.add_middleware(CORSMiddleware, **cors_kwargs)
try:
    # Add timing header middleware if available
    app.middleware("http")(timing_middleware)  # type: ignore
except Exception:
    pass


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": settings.APP_VERSION}


# Include routers with rate limiting
app.include_router(
    auth_router,
    prefix="/api/v1"
)

app.include_router(
    users_router,
    prefix="/api/v1"
)

app.include_router(
    projects_router,
    prefix="/api/v1"
)

app.include_router(
    ai_router,
    prefix="/api/v1"
)

app.include_router(
    chats_router,
    prefix="/api/v1"
)

app.include_router(
    websocket_router,
    prefix="/ws"
)

app.include_router(
    agent_router,
    prefix="/api/v1"
)


def main():
    """Run the application."""
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )


if __name__ == "__main__":
    main()
