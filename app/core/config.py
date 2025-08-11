"""
Application configuration management using Pydantic Settings.
Handles environment-based configuration for database connections, JWT secrets, and other settings.
"""
import os
from typing import Optional, List
from pydantic import field_validator
from pydantic_settings import BaseSettings
import secrets


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application settings
    APP_NAME: str = "AI PRD Generator API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Security settings
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Password settings
    PASSWORD_MIN_LENGTH: int = 8
    PASSWORD_REQUIRE_UPPERCASE: bool = True
    PASSWORD_REQUIRE_LOWERCASE: bool = True
    PASSWORD_REQUIRE_NUMBERS: bool = True
    PASSWORD_REQUIRE_SPECIAL: bool = True
    
    # Database settings
    MONGODB_URL: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "ai_prd_generator"
    
    # Redis settings
    REDIS_URL: Optional[str] = os.getenv("REDIS_URL")
    REDIS_DB: int = 0
    CACHE_TTL: int = 3600  # 1 hour default cache TTL
    
    # Cloudinary settings
    CLOUDINARY_CLOUD_NAME: str = os.getenv("CLOUDINARY_CLOUD_NAME")
    CLOUDINARY_API_KEY: str = os.getenv("CLOUDINARY_API_KEY")
    CLOUDINARY_API_SECRET: str = os.getenv("CLOUDINARY_API_SECRET")
    CLOUDINARY_FOLDER_PREFIX: str = os.getenv("CLOUDINARY_FOLDER_PREFIX")
    CLOUDINARY_SECURE_URL: bool = os.getenv("CLOUDINARY_SECURE_URL")

    # Development fallback: use local storage when Cloudinary not configured
    # USE_LOCAL_STORAGE: bool = bool(os.getenv("USE_LOCAL_STORAGE", "").lower() in ("1", "true", "yes"))
    USE_LOCAL_STORAGE: bool = False
    LOCAL_STORAGE_DIR: str = os.getenv("LOCAL_STORAGE_DIR", "./.local_storage")
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://0.0.0.0:8080",
        "https://think-prd-frontend.vercel.app/"
    ]
    ALLOWED_METHODS: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    ALLOWED_HEADERS: List[str] = ["*"]
    
    # AI Provider settings
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY") 
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY") 
    
    # Vector store / Embeddings settings
    PINECONE_API_KEY: Optional[str] = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "prd-agent-index")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    PINECONE_ENDPOINT: Optional[str] = os.getenv("PINECONE_ENDPOINT")
    NOMIC_API_KEY: Optional[str] = os.getenv("NOMIC_API_KEY")
    
    # AI Configuration
    PRIMARY_AI_PROVIDER: str = "gemini"
    SECONDARY_AI_PROVIDER: str = "groq"
    TERTIARY_AI_PROVIDER: str = "openai"

    # Model names (configurable via env)
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-70b-8192")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    # AI Rate Limiting (requests per minute per user)
    AI_RATE_LIMIT_PER_USER: int = 1000000  # effectively disabled per user decision
    AI_RATE_LIMIT_WINDOW_MINUTES: int = 60
    
    # AI Response Configuration
    AI_MAX_TOKENS: int = 4000
    AI_TEMPERATURE: float = 0.7
    AI_TIMEOUT_SECONDS: int = 60

    # PRD summary cache TTLs (seconds)
    PRD_SUMMARY_TTL_SECONDS: int = int(os.getenv("PRD_SUMMARY_TTL_SECONDS", str(24*3600)))
    PRD_SUMMARY_EPHEMERAL_TTL_SECONDS: int = int(os.getenv("PRD_SUMMARY_EPHEMERAL_TTL_SECONDS", "1800"))
    
    # Streaming Configuration
    AI_STREAM_CHUNK_SIZE: int = 1024
    AI_STREAM_TIMEOUT: int = 120

    # HITL Interrupt/Reminder configuration
    HITL_INTERRUPT_REMINDER_SECONDS: int = int(os.getenv("HITL_INTERRUPT_REMINDER_SECONDS", "120"))
    HITL_INTERRUPT_MAX_REMINDERS: int = int(os.getenv("HITL_INTERRUPT_MAX_REMINDERS", "1"))

    # Chat-mode RAG defaults
    CHAT_RAG_DEFAULT_SCOPE: str = os.getenv("CHAT_RAG_DEFAULT_SCOPE", "project")  # 'project' or 'single_file'
    CHAT_RAG_MAX_K: int = int(os.getenv("CHAT_RAG_MAX_K", "6"))
    # Agent-mode RAG controls
    AGENT_RAG_ENABLED: bool = bool(os.getenv("AGENT_RAG_ENABLED", "true").lower() in ("1", "true", "yes"))
    AGENT_RAG_DEFAULT_SCOPE: str = os.getenv("AGENT_RAG_DEFAULT_SCOPE", "project")  # 'project' or 'none'
    AGENT_RAG_MAX_K: int = int(os.getenv("AGENT_RAG_MAX_K", "6"))
    AGENT_RAG_MAX_CONTEXT_CHARS: int = int(os.getenv("AGENT_RAG_MAX_CONTEXT_CHARS", "4500"))
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @field_validator("SECRET_KEY")
    def validate_secret_key(cls, v):
        """Ensure secret key is sufficiently long."""
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")
        return v
    
    @field_validator("MONGODB_URL")
    def validate_mongodb_url(cls, v):
        """Validate MongoDB URL format."""
        if not v.startswith(("mongodb://", "mongodb+srv://")):
            raise ValueError("MONGODB_URL must start with 'mongodb://' or 'mongodb+srv://'")
        return v
    
    @field_validator("PASSWORD_MIN_LENGTH")
    def validate_password_min_length(cls, v):
        """Ensure minimum password length is reasonable."""
        if v < 6:
            raise ValueError("PASSWORD_MIN_LENGTH must be at least 6")
        return v
    
    @field_validator("REDIS_URL")
    def validate_redis_url(cls, v):
        """Validate Redis URL format."""
        if v is None or v == "":
            return v
        if not v.startswith("redis://") and not v.startswith("rediss://"):
            raise ValueError("REDIS_URL must start with 'redis://' or 'rediss://'")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()
