"""
Application configuration management using Pydantic Settings.
Handles environment-based configuration for database connections, JWT secrets, and other settings.
"""

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
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_DB: int = 0
    CACHE_TTL: int = 3600  # 1 hour default cache TTL
    
    # Cloudinary settings
    CLOUDINARY_CLOUD_NAME: str = ""
    CLOUDINARY_API_KEY: str = ""
    CLOUDINARY_API_SECRET: str = ""
    CLOUDINARY_FOLDER_PREFIX: str = "ai-prd-generator"
    CLOUDINARY_SECURE_URL: bool = True
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    ALLOWED_METHODS: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    ALLOWED_HEADERS: List[str] = ["*"]
    
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
        if not v.startswith("redis://") and not v.startswith("rediss://"):
            raise ValueError("REDIS_URL must start with 'redis://' or 'rediss://'")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()
