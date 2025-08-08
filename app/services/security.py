"""
Security services for password hashing and JWT token management.
"""

import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
import logging

from app.core.config import settings
from app.schemas.token import TokenData

logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class SecurityService:
    """Service for handling security operations."""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt."""
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode.update({
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "token_type": "access"
        })
        
        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def create_refresh_token(data: Dict[str, Any]) -> str:
        """Create a JWT refresh token."""
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

        to_encode.update({
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "token_type": "refresh"
        })
        
        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def verify_token(token: str) -> Optional[TokenData]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            
            user_id: Optional[str] = payload.get("sub")
            email: Optional[str] = payload.get("email")
            token_type: Optional[str] = payload.get("token_type")
            exp_ts = payload.get("exp")
            iat_ts = payload.get("iat")
            if exp_ts is None or iat_ts is None or user_id is None or email is None or token_type is None:
                return None
            exp: datetime = datetime.fromtimestamp(exp_ts)
            iat: datetime = datetime.fromtimestamp(iat_ts)
                
            token_data = TokenData(
                user_id=user_id,
                email=email,
                token_type=token_type,
                exp=exp,
                iat=iat
            )
            return token_data
            
        except JWTError as e:
            logger.warning(f"JWT verification failed: {e}")
            return None
    
    # Password reset token generation removed
    
    # Email verification token generation removed


    @staticmethod
    def is_token_expired(token_data: TokenData) -> bool:
        """Check if a token is expired."""
        return datetime.now(timezone.utc) > token_data.exp

    @staticmethod
    def get_token_expiry_seconds(token_type: str = "access") -> int:
        """Get token expiry time in seconds."""
        if token_type == "access":
            return settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        elif token_type == "refresh":
            return settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
        else:
            return settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60


# Global security service instance
security_service = SecurityService()
