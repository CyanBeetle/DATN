"""Authentication security utilities."""
from datetime import datetime, timezone
from typing import Optional
import bcrypt  # Direct import of bcrypt
import logging

from fastapi import Depends, HTTPException, status, Cookie, Header, Request
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext

from app.config import settings, ERROR_MESSAGES, ADMIN_ROLES
from db.models import UserInDB
from db.session import get_db
from auth.jwt import decode_token, create_access_token, create_refresh_token

# Setup logging
logger = logging.getLogger(__name__)

# Re-export for backward compatibility
SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.JWT_ALGORITHM

# Use only bcrypt for simplicity
pwd_context = CryptContext(schemes=["bcrypt"])

# OAuth2 scheme for token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    # Direct bcrypt implementation
    try:
        password_bytes = plain_password.encode('utf-8')
        hash_bytes = hashed_password.encode('utf-8') if isinstance(hashed_password, str) else hashed_password
        return bcrypt.checkpw(password_bytes, hash_bytes)
    except Exception as e:
        logger.error(f"Password verification failed: {e}")
        return False

def get_password_hash(password: str) -> str:
    """Hash a password for storing."""
    # Direct bcrypt implementation
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')

async def get_current_user(
    request: Request,
    db = Depends(get_db),
    access_token: Optional[str] = Cookie(None),
) -> UserInDB:
    """Get current user from token."""
    token = access_token
    
    # If not in cookie, try to get from authorization header via oauth2_scheme
    if not token:
        token = await oauth2_scheme(request)
    
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES["INVALID_TOKEN"],
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Decode token
    payload = decode_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES["INVALID_TOKEN"],
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Extract username
    username = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES["INVALID_TOKEN"],
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user
    user = await db.users.find_one({"username": username})
    if user is None:
        logger.warning(f"User from token not found: {username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES["USER_NOT_FOUND"],
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_model = UserInDB(**user) # Convert to Pydantic model early for easier access

    # Validate token_version from JWT against database
    jwt_token_version = payload.get("tv")
    db_token_version = user_model.token_version

    if jwt_token_version is None: # Token created before versioning was implemented
        # For enhanced security, you might want to invalidate these older tokens.
        # For now, we'll allow them but log a warning. 
        # Consider forcing re-login for such tokens if security is paramount.
        logger.warning(f"Token for user {username} lacks version info. Consider forcing re-authentication.")
    elif jwt_token_version != db_token_version:
        logger.warning(f"Invalid token version for user {username}. JWT_tv: {jwt_token_version}, DB_tv: {db_token_version}. Force re-login.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.get("SESSION_EXPIRED_SECURITY", "Session expired due to security reasons. Please login again."),
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if token was issued before user's token_version was updated (for /logout-all)
    # This check can be kept as a secondary measure or if global logout is a separate feature.
    # However, the token_version check above should be primary for password changes.
    if user_model.token_version_changed_at: # Check if the field exists and is not None
        token_issue_time = datetime.fromtimestamp(payload.get("iat", 0), tz=timezone.utc)
        token_version_changed_at = user_model.token_version_changed_at
        if isinstance(token_version_changed_at, datetime) and token_version_changed_at.tzinfo is None:
            token_version_changed_at = token_version_changed_at.replace(tzinfo=timezone.utc)
        
        # This check might be redundant if jwt_token_version != db_token_version already covers it implicitly
        # because a password change (which updates token_version_changed_at) also changes db_token_version.
        # However, it could be useful if token_version_changed_at is updated for other reasons than password change
        # without changing token_version (e.g. explicit "logout all sessions" feature).
        if jwt_token_version == db_token_version and token_issue_time < token_version_changed_at:
            logger.info(f"Token invalidated due to a global session invalidation event for user: {username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.get("SESSION_EXPIRED_GLOBAL", "Session expired. Please login again."),
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    # Check if user is active
    if not user_model.is_active:
        logger.warning(f"Inactive user attempted to access protected resource: {username}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=ERROR_MESSAGES["INACTIVE_USER"]
        )
    
    return user_model

async def get_admin_user(current_user: UserInDB = Depends(get_current_user)) -> UserInDB:
    """Check if current user is an admin."""
    if current_user.role not in ADMIN_ROLES:
        logger.warning(f"Non-admin user attempted to access admin resource: {current_user.username}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=ERROR_MESSAGES["PERMISSION_DENIED"]
        )
    return current_user


