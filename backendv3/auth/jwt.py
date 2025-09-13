"""JWT token handling utilities."""
from datetime import datetime, timedelta, timezone
import jwt  # Make sure to use PyJWT
import logging

from app.config import settings

# Setup logging
logger = logging.getLogger(__name__)

def create_access_token(data: dict) -> str:
    """Create an access token."""
    to_encode = data.copy()
    
    # Set expiration time
    current_time = datetime.now(timezone.utc)
    expire = current_time + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    # Add issued time and expiration time
    to_encode.update({
        "iat": current_time,  # Issued at time 
        "exp": expire         # Expiration time
    })
    
    # Encode JWT with algorithm
    token = jwt.encode(
        to_encode, 
        settings.SECRET_KEY, 
        algorithm=settings.JWT_ALGORITHM
    )
    
    return token

def create_refresh_token(data: dict) -> str:
    """Create a refresh token."""
    to_encode = data.copy()
    
    # Set expiration time
    current_time = datetime.now(timezone.utc)
    expire = current_time + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    
    # Add issued time and expiration time
    to_encode.update({
        "iat": current_time,  # Issued at time
        "exp": expire,        # Expiration time
        "token_type": "refresh"
    })
    
    # Encode JWT
    token = jwt.encode(
        to_encode, 
        settings.SECRET_KEY, 
        algorithm=settings.JWT_ALGORITHM
    )
    
    return token

def decode_token(token: str):
    """Decode and validate token."""
    try:
        payload = jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error decoding token: {e}")
        return None