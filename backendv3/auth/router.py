"""Authentication routes."""
from datetime import datetime, timezone, timedelta
from typing import Optional
import logging
import time

from fastapi import APIRouter, Depends, HTTPException, status, Body, Response, Cookie, Request
from fastapi.security import OAuth2PasswordRequestForm

from app.config import settings, ERROR_MESSAGES, STATUS_MESSAGES
from db.models import UserInDB, UserCreate, UserPublic, Token, LoginResponse, LoginResponseUser
from db.session import get_db
from auth.security import verify_password, get_password_hash, get_current_user
from auth.jwt import create_access_token, create_refresh_token, decode_token

# Setup logging
logger = logging.getLogger(__name__)

router = APIRouter()

# Track failed login attempts
failed_login_attempts = {}
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_TIME = 15 * 60  # 15 minutes in seconds

# Define CurrentUserRequired class for type hints in router functions
class CurrentUserRequired(UserInDB):
    """Type alias for current user dependency."""
    pass

async def get_current_authenticated_user(current_user: UserInDB = Depends(get_current_user)) -> CurrentUserRequired:
    """Get the current authenticated user."""
    return CurrentUserRequired(**current_user.model_dump())  # Use model_dump() for Pydantic v2 compatibility

@router.post("/login", response_model=LoginResponse)
async def login_for_access_token(
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db = Depends(get_db)
):
    """Login endpoint to get JWT token.
    
    This endpoint authenticates a user and returns access and refresh tokens.
    Handles the following errors:
    - Rate limiting for excessive login attempts
    - Invalid username or email
    - Invalid password
    - Inactive user account
    """
    username = form_data.username.lower().strip()
    
    # Check for rate limiting
    current_time = time.time()
    if username in failed_login_attempts:
        attempts, lockout_time = failed_login_attempts[username]
        # If user is currently locked out
        if attempts >= MAX_FAILED_ATTEMPTS and current_time < lockout_time:
            wait_time = int(lockout_time - current_time)
            logger.warning(f"Login attempt for locked out user: {username}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Too many failed login attempts. Please try again in {wait_time} seconds."
            )
        # If lockout period is over, reset the counter
        elif current_time > lockout_time:
            failed_login_attempts[username] = (0, 0)
    
    # Find the user (support login via either username or email)
    user = None
    # First, try lookup by username
    user = await db.users.find_one({"username": username})
    # If not found, try as email
    if not user and "@" in username:
        user = await db.users.find_one({"email": username})
    
    # Handle invalid username
    if not user:
        # Track failed attempt
        if username in failed_login_attempts:
            attempts, lockout_time = failed_login_attempts[username]
            failed_login_attempts[username] = (attempts + 1, 
                                              current_time + LOCKOUT_TIME if attempts + 1 >= MAX_FAILED_ATTEMPTS else 0)
        else:
            failed_login_attempts[username] = (1, 0)
        
        logger.warning(f"Failed login attempt: User '{username}' not found")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES["USER_NOT_FOUND"],
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check password
    password_valid = verify_password(form_data.password, user["hashed_password"])
    if not password_valid:
        # Track failed attempt
        if username in failed_login_attempts:
            attempts, lockout_time = failed_login_attempts[username]
            failed_login_attempts[username] = (attempts + 1, 
                                              current_time + LOCKOUT_TIME if attempts + 1 >= MAX_FAILED_ATTEMPTS else 0)
        else:
            failed_login_attempts[username] = (1, 0)
        
        logger.warning(f"Failed login attempt: Invalid password for user '{username}'")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES["INVALID_CREDENTIALS"],
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is active
    user_model = UserInDB(**user)
    if not user_model.is_active:
        logger.warning(f"Login attempt for inactive user: {username}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail=ERROR_MESSAGES["INACTIVE_USER"]
        )
    
    # Login successful, reset failed attempts
    if username in failed_login_attempts:
        del failed_login_attempts[username]
    
    # Create tokens
    token_data = {
        "sub": user_model.username, 
        "role": user_model.role,
        "tv": user_model.token_version  # Add token_version
    }
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)
    
    # Set secure cookies
    secure = settings.ENVIRONMENT == "production"
    
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=60 * settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        path="/",
        samesite="lax",
        secure=secure
    )
    
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        max_age=60 * 60 * 24 * settings.REFRESH_TOKEN_EXPIRE_DAYS,
        path="/",
        samesite="lax",
        secure=secure
    )
    
    # Update last login
    await db.users.update_one(
        {"_id": user_model.id},
        {"$set": {"last_login": datetime.now(timezone.utc)}}
    )
    
    logger.info(f"Successful login for user: {username}")
    
    # Return login response
    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        refresh_token=refresh_token,
        user=LoginResponseUser(
            id=str(user_model.id),
            username=user_model.username,
            email=user_model.email,
            role=user_model.role,
            fullname=user_model.fullname
        )
    )

@router.post("/refresh", response_model=Token)
async def refresh_token_endpoint(
    response: Response,
    refresh_token: str = Cookie(None),
    refresh_token_body: str = Body(None, embed=True),
    db = Depends(get_db)
):
    """Refresh access token using a valid refresh token."""
    # Use refresh token from cookie or request body
    token = refresh_token or refresh_token_body
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token not provided",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Decode refresh token
    payload = decode_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Extract username
    username = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user exists
    user = await db.users.find_one({"username": username})
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_model = UserInDB(**user) # Create UserInDB instance to access token_version
    if not user_model.is_active:
        # Should not happen if refresh token is valid and user was active, but good to check
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User is inactive.")

    # Create new access token
    token_data = {
        "sub": username,
        "role": user_model.role, # Include role for consistency
        "tv": user_model.token_version # Add token_version
    }
    access_token = create_access_token(token_data)
    
    # Update the access token cookie
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=60 * settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        samesite="lax",
        secure=settings.SECURE_COOKIES
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        refresh_token=token  # Return the same refresh token
    )

@router.get("/me", response_model=UserPublic)
async def read_users_me(current_user: CurrentUserRequired = Depends(get_current_authenticated_user)):
    """Get current user profile."""
    return UserPublic(
        id=str(current_user.id),
        username=current_user.username,
        email=current_user.email,
        is_active=current_user.is_active,
        role=current_user.role,
        created_at=current_user.created_at,
        last_login=current_user.last_login
    )

@router.put("/me/password")
async def update_current_user_password(
    old_password: str = Body(..., embed=True),
    new_password: str = Body(..., embed=True),
    current_user: CurrentUserRequired = Depends(get_current_authenticated_user),
    db = Depends(get_db)
):
    """Update current user's password."""
    # Verify old password
    if not verify_password(old_password, current_user.hashed_password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect old password.")

    # Hash new password
    new_hashed_password = get_password_hash(new_password)

    # Increment token version and set timestamp for token invalidation
    new_token_version = (current_user.token_version or 0) + 1
    token_version_changed_time = datetime.now(timezone.utc)

    # Update password, token_version and token_version_changed_at
    await db.users.update_one(
        {"_id": current_user.id},
        {
            "$set": {
                "hashed_password": new_hashed_password,
                "token_version": new_token_version,
                "token_version_changed_at": token_version_changed_time,
                "updated_at": token_version_changed_time # Also update the general updated_at
            }
        }
    )
    
    logger.info(f"Password updated and token version incremented for user '{current_user.username}'")
    return {"message": "Password updated successfully"}

@router.post("/register", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_create: UserCreate,
    db = Depends(get_db)
):
    """Register a new user."""
    # Check if username exists
    existing_user = await db.users.find_one({"username": user_create.username})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # Check if email exists
    existing_email = await db.users.find_one({"email": user_create.email})
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Validate role
    valid_roles = ["admin", "user"]
    if user_create.role not in valid_roles:
        raise HTTPException(status_code=400, detail=f"Invalid role. Must be one of: {', '.join(valid_roles)}")
    
    # Hash the password
    hashed_password = get_password_hash(user_create.password)
    
    # Create user document
    user_dict = user_create.model_dump()  # Use model_dump() for Pydantic v2 compatibility
    del user_dict["password"]
    user_dict["hashed_password"] = hashed_password
    user_dict["created_at"] = datetime.now(timezone.utc)
    
    # Insert into database
    result = await db.users.insert_one(user_dict)
    
    # Get the created user
    created_user = await db.users.find_one({"_id": result.inserted_id})
    
    return UserPublic(
        id=str(created_user["_id"]),
        username=created_user["username"],
        email=created_user["email"],
        is_active=created_user.get("is_active", True),
        role=created_user.get("role", "user"),
        created_at=created_user["created_at"],
        last_login=created_user.get("last_login")
    )

@router.post("/logout")
async def logout(response: Response, current_user: CurrentUserRequired = Depends(get_current_user)):
    """Logout endpoint to clear authentication cookies.
    
    This endpoint clears the authentication cookies, effectively logging out the user.
    """
    # Clear cookies
    response.delete_cookie(key="access_token", path="/")
    response.delete_cookie(key="refresh_token", path="/")
    
    logger.info(f"User logged out: {current_user.username}")
    
    return {"message": STATUS_MESSAGES["LOGOUT_SUCCESS"]}

