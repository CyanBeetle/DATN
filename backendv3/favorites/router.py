"""
API Endpoints for Managing User Favorite Cameras (UC05).
"""
import os
import sys
from fastapi import APIRouter, Depends, HTTPException, status, Body
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from bson import ObjectId # Import ObjectId

# Add parent directory to path to import from other modules if necessary
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import actual auth dependencies and DB session
from auth.security import get_current_user
from db.session import get_db
from motor.motor_asyncio import AsyncIOMotorDatabase
from db.models import UserInDB, FavoriteCamera as FavoriteCameraDBModel, FavoriteCameraCreate # Use DB models

# --- Pydantic Models (can remain similar if they serve the API contract well) ---
class FavoriteCameraAPIBase(BaseModel): # Renamed to avoid clash if any
    camera_id: str = Field(..., example="60d5ecf00000000000000000") # Example ObjectId string

class FavoriteCameraAPICreate(FavoriteCameraAPIBase):
    notifications_enabled: Optional[bool] = False # Allow setting on creation

# Response model can be different from DB model if needed, e.g., string IDs
class FavoriteCameraAPIResponse(BaseModel):
    id: str
    user_id: str
    camera_id: str
    notifications_enabled: bool
    created_at: datetime

    class Config:
        orm_mode = True # Kept if useful, but direct dict conversion is fine too

class CheckFavoriteResponse(BaseModel):
    camera_id: str
    is_favorite: bool

router = APIRouter()

# Remove Mock DB and Mock Service
# mock_db_favorites: Dict[str, List[FavoriteCameraInDB]] = {}
# class MockFavoriteService: ...
# favorite_service = MockFavoriteService()

# --- User Favorite Endpoints (UC05) ---
# Prefix: /api (from main.py)

@router.get("/user/favorites", response_model=List[FavoriteCameraAPIResponse], summary="Get all favorite cameras for the logged-in user (UC05)")
async def get_my_favorites(
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db)
):
    user_id_obj = current_user.id
    favorites_cursor = db.favorite_cameras.find({"user_id": user_id_obj})
    favorites_list = []
    async for fav_db in favorites_cursor:
        fav_db_model = FavoriteCameraDBModel(**fav_db)
        favorites_list.append(FavoriteCameraAPIResponse(
            id=str(fav_db_model.id),
            user_id=str(fav_db_model.user_id),
            camera_id=str(fav_db_model.camera_id),
            notifications_enabled=fav_db_model.notifications_enabled,
            created_at=fav_db_model.created_at
        ))
    return favorites_list

@router.post("/user/favorites", response_model=FavoriteCameraAPIResponse, status_code=status.HTTP_201_CREATED, summary="Add a camera to favorites (UC05)")
async def add_my_favorite(
    favorite_create_api: FavoriteCameraAPICreate, 
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db)
):
    user_id_obj = current_user.id
    if not ObjectId.is_valid(favorite_create_api.camera_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid camera_id format")
    camera_id_obj = ObjectId(favorite_create_api.camera_id)

    # Check if camera exists (optional, but good practice)
    camera_exists = await db.cameras.find_one({"_id": camera_id_obj, "deleted": False})
    if not camera_exists:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera to be favorited not found")

    # Check if already favorited
    existing_favorite = await db.favorite_cameras.find_one({
        "user_id": user_id_obj, 
        "camera_id": camera_id_obj
    })
    if existing_favorite:
        # Return existing one or update its notification status if needed
        # For simplicity, just return existing if found (idempotent add)
        existing_fav_model = FavoriteCameraDBModel(**existing_favorite)
        return FavoriteCameraAPIResponse(
            id=str(existing_fav_model.id),
            user_id=str(existing_fav_model.user_id),
            camera_id=str(existing_fav_model.camera_id),
            notifications_enabled=existing_fav_model.notifications_enabled,
            created_at=existing_fav_model.created_at
        )

    fav_create_data = FavoriteCameraCreate(
        user_id=user_id_obj,
        camera_id=camera_id_obj,
        notifications_enabled=favorite_create_api.notifications_enabled
    )
    
    # created_at is handled by model default
    db_fav_doc = fav_create_data.model_dump(exclude_none=True)
    
    result = await db.favorite_cameras.insert_one(db_fav_doc)
    created_fav_doc = await db.favorite_cameras.find_one({"_id": result.inserted_id})
    if not created_fav_doc:
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create favorite")
    
    created_fav_model = FavoriteCameraDBModel(**created_fav_doc)
    return FavoriteCameraAPIResponse(
        id=str(created_fav_model.id),
        user_id=str(created_fav_model.user_id),
        camera_id=str(created_fav_model.camera_id),
        notifications_enabled=created_fav_model.notifications_enabled,
        created_at=created_fav_model.created_at
    )

@router.delete("/user/favorites/{camera_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Remove a camera from favorites (UC05)")
async def remove_my_favorite(
    camera_id: str, 
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db)
):
    user_id_obj = current_user.id
    if not ObjectId.is_valid(camera_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid camera_id format")
    camera_id_obj = ObjectId(camera_id)

    result = await db.favorite_cameras.delete_one({"user_id": user_id_obj, "camera_id": camera_id_obj})
    if result.deleted_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Favorite camera not found or already removed")
    return # No content

@router.get("/user/favorites/check/{camera_id}", response_model=CheckFavoriteResponse, summary="Check if a specific camera is a favorite (UC05)")
async def check_my_favorite_status(
    camera_id: str, 
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db)
):
    user_id_obj = current_user.id
    if not ObjectId.is_valid(camera_id):
        # Allow checking even if camera_id is invalid, will just return false
        return CheckFavoriteResponse(camera_id=camera_id, is_favorite=False)
        # raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid camera_id format")
    camera_id_obj = ObjectId(camera_id)

    existing_favorite = await db.favorite_cameras.find_one({"user_id": user_id_obj, "camera_id": camera_id_obj})
    return CheckFavoriteResponse(camera_id=camera_id, is_favorite=existing_favorite is not None) 