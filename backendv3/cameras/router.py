"""
API Endpoints for Camera Management (UC13) and User Camera Viewing (UC04).
"""
import os
import sys
from fastapi import APIRouter, Depends, HTTPException, status, Body
from typing import List, Dict, Any, Optional, Tuple
from pydantic import HttpUrl, BaseModel, field_validator # Keep HttpUrl if used by models
from datetime import datetime
import uuid 
import logging
import pathlib
import shutil

# Added imports
import cv2 
import numpy as np # Added for Playwright image processing
import asyncio
from urllib.parse import urlparse, parse_qs # Added for URL parsing

# Playwright import
from playwright.async_api import async_playwright, Error, TimeoutError as PlaywrightTimeoutError, Request, Route

# Import database components
from db.session import get_db
from motor.motor_asyncio import AsyncIOMotorDatabase # For type hinting
from bson import ObjectId
from pymongo.errors import DuplicateKeyError

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import actual models from db.models
from db.models import ( 
    CameraInDB, # Use the actual DB model
    CameraCreate, # Renamed from CameraCreate in db.models to avoid clash
    CameraUpdate, # Renamed from CameraUpdate in db.models
    UserInDB,
    LocationCreate, # New model for creating locations
    LocationInDB,   # New model for locations in DB
    ROIDBModel,
    ROIPointDBModel, # Make sure this is imported if ROIAPIBody uses it
)

# Import actual auth dependencies
from auth.security import get_current_user as get_current_active_user, get_admin_user

# Add import for congestion calculator
from cameras.congestion import CongestionCalculator

# Setup logger for this module
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # You can set the level to logging.DEBUG for more verbose output

# --- Pydantic Models for API interface (can be refined based on DBCameraBase) ---
# These models define the API contract. They can be similar to db.models or subset/superset.

class LocationAPIBody(BaseModel):
    name: Optional[str] = "Unknown Location"
    latitude: float
    longitude: float

class ROIPointAPI(BaseModel): # Renamed
    x: float # Normalized (0-1)
    y: float # Normalized (0-1)

class ROIAPIBody(BaseModel): # Renamed
    points: Optional[List[ROIPointAPI]] = None 
    # normalized_points: Optional[List[List[float]]] = None # This might be internal to DB model
    roi_width_meters: Optional[float] = None 
    roi_height_meters: Optional[float] = None 

class CameraAPIBase(BaseModel):
    name: str
    description: Optional[str] = None
    stream_url: HttpUrl
    status: str # Active, Inactive, Maintenance
    location_id: Optional[str] = None # To be provided if location already exists
    location_data: Optional[LocationAPIBody] = None # To create/use a new location
    thumbnail_url: Optional[str] = None

class CameraAPICreate(CameraAPIBase):
    # Inherits all from CameraAPIBase
    # Ensure that either location_id or location_data is provided, or make both optional and handle logic
    @field_validator('location_id', mode='before', check_fields=True)
    def check_location_fields(cls, v, values):
        if v is None and values.get('location_data') is None:
            # Allowing no location for now, can be made stricter
            # raise ValueError('Either location_id or location_data must be provided')
            pass # Making location optional for flexibility
        if v is not None and values.get('location_data') is not None:
            raise ValueError('Provide either location_id or location_data, not both')
        return v

class CameraAPIUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    stream_url: Optional[HttpUrl] = None
    status: Optional[str] = None
    location_id: Optional[str] = None
    location_data: Optional[LocationAPIBody] = None # To update/create location
    thumbnail_url: Optional[str] = None
    roi: Optional[ROIAPIBody] = None

    @field_validator('location_id', mode='before', check_fields=True)
    def check_update_location_fields(cls, v, values):
        if v is not None and values.get('location_data') is not None:
            raise ValueError('Provide either location_id or location_data for update, not both')
        return v

class CameraAPIResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    stream_url: str
    status: str 
    location_id: Optional[str] = None
    location_detail: Optional[LocationAPIBody] = None # To send full location details
    thumbnail_url: Optional[str] = None
    roi: Optional[ROIAPIBody] = None
    online: bool
    deleted: bool
    created_at: datetime
    updated_at: datetime
    congestion_level: Optional[int] = None
    congestion_text: Optional[str] = None

class CameraStatusUpdateAPI(BaseModel):
    status: str # Active, Inactive, Maintenance

class CameraROIUpdateAPI(BaseModel):
    roi_points: List[ROIPointAPI] # Normalized points from frontend
    roi_dimensions: Dict[str, float] # { "width": meters, "height": meters }

class CongestionResponseItem(BaseModel):
    camera_id: str
    name: str
    congestion_level: int
    congestion_text: str
    vehicle_count: int
    vehicle_density: float
    roi_area_m2: Optional[float] = None # Added ROI area
    roi_defined: bool = False # Added flag if ROI exists
    success: bool
    error: Optional[str] = None

class AllCongestionResponse(BaseModel):
    message: str
    cameras_processed: int
    results: List[CongestionResponseItem]

router = APIRouter()

# Define static path for captured frames - UPDATED to be relative to this file's parent's parent directory
# backendv3/cameras/router.py -> backendv3/ -> backendv3/assets/
STATIC_FRAMES_DIR = pathlib.Path(__file__).resolve().parent.parent / "assets"
STATIC_FRAMES_DIR.mkdir(parents=True, exist_ok=True) # Ensure directory exists

# Initialize congestion calculator
congestion_calculator = CongestionCalculator()

# --- Cache for Congestion Data ---
# Simple in-memory cache: {camera_id: (timestamp, data)}
congestion_cache: Dict[str, Tuple[datetime, Dict[str, Any]]] = {}
CACHE_TTL_SECONDS = 60  # 1 minute

# --- Helper function to convert CameraInDB to CameraAPIResponse ---
async def db_camera_to_api_response(camera_db: CameraInDB, db: AsyncIOMotorDatabase) -> CameraAPIResponse:
    camera_data = camera_db.model_dump(exclude_none=True)
    camera_id_str = str(camera_db.id)
    camera_data["id"] = camera_id_str
    camera_data["stream_url"] = str(camera_db.stream_url) 

    # Fetch and add location details if location_id exists
    if camera_db.location_id:
        camera_data["location_id"] = str(camera_db.location_id)
        location_doc = await db.locations.find_one({"_id": camera_db.location_id})
        if location_doc:
            camera_data["location_detail"] = LocationAPIBody(**location_doc)
    
    if camera_db.roi:
        api_roi_points = None
        if camera_db.roi.points:
            api_roi_points = [ROIPointAPI(x=p.x, y=p.y) for p in camera_db.roi.points]
        
        camera_data["roi"] = ROIAPIBody(
            points=api_roi_points,
            roi_width_meters=camera_db.roi.roi_width_meters,
            roi_height_meters=camera_db.roi.roi_height_meters
        )
    else:
        camera_data["roi"] = None

    # For the API response, ensure thumbnail_url is correctly prefixed with /assets/
    if camera_data.get("thumbnail_url") and not camera_data["thumbnail_url"].startswith('/assets/'):
        camera_data["thumbnail_url"] = f"/assets/{camera_data['thumbnail_url'].lstrip('/')}"
    elif not camera_data.get("thumbnail_url"):
        camera_data["thumbnail_url"] = None # Ensure it's null if not present

    # --- Populate on-demand/cached congestion data ---
    cached_data = congestion_cache.get(camera_id_str)
    current_time = datetime.utcnow()
    
    if cached_data and (current_time - cached_data[0]).total_seconds() < CACHE_TTL_SECONDS:
        congestion_info = cached_data[1]
        # print(f"Cache HIT for camera {camera_id_str}")
    else:
        # print(f"Cache MISS for camera {camera_id_str} - Calculating congestion")
        try:
            if camera_db.roi and camera_db.roi.points and camera_db.roi.roi_width_meters and camera_db.roi.roi_height_meters:
                thumbnail_path_str = None
                if camera_db.thumbnail_url:
                    relative_thumb_path = camera_db.thumbnail_url.lstrip('/')
                    potential_thumb_path = STATIC_FRAMES_DIR / relative_thumb_path
                    if potential_thumb_path.exists() and potential_thumb_path.is_file():
                        thumbnail_path_str = str(potential_thumb_path)
                    else:
                        logger.warning(f"Thumbnail file not found for camera {camera_id_str} at {potential_thumb_path}. Congestion calculation might be inaccurate or skipped.")

                if thumbnail_path_str:
                    roi_points_dict_list = [p.model_dump() for p in camera_db.roi.points]
                    # Use the global congestion_calculator instance from congestion.py
                    temp_congestion_result = congestion_calculator.calculate_congestion(
                        frame_path=thumbnail_path_str,
                        roi_points=roi_points_dict_list,
                        roi_width_meters=camera_db.roi.roi_width_meters,
                        roi_height_meters=camera_db.roi.roi_height_meters
                    )
                    congestion_info = {
                        "congestion_level": temp_congestion_result.get("congestion_level"),
                        "congestion_text": temp_congestion_result.get("congestion_text"),
                        "vehicle_count": temp_congestion_result.get("vehicle_count", 0),
                        "vehicle_density": temp_congestion_result.get("vehicle_density", 0.0)
                    }
                else:
                    congestion_info = {"congestion_level": 0, "congestion_text": "Thumbnail not available for calculation", "vehicle_count": 0, "vehicle_density": 0.0}
            else:
                 congestion_info = {"congestion_level": 0, "congestion_text": "ROI not defined", "vehicle_count": 0, "vehicle_density": 0.0}
            congestion_cache[camera_id_str] = (current_time, congestion_info)
            # print(f"Cache UPDATED for camera {camera_id_str}")
        except Exception as e:
            print(f"Error calculating congestion for {camera_id_str}: {e}")
            congestion_info = {"congestion_level": None, "congestion_text": "Error", "vehicle_count": None, "vehicle_density": None}

    camera_data.update(congestion_info)
    
    # Ensure all fields required by CameraAPIResponse are present
    # Defaulting missing optional fields explicitly
    response_model_fields = CameraAPIResponse.model_fields
    for field_name in response_model_fields:
        if field_name not in camera_data and not response_model_fields[field_name].is_required():
            camera_data[field_name] = response_model_fields[field_name].default
        elif field_name not in camera_data and response_model_fields[field_name].is_required():
             # This case should ideally not happen if model_dump and updates are correct
            print(f"Warning: Required field {field_name} missing for CameraAPIResponse for camera {camera_id_str}")
            camera_data[field_name] = None # Or raise error

    return CameraAPIResponse(**camera_data)

# --- Helper function to get or create location ---
async def get_or_create_location_id(
    db: AsyncIOMotorDatabase, 
    location_data: Optional[LocationAPIBody] = None, 
    location_id_str: Optional[str] = None
) -> Optional[ObjectId]:
    if location_id_str:
        if not ObjectId.is_valid(location_id_str):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid location_id format")
        existing_location = await db.locations.find_one({"_id": ObjectId(location_id_str)})
        if not existing_location:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Location with id {location_id_str} not found")
        return ObjectId(location_id_str)
    
    if location_data:
        # Check if a location with similar coordinates already exists to avoid duplicates
        # This is a simple check; more sophisticated duplicate detection might be needed
        existing_location = await db.locations.find_one(
            {"latitude": location_data.latitude, "longitude": location_data.longitude}
        )
        if existing_location:
            return existing_location["_id"]
        
        # Create new location
        new_loc_data = LocationCreate(**location_data.model_dump())
        created_location = await db.locations.insert_one(new_loc_data.model_dump(exclude_none=True))
        return created_location.inserted_id
    return None

# --- Helper function for synchronous image processing ---
def _process_screenshot_sync(
    screenshot_bytes: bytes, 
    camera_id_sync: str, 
    target_width: int, 
    save_dir: pathlib.Path
) -> Optional[str]:
    try:
        nparr = np.frombuffer(screenshot_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            logger.error(f"Error: Could not decode screenshot for camera {camera_id_sync}")
            return None

        # Resize frame to thumbnail
        if frame.shape[1] == 0:
            logger.error(f"Error: Decoded frame width is 0 for camera {camera_id_sync}. Cannot resize.")
            return None
        
        target_height = int(frame.shape[0] * (target_width / frame.shape[1]))
        if target_height <= 0:
            logger.error(f"Error: Calculated thumbnail height is non-positive for camera {camera_id_sync}. Cannot resize.")
            return None
        
        thumbnail = cv2.resize(frame, (target_width, target_height))

        filename = f"{camera_id_sync}.jpg"
        filepath = save_dir / filename
        
        # Try to save the image
        success = cv2.imwrite(str(filepath), thumbnail)
        
        # Even if imwrite returns False, check if file was still created
        if os.path.exists(filepath):
            if not success:
                logger.warning(f"OpenCV imwrite returned False, but file exists for {camera_id_sync} at {filepath}")
            logger.info(f"Successfully saved image for {camera_id_sync} to {filepath}")
            return filename
        else:
            logger.error(f"Error: Failed to save thumbnail for {camera_id_sync} to {filepath}")
            return None
        
    except cv2.error as e:
        logger.error(f"OpenCV error during sync processing for camera {camera_id_sync}: {e}")
        
        # Check if the file was created despite the error
        filename = f"{camera_id_sync}.jpg"
        filepath = save_dir / filename
        if os.path.exists(filepath):
            logger.warning(f"Despite OpenCV error, file exists for {camera_id_sync}. Returning filename.")
            return filename
        return None
    except Exception as e:
        logger.error(f"Generic error during sync processing for camera {camera_id_sync}: {e}", exc_info=True)
        
        # Check if the file was created despite the error
        filename = f"{camera_id_sync}.jpg"
        filepath = save_dir / filename
        if os.path.exists(filepath):
            logger.warning(f"Despite processing error, file exists for {camera_id_sync}. Returning filename.")
            return filename
        return None

# --- Helper for Playwright capture ---
async def _capture_and_save_frame_playwright(stream_url: str, camera_id_str: str, db: AsyncIOMotorDatabase) -> Optional[str]:
    playwright_instance = None
    browser = None
    page = None
    context = None
    thumbnail_width = 640  # Increased from 320 to 640 for better quality
        
    logger.info(f"Playwright: Taking screenshot for camera {camera_id_str} from {stream_url}")

    try:
        playwright_instance = await async_playwright().start()
        browser = await playwright_instance.chromium.launch(
            headless=True,
            args=[
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-accelerated-2d-canvas',
                '--no-first-run',
                '--no-zygote',
                '--disable-gpu',
                '--mute-audio'
            ]
        )
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},  # Increased from 1280x720 for better quality
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            ignore_https_errors=True,
            java_script_enabled=True,
        )
        page = await context.new_page()

        # Enhanced event logging
        page.on("console", lambda msg: logger.info(f"PAGE CONSOLE ({camera_id_str} - {msg.type} - URL: {msg.location.get('url', 'N/A')}): {msg.text}"))
        page.on("pageerror", lambda exc: logger.error(f"PAGE ERROR ({camera_id_str}): {exc}"))
        page.on("requestfailed", lambda request: logger.warning(f"REQUEST FAILED ({camera_id_str}): {request.method} {request.url} - Failure: {request.failure().error_text if request.failure() else 'Unknown'}"))
        page.on("response", lambda response: logger.info(f"RESPONSE ({camera_id_str}): {response.status} {response.request.method} {response.url}"))

        async def permissive_route_handler(route: Route, request: Request):
            logger.debug(f"ROUTING ({camera_id_str}): {request.method} {request.url} (Nav: {request.is_navigation_request()})")
            try:
                # Always try to continue. Log if specific conditions are met or if it's just a general resource.
                if request.is_navigation_request() and request.url == stream_url:
                    logger.info(f"Playwright: Allowing main navigation request to {request.url} for camera {camera_id_str}")
                elif "giaothong.hochiminhcity.gov.vn" in request.url: # Or other relevant domains
                    logger.debug(f"Playwright: Attempting to continue request to relevant domain: {request.url}")
                else:
                    logger.debug(f"Playwright: Continuing request to other domain: {request.url}")
                
                await route.continue_()

            except Error as e: # Playwright-specific error for route.continue_() or route.abort()
                logger.error(f"Playwright: Error in route_handler for {request.url} in camera {camera_id_str}: {e}")
                # Abort non-navigation requests if continue fails, to free up resources or prevent hangs.
                if not request.is_navigation_request():
                    try:
                        # Check if response already sent or request is already handled to avoid errors on abort
                        if not route.request.response() and not route._loop.is_closed() and route.request._connection:
                            await route.abort()
                            logger.warning(f"Playwright: Aborted request {request.url} after continue failed.")
                    except Error as abort_e: # Error during abort itself
                        logger.error(f"Playwright: Error aborting request {request.url} (after continue failed): {abort_e}")
            except Exception as e: # Catch any other unexpected errors in the handler
                logger.error(f"Playwright: Generic error in route_handler for {request.url} (camera {camera_id_str}): {e}")
                if not request.is_navigation_request():
                    try:
                        if not route.request.response() and not route._loop.is_closed() and route.request._connection:
                             await route.abort()
                             logger.warning(f"Playwright: Aborted non-navigation request {request.url} due to generic error in handler.")
                    except Error as abort_e: # Error during abort itself
                         logger.error(f"Playwright: Error aborting request {request.url} (generic handler error): {abort_e}")

        await page.route("**/*", permissive_route_handler)
        
        logger.info(f"Playwright: Navigating to {stream_url} for camera {camera_id_str}")
        try:
            # Using domcontentloaded and timeout as per user instruction
            # Assuming user wants to revert to 30000ms timeout for goto based on their last message
            await page.goto(stream_url, wait_until="domcontentloaded", timeout=30000) 
        except PlaywrightTimeoutError as pte_goto:
            logger.error(f"Playwright GOTO TimeoutError for {camera_id_str} from {stream_url}: {str(pte_goto)}")
            try:
                page_content_on_error = await page.content()
                logger.info(f"""Playwright: Page content on GOTO TIMEOUT for {camera_id_str}:
{page_content_on_error[:2000]}...""") # Log first 2KB
            except Exception as ce:
                logger.error(f"Playwright: Could not get page content on GOTO TIMEOUT for {camera_id_str}: {ce}")
            raise # Re-raise to be caught by the main try-except PlaywrightTimeoutError

        # Give the page more time to fully load and render the video stream, especially after domcontentloaded
        await page.wait_for_timeout(20000)
        
        logger.info(f"Playwright: Taking screenshot for {camera_id_str}")
        screenshot_bytes = None # Initialize here
        try:
            screenshot_bytes = await page.screenshot(timeout=20000, type="jpeg", quality=90)
        except PlaywrightTimeoutError as pte_screenshot:
            logger.error(f"Playwright SCREENSHOT TimeoutError for {camera_id_str}: {str(pte_screenshot)}")
            try:
                page_content_on_error = await page.content()
                logger.info(f"""Playwright: Page content on SCREENSHOT TIMEOUT for {camera_id_str}:
{page_content_on_error[:2000]}...""")
            except Exception as ce:
                logger.error(f"Playwright: Could not get page content on SCREENSHOT TIMEOUT for {camera_id_str}: {ce}")
            raise # Re-raise to be caught by the main try-except PlaywrightTimeoutError

        if not screenshot_bytes:
            logger.warning(f"Playwright: Failed to capture image for {camera_id_str}")
            return None

        # Process the screenshot
        loop = asyncio.get_event_loop()
        thumbnail_relative_path = await loop.run_in_executor(
            None, _process_screenshot_sync, screenshot_bytes, camera_id_str, thumbnail_width, STATIC_FRAMES_DIR
        )
        
        if thumbnail_relative_path: # If screenshot processed and file supposedly saved
            logger.info(f"Playwright: Successfully processed screenshot for {camera_id_str} to {thumbnail_relative_path}")
            # Update the camera record with the new thumbnail URL
            try:
                old_thumbnail = None
                camera_record = await db.cameras.find_one({"_id": ObjectId(camera_id_str)})
                if camera_record and "thumbnail_url" in camera_record:
                    old_thumbnail = camera_record["thumbnail_url"]
                
                await db.cameras.update_one(
                    {"_id": ObjectId(camera_id_str)},
                    {"$set": {
                        "thumbnail_url": f"/{thumbnail_relative_path.lstrip('/')}", # Ensure leading slash
                        "previous_thumbnail_url": old_thumbnail,
                        "thumbnail_updated_at": datetime.utcnow()
                    }}
                )
                logger.info(f"Playwright: Database updated with new live thumbnail for {camera_id_str}")
                return thumbnail_relative_path # Return filename on success
            except Exception as e:
                logger.error(f"Playwright: Error updating thumbnail in database for {camera_id_str} after successful capture: {str(e)}")
                # Even if DB update fails, the file was saved, so we might still return the path
                # but log it as a critical error. For now, let's consider it a partial success if file exists.
                return thumbnail_relative_path # File exists, frontend might still use it temporarily
        else: # Screenshot processing failed or file not saved by _process_screenshot_sync
            logger.warning(f"Playwright: Failed to process or save live image for {camera_id_str}. Attempting to use backup.")
            # Attempt to use backup
            backup_dir = pathlib.Path("C:/Users/admin/Desktop/CapstoneApp/HoangPhi/backendv3/assets/backup")
            backup_filename = f"{camera_id_str}.jpg"
            backup_filepath = backup_dir / backup_filename
            target_copied_filepath = STATIC_FRAMES_DIR / backup_filename

            if backup_filepath.exists() and backup_filepath.is_file():
                try:
                    shutil.copy(str(backup_filepath), str(target_copied_filepath))
                    logger.info(f"Playwright: Successfully copied backup thumbnail {backup_filename} to {target_copied_filepath} for camera {camera_id_str}")
                    
                    # Update DB to point to the backup image
                    old_thumbnail = None
                    camera_record = await db.cameras.find_one({"_id": ObjectId(camera_id_str)})
                    if camera_record and "thumbnail_url" in camera_record:
                        old_thumbnail = camera_record["thumbnail_url"]

                    await db.cameras.update_one(
                        {"_id": ObjectId(camera_id_str)},
                        {"$set": {
                            "thumbnail_url": f"/{backup_filename.lstrip('/')}", # Ensure leading slash
                            "previous_thumbnail_url": old_thumbnail, # Could be the failed live attempt or older backup
                            "thumbnail_updated_at": datetime.utcnow(),
                            "used_backup_thumbnail": True
                        }}
                    )
                    logger.info(f"Playwright: Database updated with backup thumbnail for {camera_id_str}")
                    return backup_filename # Return the filename of the copied backup
                except Exception as e_copy_db:
                    logger.error(f"Playwright: Error copying or updating DB with backup for {camera_id_str}: {e_copy_db}")
                    return None # Failed to use backup effectively
            else:
                logger.warning(f"Playwright: Backup thumbnail {backup_filename} not found at {backup_filepath} for camera {camera_id_str}.")
                return None

    except PlaywrightTimeoutError as pte:
        logger.error(f"Playwright TimeoutError for {camera_id_str} from {stream_url}: {str(pte)}")
        return None
    except Error as e: 
        logger.error(f"Playwright Error for {camera_id_str} from {stream_url}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Playwright: Generic error processing {camera_id_str} from {stream_url}: {str(e)}", exc_info=True)
        return None
    finally:
        if page:
            try: await page.close()
            except Exception as e_page_close: logger.error(f"Playwright: Error closing page for {camera_id_str}: {str(e_page_close)}")
        if context:
            try: await context.close()
            except Exception as e_context_close: logger.error(f"Playwright: Error closing context for {camera_id_str}: {str(e_context_close)}")
        if browser:
            try: await browser.close()
            except Exception as e_browser_close: logger.error(f"Playwright: Error closing browser for {camera_id_str}: {str(e_browser_close)}")
        if playwright_instance:
            try: await playwright_instance.stop()
            except Exception as e_pw_stop: logger.error(f"Playwright: Error stopping playwright for {camera_id_str}: {str(e_pw_stop)}")
        logger.info(f"Playwright: Closed resources for {camera_id_str}")

@router.get("/admin/cameras", response_model=List[CameraAPIResponse], summary="List all cameras for admin (UC13)")
async def list_cameras_admin(
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: UserInDB = Depends(get_admin_user)
):
    cameras_cursor = db.cameras.find({"deleted": False}) # Filter out deleted cameras
    # return [db_camera_to_api_response(CameraInDB(**cam)) async for cam in cameras_cursor]
    # Need to pass db to the helper now
    return [await db_camera_to_api_response(CameraInDB(**cam), db) async for cam in cameras_cursor]

@router.get("/admin/cameras/{camera_id}", response_model=CameraAPIResponse, summary="Get camera details for admin (UC13)")
async def get_camera_details_admin(
    camera_id: str, 
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: UserInDB = Depends(get_admin_user)
):
    if not ObjectId.is_valid(camera_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid camera_id format")
    camera = await db.cameras.find_one({"_id": ObjectId(camera_id), "deleted": False})
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found or has been deleted")
    # return db_camera_to_api_response(CameraInDB(**camera))
    return await db_camera_to_api_response(CameraInDB(**camera), db)

@router.get("/cameras", response_model=List[CameraAPIResponse], summary="List all active cameras for users (UC04)")
async def list_cameras_user(
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user)
):
    cameras_cursor = db.cameras.find({"status": "Active", "online": True, "deleted": False})
    # return [db_camera_to_api_response(CameraInDB(**cam)) async for cam in cameras_cursor]
    return [await db_camera_to_api_response(CameraInDB(**cam), db) async for cam in cameras_cursor]

@router.get("/cameras/{camera_id}", response_model=CameraAPIResponse, summary="Get specific camera details for user (UC04)")
async def get_camera_details_user(
    camera_id: str, 
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user)
):
    if not ObjectId.is_valid(camera_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid camera_id format")
    camera = await db.cameras.find_one({
        "_id": ObjectId(camera_id), 
        "status": "Active", 
        "online": True, 
        "deleted": False
    })
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found, not active, or offline")
    # return db_camera_to_api_response(CameraInDB(**camera))
    return await db_camera_to_api_response(CameraInDB(**camera), db)

@router.post("/admin/cameras", response_model=CameraAPIResponse, status_code=status.HTTP_201_CREATED, summary="Add a new camera (UC13)")
async def add_camera(
    camera_create_api: CameraAPICreate, # Use the API model for input
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: UserInDB = Depends(get_admin_user)
):
    # Handle location_id or location_data
    final_location_id = await get_or_create_location_id(
        db, camera_create_api.location_data, camera_create_api.location_id
    )

    camera_dict = camera_create_api.model_dump(exclude_none=True, exclude={"location_data", "location_id"})
    camera_dict["location_id"] = final_location_id
    
    # Explicitly convert URL fields to strings
    if "stream_url" in camera_dict and camera_dict["stream_url"] is not None:
        camera_dict["stream_url"] = str(camera_dict["stream_url"])
    if "thumbnail_url" in camera_dict and camera_dict.get("thumbnail_url") is not None: # Check if key exists and is not None
        camera_dict["thumbnail_url"] = str(camera_dict["thumbnail_url"])
    
    # Use CameraCreate for database object creation (from db.models)
    # This step helps validate the structure based on CameraCreate Pydantic model
    db_camera_create = CameraCreate(**camera_dict)
    
    try:
        # Prepare the payload for MongoDB insertion
        # model_dump from db_camera_create should ideally use json_encoders if CameraCreate inherits from MongoBaseModel
        # However, to be absolutely sure, we'll ensure string conversion on the final dict.
        insert_payload = db_camera_create.model_dump(exclude_none=True)
        
        # Final explicit conversion for safety, directly on the payload to be inserted
        if "stream_url" in insert_payload and not isinstance(insert_payload["stream_url"], str):
            insert_payload["stream_url"] = str(insert_payload["stream_url"])
        
        if "thumbnail_url" in insert_payload and insert_payload.get("thumbnail_url") is not None and \
           not isinstance(insert_payload["thumbnail_url"], str):
            insert_payload["thumbnail_url"] = str(insert_payload["thumbnail_url"])

        # Save to DB
        result = await db.cameras.insert_one(insert_payload)
        created_camera_doc = await db.cameras.find_one({"_id": result.inserted_id})
        
        if not created_camera_doc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create or retrieve camera after insert")
        
        # Convert the document from DB (which is a dict) to CameraInDB Pydantic model
        created_camera_model = CameraInDB(**created_camera_doc)
        
        # Then convert CameraInDB to CameraAPIResponse for the API response
        return await db_camera_to_api_response(created_camera_model, db)
        
    except DuplicateKeyError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Camera with name '{db_camera_create.name}' already exists.")
    except Exception as e:
        # Log the exception e
        print(f"Error in add_camera: {e}") # Basic logging
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {str(e)}")

@router.put("/admin/cameras/{camera_id}", response_model=CameraAPIResponse, summary="Update camera details (UC13)")
async def update_camera(
    camera_id: str, 
    camera_update_api: CameraAPIUpdate, # Use the API model for input
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: UserInDB = Depends(get_admin_user)
):
    if not ObjectId.is_valid(camera_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid camera_id format")

    existing_camera = await db.cameras.find_one({"_id": ObjectId(camera_id), "deleted": False})
    if not existing_camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found or has been deleted")

    update_data = camera_update_api.model_dump(exclude_none=True, exclude={"location_data", "location_id"})

    # Handle location update
    if camera_update_api.location_id or camera_update_api.location_data:
        final_location_id = await get_or_create_location_id(
            db, camera_update_api.location_data, camera_update_api.location_id
        )
        update_data["location_id"] = final_location_id
    elif "location_id" in update_data and update_data["location_id"] is None: # Explicitly setting location_id to null
        update_data["location_id"] = None

    if not update_data: # No actual data to update
        # return db_camera_to_api_response(CameraInDB(**existing_camera))
        return await db_camera_to_api_response(CameraInDB(**existing_camera), db)

    update_data["updated_at"] = datetime.utcnow()
    
    # Use CameraUpdate for database update preparation (from db.models if specific fields are enforced)
    # For now, direct update_data is fine if CameraAPIUpdate reflects allowed fields
    await db.cameras.update_one({"_id": ObjectId(camera_id)}, {"$set": update_data})
    
    updated_camera = await db.cameras.find_one({"_id": ObjectId(camera_id)})
    if not updated_camera:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve camera after update")
    # return db_camera_to_api_response(CameraInDB(**updated_camera))
    return await db_camera_to_api_response(CameraInDB(**updated_camera), db)

@router.delete("/admin/cameras/{camera_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a camera (UC13)")
async def delete_camera_admin(
    camera_id: str, 
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: UserInDB = Depends(get_admin_user)
):
    if not ObjectId.is_valid(camera_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid camera ID format.")

    # Soft delete: mark as deleted
    result = await db.cameras.update_one(
        {"_id": ObjectId(camera_id)},
        {"$set": {"deleted": True, "online": False, "status": "Inactive", "updated_at": datetime.utcnow()}}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found.")
    
    # Optionally, attempt to delete the thumbnail file from /assets
    try:
        camera_data = await db.cameras.find_one({"_id": ObjectId(camera_id)}) # Fetch to get thumbnail_url
        if camera_data and camera_data.get("thumbnail_url"):
            thumb_filename = camera_data["thumbnail_url"].split('/')[-1]
            thumb_path = STATIC_FRAMES_DIR / thumb_filename
            if thumb_path.exists():
                thumb_path.unlink()
                print(f"Deleted thumbnail file for soft-deleted camera: {thumb_path}")
    except Exception as e:
        print(f"Error deleting thumbnail for soft-deleted camera {camera_id}: {e}")
    
    return None # No content response

@router.put("/admin/cameras/{camera_id}/status", response_model=CameraAPIResponse, summary="Update camera status (UC13)")
async def update_camera_status_admin(
    camera_id: str, 
    status_update: CameraStatusUpdateAPI, 
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: UserInDB = Depends(get_admin_user)
):
    if not ObjectId.is_valid(camera_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid camera ID format.")

    valid_statuses = ["Active", "Inactive", "Maintenance"]
    if status_update.status not in valid_statuses:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid status. Must be one of: {valid_statuses}")

    update_values = {"status": status_update.status, "updated_at": datetime.utcnow()}
    if status_update.status == "Inactive" or status_update.status == "Maintenance":
        update_values["online"] = False
    elif status_update.status == "Active":
        update_values["online"] = True
        
    result = await db.cameras.update_one(
        {"_id": ObjectId(camera_id), "deleted": False},
        {"$set": update_values}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found or has been deleted.")
        
    updated_camera_db = await db.cameras.find_one({"_id": ObjectId(camera_id)})
    return await db_camera_to_api_response(CameraInDB(**updated_camera_db), db)

@router.put("/admin/cameras/{camera_id}/roi", response_model=CameraAPIResponse, summary="Update camera ROI (UC13)")
async def update_camera_roi_admin(
    camera_id: str, 
    roi_update: CameraROIUpdateAPI, 
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: UserInDB = Depends(get_admin_user)
):
    if not ObjectId.is_valid(camera_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid camera ID format.")

    if len(roi_update.roi_points) != 4:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="ROI must have exactly 4 points.")
    
    for i, point_api in enumerate(roi_update.roi_points):
        if not (0 <= point_api.x <= 1 and 0 <= point_api.y <= 1):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"ROI point {i} coords must be normalized (0-1).")

        width = roi_update.roi_dimensions.get("width")
        height = roi_update.roi_dimensions.get("height")
    if not width or width <= 0 or not height or height <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid ROI dimensions.")
    if width > 1000 or height > 1000: # Sanity check
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="ROI dimensions too large (max 1000m).")

    # Convert ROIPointAPI list to ROIPointDBModel list
    db_roi_points = [ROIPointDBModel(x=p.x, y=p.y) for p in roi_update.roi_points]
    
    new_roi_db_model = ROIDBModel(
        points=db_roi_points,
        roi_width_meters=width,
        roi_height_meters=height
    )
    
    result = await db.cameras.update_one(
        {"_id": ObjectId(camera_id), "deleted": False},
        {"$set": {"roi": new_roi_db_model.model_dump(), "updated_at": datetime.utcnow()}}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found or has been deleted.")
        
    updated_camera_db = await db.cameras.find_one({"_id": ObjectId(camera_id)})
    return await db_camera_to_api_response(CameraInDB(**updated_camera_db), db)

@router.get("/admin/cameras/{camera_id}/capture-frame", response_model=Dict[str, str], summary="Capture/get a fresh frame for ROI setup (UC13)")
async def capture_camera_frame_admin(
    camera_id: str, 
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: UserInDB = Depends(get_admin_user),
):
    if not ObjectId.is_valid(camera_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid camera ID format.")
        
    camera = await db.cameras.find_one({"_id": ObjectId(camera_id), "deleted": False})
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found or deleted.")

    # _capture_and_save_frame_playwright now returns just the filename (e.g., "camera_id.jpg")
    # The database is updated correctly within _capture_and_save_frame_playwright to store "/camera_id.jpg"
    thumbnail_filename = await _capture_and_save_frame_playwright(str(camera["stream_url"]), str(camera["_id"]), db)
    
    if not thumbnail_filename:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to capture camera frame.")
    
    # Construct the correct web-accessible path for the frontend
    frame_url_path = f"/assets/{thumbnail_filename.lstrip('/')}"
    
    return {"frame_url": frame_url_path}

@router.post("/cameras/refresh-all-thumbnails", summary="Refresh thumbnails for all cameras (User Accessible)")
async def refresh_all_camera_thumbnails(
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user) # Changed to current_active_user
):
    cameras_cursor = db.cameras.find({"deleted": False}) # Refresh for all non-deleted cameras
    all_cameras_db = await cameras_cursor.to_list(length=None)

    if not all_cameras_db:
        return {"message": "No cameras available to refresh thumbnails.", "total_cameras": 0, "success_count": 0, "fail_count": 0}

    print(f"Admin {current_user.username} initiated refresh for {len(all_cameras_db)} camera thumbnails.")
    success_count = 0
    fail_count = 0

    # Process cameras sequentially
    for cam_db_dict in all_cameras_db:
        cam_obj = CameraInDB(**cam_db_dict) # Convert to Pydantic model
        logger.info(f"Refreshing thumbnail for camera: {cam_obj.name} (ID: {cam_obj.id})")
        try:
            thumbnail_filename = await _capture_and_save_frame_playwright(str(cam_obj.stream_url), str(cam_obj.id), db)
            if thumbnail_filename:
                success_count += 1
                logger.info(f"Successfully refreshed thumbnail for {cam_obj.name}")
            else:
                fail_count += 1
                logger.warning(f"Failed to refresh thumbnail for {cam_obj.name} (no filename returned)")
        except Exception as e:
            fail_count += 1
            logger.error(f"Exception while refreshing thumbnail for {cam_obj.name}: {str(e)}")
        # Optional: Add a small delay between requests if still facing issues
        # await asyncio.sleep(1) # e.g., 1-second delay

    return {
        "message": f"Thumbnail refresh process completed for {len(all_cameras_db)} cameras.",
        "total_cameras": len(all_cameras_db),
        "success_count": success_count,
        "fail_count": fail_count
    }

@router.post("/cameras/calculate-all-congestion", response_model=AllCongestionResponse, summary="Calculate congestion for ALL active cameras (User Accessible)")
async def calculate_all_congestion(
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user) # User accessible
):
    # As per System Spec: "Real-time Congestion Calculation: ... stateless and on-demand ... latest result per camera is calculated and served; no historical records are saved."
    # This endpoint will calculate for all *Active, Online, Non-Deleted* cameras and return the results. It will NOT update the database.
    
    cameras_cursor = db.cameras.find({"status": "Active", "deleted": False, "online": True})
    active_cameras_db = await cameras_cursor.to_list(length=None)

    if not active_cameras_db:
        return AllCongestionResponse(message="No active cameras found to calculate congestion.", cameras_processed=0, results=[])

    results_list = []
    processed_count = 0

    for cam_dict in active_cameras_db:
        cam = CameraInDB(**cam_dict)
        processed_count += 1
        
        roi_defined = False
        roi_points_api = []
        roi_width_m = 0.0
        roi_height_m = 0.0

        if cam.roi and cam.roi.points and len(cam.roi.points) == 4 and cam.roi.roi_width_meters and cam.roi.roi_height_meters:
            roi_defined = True
            # Convert ROIPointDBModel to Dict for congestion_calculator
            roi_points_api = [{"x": p.x, "y": p.y} for p in cam.roi.points]
            roi_width_m = cam.roi.roi_width_meters
            roi_height_m = cam.roi.roi_height_meters
        
        calc_success = False
        error_msg = None
        congestion_data = {
            "congestion_level": 0, 
            "congestion_text": "No Data", 
            "vehicle_count": 0, 
            "vehicle_density": 0.0,
            "roi_area_m2": None
        }

        if not cam.thumbnail_url:
            error_msg = "Thumbnail not available for congestion calculation. Please refresh thumbnails."
        elif not roi_defined:
            error_msg = "ROI not defined or incomplete for this camera."
            congestion_data["congestion_text"] = "No ROI"
        else:
            frame_path = str(STATIC_FRAMES_DIR / cam.thumbnail_url.split('/')[-1])
            if not os.path.exists(frame_path):
                error_msg = f"Thumbnail file not found at {frame_path}. Please refresh thumbnails."
            else:
                try:
                    # Call the refactored calculate_congestion
                    raw_congestion_result = congestion_calculator.calculate_congestion(
                        frame_path=frame_path,
                        roi_points=roi_points_api, # List of Dicts
                        roi_width_meters=roi_width_m,
                        roi_height_meters=roi_height_m
                    )

                    if "error" in raw_congestion_result:
                        calc_success = False
                        error_msg = raw_congestion_result.get("error", "Unknown calculation error")
                        congestion_data.update({
                            "congestion_text": "Error",
                            "congestion_level": 0, # Ensure defaults on error
                            "vehicle_count": 0,
                            "vehicle_density": 0.0
                        })
                    else:
                        calc_success = True
                        congestion_data.update({
                            "congestion_level": raw_congestion_result.get("congestion_level"),
                            "congestion_text": raw_congestion_result.get("congestion_text"),
                            "vehicle_count": raw_congestion_result.get("vehicle_count"),
                            "vehicle_density": raw_congestion_result.get("vehicle_density"),
                            "roi_area_m2": raw_congestion_result.get("roi_area_m2")
                        })
                except Exception as e:
                    error_msg = f"Internal error during calculation: {str(e)}"
                    congestion_data["congestion_text"] = "Error"
                    calc_success = False
        
        results_list.append(CongestionResponseItem(
            camera_id=str(cam.id),
            name=cam.name,
            congestion_level=congestion_data["congestion_level"],
            congestion_text=congestion_data["congestion_text"],
            vehicle_count=congestion_data["vehicle_count"],
            vehicle_density=congestion_data["vehicle_density"],
            roi_area_m2=congestion_data["roi_area_m2"],
            roi_defined=roi_defined,
            success=calc_success,
            error=error_msg
        ))

    return AllCongestionResponse(
        message=f"Calculated congestion for {processed_count} active cameras.",
        cameras_processed=processed_count,
        results=results_list
    )