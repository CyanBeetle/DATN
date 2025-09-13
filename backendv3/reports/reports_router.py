from typing import List, Optional

from auth.security import get_admin_user, get_current_user
from bson import ObjectId
from db.models import (LocationCreate, PyObjectId, ReportCreate,
                       ReportNotificationPublic, ReportPublic,
                       ReportStatusLiteral, ReportTypeLiteral,
                       ReportUpdateAdmin, UserInDB)
from db.session import get_db
from fastapi import (APIRouter, Depends, File, Form, HTTPException, Query,
                     UploadFile)
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel

from .report_service import ReportService

router = APIRouter()

# --- Helper to get ReportService instance ---
def get_report_service(db: AsyncIOMotorDatabase = Depends(get_db)) -> ReportService:
    return ReportService(db)

# Define a simple Location model for report submission if not using one from db.models directly for API body
class ReportLocationData(BaseModel):
    name: Optional[str] = None
    latitude: float
    longitude: float

# --- User-facing Report Endpoints (UC09) ---
@router.post("/reports", response_model=ReportPublic, status_code=201)
async def submit_report(
    report_title: str = Form(...),
    report_description: str = Form(...),
    report_type: ReportTypeLiteral = Form(...),
    location_id: Optional[str] = Form(None),
    location_name: Optional[str] = Form(None),
    location_latitude: Optional[float] = Form(None),
    location_longitude: Optional[float] = Form(None),
    image: Optional[UploadFile] = File(None),
    current_user: UserInDB = Depends(get_current_user),
    report_service: ReportService = Depends(get_report_service)
):
    """Submit a new report. Image is optional."""
    
    report_location_data: Optional[LocationCreate] = None
    final_location_id_for_report: Optional[PyObjectId] = None

    if location_id:
        if not PyObjectId.is_valid(location_id): # Changed from ObjectId to PyObjectId
            raise HTTPException(status_code=400, detail="Invalid location_id format provided.")
        final_location_id_for_report = PyObjectId(location_id) # Changed from ObjectId to PyObjectId
        # Service should verify this location_id exists
    elif location_latitude is not None and location_longitude is not None:
        report_location_data = LocationCreate(
            name=location_name,
            latitude=location_latitude,
            longitude=location_longitude
        )
        # Service will handle get_or_create for this LocationCreate data
    
    report_data = ReportCreate(
        title=report_title,
        description=report_description,
        report_type=report_type,
        # location_id will be handled by the service based on final_location_id_for_report or report_location_data
        # Temporarily pass None, service layer will populate it.
        location_id=None 
    )
    # The service method create_report will need to be updated to accept 
    # final_location_id_for_report and report_location_data (or a consolidated location input)
    return await report_service.create_report(
        report_data, 
        current_user, 
        image_file=image, 
        # Pass location info to service layer
        direct_location_id=final_location_id_for_report,
        location_create_data=report_location_data
    )

@router.get("/reports/my-reports", response_model=List[ReportPublic])
async def get_my_reports(
    skip: int = 0,
    limit: int = 20,
    current_user: UserInDB = Depends(get_current_user),
    report_service: ReportService = Depends(get_report_service)
):
    """Get all reports submitted by the currently authenticated user."""
    return await report_service.get_user_reports(current_user, skip=skip, limit=limit)

@router.get("/reports/{report_id}", response_model=ReportPublic)
async def get_report_details(
    report_id: str,
    current_user: UserInDB = Depends(get_current_user),
    report_service: ReportService = Depends(get_report_service)
):
    """Get details of a specific report. 
    Users can view their own reports. Admins can view any report.
    """
    report = await report_service.get_report_by_id(report_id, current_user)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    # Authorization: Admin can view any report. User can only view their own.
    if current_user.role != "admin" and report.created_by != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to view this report")
    
    return report

# --- Admin-facing Report Management Endpoints (UC14) ---
@router.get("/admin/reports", response_model=List[ReportPublic])
async def get_all_reports_for_admin(
    status: Optional[ReportStatusLiteral] = Query(None),
    report_type: Optional[ReportTypeLiteral] = Query(None),
    search: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    admin_user: UserInDB = Depends(get_admin_user),
    report_service: ReportService = Depends(get_report_service)
):
    """Get all reports for admin, with filtering and pagination."""
    return await report_service.get_all_reports_admin(
        admin_user, status, report_type, search, skip, limit
    )

@router.patch("/admin/reports/{report_id}", response_model=ReportPublic)
async def update_report_by_admin(
    report_id: str,
    report_update: ReportUpdateAdmin,
    admin_user: UserInDB = Depends(get_admin_user),
    report_service: ReportService = Depends(get_report_service)
):
    updated_report = await report_service.update_report_admin(report_id, report_update, admin_user)
    if not updated_report:
        raise HTTPException(status_code=404, detail="Report not found or update failed")
    return updated_report

# --- Admin Notification Endpoints (UC14 related) ---
@router.get("/admin/notifications", response_model=List[ReportNotificationPublic])
async def get_report_notifications_for_admin(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    admin_user: UserInDB = Depends(get_admin_user),
    report_service: ReportService = Depends(get_report_service)
):
    """Get report submission notifications for admins."""
    return await report_service.get_admin_notifications(admin_user, skip, limit)

@router.patch("/admin/notifications/{notification_id}/read", response_model=ReportNotificationPublic)
async def mark_report_notification_read(
    notification_id: str,
    admin_user: UserInDB = Depends(get_admin_user),
    report_service: ReportService = Depends(get_report_service)
):
    """Mark a report notification as read (admin only)."""
    notification = await report_service.mark_notification_as_read(notification_id, admin_user)
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found or update failed")
    return notification