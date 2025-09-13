import logging
import os
import shutil
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.config import \
    settings  # Corrected import path, using the lowercase 'settings' instance
from auth.security import get_current_user  # Corrected import path
from bson import ObjectId
from db.models import (LocationCreate, LocationInDB, ReportCreate, ReportInDB,
                       ReportNotificationInDB, ReportNotificationPublic,
                       ReportPublic, ReportStatusLiteral, ReportTypeLiteral,
                       ReportUpdateAdmin, UserInDB)
from db.session import get_db  # Not directly used here, but good for context
from fastapi import File, Form, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ReturnDocument

# Configure logging
logger = logging.getLogger(__name__)

# Placeholder for file storage logic if not using a dedicated service
UPLOAD_DIR = "backendv3/app/uploads/reports"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Define a helper that might be similar to the one in cameras.router if general enough,
# or keep it specific here for report location handling.
async def get_or_create_location_for_report(
    db: AsyncIOMotorDatabase, 
    location_create_data: Optional[LocationCreate] = None, 
    direct_location_id: Optional[ObjectId] = None
) -> Optional[ObjectId]:
    if direct_location_id:
        existing_location = await db.locations.find_one({"_id": direct_location_id})
        if not existing_location:
            raise HTTPException(status_code=404, detail=f"Provided location_id {direct_location_id} not found")
        return direct_location_id
    
    if location_create_data:
        # Optional: Check for existing location by coordinates to avoid duplicates
        existing_location_by_coords = await db.locations.find_one(
            {"latitude": location_create_data.latitude, "longitude": location_create_data.longitude}
        )
        if existing_location_by_coords:
            return existing_location_by_coords["_id"]
        
        # Create new location
        new_loc_doc = location_create_data.model_dump(exclude_none=True)
        new_loc_doc["created_at"] = datetime.now(timezone.utc)
        new_loc_doc["updated_at"] = datetime.now(timezone.utc)
        created_location = await db.locations.insert_one(new_loc_doc)
        return created_location.inserted_id
    return None

class ReportService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.reports_collection = db["reports"]
        self.notifications_collection = db["report_notifications"]

    async def _save_report_image(self, report_id: ObjectId, image_file: UploadFile) -> Optional[str]:
        """Saves an uploaded image file for a report."""
        try:
            # Ensure filename is safe and unique if needed, here using report_id
            # For simplicity, directly using image_file.filename but consider sanitizing/uniquifying
            filename_ext = os.path.splitext(image_file.filename)[1]
            # Generate a more unique filename using report_id to avoid collisions
            unique_filename = f"{str(report_id)}{filename_ext}"
            
            file_path = os.path.join(UPLOAD_DIR, unique_filename)
            
            async with image_file.file as source_file:
                with open(file_path, "wb") as buffer:
                    buffer.write(source_file.read()) # Read from the async file object
            
            # Return the relative URL path that will be served by StaticFiles
            image_url_path = f"/uploads/reports/{unique_filename}"
            logger.info(f"Successfully saved image for report {report_id} to {file_path}, URL: {image_url_path}")
            return image_url_path
        except Exception as e:
            logger.error(f"Error saving image for report {report_id} (filename: {image_file.filename}): {e}")
            return None

    async def create_report(
        self,
        report_data: ReportCreate,
        current_user: UserInDB,
        image_file: Optional[UploadFile] = None,
        direct_location_id: Optional[ObjectId] = None,
        location_create_data: Optional[LocationCreate] = None,
    ) -> ReportPublic:
        """Creates a new report with optional image upload."""
        logger.info(f"User {current_user.username} attempting to create report: {report_data.title}")
        
        final_location_id = await get_or_create_location_for_report(
            self.db, location_create_data, direct_location_id
        )

        image_url_str: Optional[str] = None
        if image_file:
            if image_file.content_type not in ["image/jpeg", "image/png", "image/gif"]:
                raise HTTPException(
                    status_code=400, 
                    detail="Invalid image file type. Allowed types: JPEG, PNG, GIF."
                )
            # Max size check (e.g., 5MB)
            MAX_IMAGE_SIZE = 5 * 1024 * 1024 # 5MB
            size = await image_file.read() # Read to get size
            await image_file.seek(0) # Reset cursor to beginning for _save_report_image
            if len(size) > MAX_IMAGE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"Image size exceeds the maximum limit of {MAX_IMAGE_SIZE // (1024*1024)}MB."
                )

            image_url_str = await self._save_report_image(ObjectId(), image_file)
            if not image_url_str:
                # Decide if this is a critical failure. For now, we'll proceed without image if saving fails.
                logger.warning(f"Could not save image for report {report_data.title}, proceeding without image.")
                # Alternatively, raise HTTPException(status_code=500, detail="Failed to save report image.")
        
        report_doc = ReportInDB(
            **report_data.model_dump(exclude_none=True),
            user_id=current_user.id, # Changed from created_by
            created_by_username=current_user.username,
            submitted_at=datetime.now(timezone.utc), # Changed from created_at
            # status=ReportStatusLiteral("New"), # Ensure status is set using the Literal
            image_url=image_url_str, # Use the string path directly
            location_id=final_location_id # Set the resolved location_id
        )

        # report_doc_dict = report_doc.model_dump(exclude_none=True, by_alias=True)
        # Remove id if it's None and by_alias=True is used for _id, to avoid inserting null _id
        report_doc_dict = report_doc.model_dump(exclude_none=True)
        if 'id' in report_doc_dict and report_doc_dict['id'] is None:
            del report_doc_dict['id']

        try:
            result = await self.reports_collection.insert_one(report_doc_dict)
            created_report = await self.reports_collection.find_one({"_id": result.inserted_id})
            if not created_report:
                raise HTTPException(status_code=500, detail="Failed to create or retrieve report after insert")
            
            # Create notification for admins
            # For simplicity, assuming all admins should be notified. 
            # A more complex system might target specific admin groups.
            admin_users = await self.db.users.find({"role": "admin"}).to_list(length=None)
            if admin_users:
                for admin in admin_users:
                    notification_doc = ReportNotificationInDB(
                        report_id=created_report["_id"],
                        report_title=created_report["title"],
                        report_type=created_report["report_type"],
                        message=f"New report '{created_report['title']}' submitted by {created_report['created_by_username']}.",
                        admin_id=admin["_id"] # Link to specific admin
                    )
                    await self.notifications_collection.insert_one(notification_doc.model_dump(exclude_none=True, by_alias=True))
            
            return ReportPublic(**created_report)
        except Exception as e:
            # Log the exception details (e.g., logger.error(f"Error creating report: {e}"))
            raise HTTPException(status_code=500, detail=f"Database error during report creation: {str(e)}")

    async def get_report_by_id(self, report_id: str, current_user: UserInDB) -> Optional[ReportPublic]:
        """Fetches a single report by its ID."""
        try:
            report_oid = ObjectId(report_id)
        except Exception:
            logger.warning(f"Invalid report_id format: {report_id}")
            return None # Or raise HTTPException(status_code=400, detail="Invalid report ID format")

        report_db = await self.reports_collection.find_one({"_id": report_oid})
        if report_db:
            # Authorization: Admin can view any report. User can only view their own.
            if current_user.role != "admin" and report_db.get("user_id") != current_user.id:
                raise HTTPException(status_code=403, detail="Not authorized to view this report")
            logger.info(f"Report {report_id} fetched by user {current_user.username}")
            return ReportPublic(**report_db)
        logger.warning(f"Report {report_id} not found for user {current_user.username}")
        return None

    async def get_user_reports(self, current_user: UserInDB, skip: int = 0, limit: int = 20) -> List[ReportPublic]:
        """Fetches all reports submitted by the current user."""
        query = {"user_id": current_user.id}
        reports_cursor = self.reports_collection.find(query).sort("submitted_at", -1).skip(skip).limit(limit)
        reports_db = await reports_cursor.to_list(length=limit)
        
        reports = [ReportPublic(**report) for report in reports_db]
        logger.info(f"User {current_user.username} fetched {len(reports)} reports (skip={skip}, limit={limit}).")
        return reports

    async def get_all_reports_admin(
        self, 
        current_user: UserInDB, # Parameter kept for signature consistency, but admin check bypassed
        status: Optional[ReportStatusLiteral] = None,
        report_type: Optional[ReportTypeLiteral] = None, 
        search: Optional[str] = None,
        skip: int = 0, 
        limit: int = 100
    ) -> List[ReportPublic]:
        """Fetches all reports for admin, with optional filters."""
        # if not current_user.is_admin:
        #     logger.warning(f"Non-admin user {current_user.username} attempted to access get_all_reports_admin.")
        #     raise HTTPException(status_code=403, detail="Not authorized to perform this action")

        query = {}
        if status:
            query["status"] = status
        if report_type:
            query["report_type"] = report_type
        if search:
            # Simple search on title and description. For more complex search, consider text indexes.
            query["$or"] = [
                {"title": {"$regex": search, "$options": "i"}},
                {"description": {"$regex": search, "$options": "i"}},
            ]

        reports_cursor = self.reports_collection.find(query).skip(skip).limit(limit).sort("submitted_at", -1)
        reports_db = await reports_cursor.to_list(length=limit)
        reports = [ReportPublic(**report) for report in reports_db]
        logger.info(f"Admin {current_user.username} fetched {len(reports)} reports with filters (status={status}, type={report_type}, search='{search}', skip={skip}, limit={limit}).")
        return reports

    async def update_report_admin(
        self, 
        report_id: str, 
        report_update: ReportUpdateAdmin, 
        admin_user: UserInDB  # Parameter kept for signature consistency, but admin check bypassed
    ) -> Optional[ReportPublic]:
        if not ObjectId.is_valid(report_id):
            raise HTTPException(status_code=400, detail="Invalid report ID format")

        update_fields = report_update.model_dump(exclude_unset=True)
        if not update_fields:
            raise HTTPException(status_code=400, detail="No update data provided")

        update_fields["processed_at"] = datetime.now(timezone.utc)
        update_fields["handled_by_admin_id"] = admin_user.id

        result = await self.reports_collection.update_one(
            {"_id": ObjectId(report_id)},
            {"$set": update_fields}
        )

        if result.modified_count > 0:
            logger.info(f"Report {report_id} updated by admin {admin_user.username}. Updated fields: {update_fields}")

            # Gửi thông báo cho user tạo báo cáo
            if update_fields.get("status") and report_update.admin_reply and report_update.admin_reply.strip():
                try:
                    report_doc = await self.reports_collection.find_one({"_id": ObjectId(report_id)})
                    if report_doc and report_doc.get("user_id"):
                        await self.create_user_notification_for_report_update(
                            report_id=str(report_id),
                            user_id_to_notify=str(report_doc["user_id"]),
                            message=f"Admin replied to your report: '{report_update.admin_reply}'"
                        )
                        logger.info(f"Notification created for user {report_doc['user_id']} for report {report_id}.")
                except Exception as e:
                    logger.error(f"Failed to create notification for report {report_id}: {e}")

            updated_report = await self.reports_collection.find_one({"_id": ObjectId(report_id)})
            if updated_report:
                return ReportPublic(**updated_report)

        logger.error(f"Failed to update report {report_id} in database for admin {admin_user.username}.")
        return None

    async def _create_admin_notification_for_new_report(self, report: ReportInDB) -> Optional[ReportNotificationPublic]:
        """Creates a notification for ALL admins about a newly created report."""
        # This method is intended to notify admins when a new report is submitted.
        # The `user_id` in ReportNotificationInDB will be the ID of the admin to be notified.
        # Since we want to notify all admins, this might involve fetching all admin users
        # or having a generic "admin group" notification. For simplicity, we'll create a single
        # notification that isn't tied to a specific admin user_id for now, but is marked as an admin notification.
        # A more robust system would iterate through all users with an 'admin' role.

        if not report.id:
            logger.error("Cannot create notification for report without ID.")
            return None

        notification_doc = ReportNotificationInDB(
            report_id=report.id, 
            user_id=None, # This notification is for admins, not a specific user.
            message=f"New report submitted: '{report.title}' by {report.created_by_username}.",
            is_read=False,
            is_admin_notification=True, # Mark this as an admin-level notification
            created_at=datetime.utcnow()
        )
        try:
            inserted_notification = await self.notifications_collection.insert_one(jsonable_encoder(notification_doc))
            if inserted_notification.inserted_id:
                logger.info(f"Admin notification created for new report ID {report.id}")
                # Fetch and return the created notification as ReportNotificationPublic
                created_notification_db = await self.notifications_collection.find_one({"_id": inserted_notification.inserted_id})
                if created_notification_db:
                    return ReportNotificationPublic(**created_notification_db)
            logger.error(f"Failed to insert admin notification for report ID {report.id}")
            return None
        except Exception as e:
            logger.error(f"Error creating admin notification for report ID {report.id}: {e}")
            return None

    async def create_user_notification_for_report_update(
        self, 
        report_id: str, 
        user_id_to_notify: str,
        message: str
    ) -> Optional[ReportNotificationPublic]:
        """Creates a notification for a specific user about an update to their report."""
        try:
            report_obj_id = ObjectId(report_id)
            user_obj_id = ObjectId(user_id_to_notify)
        except Exception as e:
            logger.error(f"Invalid ObjectId format for report_id ({report_id}) or user_id_to_notify ({user_id_to_notify}): {e}")
            return None

        notification_doc = ReportNotificationInDB(
            report_id=report_obj_id,
            user_id=user_obj_id, # The user who submitted the report and should be notified
            message=message,
            is_read=False,
            is_admin_notification=False, # This is a user-specific notification
            created_at=datetime.utcnow()
        )
        try:
            inserted_notification = await self.notifications_collection.insert_one(jsonable_encoder(notification_doc))
            if inserted_notification.inserted_id:
                logger.info(f"User notification created for report ID {report_id} for user {user_id_to_notify}")
                created_notification_db = await self.notifications_collection.find_one({"_id": inserted_notification.inserted_id})
                if created_notification_db:
                    return ReportNotificationPublic(**created_notification_db)
            logger.error(f"Failed to insert user notification for report ID {report_id}")
            return None
        except Exception as e:
            logger.error(f"Error creating user notification for report ID {report_id}: {e}")
            return None

    async def get_admin_notifications(self, admin_user: UserInDB, skip: int = 0, limit: int = 20) -> List[ReportNotificationPublic]:
        """Fetches notifications for admins (i.e., new report submissions)."""
        # if not admin_user.is_admin:
        #     logger.warning(f"Non-admin user {admin_user.username} attempted to access admin notifications.")
        #     raise HTTPException(status_code=403, detail="Not authorized to perform this action")

        # Fetches notifications specifically marked for admins (e.g., new reports)
        query = {"is_admin_notification": True}
        
        notifications_cursor = self.notifications_collection.find(query).skip(skip).limit(limit).sort("created_at", -1)
        notifications_db = await notifications_cursor.to_list(length=limit)
        
        notifications = [ReportNotificationPublic(**notif) for notif in notifications_db]
        logger.info(f"Admin {admin_user.username} fetched {len(notifications)} admin notifications (skip={skip}, limit={limit}).")
        return notifications

    async def mark_notification_as_read(
        self, notification_id: str, current_user: UserInDB # Changed to current_user (expecting admin from router)
    ) -> Optional[ReportNotificationPublic]:
        """Marks a specific notification as read by the current user.
           Admins can mark any admin notification. Users can only mark their own non-admin notifications.
        """
        if not ObjectId.is_valid(notification_id):
            raise HTTPException(status_code=400, detail="Invalid notification ID format")

        # Ensure the user is an admin before allowing to mark as read
        if current_user.role != "admin":
             raise HTTPException(status_code=403, detail="Only admins can mark notifications as read.")

        notification = await self.notifications_collection.find_one_and_update(
            {"_id": ObjectId(notification_id), "admin_id": current_user.id}, # Ensure admin owns it or it's a general admin notification
            {"$set": {"is_read": True, "updated_at": datetime.now(timezone.utc)}},
            return_document=True
        )
        if not notification:
            # Check if it exists but doesn't belong to this admin (if notifications are admin-specific)
            # For now, if admin_id matches, it should find it.
            # If not found, it could be wrong ID or not targeted to this admin.
            existing_notif = await self.notifications_collection.find_one({"_id": ObjectId(notification_id)})
            if existing_notif and existing_notif.get("admin_id") != current_user.id:
                 raise HTTPException(status_code=403, detail="Notification does not belong to this admin.")
            raise HTTPException(status_code=404, detail="Notification not found.")
            
        return ReportNotificationPublic(**notification) 