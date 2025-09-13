import os
import json
import uuid
import sys
import shutil
from fastapi import APIRouter, BackgroundTasks, UploadFile, File, Form, HTTPException, Depends, status
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel
from bson import ObjectId

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video.processor import VideoProcessor
from auth.security import get_current_user
from db.models import VideoTask, ROI, UserInDB, VideoAnalysisData
from db.session import get_db
from app.config import settings

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.RESULTS_DIR, exist_ok=True)
os.makedirs(settings.VIDEO_INPUT_DIR, exist_ok=True)
os.makedirs(settings.VIDEO_OUTPUT_DIR, exist_ok=True)

router = APIRouter()

# --- Models ---
class ROIUpload(BaseModel):
    """ROI upload request."""
    points: List[List[int]]
    name: str

class VideoUploadForm(BaseModel):
    """Video upload form data."""
    roi_id: Optional[str] = None
    frame_skip: Optional[int] = 10
    max_duration: Optional[int] = None

class TaskResponse(BaseModel):
    """Task status response."""
    task_id: str
    status: str
    progress: int = 0
    filename: str
    start_time: str
    end_time: Optional[str] = None
    error: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None

# --- Helper Functions ---
def get_default_roi():
    """Generate default ROI if none is specified"""
    # Basic default ROI - covers roughly 80% of a standard frame
    return [
        [100, 100],   # top-left
        [540, 100],   # top-right
        [540, 380],   # bottom-right
        [100, 380]    # bottom-left
    ]

async def update_task_progress(db, task_id, progress, status=None):
    """Update task progress in database"""
    update_data = {"progress": progress}
    if status:
        update_data["status"] = status
    
    await db.video_tasks.update_one(
        {"task_id": task_id},
        {"$set": update_data}
    )

async def process_video_background_task(
    task_id: str,
    input_path: str,
    output_path: str,
    roi_points: List[List[int]],
    target_width: float,
    target_height: float,
    frame_skip: int,
    max_duration: Optional[int],
    db
):
    """Background task to process video."""
    try:
        # Update task status to processing
        await update_task_progress(db, task_id, 1, "processing")
        
        # Initialize processor
        processor = VideoProcessor(
            model_name="yolov8x.pt",
            confidence=0.3,
            use_gpu=True
        )
        
        # Define progress update callback
        async def update_progress(progress):
            await update_task_progress(db, task_id, progress)
        
        # Process the video
        stats = processor.process_video(
            input_path=input_path,
            output_path=output_path,
            roi_points=roi_points,
            target_width=target_width,
            target_height=target_height,
            frame_skip=frame_skip,
            max_duration=max_duration,
            progress_callback=update_progress,
            result_prefix=task_id
        )
        
        # Extract results path from stats
        json_path = stats.get("json_path")
        if json_path and os.path.exists(json_path):
            with open(json_path, 'r') as f:
                result_data = json.load(f)
            
            # Create VideoAnalysisData document
            video_analysis = VideoAnalysisData(
                task_id=task_id,
                filename=os.path.basename(input_path),
                created_at=datetime.utcnow(),
                processed_date=datetime.now().strftime("%d/%m/%Y/%H/%M/%S"),
                duration_seconds=stats.get("processing_time", 0),
                video_info={
                    "width": stats.get("frame_width", 0),
                    "height": stats.get("frame_height", 0),
                    "fps": stats.get("fps", 0),
                    "total_frames": stats.get("total_frames", 0),
                    "processed_frames": stats.get("processed_frames", 0)
                },
                roi_config={
                    "points": roi_points,
                    "width_meters": target_width,
                    "height_meters": target_height
                },
                unique_vehicles=result_data.get("unique_vehicles", []),
                vehicle_count=stats.get("unique_vehicles", 0),
                avg_speed=None,  # Will be calculated if data is available
                time_intervals=result_data.get("time_intervals", [])
            )
            
            # Save to database
            await db.video_analysis_data.insert_one(
                video_analysis.model_dump(by_alias=True, exclude={"id"})
            )
        
        # Mark task as completed
        await update_task_progress(db, task_id, 100, "completed")
        
        # Trigger prediction job (Phase 3)
        from ml.prediction_service import trigger_prediction_job
        await trigger_prediction_job(task_id, db)
        
        # Update task with stats
        await db.video_tasks.update_one(
            {"task_id": task_id},
            {"$set": {
                "stats": stats,
                "completed_at": datetime.utcnow(),
                "output_path": output_path,
                "json_path": json_path
            }}
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        # Update task as failed
        await db.video_tasks.update_one(
            {"task_id": task_id},
            {"$set": {
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.utcnow()
            }}
        )

# --- Routes ---
@router.post("/upload-roi")
async def upload_roi(
    roi_data: ROIUpload,
    current_user: UserInDB = Depends(get_current_user),
    db = Depends(get_db)
):
    """Save ROI points and return ID."""
    # Validate ROI points
    if len(roi_data.points) != 4:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ROI must have exactly 4 points"
        )

    # Create ROI object
    roi_doc = ROI(
        name=roi_data.name,
        points=roi_data.points,
        created_by=current_user.username,
        created_at=datetime.utcnow()
    )

    # Insert into database
    result = await db.roi.insert_one(roi_doc.model_dump(by_alias=True, exclude={'id'}))

    return {"roi_id": str(result.inserted_id)}

@router.get("/rois")
async def get_rois(
    current_user: UserInDB = Depends(get_current_user),
    db = Depends(get_db)
):
    """Get all ROIs."""
    cursor = db.roi.find({})
    rois = await cursor.to_list(length=100)
    
    # Process for response
    for roi in rois:
        if "_id" in roi:
            roi["id"] = str(roi["_id"])
            del roi["_id"]
    
    return rois

@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    current_user: UserInDB = Depends(get_current_user),
    db = Depends(get_db)
):
    """Upload a video file for later processing."""
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    safe_filename = f"{timestamp}_{unique_id}_{file.filename.replace('..', '')}"
    file_path = os.path.join(settings.UPLOAD_DIR, safe_filename)
    
    # Save uploaded file
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")
    
    # Create task entry
    task = VideoTask(
        task_id=task_id,
        status="uploaded",
        progress=0,
        filename=file.filename,
        file_path=file_path,
        created_by=current_user.username,
        created_at=datetime.utcnow()
    )
    
    await db.video_tasks.insert_one(task.model_dump(by_alias=True, exclude={"id"}))
    
    return {
        "task_id": task_id,
        "filename": file.filename,
        "file_path": file_path,
        "message": "Video uploaded successfully"
    }

@router.post("/process/{task_id}")
async def process_video(
    task_id: str,
    background_tasks: BackgroundTasks,
    roi_id: Optional[str] = Form(None),
    frame_skip: int = Form(10),
    max_duration: Optional[int] = Form(None),
    target_width: float = Form(24.0),  # Default ROI width in meters
    target_height: float = Form(30.0), # Default ROI height in meters
    current_user: UserInDB = Depends(get_current_user),
    db = Depends(get_db)
):
    """Process a previously uploaded video."""
    # Find task
    task = await db.video_tasks.find_one({"task_id": task_id})
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Check if task is already processing or completed
    if task["status"] in ["processing", "completed"]:
        return {
            "task_id": task_id,
            "status": task["status"],
            "message": f"Video is already {task['status']}"
        }
    
    # Get ROI points
    roi_points = None
    if roi_id:
        try:
            roi_object_id = ObjectId(roi_id)
            roi_doc = await db.roi.find_one({"_id": roi_object_id})
            if roi_doc:
                roi_points = roi_doc["points"]
        except Exception:
            pass
    
    # Use default ROI if none provided
    if roi_points is None:
        roi_points = get_default_roi()
        print(f"Using default ROI: {roi_points}")
    
    # Set output paths
    input_path = task["file_path"]
    output_filename = f"processed_{os.path.basename(input_path)}"
    output_path = os.path.join(settings.VIDEO_OUTPUT_DIR, output_filename)
    
    # Update task status
    await db.video_tasks.update_one(
        {"task_id": task_id},
        {"$set": {
            "status": "pending",
            "progress": 0,
            "roi_points": roi_points,
            "frame_skip": frame_skip,
            "max_duration": max_duration,
            "target_width": target_width,
            "target_height": target_height,
            "updated_at": datetime.utcnow()
        }}
    )
    
    # Start background processing task
    background_tasks.add_task(
        process_video_background_task,
        task_id=task_id,
        input_path=input_path,
        output_path=output_path,
        roi_points=roi_points,
        target_width=target_width,
        target_height=target_height,
        frame_skip=frame_skip,
        max_duration=max_duration,
        db=db
    )
    
    return {
        "task_id": task_id,
        "status": "pending",
        "message": "Video processing started"
    }

@router.post("/upload_process")
async def upload_and_process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    roi_id: Optional[str] = Form(None),
    frame_skip: int = Form(10),
    max_duration: Optional[int] = Form(None),
    target_width: float = Form(24.0),  # Default ROI width in meters
    target_height: float = Form(30.0), # Default ROI height in meters
    current_user: UserInDB = Depends(get_current_user),
    db = Depends(get_db)
):
    """Upload and immediately process a video file in one step."""
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    safe_filename = f"{timestamp}_{unique_id}_{file.filename.replace('..', '')}"
    file_path = os.path.join(settings.UPLOAD_DIR, safe_filename)
    
    # Save uploaded file
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")
    
    # Get ROI points
    roi_points = None
    if roi_id:
        try:
            roi_object_id = ObjectId(roi_id)
            roi_doc = await db.roi.find_one({"_id": roi_object_id})
            if roi_doc:
                roi_points = roi_doc["points"]
        except Exception:
            pass
    
    # Use default ROI if none provided
    if roi_points is None:
        roi_points = get_default_roi()
    
    # Set output paths
    output_filename = f"processed_{os.path.basename(file_path)}"
    output_path = os.path.join(settings.VIDEO_OUTPUT_DIR, output_filename)
    
    # Create task entry
    task = VideoTask(
        task_id=task_id,
        status="pending",  # Set directly to pending as we'll process immediately
        progress=0,
        filename=file.filename,
        file_path=file_path,
        roi_points=roi_points,
        frame_skip=frame_skip,
        max_duration=max_duration,
        target_width=target_width,
        target_height=target_height,
        created_by=current_user.username,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    await db.video_tasks.insert_one(task.model_dump(by_alias=True, exclude={"id"}))
    
    # Start background processing task
    background_tasks.add_task(
        process_video_background_task,
        task_id=task_id,
        input_path=file_path,
        output_path=output_path,
        roi_points=roi_points,
        target_width=target_width,
        target_height=target_height,
        frame_skip=frame_skip,
        max_duration=max_duration,
        db=db
    )
    
    return {
        "task_id": task_id,
        "filename": file.filename,
        "file_path": file_path,
        "status": "pending",
        "message": "Video upload complete and processing started"
    }

@router.get("/tasks/{task_id}")
async def get_task(
    task_id: str,
    current_user: UserInDB = Depends(get_current_user),
    db = Depends(get_db)
):
    """Get task status."""
    task = await db.video_tasks.find_one({"task_id": task_id})
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Convert ObjectId to string
    if "_id" in task:
        task["id"] = str(task["_id"])
        del task["_id"]
    
    # Format datetime objects
    for date_field in ['created_at', 'updated_at', 'completed_at']:
        if date_field in task and isinstance(task[date_field], datetime):
            task[date_field] = task[date_field].isoformat()
    
    return task

@router.get("/tasks")
async def get_tasks(
    limit: int = 20,
    current_user: UserInDB = Depends(get_current_user),
    db = Depends(get_db)
):
    """Get list of video tasks."""
    cursor = db.video_tasks.find().sort("created_at", -1).limit(limit)
    tasks = await cursor.to_list(length=limit)
    
    # Process tasks for response
    for task in tasks:
        # Convert ObjectId to string
        if "_id" in task:
            task["id"] = str(task["_id"])
            del task["_id"]
        
        # Format datetime objects
        for date_field in ['created_at', 'updated_at', 'completed_at']:
            if date_field in task and isinstance(task[date_field], datetime):
                task[date_field] = task[date_field].isoformat()
    
    return tasks

@router.get("/analysis/{task_id}")
async def get_analysis(
    task_id: str,
    current_user: UserInDB = Depends(get_current_user),
    db = Depends(get_db)
):
    """Get video analysis results."""
    analysis = await db.video_analysis_data.find_one({"task_id": task_id})
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis data not found")
    
    # Convert ObjectId to string
    if "_id" in analysis:
        analysis["id"] = str(analysis["_id"])
        del analysis["_id"]
    
    # Format datetime objects
    if "created_at" in analysis and isinstance(analysis["created_at"], datetime):
        analysis["created_at"] = analysis["created_at"].isoformat()
    
    return analysis

@router.get("/results/{task_id}/video")
async def get_video_result(
    task_id: str,
    current_user: UserInDB = Depends(get_current_user),
    db = Depends(get_db)
):
    """Get processed video file."""
    task = await db.video_tasks.find_one({"task_id": task_id})
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if "output_path" not in task or not os.path.exists(task["output_path"]):
        raise HTTPException(status_code=404, detail="Processed video not found")
    
    return FileResponse(path=task["output_path"], filename=os.path.basename(task["output_path"]))

@router.get("/results/{task_id}/json")
async def get_json_result(
    task_id: str,
    current_user: UserInDB = Depends(get_current_user),
    db = Depends(get_db)
):
    """Get JSON analysis data."""
    task = await db.video_tasks.find_one({"task_id": task_id})
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if "json_path" not in task or not os.path.exists(task["json_path"]):
        raise HTTPException(status_code=404, detail="JSON result not found")
    
    return FileResponse(path=task["json_path"], filename=os.path.basename(task["json_path"]))