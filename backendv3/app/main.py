"""Main application module for the Traffic Monitoring API."""

import sys
import os
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import pathlib # Added for explicit path handling

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration
from app.config import settings
# API_PREFIX might not be used if we simplify routes, but keep for now
# from app.api_standards import API_VERSION as API_PREFIX

# Import database utilities
from db.session import get_db, connect_to_mongo, close_mongo_connection

# Import auth router
from auth.router import router as auth_router
# Import new routers
from cameras.router import router as cameras_router
from favorites.router import router as favorites_router
from reports.reports_router import router as reports_router # Corrected import

# Import the new ML routers for UC6, UC11, UC12
from ml.routers.forecast_router import router as ml_forecast_router
from ml.routers.admin_model_router import router as ml_admin_model_router

# Removed other router imports:
# from admin.router import router as admin_router
# from video.router import router as video_router
# from ml.router import router as ml_router
# from map.router import router as map_router
# from external.router import router as external_router
# from forecasts.router import router as forecasts_router

# Import routers for UC12 and UC6
# from ml.routers.prediction_model_router import router as prediction_model_admin_router # UC12
# from forecasts.router import router as forecasts_user_router # UC6

# Define lifespan context manager to replace on_event handlers
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Connect to database
    await connect_to_mongo()
    yield
    # Shutdown: Close database connection
    await close_mongo_connection()

# Create FastAPI app with lifespan
app = FastAPI(
    title=settings.API_TITLE + " - Minimal (Auth Only)", # Modified title
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Setup CORS with configurable origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["Content-Type", "Set-Cookie", "Access-Control-Allow-Headers", 
                  "Access-Control-Allow-Origin", "Authorization", "X-Requested-With"],
    expose_headers=["Content-Type", "Set-Cookie"],
    max_age=600,  # 10 minutes cache for preflight requests
)
print("CORS Middleware added with origins:", settings.CORS_ORIGINS)

# --- Mount for camera thumbnails in backendv3/assets ---
# backendv3/app/main.py -> .. -> backendv3/ -> backendv3/assets/
camera_assets_dir = pathlib.Path(__file__).resolve().parent.parent / "assets"
camera_assets_dir.mkdir(parents=True, exist_ok=True) 
app.mount("/assets", StaticFiles(directory=camera_assets_dir), name="camera_assets")

# Create essential directories for general static files (if used elsewhere)
# This creates backendv3/app/static if settings.BASE_DIR is not a full path or is relative
# For clarity, ensure settings.BASE_DIR points to the intended root (e.g., backendv3 folder)
# If settings.BASE_DIR is backendv3, then os.path.join(settings.BASE_DIR, "static") is backendv3/static
# general_static_dir_path = os.path.join(settings.BASE_DIR, "static") # Assuming settings.BASE_DIR is backendv3
# For this example, let's assume settings.BASE_DIR is setup correctly for 'backendv3/static'
# If you only have 'backendv3/assets' and no 'backendv3/static', you might not need the /static mount.
# For this change, we focus on adding /assets. The original /static mount remains.

# Ensure the target directory for the original /static mount exists
# This depends on what settings.BASE_DIR resolves to.
# If settings.BASE_DIR points to your project root (e.g. HoangPhi) then this creates HoangPhi/static
# If settings.BASE_DIR points to backendv3, then this creates backendv3/static
target_static_path = pathlib.Path(settings.BASE_DIR) / "static"
target_static_path.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=target_static_path), name="static")

# Mount for report uploads
report_uploads_dir = pathlib.Path(__file__).resolve().parent / "uploads" / "reports"
report_uploads_dir.mkdir(parents=True, exist_ok=True)
app.mount("/uploads/reports", StaticFiles(directory=report_uploads_dir), name="report_images")

uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(uploads_dir, exist_ok=True)

# Add a simple health check route that follows CORS rules
@app.get("/api/health-check", tags=["Health"])
async def health_check():
    """Simple health check endpoint to verify CORS is working."""
    return {"status": "ok", "cors": "enabled"}

# Include routers
app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])
# Assuming auth_router also handles user creation which is linked to auth.
# If /api/users prefix serves other user management apart from registration,
# and those are meant to be removed, this line should be reviewed.
# For now, assuming it's for user registration tied to auth.
# app.include_router(auth_router, prefix="/api/users", tags=["User Management (via Auth)"]) # This line should be removed or commented out
# Include new routers
app.include_router(cameras_router, prefix="/api", tags=["Cameras"])
app.include_router(favorites_router, prefix="/api", tags=["Favorites"])
app.include_router(reports_router, prefix="/api", tags=["Reports & Notifications"])

# Include the new ML routers for UC6, UC11, UC12
app.include_router(ml_forecast_router, prefix="/api/ml", tags=["ML - User Forecasts"]) # UC06 related
app.include_router(ml_admin_model_router, prefix="/api/admin/ml", tags=["ML - Admin Model Overview"]) # UC11, UC12 related

# Include the new admin training data router
# app.include_router(
#     training_data_admin_router, 
#     prefix="/api/admin/training-data", 
#     tags=["Admin - Training Data Management"]
# )

# Include routers for UC12 (Prediction Model Management - Admin)
# app.include_router(
#     prediction_model_admin_router, 
#     prefix="/api/admin/prediction-models", 
#     tags=["Admin - Prediction Model Management"]
# )

# Include router for UC6 (Traffic Forecast - User)
# app.include_router(
#     forecasts_user_router, 
#     prefix="/api/forecasts", 
#     tags=["User - Traffic Forecasts"]
# )

# Removed other app.include_router calls:
# app.include_router(admin_router, prefix="/api/admin", tags=["Admin"])
# app.include_router(video_router, prefix="/api/videos", tags=["Video Processing"])
# app.include_router(ml_router, prefix="/api/ml", tags=["Machine Learning"])
# app.include_router(camera_router, prefix="/api/cameras", tags=["Cameras"])
# app.include_router(map_router, prefix="/api/map", tags=["Map"])
# app.include_router(external_router, prefix="/api/external", tags=["External Data"])
# app.include_router(forecasts_router, prefix="/api/forecasts", tags=["Forecasts"])

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Traffic Monitoring API",
        "version": settings.API_VERSION,
        "status": "active",
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "api_version": settings.API_VERSION,
        "environment": settings.ENVIRONMENT,
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled exceptions."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": f"An unexpected error occurred: {str(exc)}",
            "type": "server_error",
        },
        headers={"Access-Control-Allow-Origin": "http://localhost:3000"},
    )

# Add this for direct execution
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
