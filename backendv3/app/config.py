"""
Configuration for the Traffic Monitoring API

This module loads and manages configuration from environment variables
and provides configuration constants for the application.
"""

import os
from pydantic_settings import BaseSettings
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Define API version directly instead of importing from backendv3.api.api_endpoints
API_VERSION = "/api"

class Settings(BaseSettings):
    """Settings loaded from environment variables."""
    
    # API Configuration
    API_PREFIX: str = API_VERSION
    API_TITLE: str = "Traffic Monitoring API"
    API_DESCRIPTION: str = "API for traffic monitoring system with video processing and congestion prediction"
    API_VERSION: str = "1.0.0"
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")  # "development", "testing", "production"
      # CORS Configuration
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173", "http://127.0.0.1:8000"]
    
    # Authentication
    SECRET_KEY: str = os.getenv("SECRET_KEY", "supersecretkey123!@#")  # Change in production
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 1 day
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7  # 7 days
    SECURE_COOKIES: bool = os.getenv("ENVIRONMENT", "development") == "production"
    
    # Database
    MONGO_CONNECTION_STRING: str = os.getenv("MONGO_CONNECTION_STRING", "mongodb://localhost:27017")
    DBNAME: str = os.getenv("DBNAME", "traffic_monitoring")
    
    # Directories
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    UPLOAD_DIR: str = os.path.join(BASE_DIR, "uploads") # General uploads if needed by other modules
    RESULTS_DIR: str = os.path.join(BASE_DIR, "results") # General results if needed by other modules
    STATIC_DIR: str = os.path.join(BASE_DIR, "static")
    
    # ML Data and Model Directories (Centralized in backendv3/ml_data)
    ML_DATA_BASE_DIR: str = os.path.join(BASE_DIR, "ml_data")
    RAW_TRAINING_VIDEO_DIR: str = os.path.join(ML_DATA_BASE_DIR, "raw_training_videos")
    PROCESSED_VIDEO_JSON_DIR: str = os.path.join(ML_DATA_BASE_DIR, "processed_video_json")
    ML_READY_DATA_DIR: str = os.path.join(ML_DATA_BASE_DIR, "ml_ready_datasets")
    MODEL_DIR: str = os.path.join(ML_DATA_BASE_DIR, "saved_models") # Consolidated model directory
    
    # Video processing (Legacy or if specific non-ML video outputs are needed)
    # These might become obsolete or point to subdirs within ML_DATA_BASE_DIR if all video work is for ML
    VIDEO_INPUT_DIR: str = os.path.join(UPLOAD_DIR, "videos") # Example: if non-ML videos are uploaded elsewhere
    VIDEO_OUTPUT_DIR: str = os.path.join(RESULTS_DIR, "videos") # Example: if non-ML video results go elsewhere
    
    # ML models (Legacy paths, superseded by MODEL_DIR under ML_DATA_BASE_DIR)
    # OLD_MODEL_DIR: str = os.path.join(BASE_DIR, "ml", "models", "saved_models") # Kept for reference, prefer new MODEL_DIR
    PREDICTION_DIR: str = os.path.join(ML_DATA_BASE_DIR, "predictions") # For storing prediction outputs
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"  # Ignore extra fields from .env file
    }

# Create settings instance
settings = Settings()

# Function to get settings (for consistency across modules)
def get_settings():
    """Return the settings instance."""
    return settings

# Additional non-settings configuration
ADMIN_ROLES = ["admin", "superadmin"]
USER_ROLES = ["user"]
ALL_ROLES = ADMIN_ROLES + USER_ROLES

# File upload settings
MAX_UPLOAD_SIZE = 1024 * 1024 * 1024  # 1GB
ALLOWED_VIDEO_TYPES = ["video/mp4", "video/avi", "video/x-msvideo", "video/quicktime"]

# Status messages for consistent response
STATUS_MESSAGES = {
    "USER_CREATED": "User created successfully",
    "USER_UPDATED": "User updated successfully",
    "USER_DELETED": "User deleted successfully",
    "LOGIN_SUCCESS": "Login successful",
    "LOGIN_FAILED": "Invalid username or password",
    "UNAUTHORIZED": "Not authorized to access this resource",
    "VIDEO_UPLOADED": "Video uploaded successfully",
    "PROCESSING_STARTED": "Video processing started",
    "REPORT_SUBMITTED": "Report submitted successfully",
    "REPORT_PROCESSED": "Report processed successfully",
    "LOGOUT_SUCCESS": "Logout successful",
}

# Error messages
ERROR_MESSAGES = {
    "INVALID_CREDENTIALS": "Invalid username or password",
    "USER_EXISTS": "User already exists",
    "USER_NOT_FOUND": "User not found",
    "INVALID_TOKEN": "Invalid authentication token",
    "TOKEN_EXPIRED": "Token has expired",
    "PERMISSION_DENIED": "Permission denied",
    "RESOURCE_NOT_FOUND": "Resource not found",
    "INVALID_FILE": "Invalid file format",
    "FILE_TOO_LARGE": "File size exceeds the maximum allowed size",
    "INACTIVE_USER": "Account is inactive. Please contact an administrator.",
    "SERVER_ERROR": "An unexpected error occurred. Please try again later.",
    "TOO_MANY_REQUESTS": "Too many requests. Please try again later.",
    "DATABASE_ERROR": "Database connection error. Please try again later.",
}

# API rate limits
RATE_LIMITS = {
    "LOGIN_MAX_ATTEMPTS": 5,
    "LOGIN_LOCKOUT_MINUTES": 15,
    "API_DEFAULT_RATE": "60/minute",
}
