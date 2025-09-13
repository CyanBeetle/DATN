"""MongoDB data models."""
from bson import ObjectId
from typing import Any, List, Optional, Union, Literal # Ensure Any, Literal are imported
from pydantic import BaseModel, Field, EmailStr, GetJsonSchemaHandler, ConfigDict, field_validator, AnyHttpUrl, HttpUrl # Keep existing pydantic imports and add new ones
from pydantic_core import core_schema
from pydantic.json_schema import JsonSchemaValue # Add this import if not present
from datetime import datetime
from enum import Enum # Add Enum import

# --- Custom ObjectId field for MongoDB integration ---
class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetJsonSchemaHandler
    ) -> core_schema.CoreSchema:
        
        def validate_object_id(value: Any) -> ObjectId:
            """Validates if the value is a valid ObjectId or can be converted to one."""
            if isinstance(value, ObjectId):
                return value
            if isinstance(value, str) and ObjectId.is_valid(value):
                return ObjectId(value)
            raise ValueError("value is not a valid ObjectId")

        # Schema for validating input from Python code
        python_schema = core_schema.union_schema(
            [
                core_schema.is_instance_schema(ObjectId), # Accepts ObjectId instances directly
                core_schema.chain_schema( # For string input that needs validation
                    [
                        core_schema.str_schema(),
                        core_schema.no_info_plain_validator_function(validate_object_id),
                    ]
                ),
            ],
            custom_error_type='ObjectIdError', # Optional: for more specific error handling
            custom_error_message='value is not a valid ObjectId'
        )
        
        # Defines how PyObjectId is handled for JSON (input) and Python (usage),
        # and how it's serialized.
        return core_schema.json_or_python_schema(
            # For JSON input (e.g., from HTTP request), expect a string that validates to ObjectId
            json_schema=core_schema.no_info_plain_validator_function(validate_object_id),
            # For Python input, use the more flexible python_schema defined above
            python_schema=python_schema,
            # For serialization (e.g., to JSON response), convert ObjectId to string
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda v: str(v) if isinstance(v, ObjectId) else v,
                info_arg=False, # No extra info needed for this serializer
                when_used='json', # Apply only during JSON serialization
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue: # JsonSchemaValue is a type hint for the return dict
        # For OpenAPI (JSON Schema), PyObjectId is represented as a string.
        # The pattern ensures it's a 24-character hex string.
        # Example is good for documentation.
        return {"type": "string", "pattern": "^[0-9a-fA-F]{24}$", "example": "507f1f77bcf86cd799439011"}

# --- Base Model with MongoDB compatibility ---
class MongoBaseModel(BaseModel):
    """Base model for MongoDB documents."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        json_encoders={
            ObjectId: str,
            HttpUrl: str,  # Ensure HttpUrl is converted to string
            AnyHttpUrl: str  # Also handle AnyHttpUrl if used
        }
    )
    fullname: Optional[str] = None  # Added for full name display
    display_name: Optional[str] = None # Ensure display_name is present as per docs

# --- User Models ---
class UserBase(BaseModel):
    """Base user model."""
    username: str
    email: EmailStr
    is_active: bool = True
    role: str = "user"  # Default role is "user"
    display_name: Optional[str] = None # Ensure display_name is present as per docs

class UserCreate(UserBase):
    """User creation model."""
    password: str

class UserUpdate(BaseModel):
    """User update model."""
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None
    role: Optional[str] = None
    fullname: Optional[str] = None

class UserInDB(UserBase, MongoBaseModel):
    """User model as stored in the database."""
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    display_name: Optional[str] = None  # For display purposes
    phone: Optional[str] = None  # Added for contact information
    token_version: Optional[int] = 1
    token_version_changed_at: Optional[datetime] = None

class UserPublic(BaseModel):
    """User model for public responses."""
    id: str
    username: str
    email: EmailStr
    is_active: bool
    role: str
    created_at: datetime
    last_login: Optional[datetime] = None
    fullname: Optional[str] = None
    display_name: Optional[str] = None

# --- User Preferences Model ---
# class UserPreference(MongoBaseModel):
#     """User preferences for personalization."""
#     id: Optional[PyObjectId] = Field(default=None, alias="_id")
#     user_id: str
#     notification_enabled: bool = True
#     email_notifications: bool = False
#     default_map_view: Dict[str, float] = Field(default_factory=lambda: {"lat": 10.7769, "lng": 106.7009, "zoom": 13})  # Default to HCMC
#     favorite_locations: List[str] = Field(default_factory=list)  # List of location IDs
#     theme: str = "light"
#     language: str = "en"
#     updated_at: datetime = Field(default_factory=datetime.utcnow)

# --- Location Models (NEW) ---
class LocationBase(BaseModel):
    """Base location model."""
    name: Optional[str] = None
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)

class LocationCreate(LocationBase):
    """Location creation model."""
    pass

class LocationInDB(MongoBaseModel, LocationBase):
    """Location model as stored in the database."""
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class LocationPublic(LocationInDB):
    """Public representation of a location."""
    id: str # Override to string for public API

    @field_validator('id', mode='before')
    def stringify_id(cls, v):
        return str(v) if v is not None else None

# --- Camera Models ---
class CameraStatus(str, Enum):
    """Camera status enum."""
    ACTIVE = "Active"
    MAINTENANCE = "Maintenance"
    INACTIVE = "Inactive"

class LocationDBModel(BaseModel):
    """Embedded location model for Camera."""
    name: Optional[str] = "Unknown Location"
    latitude: float
    longitude: float

class ROIPointDBModel(BaseModel):
    """Normalized ROI point."""
    x: float = Field(..., ge=0, le=1)  # Normalized (0-1)
    y: float = Field(..., ge=0, le=1)  # Normalized (0-1)

class ROIDBModel(BaseModel):
    """ROI data for a camera."""
    points: Optional[List[ROIPointDBModel]] = None # List of 4 normalized points
    roi_width_meters: Optional[float] = None
    roi_height_meters: Optional[float] = None

class CameraBase(BaseModel):
    """Base camera model for core camera attributes."""
    name: str
    description: Optional[str] = None
    stream_url: AnyHttpUrl # Changed from HttpUrl to AnyHttpUrl
    status: CameraStatus = CameraStatus.INACTIVE
    location_id: Optional[PyObjectId] = None # Reference to Location collection
    thumbnail_url: Optional[str] = None # Changed from AnyHttpUrl to str
    previous_thumbnail_url: Optional[str] = None # Track the previous thumbnail for comparison
    thumbnail_updated_at: Optional[datetime] = None # When was the thumbnail last updated
    roi: Optional[ROIDBModel] = None
    online: bool = Field(default=True)
    deleted: bool = Field(default=False)

class CameraCreate(CameraBase):
    """Model for creating a new camera. Inherits all from CameraBase."""
    pass

class CameraUpdate(BaseModel):
    """Model for updating an existing camera. All fields are optional."""
    name: Optional[str] = None
    description: Optional[str] = None
    stream_url: Optional[AnyHttpUrl] = None # Changed from HttpUrl to AnyHttpUrl
    status: Optional[CameraStatus] = None
    location_id: Optional[PyObjectId] = None # Reference to Location collection
    thumbnail_url: Optional[str] = None # Changed from AnyHttpUrl to str
    previous_thumbnail_url: Optional[str] = None # Track previous thumbnail
    thumbnail_updated_at: Optional[datetime] = None # When was the thumbnail last updated
    roi: Optional[ROIDBModel] = None
    online: Optional[bool] = None
    deleted: Optional[bool] = None

class CameraInDB(MongoBaseModel, CameraBase):
    """Camera model as stored in the database."""
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

# --- Favorite Camera Models ---
class FavoriteCameraBase(BaseModel):
    """Base favorite camera model."""
    user_id: PyObjectId  # Reference to User
    camera_id: PyObjectId  # Reference to Camera
    notifications_enabled: bool = False # Added from DB Design txt, was missing in base

class FavoriteCameraCreate(FavoriteCameraBase):
    """Favorite camera creation model."""
    pass

class FavoriteCamera(MongoBaseModel, FavoriteCameraBase):
    """Favorite camera model as stored in the database."""
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow) # Aligned with DB Design txt

# --- Training Dataset Models (UC11) ---
# class VideoDimensionsModel(BaseModel):
#     width: int = Field(..., gt=0, description="Width of the video in pixels")
#     height: int = Field(..., gt=0, description="Height of the video in pixels")
# 
# class DatasetStatus(str, Enum):
#     PENDING_UPLOAD = "PendingUpload"
#     UPLOADED = "Uploaded"  # File is on server, DB record created
#     METADATA_EXTRACTION_PENDING = "MetadataExtractionPending" # if we have a step for this
#     ROI_PENDING = "ROIPending" # For video datasets needing ROI definition
#     READY_FOR_PROCESSING = "ReadyForProcessing" # ROI defined, ready for VideoProcessor
#     VIDEO_PROCESSING = "VideoProcessing" # VideoProcessor is running
#     VIDEO_PROCESSED_JSON_AVAILABLE = "VideoProcessedJSONAvailable" # JSON output from VideoProcessor is ready
#     ML_DATASET_CREATION_PENDING = "MLDatasetCreationPending" # Ready for DatasetCreator/Preprocessor
#     ML_DATASET_CREATION_IN_PROGRESS = "MLDatasetCreationInProgress"
#     ML_BUNDLE_AVAILABLE = "MLBundleAvailable"  # .npy files, scaler, feature_names are ready
#     ARCHIVED = "Archived"
#     ERROR = "Error"
# 
# class TrainingDatasetBase(BaseModel):
#     name: constr(min_length=3, max_length=100) # type: ignore
#     description: Optional[constr(max_length=500)] = None # type: ignore
#     data_type: str = Field(..., description="Type of dataset (e.g., 'RawVideo', 'ProcessedVideoJSON', 'ML_BUNDLE', 'Log', 'Other')")
#     
#     # ROI fields, specific to video-type datasets
#     roi_config: Optional[ROIDBModel] = Field(default=None, description="ROI configuration for video datasets")
#     video_dimensions: Optional[VideoDimensionsModel] = Field(default=None, description="Original video dimensions for context")
# 
#     # Fields related to ML Bundle type
#     ml_dataset_path_prefix: Optional[str] = Field(default=None, description="Path prefix for ML bundle files (X_train.npy, scaler.pkl etc.)")
#     training_parameters: Optional[Dict[str, Any]] = Field(default=None, description="Parameters used or derived during ML dataset creation, e.g., interval_seconds, sequence_length, feature_list")
#     
#     source_video_dataset_ids: Optional[List[PyObjectId]] = Field(default_factory=list, description="For ML_BUNDLE, list of processed video JSON dataset IDs used to create it")
#     source_ml_bundle_ids: Optional[List[PyObjectId]] = Field(default_factory=list, description="For PredictionModel, list of ML_BUNDLE dataset IDs used for training")
# 
# 
#     tags: Optional[List[str]] = Field(default_factory=list)
#     metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Flexible key-value store for other metadata")
#     status: DatasetStatus = DatasetStatus.PENDING_UPLOAD
#     error_message: Optional[str] = None
#     
#     # Timestamps
#     created_at: datetime = Field(default_factory=datetime.utcnow)
#     updated_at: datetime = Field(default_factory=datetime.utcnow)
#     
#     # User responsible
#     uploaded_by_admin_id: Optional[PyObjectId] = None # Assuming PyObjectId is defined and imported
# 
#     # File specific info - might be populated after upload and inspection
#     file_name: Optional[str] = None
#     file_path_on_server: Optional[str] = None # Internal path on server
#     file_size_bytes: Optional[int] = None
#     
#     # For video processing results specifically
#     processed_video_json_path: Optional[str] = Field(default=None, description="Path to the JSON output from VideoProcessor")
# 
# 
#     model_config = ConfigDict(
#         use_enum_values=True,
#         json_encoders={
#             datetime: lambda dt: dt.isoformat(),
#             PyObjectId: lambda oid: str(oid),
#         },
#         arbitrary_types_allowed=True
#     )
# 
# 
# class TrainingDatasetCreate(TrainingDatasetBase):
#     pass # At creation, ROI might not be set yet if it's a separate step after upload.
#          # File details also set by service layer after file is received.
# 
# 
# class TrainingDatasetUpdate(BaseModel): # More targeted update model
#     name: Optional[constr(min_length=3, max_length=100)] = None # type: ignore
#     description: Optional[constr(max_length=500)] = None # type: ignore
#     tags: Optional[List[str]] = None
#     metadata: Optional[Dict[str, Any]] = None
#     status: Optional[DatasetStatus] = None # Status updates are common
#     error_message: Optional[str] = None
#     # ROI can be updated via a specific endpoint/model
#     roi_config: Optional[ROIDBModel] = None 
#     video_dimensions: Optional[VideoDimensionsModel] = None
#     # ML bundle related paths/params updated by services
#     ml_dataset_path_prefix: Optional[str] = None
#     training_parameters: Optional[Dict[str, Any]] = None
#     processed_video_json_path: Optional[str] = None
# 
#     updated_at: datetime = Field(default_factory=datetime.utcnow)
# 
#     model_config = ConfigDict(
#         use_enum_values=True,
#         json_encoders={
#             datetime: lambda dt: dt.isoformat(),
#             PyObjectId: lambda oid: str(oid),
#         },
#         arbitrary_types_allowed=True
#     )
# 
# 
# class TrainingDatasetInDB(TrainingDatasetBase):
#     id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
# 
#     model_config = ConfigDict(
#         use_enum_values=True,
#         json_encoders={
#             datetime: lambda dt: dt.isoformat(),
#             PyObjectId: lambda oid: str(oid),
#         },
#         arbitrary_types_allowed=True
#     )

# --- Weather & News Models (UC07) ---
# class WeatherData(MongoBaseModel):
#     """Weather information model."""
#     id: Optional[PyObjectId] = Field(default=None, alias="_id")
#     location_id: str  # Reference to Location
#     temperature: float  # In Celsius
#     description: str  # Clear, rainy, etc.
#     humidity: float  # Percentage
#     wind_speed: float  # km/h
#     precipitation: Optional[float] = None  # mm
#     timestamp: datetime = Field(default_factory=datetime.utcnow)
#     forecast_for: Optional[datetime] = None  # If this is forecast data
#     icon: Optional[str] = None  # Icon code for UI
# 
# class NewsArticle(MongoBaseModel):
#     """News article model."""
#     id: Optional[PyObjectId] = Field(default=None, alias="_id")
#     title: str
#     content: str
#     summary: str
#     source: str
#     author: Optional[str] = None
#     published_at: datetime
#     url: Optional[str] = None
#     image_url: Optional[str] = None
#     categories: List[str] = Field(default_factory=list)
#     location_id: Optional[str] = None  # If news is related to a specific location
#     is_traffic_related: bool = False
#     added_to_system_at: datetime = Field(default_factory=datetime.utcnow)

# --- Chatbot Models (UC10) ---
# class ChatMessage(BaseModel):
#     """Chat message model for support chatbot."""
#     content: str
#     sender_type: Literal['user', 'bot']
#     timestamp: datetime = Field(default_factory=datetime.utcnow)
# 
# class ChatSession(MongoBaseModel):
#     """Chatbot session model."""
#     id: Optional[PyObjectId] = Field(default=None, alias="_id")
#     user_id: str
#     messages: List[ChatMessage] = Field(default_factory=list)
#     created_at: datetime = Field(default_factory=datetime.utcnow)
#     updated_at: datetime = Field(default_factory=datetime.utcnow)
#     is_active: bool = True
#     session_data: Optional[Dict[str, Any]] = None  # For storing context or state

# --- Token & Auth Models (UC01) ---
class LoginResponseUser(BaseModel):
    """User model for login response."""
    id: str
    username: str
    email: EmailStr
    role: str
    fullname: Optional[str] = None

class LoginResponse(BaseModel):
    """Login response model with user and token data."""
    access_token: str
    token_type: str
    refresh_token: str
    user: LoginResponseUser

# --- Auth Models ---
class Token(BaseModel):
    """JWT token model."""
    access_token: str
    token_type: str
    refresh_token: Optional[str] = None

class TokenData(BaseModel):
    """Token data model."""
    username: str
    exp: Optional[int] = None

# --- Video Processing & Analysis Models (Related to UC11 training data, UC04 camera view processing) ---
# These might be used by tests or utility components, but are not core to UC01.
# Commenting out for now to strictly adhere to "reduce every UC to zero except UC01".

# class ROI(MongoBaseModel):
#     """Region of Interest model."""
#     id: Optional[PyObjectId] = Field(default=None, alias="_id")
#     name: str
#     points: List[List[int]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
#     target_width: float = 50.0  # Real-world width in meters
#     target_height: float = 100.0  # Real-world height in meters
#     created_by: Optional[str] = None
#     created_at: datetime = Field(default_factory=datetime.now)
#     location_id: Optional[str] = None  # Link to a location if applicable
# 
# class VideoTask(MongoBaseModel):
#     """Video processing task model."""
#     id: Optional[PyObjectId] = Field(default=None, alias="_id")
#     task_id: str
#     status: str  # "pending", "processing", "completed", "failed"
#     progress: int = 0
#     filename: str
#     file_path: str
#     output_path: Optional[str] = None
#     roi_points: List[List[int]]
#     frame_skip: int = 10
#     max_duration: Optional[int] = None
#     created_by: Optional[str] = None
#     created_at: datetime = Field(default_factory=datetime.now)
#     completed_at: Optional[datetime] = None
#     error: Optional[str] = None
#     stats: Optional[Dict[str, Any]] = None
#     description: Optional[str] = None  # Task description provided by user
#     has_predictions: bool = False  # Whether predictions have been generated
#     prediction_id: Optional[str] = None  # Reference to prediction results
# 
# class VehicleDetection(MongoBaseModel):
#     """Vehicle detection model."""
#     id: Optional[PyObjectId] = Field(default=None, alias="_id")
#     task_id: str
#     frame_number: int
#     timestamp: float
#     tracker_id: int
#     class_id: int
#     class_name: str
#     confidence: float
#     bbox: List[float]  # [x1, y1, x2, y2]
#     speed_kmh: Optional[float] = None
# 
# class VideoAnalysisData(MongoBaseModel):
#     """Video analysis results model."""
#     id: Optional[PyObjectId] = Field(default=None, alias="_id")
#     task_id: str
#     filename: str
#     created_at: datetime = Field(default_factory=datetime.now)
#     processed_date: str
#     duration_seconds: float
#     video_info: Dict[str, Any]
#     roi_config: Dict[str, Any]
#     unique_vehicles: List[Dict[str, Any]]
#     vehicle_count: int
#     avg_speed: Optional[float] = None
#     time_intervals: Optional[List[Dict[str, Any]]] = None
#     location_id: Optional[str] = None  # Reference to Location where video was recorded

# --- Report Models (UC09, UC14) ---
ReportTypeLiteral = Literal["incident", "infrastructure"]
ReportStatusLiteral = Literal[
    "New", # Was "Submitted"
    "Processing",
    "Verified", # Added
    "Invalid",  # Added
    "Resolved",
    "Rejected",
    "Archived" # Kept, can be for long-term storage if needed
]

class ReportBase(BaseModel):
    """Base model for report data common to create and DB."""
    title: str = Field(..., min_length=5, max_length=150)
    description: str = Field(..., min_length=10, max_length=1000)
    report_type: ReportTypeLiteral # Using updated Literal
    location_id: Optional[PyObjectId] = None # Reference to Location collection

class ReportCreate(ReportBase):
    """Model for user submitting a new report."""
    # created_by, created_by_username are added by the backend from authenticated user.
    # image_url is handled if a file is uploaded.
    pass

class ReportUpdateAdmin(BaseModel):
    """Model for admin updating a report's status and resolution notes."""
    status: ReportStatusLiteral # Using updated Literal
    admin_reply: Optional[str] = Field(None, max_length=1000) # Aligned with DB Design txt

class ReportInDB(MongoBaseModel, ReportBase):
    """Report model as stored in the database."""
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    image_url: Optional[AnyHttpUrl] = None # Changed from HttpUrl to AnyHttpUrl
    status: ReportStatusLiteral = Field(default="New") # Default to "New"
    user_id: PyObjectId # ObjectId of the user who created the report (Aligned with DB Design)
    created_by_username: str  # Username for easy display
    submitted_at: datetime = Field(default_factory=datetime.now) # Aligned with DB Design
    processed_at: Optional[datetime] = None
    handled_by_admin_id: Optional[PyObjectId] = None  # ObjectId of the admin who processed
    admin_reply: Optional[str] = None # Aligned with DB Design

class ReportPublic(ReportInDB): # For API responses, might be same as InDB or selected fields
    """Report model for API responses."""
    pass 

# --- Report Notification Models (Related to UC14) ---
class ReportNotificationBase(BaseModel):
    """Base for report notifications."""
    report_id: PyObjectId
    report_title: str
    report_type: ReportTypeLiteral # Aligned with ReportTypeLiteral for consistency
    message: str # Added message field for more context
    is_read: bool = False
    admin_id: Optional[PyObjectId] = None # Added admin_id as per documentation

class ReportNotificationInDB(MongoBaseModel, ReportNotificationBase):
    """Report Notification model as stored in the database."""
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ReportNotificationPublic(ReportNotificationInDB):
    """Report Notification model for API responses."""
    pass

# --- Notification Models (UC05 and others) ---
# class NotificationType(str, Enum):
#     """Notification type enum."""
#     TRAFFIC_ALERT = "traffic_alert"
#     CAMERA_STATUS = "camera_status"
#     SYSTEM_ALERT = "system_alert"
#     INCIDENT = "incident"
# 
# class NotificationBase(BaseModel):
#     """Base notification model."""
#     type: str
#     title: str
#     message: str
#     data: Optional[Dict[str, Any]] = None
#     created_at: datetime = Field(default_factory=datetime.utcnow)
# 
# class Notification(MongoBaseModel, NotificationBase):
#     """Notification model as stored in the database."""
#     id: Optional[PyObjectId] = Field(default=None, alias="_id")
#     user_id: PyObjectId # This was PyObjectId, should likely be str if user_id is generally a string id
#     read: bool = False
#     read_at: Optional[datetime] = None
#     priority: str = "normal"  # Priority level: "high", "normal", "low"
#     
#     # Additional fields for specific notification types
#     camera_id: Optional[str] = None  # For camera-related notifications
#     report_id: Optional[str] = None  # For report-related notifications
#     location_id: Optional[str] = None  # For location-related notifications
#     created_at: datetime = Field(default_factory=datetime.now) # Duplicate created_at? One from Base, one here.
# 
# class NotificationSettings(MongoBaseModel):
#     """User notification settings."""
#     id: Optional[PyObjectId] = Field(default=None, alias="_id")
#     user_id: str
#     email_enabled: bool = False
#     push_enabled: bool = True
#     congestion_alerts: bool = True
#     favorite_camera_alerts: bool = True
#     report_updates: bool = True
#     news_alerts: bool = False
#     weather_alerts: bool = False
#     updated_at: datetime = Field(default_factory=datetime.now)

# --- Prediction Model Models (UC06, UC12) ---
# class ModelStatus(str, Enum):
#     """Status of a prediction model."""
#     CREATING = "Creating" # Initial state before first training kicks off
#     PENDING_TRAINING = "PendingTraining" # Added this new status
#     TRAINING = "Training"
#     TRAINING_FAILED = "TrainingFailed"
#     ACTIVE = "Active" # Trained and can be used for predictions
#     INACTIVE = "Inactive" # Trained but not currently used
#     ERROR = "Error" # Generic error state
# 
# class ModelType(str, Enum):
#     """Type of prediction model."""
#     ARIMA = "ARIMA"
#     LSTM = "LSTM"
#     BILSTM = "BiLSTM"
#     ENSEMBLE = "Ensemble" # Could be a combination of models
#     OTHER = "Other"
# 
# class PerformanceMetrics(BaseModel):
#     """Stores performance metrics of a trained model."""
#     accuracy: Optional[float] = None
#     precision: Optional[float] = None
#     recall: Optional[float] = None
#     f1_score: Optional[float] = None
#     mae: Optional[float] = None # Mean Absolute Error
#     mse: Optional[float] = None # Mean Squared Error
#     rmse: Optional[float] = None # Root Mean Squared Error
#     r2_score: Optional[float] = None
#     custom_metrics: Optional[Dict[str, Any]] = None # For any other metrics
# 
# class PredictionModelBase(MongoBaseModel):
#     """Base model for prediction models."""
#     name: str = Field(..., min_length=3, max_length=100)
#     description: Optional[str] = Field(default=None, max_length=500)
#     model_type: str = Field(..., description="Type of the model, e.g., ARIMA, LSTM, Prophet") # Changed to simple str
#     status: str = Field(default="Creating") # Changed to simple str, e.g. ModelStatus.CREATING.value
#     version: str = Field(default="1.0.0")
#     parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Hyperparameters or configuration")
#     metrics: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Performance metrics, e.g., MAE, RMSE") # Or use PerformanceMetrics model
#     trained_at: Optional[datetime] = None
#     deployed_at: Optional[datetime] = None
#     file_path: Optional[str] = Field(default=None, description="Path to the serialized model file") # Relative to a base model storage
#     # Added fields from Database Design Txt
#     is_default: bool = False 
#     training_data_ids: Optional[List[PyObjectId]] = Field(default_factory=list, description="List of TrainingDataset IDs used for training") # Changed from training_data_id
#     created_by_admin_id: Optional[PyObjectId] = None
#     is_baseline_arima: bool = False # New field for UC12 baseline ARIMA
# 
# class PredictionModelCreate(PredictionModelBase):
#     # created_by_admin_id should be set by the service from the authenticated admin
#     pass
# 
# class PredictionModelUpdate(PredictionModelBase): # Make all fields optional for update
#     name: Optional[str] = Field(default=None, min_length=3, max_length=100)
#     description: Optional[str] = Field(default=None, max_length=500)
#     model_type: Optional[str] = Field(default=None) # Changed to simple str
#     status: Optional[str] = Field(default=None) # Changed to simple str
#     version: Optional[str] = Field(default=None)
#     parameters: Optional[Dict[str, Any]] = Field(default=None)
#     metrics: Optional[Dict[str, Any]] = Field(default=None) # Or use PerformanceMetrics model
#     file_path: Optional[str] = Field(default=None)
#     is_default: Optional[bool] = None
#     trained_at: Optional[datetime] = None # Can be set upon successful training
#     # training_data_ids could potentially be updated if a model is retrained with different data
#     training_data_ids: Optional[List[PyObjectId]] = None 
# 
# class PredictionModelInDB(PredictionModelBase):
#     id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")

# --- Forecast Models (UC06) ---
# Placeholder for Forecast related Pydantic models if needed.
# The Database Design doc has a `forecasts` collection.
# class ForecastInputParams(BaseModel):
#     """Input parameters used for generating a forecast."""
#     location_name: Optional[str] = None # e.g. "Đường Mai Chí Thọ - đầu hầm Thủ Thiêm"
#     time_horizon_minutes: int 
#     weather_conditions: Optional[str] = "normal" # "normal", "rainy", "holiday"
#     # Add other relevant parameters the model might use
# 
# class ForecastResultItem(BaseModel):
#     """A single data point in a forecast sequence."""
#     timestamp: datetime # The future time for this prediction point
#     predicted_speed: Optional[float] = None # km/h
#     predicted_density: Optional[float] = None # vehicles per 100m or similar unit
#     congestion_level: Optional[int] = None # 1-5
#     confidence_lower: Optional[float] = None # Lower bound of confidence interval for speed/density
#     confidence_upper: Optional[float] = None # Upper bound of confidence interval for speed/density
# 
# class ForecastInDB(MongoBaseModel): # Storing forecast results
#     """Model for storing traffic forecast results in the database."""
#     id: Optional[PyObjectId] = Field(default=None, alias="_id")
#     location_id: PyObjectId # Reference to locations collection
#     model_id: PyObjectId # Reference to prediction_models collection used for this forecast
#     input_params: ForecastInputParams # The parameters used to generate this forecast
#     forecast_time: datetime = Field(default_factory=datetime.utcnow) # When the forecast was generated
#     # Store forecast as a sequence of results
#     forecast_sequence: List[ForecastResultItem] = Field(default_factory=list) 
#     # Overall summary or metadata for the forecast run
#     metadata: Optional[Dict[str, Any]] = None
#     created_at: datetime = Field(default_factory=datetime.utcnow) # Redundant with forecast_time? Usually one is enough.
# 

# --- Video Processing Task Models (If not already defined elsewhere) ---