"""MongoDB database connection management."""
import logging
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
import certifi
from pymongo.errors import DuplicateKeyError, OperationFailure

# Import config variables
from app.config import settings

# Setup logging
logger = logging.getLogger(__name__)

# Global client/db variables
_client: AsyncIOMotorClient = None
_db: AsyncIOMotorDatabase = None
_lock = asyncio.Lock()  # Add lock for thread safety

async def connect_to_mongo():
    """Establishes MongoDB connection."""
    global _client, _db
    
    async with _lock:
        if _db is not None:
            return _db
        
        try:
            logger.info(f"Connecting to MongoDB at {settings.MONGO_CONNECTION_STRING}")
            # Use certifi for TLS/SSL certificates with MongoDB Atlas
            _client = AsyncIOMotorClient(
                settings.MONGO_CONNECTION_STRING,
                tlsCAFile=certifi.where(),
                serverSelectionTimeoutMS=5000
            )
            
            # Test connection
            await _client.admin.command('ping')
            _db = _client[settings.DBNAME]
            
            logger.info(f"Connected to MongoDB database: {settings.DBNAME}")
            
            # Create indexes - handle exceptions gracefully
            try:
                # Check if index exists first before creating
                indexes = await _db.users.index_information()
                if 'username_1' not in indexes:
                    await _db.users.create_index("username", unique=True)
                    logger.info("Created index on users.username")
                else:
                    logger.info("Index users.username already exists")
            except Exception as e:
                logger.warning(f"Issue with users.username index: {str(e)}")
            
            try:
                # Check if index exists first before creating
                indexes = await _db.users.index_information()
                if 'email_1' not in indexes:
                    await _db.users.create_index("email", unique=True)
                    logger.info("Created index on users.email")
                else:
                    logger.info("Index users.email already exists")
            except Exception as e:
                logger.warning(f"Issue with users.email index: {str(e)}")
            
            try:
                # Check if index exists first before creating
                indexes = await _db.video_tasks.index_information()
                if 'task_id_1' not in indexes:
                    await _db.video_tasks.create_index("task_id", unique=True)
                    logger.info("Created index on video_tasks.task_id")
                else:
                    logger.info("Index video_tasks.task_id already exists")
            except Exception as e:
                logger.warning(f"Issue with video_tasks.task_id index: {str(e)}")
                
            # Create additional indexes for camera management
            try:
                # Create index on cameras collection
                await _db.cameras.create_index("name", unique=True)
                logger.info("Created index on cameras.name")
            except Exception as e:
                logger.warning(f"Issue with cameras.name index: {str(e)}")
                
            try:
                # Create index for favorite cameras
                await _db.favorite_cameras.create_index([("user_id", 1), ("camera_id", 1)], unique=True)
                logger.info("Created index on favorite_cameras.user_id and camera_id")
            except Exception as e:
                logger.warning(f"Issue with favorite_cameras index: {str(e)}")
                
            try:
                # Create index for locations
                await _db.locations.create_index([("latitude", 1), ("longitude", 1)])
                logger.info("Created index on locations.latitude and longitude")
            except Exception as e:
                logger.warning(f"Issue with locations index: {str(e)}")

            # Additional indexes for cameras collection based on Database Design.txt
            try:
                camera_indexes = await _db.cameras.index_information()
                if 'status_1' not in camera_indexes:
                    await _db.cameras.create_index("status")
                    logger.info("Created index on cameras.status")
                if 'online_1' not in camera_indexes:
                    await _db.cameras.create_index("online")
                    logger.info("Created index on cameras.online")
                if 'deleted_1' not in camera_indexes:
                    await _db.cameras.create_index("deleted")
                    logger.info("Created index on cameras.deleted")
            except Exception as e:
                logger.warning(f"Issue with additional camera indexes (status, online, deleted): {str(e)}")

            # Indexes for Reports Collection
            try:
                report_indexes = await _db.reports.index_information()
                if 'status_1' not in report_indexes:
                    await _db.reports.create_index("status")
                    logger.info("Created index on reports.status")
                if 'created_by_1' not in report_indexes:
                    await _db.reports.create_index("created_by")
                    logger.info("Created index on reports.created_by")
                if 'report_type_1' not in report_indexes:
                    await _db.reports.create_index("report_type")
                    logger.info("Created index on reports.report_type")
                if 'created_at_1' not in report_indexes: # For sorting
                    await _db.reports.create_index("created_at")
                    logger.info("Created index on reports.created_at") 
            except Exception as e:
                logger.warning(f"Issue with reports indexes: {str(e)}")

            # Indexes for Report Notifications Collection
            try:
                notification_indexes = await _db.report_notifications.index_information()
                if 'is_read_1' not in notification_indexes:
                    await _db.report_notifications.create_index("is_read")
                    logger.info("Created index on report_notifications.is_read")
                if 'created_at_1' not in notification_indexes: # For sorting & TTL
                    await _db.report_notifications.create_index("created_at")
                    logger.info("Created index on report_notifications.created_at")
                if 'report_id_1' not in notification_indexes:
                    await _db.report_notifications.create_index("report_id")
                    logger.info("Created index on report_notifications.report_id")
            except Exception as e:
                logger.warning(f"Issue with report_notifications indexes: {str(e)}")
            
            return _db
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            if _client:
                _client.close()
            _client = None
            _db = None
            raise

async def close_mongo_connection():
    """Closes the MongoDB connection."""
    global _client, _db
    async with _lock:
        if _client:
            _client.close()
            _client = None
            _db = None
            logger.info("MongoDB connection closed")

async def get_db() -> AsyncIOMotorDatabase:
    """FastAPI dependency to get database instance."""
    if _db is None:
        await connect_to_mongo()
    return _db