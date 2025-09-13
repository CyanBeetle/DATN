"""Utility for cleaning up temporary files and resources."""
import os
import shutil
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def cleanup_temp_files(directory, max_age_days=7):
    """Remove files older than specified days from directory."""
    if not os.path.exists(directory):
        logger.warning(f"Directory does not exist: {directory}")
        return
        
    threshold = datetime.now() - timedelta(days=max_age_days)
    removed_count = 0
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        # Skip directories
        if os.path.isdir(filepath):
            continue
            
        # Check file age
        file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
        if file_time < threshold:
            try:
                os.remove(filepath)
                removed_count += 1
                logger.debug(f"Removed old file: {filepath}")
            except Exception as e:
                logger.error(f"Error removing file {filepath}: {str(e)}")
    
    if removed_count > 0:
        logger.info(f"Removed {removed_count} old files from {directory}")

def schedule_cleanup_task(app):
    """Schedule cleanup task to run on application startup."""
    from app.config import UPLOAD_DIR, RESULTS_DIR
    
    @app.on_event("startup")
    async def cleanup_on_startup():
        """Run cleanup when application starts."""
        logger.info("Running startup file cleanup")
        cleanup_temp_files(UPLOAD_DIR, max_age_days=7)
        cleanup_temp_files(RESULTS_DIR, max_age_days=30)