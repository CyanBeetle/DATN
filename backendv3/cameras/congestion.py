"""
Congestion level calculation service for camera frames.
This module provides functions to calculate traffic congestion levels from camera frames
using vehicle detection with YOLOv8 and configurable ROI-based density calculation.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from video.detector import VehicleDetector
from utils.roi import ROIHandler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CongestionCalculator:
    """Class to calculate congestion levels for camera frames based on ROI and vehicle detection"""
    
    def __init__(self, model_path: str = "yolov8x.pt", confidence: float = 0.35):
        """
        Initialize the congestion calculator with YOLOv8 detector
        
        Args:
            model_path: Path to YOLOv8 model (can be a name, relative path, or absolute path)
            confidence: Detection confidence threshold
        """
        logger.info(f"Initializing CongestionCalculator with requested model: {model_path}")
        
        potential_model_path = Path(model_path)
        resolved_model_path = None
        
        # Determine the root of the backendv3 application based on this file's location
        # Path(__file__) is backendv3/cameras/congestion.py
        # Path(__file__).parent is backendv3/cameras/
        # Path(__file__).parent.parent is backendv3/
        app_root = Path(__file__).parent.parent 
        
        # Get the filename part, e.g., "yolov8x.pt"
        model_filename = potential_model_path.name

        if potential_model_path.is_absolute():
            if potential_model_path.exists() and potential_model_path.is_file():
                resolved_model_path = potential_model_path
            else:
                raise FileNotFoundError(f"Absolute model path specified but not found: {potential_model_path}")
        else:
            # 1. Try path relative to the application root (backendv3 directory)
            path_in_app_root = app_root / model_filename
            if path_in_app_root.exists() and path_in_app_root.is_file():
                resolved_model_path = path_in_app_root
            else:
                # 2. Try path relative to Current Working Directory (CWD) as a fallback
                # This handles the case where model_path was like "yolov8x.pt" and CWD has it.
                path_in_cwd = Path.cwd() / potential_model_path 
                if path_in_cwd.exists() and path_in_cwd.is_file():
                    resolved_model_path = path_in_cwd.resolve() # Resolve to make it absolute
                    logger.warning(
                        f"Model '{model_filename}' found in CWD '{Path.cwd()}' using input path '{potential_model_path}'. "
                        f"Prefer placing models within the application structure (e.g., '{app_root}') or using absolute paths."
                    )
                else:
                    # Construct error message for all checked relative paths
                    checked_paths_str = f"1. Relative to application root: {path_in_app_root}\\n"
                    checked_paths_str += f"2. Relative to CWD (using input '{potential_model_path}'): {path_in_cwd}"
                    raise FileNotFoundError(
                        f"Model file '{model_filename}' (from input '{model_path}') not found. Checked locations:\\n{checked_paths_str}"
                    )
                        
        logger.info(f"Using model file: {resolved_model_path}")
        
        # Initialize vehicle detector
        self.detector = VehicleDetector(
            model_name=str(resolved_model_path),
            confidence=confidence,
            use_gpu=True
        )
        
        # Initialize ROI handler
        self.roi_handler = ROIHandler()
        
        # Congestion thresholds (vehicles per 100m²) - UPDATED TO 5 LEVELS
        # Level 1: Density < 2
        # Level 2: 2 <= Density < 6
        # Level 3: 6 <= Density < 10
        # Level 4: 10 <= Density < 15
        # Level 5: Density >= 15
        self.congestion_thresholds = [
            2.0,   # Threshold for Level 1 (< 2.0)
            6.0,   # Threshold for Level 2 (< 6.0)
            10.0,  # Threshold for Level 3 (< 10.0)
            15.0   # Threshold for Level 4 (< 15.0)
                   # Level 5 is >= 15.0
        ]
        
    def calculate_congestion(self, frame_path: str, roi_points: List[Dict[str, float]], 
                             roi_width_meters: float, roi_height_meters: float) -> Dict[str, Any]:
        """
        Calculate congestion level for a camera frame within the ROI
        
        Args:
            frame_path: Path to camera frame image
            roi_points: List of ROI points [{"x": float, "y": float}, ...]
            roi_width_meters: Width of ROI in meters
            roi_height_meters: Height of ROI in meters
            
        Returns:
            Dictionary with congestion calculation results
        """
        try:
            # Load frame
            frame = cv2.imread(frame_path)
            if frame is None:
                raise ValueError(f"Could not load frame from {frame_path}")
            
            # Convert ROI points to numpy array
            np_points = np.array([(p["x"], p["y"]) for p in roi_points])
            
            # Scale coordinates to image dimensions
            roi_polygon = np.array([
                (int(p[0] * frame.shape[1]), int(p[1] * frame.shape[0])) 
                for p in np_points
            ], dtype=np.int32)
            
            # Detect vehicles using YOLOv8
            detections = self.detector.detect(frame)
            
            # Create polygon mask for ROI
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [roi_polygon], 255)
            
            # Count vehicles within ROI
            vehicles_in_roi = self._count_vehicles_in_roi(detections, mask, frame.shape)
            
            # Calculate ROI area in square meters
            roi_area_m2 = roi_width_meters * roi_height_meters
            
            # Calculate vehicle density per 100m²
            # Handle potential division by zero if ROI area is invalid
            density = (vehicles_in_roi / roi_area_m2) * 100 if roi_area_m2 > 0.01 else 0 # Use small threshold for validity
            
            # Determine congestion level (1-5)
            congestion_level = self._get_congestion_level(density)
            
            # Prepare result with additional information
            result = {
                "congestion_level": congestion_level,
                "congestion_text": self._get_congestion_text(congestion_level),
                "vehicle_count": vehicles_in_roi,
                "roi_area_m2": round(roi_area_m2, 2),
                "vehicle_density": round(density, 2),  # Vehicles per 100m²
                "calculation_timestamp": datetime.utcnow().isoformat(),
                "detection_confidence": self.detector.confidence
            }
            
            logger.info(f"Calculated congestion level {congestion_level} (Density: {density:.2f}) for {frame_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating congestion: {str(e)}", exc_info=True)
            return {
                "error": str(e),
                "congestion_level": 0,
                "congestion_text": "Error",
                "vehicle_count": 0
            }
    
    def _count_vehicles_in_roi(self, detections, mask, frame_shape) -> int:
        """
        Count vehicles that have their center points within the ROI
        
        Args:
            detections: Vehicle detections from YOLOv8
            mask: Binary mask for ROI
            frame_shape: Shape of the frame
            
        Returns:
            Count of vehicles inside the ROI
        """
        if not hasattr(detections, 'xyxy') or detections.xyxy.size == 0:
            return 0
        
        vehicle_count = 0
        
        # Process each detection
        for i, bbox in enumerate(detections.xyxy):
            # Get detection class
            class_id = detections.class_id[i] if hasattr(detections, 'class_id') else None
            
            # Skip if not a vehicle (class filtering can be extended as needed)
            if class_id is not None and class_id == 0:  # Class 0 is person in COCO
                continue
                
            # Get center point of bounding box
            x1, y1, x2, y2 = map(int, bbox[:4])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Check if center point is within ROI
            if 0 <= center_x < frame_shape[1] and 0 <= center_y < frame_shape[0]:
                if mask[center_y, center_x] > 0:
                    vehicle_count += 1
        
        return vehicle_count
    
    def _get_congestion_level(self, density: float) -> int:
        """
        Determine congestion level (1-5) based on vehicle density per 100m²
        
        Args:
            density: Vehicle density (vehicles per 100m²)
            
        Returns:
            Congestion level (1-5)
        """
        for level, threshold in enumerate(self.congestion_thresholds, 1):
            # Levels 1, 2, 3, 4 are determined by being LESS than the threshold
            if density < threshold:
                return level
                
        # If density is not less than the last threshold (15.0), it's Level 5
        return len(self.congestion_thresholds) + 1 # This will be 4 + 1 = 5
    
    def _get_congestion_text(self, level: int) -> str:
        """
        Get textual description of congestion level (1-5)
        
        Args:
            level: Congestion level (1-5)
            
        Returns:
            Textual description
        """
        # UPDATED for 5 levels based on provided table
        congestion_texts = {
            1: "Mức 1 (thông thoáng)",    # < 2
            2: "Mức 2 (ổn định)",       # 2 to < 6
            3: "Mức 3 (ùn ứ)",          # 6 to < 10
            4: "Mức 4 (tắc nghẽn)",     # 10 to < 15
            5: "Mức 5 (tê liệt)"        # >= 15
        }
        
        return congestion_texts.get(level, "Unknown")


# Create a singleton instance of the congestion calculator
congestion_calculator = CongestionCalculator()

def debug_draw_roi_and_detections(frame_path, roi_points, detections, output_path=None):
    """
    Debug function to draw ROI and detections on an image
    
    Args:
        frame_path: Path to image file
        roi_points: ROI points
        detections: YOLOv8 detections
        output_path: Path to save the debug image
        
    Returns:
        Path to saved debug image
    """
    frame = cv2.imread(frame_path)
    if frame is None:
        return None
    
    # Convert normalized ROI points to pixel coordinates
    h, w = frame.shape[:2]
    roi_polygon = np.array([
        (int(p["x"] * w), int(p["y"] * h)) for p in roi_points
    ], dtype=np.int32)
    
    # Draw ROI
    cv2.polylines(frame, [roi_polygon], True, (0, 255, 0), 2)
    
    # Draw detections
    if hasattr(detections, 'xyxy') and detections.xyxy.size > 0:
        for i, bbox in enumerate(detections.xyxy):
            x1, y1, x2, y2 = map(int, bbox[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Get center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 4, (255, 0, 0), -1)
    
    # Save debug image
    if output_path is None:
        output_dir = Path(frame_path).parent / "debug"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"debug_{Path(frame_path).name}"
    
    cv2.imwrite(str(output_path), frame)
    return str(output_path)