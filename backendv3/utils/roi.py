"""Region of Interest (ROI) handling utilities for video processing."""
import cv2
import numpy as np
import json
import os
import datetime
from typing import List, Tuple, Dict, Any, Optional


class ROIHandler:
    """Comprehensive utilities for handling regions of interest in videos."""
    
    # Default directory for storing ROI data
    DEFAULT_ROI_DIR = os.path.join(os.getcwd(), "data", "roi")
    
    @staticmethod
    def validate_roi(points: List[List[float]]) -> Tuple[bool, str]:
        """
        Validate that ROI points form a proper quadrilateral.
        
        Args:
            points: List of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            
        Returns:
            tuple: (is_valid, message)
        """
        if not isinstance(points, list) or len(points) != 4:
            return False, "ROI must have exactly 4 points"
        
        # Check each point has 2 coordinates
        for p in points:
            if not isinstance(p, list) or len(p) != 2:
                return False, "Each ROI point must have x,y coordinates"
                
        # Check for convex quadrilateral using cross product method
        def cross_product(p1, p2, p3):
            return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        
        # Check each corner
        signs = []
        for i in range(4):
            p1 = points[i]
            p2 = points[(i + 1) % 4]
            p3 = points[(i + 2) % 4]
            signs.append(cross_product(p1, p2, p3) > 0)
        
        # All signs must be the same for a convex quadrilateral
        if not (all(signs) or not any(signs)):
            return False, "ROI points must form a convex quadrilateral"
        
        # Check minimum area (to avoid degenerate ROIs)
        area = 0
        for i in range(4):
            j = (i + 1) % 4
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        area = abs(area) / 2
        
        if area < 1000:  # Arbitrary minimum size
            return False, "ROI area is too small"
        
        return True, "Valid ROI"
    
    @staticmethod
    def generate_default_roi(video_path: str) -> np.ndarray:
        """
        Generate default ROI for a video as a trapezoid in the lower half of frame.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            np.ndarray: Array of ROI points
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Create trapezoid in lower half of frame
        margin_top = width // 6
        margin_bottom = width // 12
        
        roi = np.array([
            [margin_top, height//2],         # top-left
            [width-margin_top, height//2],   # top-right
            [width-margin_bottom, height],   # bottom-right
            [margin_bottom, height]          # bottom-left
        ])
        
        return roi
    
    # Alias for backward compatibility with existing code
    get_default_roi = generate_default_roi
    
    @staticmethod
    def create_transformation_matrix(src_points: List[List[float]],
                                   target_width: int = 50,
                                   target_height: int = 100) -> np.ndarray:
        """
        Create perspective transform matrix for ROI.
        
        Args:
            src_points: Source points in image space
            target_width: Width in meters of the target area
            target_height: Height in meters of the target area
            
        Returns:
            np.ndarray: Transformation matrix
        """
        # Define target points in normalized space
        target = np.array([
            [0, 0],  # top-left
            [target_width - 1, 0],  # top-right
            [target_width - 1, target_height - 1],  # bottom-right
            [0, target_height - 1]  # bottom-left
        ], dtype=np.float32)
        
        # Convert source points to float32
        source = np.array(src_points, dtype=np.float32)
        
        # Calculate perspective transform
        transform_matrix = cv2.getPerspectiveTransform(source, target)
        return transform_matrix
    
    @staticmethod
    def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """
        Transform points from image space to real-world space.
        
        Args:
            points: Points to transform
            matrix: Transformation matrix
            
        Returns:
            np.ndarray: Transformed points
        """
        # Reshape for transformation
        points = np.array(points, dtype=np.float32)
        if points.size == 0:
            return np.array([])
        
        points = points.reshape(-1, 1, 2)
        
        # Apply transformation
        transformed = cv2.perspectiveTransform(points, matrix)
        return transformed.reshape(-1, 2)
    
    @staticmethod
    def save_roi(roi_id: str, roi_points: List[List[float]], 
                user_id: Optional[str] = None, 
                metadata: Optional[Dict[str, Any]] = None, 
                roi_dir: Optional[str] = None) -> str:
        """
        Save ROI points to a file.
        
        Args:
            roi_id: Unique identifier for the ROI
            roi_points: List of ROI points
            user_id: User ID who created the ROI
            metadata: Additional metadata about the ROI
            roi_dir: Directory to save the ROI data (defaults to DEFAULT_ROI_DIR)
            
        Returns:
            str: Path to the saved ROI file
        """
        # Use provided roi_dir or default
        if roi_dir is None:
            roi_dir = ROIHandler.DEFAULT_ROI_DIR
        
        os.makedirs(roi_dir, exist_ok=True)
        
        roi_data = {
            "id": roi_id,
            "points": roi_points,
            "created_by": user_id,
            "created_at": str(datetime.datetime.now()),
            "metadata": metadata or {}
        }
        
        roi_path = os.path.join(roi_dir, f"{roi_id}.json")
        with open(roi_path, "w") as f:
            json.dump(roi_data, f, indent=2)
            
        return roi_path
        
    @staticmethod
    def load_roi(roi_id: str, roi_dir: Optional[str] = None) -> Optional[List[List[float]]]:
        """
        Load ROI points from a file.
        
        Args:
            roi_id: Unique identifier for the ROI
            roi_dir: Directory where ROI data is stored (defaults to DEFAULT_ROI_DIR)
            
        Returns:
            list: ROI points or None if not found
        """
        # Use provided roi_dir or default
        if roi_dir is None:
            roi_dir = ROIHandler.DEFAULT_ROI_DIR
            
        roi_path = os.path.join(roi_dir, f"{roi_id}.json")
        
        if not os.path.exists(roi_path):
            return None
            
        with open(roi_path, "r") as f:
            roi_data = json.load(f)
            
        return roi_data["points"]


# For backward compatibility, create alias ROIUtils pointing to the same class
ROIUtils = ROIHandler