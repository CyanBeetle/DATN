import cv2
import numpy as np
from collections import defaultdict, deque

class SpeedCalculator:
    """Calculate vehicle speeds using perspective transformation and tracking"""
    
    def __init__(self, roi_points, target_width=50, target_height=100, fps=30):
        """
        Initialize speed calculator
        
        Args:
            roi_points: Region of interest points for perspective transformation
            target_width: Real-world width in meters of the ROI
            target_height: Real-world height in meters of the ROI
            fps: Video frames per second
        """
        self.fps = fps
        self.target_width = target_width
        self.target_height = target_height
        
        # Create transformation matrix
        self.transformer = self._create_transformer(roi_points)
        
        # Store positions for speed calculation
        self.positions = defaultdict(lambda: deque(maxlen=int(fps)))
    
    def _create_transformer(self, roi_points):
        """
        Create perspective transform matrix
        
        Args:
            roi_points: Four ROI points defining the region
            
        Returns:
            Perspective transformation matrix
        """
        # Define target points in normalized space (birds-eye view)
        target = np.array([
            [0, 0],  # top-left
            [self.target_width - 1, 0],  # top-right
            [self.target_width - 1, self.target_height - 1],  # bottom-right
            [0, self.target_height - 1]  # bottom-left
        ], dtype=np.float32)
        
        # Convert ROI points to float32
        source = np.array(roi_points, dtype=np.float32)
        
        # Calculate perspective transform
        transform_matrix = cv2.getPerspectiveTransform(source, target)
        return transform_matrix
    
    def transform_points(self, points):
        """
        Transform points from image space to real-world space
        
        Args:
            points: Points in image space
            
        Returns:
            Transformed points in real-world space
        """
        # Reshape for transformation
        points = np.array(points, dtype=np.float32)
        if points.size == 0:
            return np.array([])
        
        points = points.reshape(-1, 1, 2)
        
        # Apply transformation
        transformed = cv2.perspectiveTransform(points, self.transformer)
        return transformed.reshape(-1, 2)
    
    def update_position(self, tracker_id, position):
        """
        Update tracked position for a vehicle
        
        Args:
            tracker_id: Unique ID of tracked vehicle
            position: Current position (transformed coordinates)
        """
        self.positions[tracker_id].append(position)
    
    def calculate_speed(self, tracker_id):
        """
        Calculate speed in km/h for tracked vehicle
        
        Args:
            tracker_id: Unique ID of tracked vehicle
            
        Returns:
            Speed in km/h or None if not enough data
        """
        positions = self.positions[tracker_id]
        if len(positions) < self.fps / 2:  # Need at least half a second of tracking
            return None
        
        # Calculate distance in meters
        first_pos = positions[0]
        last_pos = positions[-1]
        distance = np.linalg.norm(np.array(last_pos) - np.array(first_pos))
        
        # Calculate time difference in seconds
        time_diff = len(positions) / self.fps
        
        # Calculate speed in km/h (distance in meters, time in seconds)
        if time_diff > 0:
            speed = distance / time_diff * 3.6  # Convert m/s to km/h
            return speed
        
        return None
    
    def get_all_speeds(self):
        """
        Get speeds for all tracked vehicles
        
        Returns:
            Dictionary of tracker IDs and their speeds
        """
        speeds = {}
        for tracker_id in self.positions:
            speed = self.calculate_speed(tracker_id)
            if speed is not None:
                speeds[tracker_id] = speed
        
        return speeds