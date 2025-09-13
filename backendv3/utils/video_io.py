"""Video input/output utilities."""
import os
import cv2
import tempfile
import logging
import time
import json
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class VideoReader:
    """Video file reader with efficient frame extraction."""
    
    def __init__(self, video_path):
        """
        Initialize video reader.
        
        Args:
            video_path (str): Path to the video file
        """
        self.video_path = video_path
        self.cap = None
        
        # Try to open video
        self._open_video()
        
        if not self.is_opened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        
    def _open_video(self):
        """Open the video file."""
        if self.cap is not None:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(self.video_path)
        
    def is_opened(self):
        """Check if video is successfully opened."""
        return self.cap is not None and self.cap.isOpened()
    
    def get_frame(self, frame_number=None):
        """
        Get a specific frame or the next frame.
        
        Args:
            frame_number (int, optional): Specific frame number to get
            
        Returns:
            tuple: (success, frame)
        """
        if not self.is_opened():
            return False, None
            
        # Get specific frame if requested
        if frame_number is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
        success, frame = self.cap.read()
        return success, frame
    
    def get_frames(self, start_frame=0, end_frame=None, step=1):
        """
        Get a range of frames.
        
        Args:
            start_frame (int): Starting frame number
            end_frame (int, optional): Ending frame number (inclusive)
            step (int): Step size (every Nth frame)
            
        Yields:
            tuple: (frame_number, frame)
        """
        if not self.is_opened():
            return
            
        if end_frame is None:
            end_frame = self.frame_count - 1
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_number in range(start_frame, end_frame + 1, step):
            success, frame = self.cap.read()
            
            if not success:
                break
                
            yield frame_number, frame
            
            # Skip frames according to step
            if step > 1 and frame_number + step <= end_frame:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number + step)
    
    def analyze_video_quality(self):
        """
        Analyze the video to determine optimal processing parameters.
        
        Returns:
            dict: Analysis results including recommended frame skip
        """
        # Sample a few frames to analyze
        sample_count = min(20, self.frame_count)
        sample_indices = np.linspace(0, self.frame_count - 1, sample_count, dtype=int)
        
        # Analyze brightness, motion, etc.
        brightness_values = []
        
        for idx in sample_indices:
            success, frame = self.get_frame(idx)
            if success:
                # Calculate average brightness
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness_values.append(np.mean(gray))
        
        avg_brightness = np.mean(brightness_values) if brightness_values else 0
        brightness_std = np.std(brightness_values) if len(brightness_values) > 1 else 0
        
        # Recommend frame skip based on FPS
        # Higher FPS videos can skip more frames
        if self.fps > 30:
            recommended_skip = 15
        elif self.fps > 20:
            recommended_skip = 10
        else:
            recommended_skip = 5
            
        results = {
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration": self.duration,
            "resolution": f"{self.width}x{self.height}",
            "avg_brightness": avg_brightness,
            "brightness_std": brightness_std,
            "recommended_skip": recommended_skip
        }
        
        return results
    
    def close(self):
        """Close the video file."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class VideoWriter:
    """Video file writer with flexible options."""
    
    def __init__(self, output_path, fps, frame_size, fourcc='mp4v'):
        """
        Initialize video writer.
        
        Args:
            output_path (str): Path to save the output video
            fps (float): Frames per second
            frame_size (tuple): Frame size (width, height)
            fourcc (str): Four-character code for codec
        """
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Create VideoWriter
        fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
        self.writer = cv2.VideoWriter(output_path, fourcc_code, fps, frame_size)
        
        if not self.writer.isOpened():
            raise ValueError(f"Could not create video writer for {output_path}")
    
    def write_frame(self, frame):
        """
        Write a frame to the video.
        
        Args:
            frame (numpy.ndarray): Frame to write
        """
        if frame.shape[1::-1] != self.frame_size:
            frame = cv2.resize(frame, self.frame_size)
            
        self.writer.write(frame)
    
    def close(self):
        """Close the video writer."""
        if self.writer is not None:
            self.writer.release()


def get_video_info(video_path):
    """
    Get basic information about a video file.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        dict: Video information or None if file couldn't be opened
    """
    if not os.path.exists(video_path):
        return None
        
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        info = {
            "path": video_path,
            "filename": os.path.basename(video_path),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "format": os.path.splitext(video_path)[1].lower().lstrip("."),
            "file_size_mb": os.path.getsize(video_path) / (1024 * 1024)
        }
        
        info["duration"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
        cap.release()
        
        return info
    except Exception as e:
        logger.error(f"Error getting video info for {video_path}: {str(e)}")
        return None