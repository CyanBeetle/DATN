import os
import cv2
import time
import json
import numpy as np
import supervision as sv
from tqdm import tqdm
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

from video.detector import VehicleDetector
from video.tracker import VehicleTracker
from video.speed import SpeedCalculator
from utils.roi import ROIHandler
from utils.gpu import ResourceMonitor

class VideoProcessor:
    """Main video processing pipeline for vehicle detection and speed calculation"""
    
    def __init__(self, 
                 model_name="yolov8x.pt", 
                 confidence=0.3, 
                 iou_threshold=0.5,
                 resolution=1280, 
                 use_gpu=True,
                 batch_size=8,
                 results_dir=None):
        """
        Initialize video processor
        
        Args:
            model_name: Name or path of YOLO model
            confidence: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            resolution: Input resolution for model
            use_gpu: Whether to use GPU if available
            batch_size: Batch size for processing multiple frames at once
            results_dir: Directory to save results (defaults to './results')
        """
        self.detector = VehicleDetector(
            model_name=model_name,
            confidence=confidence,
            use_gpu=use_gpu,
            batch_size=batch_size
        )
        
        self.iou_threshold = iou_threshold
        self.resolution = resolution
        self.roi_handler = ROIHandler()
        self.resource_monitor = ResourceMonitor()
        self.batch_size = batch_size
        self.results_dir = results_dir or os.path.join(os.getcwd(), "results")
        
        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)
    
    def process_video(self, 
                      input_path, 
                      output_path=None, 
                      roi_points=None, 
                      target_width=50, 
                      target_height=100, 
                      frame_skip=None, 
                      max_duration=None, 
                      progress_callback=None,
                      result_prefix=None):
        """
        Process video to detect vehicles and calculate speeds
        
        Args:
            input_path: Path to input video
            output_path: Path to output video (None to skip video generation)
            roi_points: ROI points for perspective transform
            target_width: Width of ROI in meters
            target_height: Height of ROI in meters
            frame_skip: Process every N-th frame (None for auto)
            max_duration: Maximum duration to process in seconds
            progress_callback: Callback function for progress updates
            result_prefix: Optional prefix to add to result JSON filename
            
        Returns:
            Dictionary with processing statistics
        """
        # Get ROI points if not provided
        if roi_points is None:
            roi_points = self.roi_handler.get_default_roi(input_path)
        else:
            valid, message = self.roi_handler.validate_roi(roi_points)
            if not valid:
                raise ValueError(f"Invalid ROI: {message}")
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Auto-calculate frame skip if not provided
        if frame_skip is None:
            frame_skip = max(1, int(fps / 2))  # Target about 2 FPS processing
            print(f"Auto-calculated frame skip: {frame_skip} for {fps} fps video (targeting 2 fps)")
        
        # Limit duration if specified
        max_frames = total_frames
        if max_duration:
            max_frames = min(max_frames, int(max_duration * fps))
        
        # Setup output video if needed
        out = None
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Initialize tracker and speed calculator
        tracker = VehicleTracker(fps=fps)
        speed_calc = SpeedCalculator(roi_points, target_width, target_height, fps)
        
        # Create polygon zone for filtering - handle different Supervision versions
        try:
            # Try with resolution parameter (newer versions)
            polygon_zone = sv.PolygonZone(polygon=np.array(roi_points), resolution=(width, height))
        except TypeError:
            # Fallback for older versions that don't support resolution parameter
            polygon_zone = sv.PolygonZone(polygon=np.array(roi_points))
            
        # Initialize statistics
        stats = {
            "processed_frames": 0,
            "total_detections": 0,
            "unique_vehicles": set(),
            "start_time": time.time(),
            "frame_width": width,
            "frame_height": height,
            "fps": fps,
            "roi_points": roi_points.tolist() if isinstance(roi_points, np.ndarray) else roi_points,
        }
        
        # Initialize results storage
        detection_data = []
        
        # Calculate total frames to process for progress tracking
        frames_to_process = max_frames // frame_skip
        
        # Define custom progress bar function
        def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
            """
            Call in a loop to create terminal progress bar
            """
            percent = ("{0:.1f}").format(100 * (iteration / float(total)))
            filled_length = int(length * iteration // total)
            bar = fill * filled_length + '-' * (length - filled_length)
            print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
            # Print New Line on Complete
            if iteration == total: 
                print()
        
        # Process frames
        try:
            frame_idx = 0
            processed_count = 0
            batch_frames = []
            batch_indices = []
            
            print_progress_bar(0, frames_to_process, prefix='Processing video:', suffix='Complete', length=50)
            
            while frame_idx < max_frames:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every Nth frame
                if frame_idx % frame_skip == 0:
                    # Add frame to batch
                    batch_frames.append(frame)
                    batch_indices.append(frame_idx)
                    
                    # Process batch when it reaches batch size or at the end
                    if len(batch_frames) >= self.batch_size or frame_idx + frame_skip >= max_frames:
                        # Run batch detection
                        all_detections = self.detector.detect_batch(batch_frames)
                        
                        for i, (batch_idx, detections) in enumerate(zip(batch_indices, all_detections)):
                            frame = batch_frames[i]
                            
                            # Filter by ROI
                            detections = detections[polygon_zone.trigger(detections)]
                            
                            # Apply NMS
                            detections = self.detector.safe_nms(detections, self.iou_threshold)
                            
                            # Track objects
                            detections = tracker.update(detections)
                            
                            # Calculate positions and speeds
                            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                            transformed_points = speed_calc.transform_points(points)
                            
                            # Process each detection
                            labels = []
                            vehicle_data = []
                            
                            for j, tracker_id in enumerate(detections.tracker_id):
                                # Update position for speed calculation
                                if transformed_points.size > 0:
                                    speed_calc.update_position(tracker_id, transformed_points[j])
                                
                                # Get detection details
                                xyxy = detections.xyxy[j]
                                class_id = detections.class_id[j]
                                confidence = detections.confidence[j]
                                
                                # Create vehicle data entry
                                vehicle_info = {
                                    "tracker_id": int(tracker_id),
                                    "class_id": int(class_id),
                                    "class_name": self.detector.class_names[class_id],
                                    "confidence": float(confidence),
                                    "bbox": [float(x) for x in xyxy],
                                }
                                
                                # Calculate speed if enough tracking data is available
                                speed = speed_calc.calculate_speed(tracker_id)
                                if speed is None:
                                    labels.append(f"#{tracker_id}")
                                else:
                                    labels.append(f"#{tracker_id} {int(speed)} km/h")
                                    vehicle_info["speed_kmh"] = float(speed)
                                    stats["unique_vehicles"].add(tracker_id)
                                
                                # Add vehicle to the list
                                vehicle_data.append(vehicle_info)
                            
                            # Save frame data
                            frame_data = {
                                "frame_number": batch_idx,
                                "timestamp": batch_idx / fps,
                                "vehicles": vehicle_data
                            }
                            detection_data.append(frame_data)
                            
                            # Create annotated frame if needed
                            if out is not None:
                                annotated_frame = frame.copy()
                                
                                # Draw ROI - ensure proper format for OpenCV
                                try:
                                    # Make sure the polygon points are in the right format for OpenCV
                                    # Convert to numpy array with int32 type and reshape to expected format
                                    polygon_pts = np.array(roi_points, dtype=np.int32).reshape((-1, 1, 2))
                                    cv2.polylines(
                                        annotated_frame, 
                                        [polygon_pts], 
                                        isClosed=True, 
                                        color=(0, 0, 255),  # Red color in BGR
                                        thickness=2
                                    )
                                except Exception as e:
                                    print(f"Warning: Could not draw ROI polygon: {e}")
                                
                                # Draw bounding boxes
                                box_annotator = sv.BoxAnnotator()
                                label_annotator = sv.LabelAnnotator()
                                
                                annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
                                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
                                
                                out.write(annotated_frame)
                            
                            # Update statistics
                            stats["total_detections"] += len(detections)
                            stats["processed_frames"] += 1
                            
                            processed_count += 1
                        
                        # Update progress
                        print_progress_bar(min(processed_count, frames_to_process), frames_to_process, 
                                         prefix='Processing video:', suffix='Complete', length=50)
                        
                        # Call progress callback if provided
                        if progress_callback:
                            progress = min(int(100 * processed_count / frames_to_process), 100)
                            progress_callback(progress)
                        
                        # Clear batch
                        batch_frames = []
                        batch_indices = []
                
                # Check resource usage periodically
                if frame_idx % 100 == 0:
                    self.resource_monitor.cool_down_if_needed()
                
                frame_idx += 1
                
        finally:
            # Clean up
            print()  # Ensure new line after progress bar
            if cap.isOpened():
                cap.release()
            if out is not None:
                out.release()
        
        # Finalize statistics
        stats["unique_vehicles"] = len(stats["unique_vehicles"])
        stats["processing_time"] = time.time() - stats["start_time"]
        
        # Save results to JSON
        json_path = None
        if detection_data:
            json_path = self._save_results(
                input_path, detection_data, roi_points, 
                target_width, target_height, 
                fps, frame_skip, width, height, 
                total_frames, max_frames, result_prefix
            )
            stats["json_path"] = json_path
        
        return stats
    
    def _save_results(self, input_path, detection_data, roi_points, 
                     target_width, target_height,
                     fps, frame_skip, width, height, 
                     total_frames, processed_frames, result_prefix=None):
        """
        Save detection results to JSON file optimized for congestion prediction
        
        Args:
            input_path: Path to input video
            detection_data: List of detection data for each frame
            roi_points: ROI points used
            target_width: Width in meters
            target_height: Height in meters
            fps: Frames per second
            frame_skip: Frame skip value
            width: Video width
            height: Video height
            total_frames: Total frames in video
            processed_frames: Number of frames processed
            result_prefix: Optional prefix to add to result JSON filename
            
        Returns:
            Path to saved JSON file
        """
        # Extract video info
        filename = os.path.basename(input_path)
        
        # Record time is when the video is processed (current time)
        processed_date = datetime.now().strftime("%d/%m/%Y/%H/%M/%S")
        
        # Track unique vehicles and their data
        unique_vehicles = {}
        vehicle_appearances = {}
        
        # Process frame data to collect vehicle information
        for frame in detection_data:
            frame_time = frame["timestamp"]  # Time in seconds from start of video (starts at 0)
            
            for vehicle in frame["vehicles"]:
                tracker_id = vehicle.get("tracker_id")
                if tracker_id is None:
                    continue
                    
                # Get or initialize vehicle data
                if tracker_id not in unique_vehicles:
                    unique_vehicles[tracker_id] = {
                        "id": tracker_id,
                        "type": vehicle.get("class_name", "unknown"),
                        "first_seen_time": frame_time,
                        "last_seen_time": frame_time,
                        "total_tracked_time": 0.0,
                        "speeds": [],
                        "avg_speed_kmh": None
                    }
                
                # Update vehicle data
                vehicle_data = unique_vehicles[tracker_id]
                vehicle_data["last_seen_time"] = frame_time
                vehicle_data["total_tracked_time"] = round(vehicle_data["last_seen_time"] - vehicle_data["first_seen_time"], 2)
                
                # Add speed if available
                if "speed_kmh" in vehicle:
                    speed = vehicle["speed_kmh"]
                    vehicle_data["speeds"].append(speed)
                    vehicle_data["avg_speed_kmh"] = round(sum(vehicle_data["speeds"]) / len(vehicle_data["speeds"]), 2)
                
                # Track vehicle appearances in multiple time intervals for ML analysis
                # Using relative time intervals (in seconds from video start)
                
                # 1-minute interval (60 seconds) for fine-grained analysis
                interval_1min = int(frame_time / 60)  
                interval_key_1min = f"1min_{interval_1min}"
                
                if interval_key_1min not in vehicle_appearances:
                    vehicle_appearances[interval_key_1min] = {
                        "interval_type": "1min",
                        "interval_index": interval_1min,
                        "start_time": interval_1min * 60,
                        "end_time": (interval_1min + 1) * 60,
                        "vehicles": set(),
                        "speed_sum": 0,
                        "speed_count": 0,
                        "vehicle_counts": {},
                        "cumulative_count": 0
                    }
                
                # 3-minute interval (180 seconds) - standard for traffic analysis
                interval_3min = int(frame_time / 180)
                interval_key_3min = f"3min_{interval_3min}"
                
                if interval_key_3min not in vehicle_appearances:
                    vehicle_appearances[interval_key_3min] = {
                        "interval_type": "3min",
                        "interval_index": interval_3min,
                        "start_time": interval_3min * 180,
                        "end_time": (interval_3min + 1) * 180,
                        "vehicles": set(),
                        "speed_sum": 0,
                        "speed_count": 0,
                        "vehicle_counts": {},
                        "cumulative_count": 0
                    }
                
                # 5-minute interval (300 seconds) for smoother trends
                interval_5min = int(frame_time / 300)
                interval_key_5min = f"5min_{interval_5min}"
                
                if interval_key_5min not in vehicle_appearances:
                    vehicle_appearances[interval_key_5min] = {
                        "interval_type": "5min",
                        "interval_index": interval_5min,
                        "start_time": interval_5min * 300,
                        "end_time": (interval_5min + 1) * 300,
                        "vehicles": set(),
                        "speed_sum": 0,
                        "speed_count": 0,
                        "vehicle_counts": {},
                        "cumulative_count": 0
                    }
                
                # Add vehicle to all interval types
                for interval_key in [interval_key_1min, interval_key_3min, interval_key_5min]:
                    interval_data = vehicle_appearances[interval_key]
                    
                    # Only count new vehicles in this interval
                    if tracker_id not in interval_data["vehicles"]:
                        interval_data["vehicles"].add(tracker_id)
                        interval_data["cumulative_count"] += 1
                        
                        # Update vehicle type counts
                        vehicle_type = vehicle.get("class_name", "unknown")
                        if vehicle_type not in interval_data["vehicle_counts"]:
                            interval_data["vehicle_counts"][vehicle_type] = 0
                        interval_data["vehicle_counts"][vehicle_type] += 1
                    
                    # Update speed data if available
                    if "speed_kmh" in vehicle:
                        interval_data["speed_sum"] += vehicle["speed_kmh"]
                        interval_data["speed_count"] += 1
        
        # Calculate average speeds for intervals
        for interval_data in vehicle_appearances.values():
            if interval_data["speed_count"] > 0:
                interval_data["avg_speed"] = interval_data["speed_sum"] / interval_data["speed_count"]
            else:
                interval_data["avg_speed"] = None
            
            # Convert set to count for serialization
            interval_data["vehicle_count"] = len(interval_data["vehicles"])
            interval_data["vehicles"] = list(interval_data["vehicles"])
        
        # Calculate video duration in seconds
        duration_seconds = total_frames / fps if fps > 0 else 0
        
        # Create optimized result structure
        result = {
            "metadata": {
                "filename": filename,
                "recording_date": None,  # No specific recording date needed
                "duration_seconds": round(duration_seconds, 3),
                "processed_date": processed_date,
                "video_info": {
                    "resolution": f"{width}x{height}",
                    "original_fps": round(fps, 2),
                    "processed_fps": round(fps / frame_skip, 2) if frame_skip > 0 else round(fps, 2),
                    "total_frames": total_frames,
                    "processed_frames": processed_frames
                }
            },
            "roi_config": {
                "points": roi_points,
                "width_meters": target_width,
                "height_meters": target_height
            },
            "unique_vehicles": list(unique_vehicles.values())
        }
        
        # Save JSON file
        base_name = os.path.splitext(filename)[0]
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"{result_prefix}_{base_name}_{timestamp_str}_analysis.json" if result_prefix else f"{base_name}_{timestamp_str}_analysis.json"
        json_path = os.path.join(self.results_dir, json_filename)
        
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Saved analysis results to {json_path}")
        return json_path

# Mock processing function - replace with actual implementation
async def process_video_task(
    input_path: str,
    output_path: str,
    task_id: str,
    user_id: str
) -> Dict[str, Any]:
    """
    Process video file using computer vision techniques.
    
    This is a placeholder for the actual video processing implementation.
    """
    logger.info(f"Starting video processing task {task_id} for file {input_path}")
    
    try:
        # TODO: Replace with actual video processing code
        # Simulating processing time
        time.sleep(5)
        
        # Create a mock output file
        with open(output_path, "w") as f:
            f.write(f"Processed by task {task_id}\n")
            f.write(f"Original file: {input_path}\n")
            f.write(f"Processed at: {time.ctime()}\n")
        
        logger.info(f"Video processing completed for task {task_id}")
        return {
            "task_id": task_id,
            "status": "completed",
            "input_file": os.path.basename(input_path),
            "output_file": os.path.basename(output_path)
        }
        
    except Exception as e:
        logger.error(f"Error processing video for task {task_id}: {str(e)}")
        return {
            "task_id": task_id,
            "status": "failed",
            "error": str(e)
        }