import torch
from ultralytics import YOLO
import supervision as sv
import numpy as np

class VehicleDetector:
    """YOLOv8 vehicle detector with GPU support"""
    
    def __init__(self, model_name="yolov8x.pt", confidence=0.3, use_gpu=True, batch_size=8):
        """
        Initialize the vehicle detector
        
        Args:
            model_name: YOLO model name or path
            confidence: Detection confidence threshold
            use_gpu: Whether to use GPU if available
            batch_size: Batch size for processing multiple frames at once
        """
        # Setup device for model
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device} for detection")
        
        # Load model
        self.model = YOLO(model_name)
        self.model.to(self.device)
        
        # Parameters
        self.confidence = confidence
        self.class_names = self.model.names
        self.batch_size = batch_size
        
    def detect(self, frame, resolution=1280):
        """
        Detect objects in a video frame
        
        Args:
            frame: Input video frame (numpy array)
            resolution: Input resolution for the model
            
        Returns:
            Detections object with filtered results
        """
        # Perform detection
        result = self.model(frame, imgsz=resolution, verbose=False, device=self.device)[0]
        detections = sv.Detections.from_ultralytics(result)
        
        # Filter by confidence and class (remove people - class 0)
        detections = detections[detections.confidence > self.confidence]
        detections = detections[detections.class_id != 0]  # Filter out people
        
        return detections
        
    def detect_batch(self, frames, resolution=1280):
        """
        Detect objects in a batch of video frames
        
        Args:
            frames: List of input video frames (numpy arrays)
            resolution: Input resolution for the model
            
        Returns:
            List of Detections objects with filtered results
        """
        if not frames:
            return []
            
        # Perform detection on batch
        results = self.model(frames, imgsz=resolution, verbose=False, device=self.device)
        
        all_detections = []
        for result in results:
            detections = sv.Detections.from_ultralytics(result)
            
            # Filter by confidence and class (remove people - class 0)
            detections = detections[detections.confidence > self.confidence]
            detections = detections[detections.class_id != 0]  # Filter out people
            
            all_detections.append(detections)
            
        return all_detections
    
    def safe_nms(self, detections, iou_threshold=0.5):
        """
        Apply Non-Maximum Suppression with CPU fallback
        
        Args:
            detections: Supervision Detections object
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Filtered detections
        """
        try:
            # Try with current device (potentially CUDA)
            return detections.with_nms(iou_threshold)
        except RuntimeError as e:
            if "torchvision::nms" in str(e) and "CUDA" in str(e):
                # CPU fallback for NMS if CUDA version fails
                print("⚠️ CUDA NMS failed, falling back to CPU for NMS operation")
                
                # Move tensors to CPU for NMS
                cpu_detections = detections
                try:
                    if hasattr(cpu_detections, 'xyxy') and hasattr(cpu_detections.xyxy, 'device'):
                        cpu_detections.xyxy = cpu_detections.xyxy.cpu()
                    if hasattr(cpu_detections, 'confidence') and hasattr(cpu_detections.confidence, 'device'):
                        cpu_detections.confidence = cpu_detections.confidence.cpu()
                    
                    return cpu_detections.with_nms(iou_threshold)
                except Exception as inner_e:
                    print(f"CPU fallback also failed: {inner_e}")
                    # Return original detections if CPU fallback also fails
                    return detections
            else:
                # Re-raise other errors
                raise