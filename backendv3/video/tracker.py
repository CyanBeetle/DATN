import supervision as sv

class VehicleTracker:
    """ByteTrack based vehicle tracker"""
    
    def __init__(self, fps=30.0):
        """
        Initialize tracker
        
        Args:
            fps: Video frames per second (used for tracking parameters)
        """
        self.fps = fps
        
        try:
            # For newer supervision versions
            self.byte_track = sv.ByteTrack(frame_rate=fps)
        except TypeError:
            # Fallback for older versions
            self.byte_track = sv.ByteTrack()
    
    def update(self, detections):
        """
        Track detections between frames
        
        Args:
            detections: Detections from current frame
            
        Returns:
            Updated detections with tracking IDs
        """
        return self.byte_track.update_with_detections(detections)