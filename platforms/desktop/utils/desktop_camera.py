import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from core.detection.object_tracker import BaseCamera

class DesktopCamera(BaseCamera):
    """Camera implementation for desktop platforms using OpenCV."""
    
    def __init__(self, camera_id: int = 0, resolution: Optional[Tuple[int, int]] = None):
        """
        Initialize the desktop camera.
        
        Args:
            camera_id: ID of the camera to use (typically 0 for built-in webcam)
            resolution: Optional tuple of (width, height) to set camera resolution
        """
        self.camera_id = camera_id
        self.resolution = resolution
        self.capture = None
        self.initialize()
    
    def initialize(self) -> bool:
        """
        Initialize the camera.
        
        Returns:
            Success status
        """
        try:
            self.capture = cv2.VideoCapture(self.camera_id)
            
            # Set resolution if specified
            if self.resolution is not None:
                width, height = self.resolution
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Check if camera opened successfully
            if not self.capture.isOpened():
                print(f"Error: Could not open camera {self.camera_id}")
                return False
            
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def read(self) -> Tuple[bool, np.ndarray]:
        """
        Read a frame from the camera.
        
        Returns:
            Tuple of (success, frame)
        """
        if self.capture is None or not self.capture.isOpened():
            return False, np.zeros((480, 640, 3), dtype=np.uint8)
        
        return self.capture.read()
    
    def release(self) -> None:
        """Release camera resources."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
    
    def get_properties(self) -> Dict[str, float]:
        """
        Get camera properties.
        
        Returns:
            Dictionary of camera properties
        """
        if self.capture is None:
            return {}
        
        return {
            'width': self.capture.get(cv2.CAP_PROP_FRAME_WIDTH),
            'height': self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
            'fps': self.capture.get(cv2.CAP_PROP_FPS),
            'brightness': self.capture.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': self.capture.get(cv2.CAP_PROP_CONTRAST),
            'saturation': self.capture.get(cv2.CAP_PROP_SATURATION),
            'hue': self.capture.get(cv2.CAP_PROP_HUE),
            'exposure': self.capture.get(cv2.CAP_PROP_EXPOSURE),
        }
    
    def set_property(self, prop_id: int, value: float) -> bool:
        """
        Set camera property.
        
        Args:
            prop_id: Property ID (from cv2.CAP_PROP_*)
            value: Property value
            
        Returns:
            Success status
        """
        if self.capture is None:
            return False
        
        return self.capture.set(prop_id, value)
    
    def __del__(self):
        """Clean up when destroyed."""
        self.release()