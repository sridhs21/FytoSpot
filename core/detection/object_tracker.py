import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
from collections import deque
from abc import ABC, abstractmethod

class BaseCamera(ABC):
    """
    Abstract base class for camera interfaces.
    Platform-specific implementations will subclass this.
    """
    @abstractmethod
    def read(self) -> Tuple[bool, np.ndarray]:
        """
        Read a frame from the camera
        
        Returns:
            Tuple of (success, frame)
        """
        pass
    
    @abstractmethod
    def release(self) -> None:
        """Release camera resources"""
        pass

class ObjectTracker:
    """
    Platform-independent object tracker for detecting and tracking plants.
    Works with multiple detection methods and supports recording.
    """
    def __init__(self, camera: BaseCamera, detector):
        """
        Initialize the object tracker.
        
        Args:
            camera: Camera implementation (must subclass BaseCamera)
            detector: Detection algorithm implementation
        """
        # Store camera
        self.capture = camera
        
        # Video writer setup
        self.recording = False
        self.out = None
        
        # Tracking variables
        self.tracker = None
        self.tracking = False
        self.detection_bbox = None
        
        # Smoothing filter for bounding boxes
        self.bbox_history = deque(maxlen=5)
        
        # Initialize plant detector
        self.plant_detector = detector
        
        # Detection method
        self.detection_method = 'multi'  # 'multi', 'color', 'texture', 'contour' or 'manual'
    
    def start_recording(self, filename: str = 'output.mp4', 
                       fps: int = 20, 
                       frame_size: Tuple[int, int] = None) -> None:
        """
        Start recording video to file
        
        Args:
            filename: Output filename
            fps: Frames per second
            frame_size: Frame dimensions (width, height)
        """
        if frame_size is None:
            # Try to get frame size from camera
            ret, frame = self.capture.read()
            if ret:
                height, width = frame.shape[:2]
                frame_size = (width, height)
            else:
                # Default size if can't get frame
                frame_size = (640, 480)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(filename, fourcc, fps, frame_size)
        self.recording = True
        
    def stop_recording(self) -> None:
        """Stop recording video"""
        if self.recording and self.out is not None:
            self.out.release()
            self.recording = False
    
    def smooth_bbox(self, bbox: Optional[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        """
        Apply smoothing to bounding box coordinates
        
        Args:
            bbox: Bounding box (x, y, w, h)
            
        Returns:
            Smoothed bounding box
        """
        if bbox is None:
            return None
            
        self.bbox_history.append(bbox)
        
        if len(self.bbox_history) < 2:
            return bbox
            
        # Calculate the average of recent bounding boxes
        x_avg = sum(b[0] for b in self.bbox_history) / len(self.bbox_history)
        y_avg = sum(b[1] for b in self.bbox_history) / len(self.bbox_history)
        w_avg = sum(b[2] for b in self.bbox_history) / len(self.bbox_history)
        h_avg = sum(b[3] for b in self.bbox_history) / len(self.bbox_history)
        
        return (int(x_avg), int(y_avg), int(w_avg), int(h_avg))
    
    def detect_object(self, frame: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], np.ndarray]:
        """
        Detect objects using combined methods based on detection_method
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (bounding box, debug frame)
        """
        if self.detection_method == 'multi':
            # Use combined detection approach
            bbox, confidence = self.plant_detector.detect(frame)
            
        elif self.detection_method == 'color':
            # Use only color filtering from the plant detector
            color_mask = self.plant_detector.color_filter(frame)
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.plant_detector.min_detection_area]
            
            bbox = None
            confidence = 0.0
            
            if valid_contours:
                largest_contour = max(valid_contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                bbox = (x, y, w, h)
                
                # Use area ratio as a confidence measure
                area_ratio = cv2.contourArea(largest_contour) / (frame.shape[0] * frame.shape[1])
                confidence = min(area_ratio * 10, 1.0)  # Scale to 0-1 range
            
            # Create debug frame
            debug_frame = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
            
            # Add debug title
            debug_frame = cv2.putText(debug_frame, "Color", (10, 30), 
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            return bbox, debug_frame
            
        elif self.detection_method == 'texture':
            # Use only texture filtering from the plant detector
            texture_mask = self.plant_detector.texture_filter(frame)
            contours, _ = cv2.findContours(texture_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.plant_detector.min_detection_area]
            
            bbox = None
            confidence = 0.0
            
            if valid_contours:
                largest_contour = max(valid_contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                bbox = (x, y, w, h)
                
                # Use area ratio as a confidence measure
                area_ratio = cv2.contourArea(largest_contour) / (frame.shape[0] * frame.shape[1])
                confidence = min(area_ratio * 10, 1.0)  # Scale to 0-1 range
            
            # Create debug frame
            debug_frame = cv2.cvtColor(texture_mask, cv2.COLOR_GRAY2BGR)
            
            # Add debug title
            debug_frame = cv2.putText(debug_frame, "Texture", (10, 30), 
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Draw on debug frame if bbox exists
            if bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            return bbox, debug_frame
            
        elif self.detection_method == 'contour':
            # Use only contour analysis from the plant detector
            contour_mask = self.plant_detector.contour_analysis(frame)
            contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.plant_detector.min_detection_area]
            
            bbox = None
            confidence = 0.0
            
            if valid_contours:
                largest_contour = max(valid_contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                bbox = (x, y, w, h)
                
                # Use area ratio as a confidence measure
                area_ratio = cv2.contourArea(largest_contour) / (frame.shape[0] * frame.shape[1])
                confidence = min(area_ratio * 10, 1.0)  # Scale to 0-1 range
            
            # Create debug frame
            debug_frame = cv2.cvtColor(contour_mask, cv2.COLOR_GRAY2BGR)
            
            # Add debug title
            debug_frame = cv2.putText(debug_frame, "Contour", (10, 30), 
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Draw on debug frame if bbox exists
            if bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            return bbox, debug_frame
        
        # If using multi-method detection, use the plant detector's visualize method
        if self.detection_method == 'multi':
            if bbox is not None:
                self.detection_bbox = bbox
                
            # Return visualization frames for display
            display_frame, debug_frame = self.plant_detector.visualize(frame, bbox, confidence)
            
            return bbox, debug_frame
        
        # Fallback for other methods
        return None, frame.copy()
    
    def start_tracking(self, bbox=None):
        """
        Start tracking the object in the given bounding box
        
        Args:
            bbox: Bounding box to track (x, y, w, h). If None, uses last detected bbox.
            
        Returns:
            Success status
        """
        if bbox is None and self.detection_bbox is None:
            return False
        
        # Use provided bbox or detection_bbox
        box_to_track = bbox if bbox is not None else self.detection_bbox
        
        # Create and initialize tracker
        # CSRT is more accurate but slower, KCF is faster but less accurate
        self.tracker = cv2.TrackerCSRT_create()  # More accurate but slower
        # self.tracker = cv2.TrackerKCF_create()  # Faster but less accurate

        ret, frame = self.capture.read()
        if ret:
            success = self.tracker.init(frame, box_to_track)
            self.tracking = success
            # Clear bbox history for new tracking
            self.bbox_history.clear()
            return success
        return False
    
    def stop_tracking(self):
        """Stop tracking"""
        self.tracking = False
        self.tracker = None
        self.bbox_history.clear()
    
    def process_frame(self):
        """
        Process a frame from the camera
        
        Returns:
            Tuple of (display frame, debug frame)
        """
        ret, frame = self.capture.read()
        if not ret:
            return None, None
        
        # Make a copy for display
        display_frame = frame.copy()
        debug_frame = None
        
        # If tracking, update the tracker
        if self.tracking and self.tracker is not None:
            success, bbox = self.tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                
                # Add to bbox history for potential smoothing
                smooth_bbox = self.smooth_bbox((x, y, w, h))
                if smooth_bbox:
                    x, y, w, h = smooth_bbox
                
                # Update detection_bbox for reference
                self.detection_bbox = (x, y, w, h)
                
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(display_frame, "Tracking", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                cv2.putText(display_frame, "Tracking failed", (100, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                self.tracking = False
        
        # If not tracking, detect objects
        else:
            bbox, debug_frame = self.detect_object(frame)
            
            if bbox is not None:
                x, y, w, h = bbox
                self.detection_bbox = bbox
                
                # Draw on display frame
                if self.detection_method == 'multi':
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(display_frame, "Plant Detected", (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                elif self.detection_method == 'color':
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(display_frame, "Color Detection", (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                elif self.detection_method == 'contour':
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
                    cv2.putText(display_frame, "Contour Detection", (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                elif self.detection_method == 'texture':
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                    cv2.putText(display_frame, "Texture Detection", (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Display the detection method
        method_text = f"Mode: {self.detection_method.upper()}"
        cv2.putText(display_frame, method_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Record frame if recording
        if self.recording and self.out is not None:
            self.out.write(display_frame)
        
        # Convert to RGB format for display in customtkinter
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        
        return rgb_frame, debug_frame
    
    def set_detection_method(self, method):
        """
        Set the detection method
        
        Args:
            method: Detection method ('multi', 'color', 'texture', 'contour', 'manual')
        """
        valid_methods = ['multi', 'color', 'texture', 'contour', 'manual']
        
        # Map legacy methods to new methods
        method_mapping = {
            'grabcut': 'multi',  # Map grabcut to multi-detection
            'motion': 'multi',   # Map motion to multi-detection
            'yolo': 'contour'    # Replace YOLO with contour analysis
        }
        
        # Check if it's a legacy method and map it
        if method in method_mapping:
            mapped_method = method_mapping[method]
            print(f"Method '{method}' mapped to '{mapped_method}'")
            self.detection_method = mapped_method
            return
            
        # Check if it's a valid method
        if method in valid_methods:
            self.detection_method = method
            print(f"Detection method set to: {method}")
        else:
            print(f"Invalid detection method: {method}. Valid options are: {valid_methods}")
            # Default to 'multi' as fallback
            self.detection_method = 'multi'
            print(f"Defaulting to 'multi' detection method")
    
    def manually_select_roi(self, callback=None):
        """
        Prepare to manually select ROI
        
        Args:
            callback: Function to call with the selected ROI
            
        Returns:
            Selected ROI or None
        """
        self.detection_method = 'manual'
        
        # Stop tracking if active
        if self.tracking:
            self.stop_tracking()
        
        # Get a frame for selection
        ret, frame = self.capture.read()
        if ret:
            # OpenCV's selectROI function
            bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select Object")
            
            if bbox[2] > 0 and bbox[3] > 0:  # If width and height are valid
                self.detection_bbox = bbox
                if callback:
                    callback(bbox)
                return bbox
        
        return None
    
    def cleanup(self):
        """Release resources"""
        self.stop_recording()
        self.capture.release()