import cv2
import numpy as np
import torch
from typing import Tuple, Optional, List, Dict

class PlantDetector:
    """
    Detector for identifying plants in images using multiple methods.
    Uses a combination of color filtering, texture analysis, and contour analysis
    to detect plants.
    """
    def __init__(self):
        
        self.yolo_available = False
        
        
        self.color_ranges = [
            
            (np.array([35, 40, 40]), np.array([85, 255, 255])),
            
            (np.array([0, 40, 40]), np.array([10, 255, 255])),
            (np.array([170, 40, 40]), np.array([180, 255, 255])),
            
            (np.array([15, 40, 40]), np.array([35, 255, 255])),
            
            (np.array([90, 40, 40]), np.array([140, 255, 255])),
            
            (np.array([10, 20, 40]), np.array([20, 100, 150]))
        ]
        
        
        self.texture_kernel_size = 5
        self.texture_canny_lower = 50
        self.texture_canny_upper = 150
        
        
        self.bbox_history = []
        self.max_history = 5  
        self.min_detection_area = 800  
        
        
        self.detection_cooldown = 0
        self.cooldown_frames = 3  
        
        
        self.weights = {
            'color': 0.5,
            'texture': 0.3,
            'contour': 0.2
        }
        
        
        self.detection_evidence = {
            'color': None,
            'texture': None,
            'contour': None,
            'combined': None
        }

    def color_filter(self, frame: np.ndarray) -> np.ndarray:
        """
        Filter frame to keep regions matching multiple plant color profiles
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Binary mask with color-matched regions
        """
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        
        combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        
        for lower, upper in self.color_ranges:
            
            color_mask = cv2.inRange(hsv, lower, upper)
            
            
            combined_mask = cv2.bitwise_or(combined_mask, color_mask)
        
        
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        return combined_mask

    def texture_filter(self, frame: np.ndarray) -> np.ndarray:
        """
        Filter frame based on texture features typical of plants
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Binary mask with plant-like textures
        """
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        
        blur = cv2.GaussianBlur(gray, (self.texture_kernel_size, self.texture_kernel_size), 0)
        
        
        laplacian = cv2.Laplacian(blur, cv2.CV_8U, ksize=3)
        
        
        _, texture_mask = cv2.threshold(laplacian, 10, 255, cv2.THRESH_BINARY)
        
        
        edges = cv2.Canny(blur, self.texture_canny_lower, self.texture_canny_upper)
        
        
        texture_mask = cv2.bitwise_or(texture_mask, edges)
        
        
        kernel = np.ones((3, 3), np.uint8)
        texture_mask = cv2.morphologyEx(texture_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        texture_mask = cv2.dilate(texture_mask, kernel, iterations=2)
        
        return texture_mask
        
    def contour_analysis(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect plants by analyzing contour properties
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Binary mask with plant-like contours
        """
        
        color_mask = self.color_filter(frame)
        
        
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        
        shape_mask = np.zeros_like(color_mask)
        
        for contour in contours:
            
            if cv2.contourArea(contour) < self.min_detection_area:
                continue
                
            
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            
            
            if perimeter == 0:
                continue
                
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            
            
            if (circularity < 0.8 and solidity > 0.4 and solidity < 0.95) or \
               (aspect_ratio > 0.3 and aspect_ratio < 3.0):
                cv2.drawContours(shape_mask, [contour], 0, 255, -1)
        
        
        kernel = np.ones((5, 5), np.uint8)
        shape_mask = cv2.morphologyEx(shape_mask, cv2.MORPH_CLOSE, kernel)
        
        return shape_mask

    def get_contour_bboxes(self, mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Get bounding boxes from contours in a binary mask
        
        Args:
            mask: Binary mask
            
        Returns:
            List of bounding boxes (x, y, w, h)
        """
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_detection_area]
        
        
        bboxes = []
        for cnt in valid_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            bboxes.append((x, y, w, h))
        
        return bboxes

    def combine_bboxes(self, bboxes: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        """
        Combine multiple bounding boxes into one
        
        Args:
            bboxes: List of bounding boxes
            
        Returns:
            Combined bounding box
        """
        if not bboxes:
            return None
        
        if len(bboxes) == 1:
            return bboxes[0]
        
        
        min_x = min(bbox[0] for bbox in bboxes)
        min_y = min(bbox[1] for bbox in bboxes)
        max_x = max(bbox[0] + bbox[2] for bbox in bboxes)
        max_y = max(bbox[1] + bbox[3] for bbox in bboxes)
        
        return (min_x, min_y, max_x - min_x, max_y - min_y)

    def refine_detection(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Refine detection using color and texture filtering
        
        Args:
            frame: Input BGR frame
            bbox: Initial bounding box (x, y, w, h)
            
        Returns:
            Refined bounding box
        """
        x, y, w, h = bbox
        
        
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return bbox
        
        
        color_mask = self.color_filter(roi)
        
        
        texture_mask = self.texture_filter(roi)
        
        
        combined_mask = cv2.bitwise_and(color_mask, texture_mask)
        
        
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return bbox
        
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        
        x_roi, y_roi, w_roi, h_roi = cv2.boundingRect(largest_contour)
        
        
        refined_bbox = (x + x_roi, y + y_roi, w_roi, h_roi)
        
        return refined_bbox
    
    def smooth_bbox(self, bbox: Optional[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        """
        Apply temporal smoothing to bounding box
        
        Args:
            bbox: Current detection bounding box
            
        Returns:
            Smoothed bounding box
        """
        if bbox is None:
            if self.bbox_history:
                
                return self.bbox_history[-1]
            return None
        
        
        self.bbox_history.append(bbox)
        
        
        if len(self.bbox_history) > self.max_history:
            self.bbox_history.pop(0)
        
        if len(self.bbox_history) < 2:
            return bbox
        
        
        x_avg = sum(b[0] for b in self.bbox_history) / len(self.bbox_history)
        y_avg = sum(b[1] for b in self.bbox_history) / len(self.bbox_history)
        w_avg = sum(b[2] for b in self.bbox_history) / len(self.bbox_history)
        h_avg = sum(b[3] for b in self.bbox_history) / len(self.bbox_history)
        
        return (int(x_avg), int(y_avg), int(w_avg), int(h_avg))

    def compute_confidence(self, 
                          color_area_ratio: float,
                          texture_area_ratio: float,
                          contour_area_ratio: float) -> float:
        """
        Compute overall confidence based on multiple detection methods
        
        Args:
            color_area_ratio: Ratio of color-matched area to frame area
            texture_area_ratio: Ratio of texture-matched area to frame area
            contour_area_ratio: Ratio of contour-matched area to frame area
            
        Returns:
            Combined confidence score
        """
        
        weighted_color = min(color_area_ratio * 5, 1.0) * self.weights['color']  
        weighted_texture = min(texture_area_ratio * 8, 1.0) * self.weights['texture']  
        weighted_contour = min(contour_area_ratio * 5, 1.0) * self.weights['contour']  
        
        
        combined_confidence = weighted_color + weighted_texture + weighted_contour
        
        return min(combined_confidence, 1.0)

    def detect(self, frame: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        """
        Detect plants in the frame using multiple methods
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Tuple of (bounding box, confidence)
        """
        
        if self.detection_cooldown > 0:
            self.detection_cooldown -= 1
            
            return self.smooth_bbox(None), 0.0
        
        
        self.detection_cooldown = self.cooldown_frames
        
        
        color_bboxes = []
        texture_bboxes = []
        contour_bboxes = []
        frame_area = frame.shape[0] * frame.shape[1]
        color_area_ratio = 0.0
        texture_area_ratio = 0.0
        contour_area_ratio = 0.0
        
        
        color_mask = self.color_filter(frame)
        self.detection_evidence['color'] = color_mask  
        
        
        color_contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color_contours = [cnt for cnt in color_contours if cv2.contourArea(cnt) > self.min_detection_area]
        
        if color_contours:
            
            color_area = sum(cv2.contourArea(cnt) for cnt in color_contours)
            color_area_ratio = color_area / frame_area
            
            
            color_bboxes = [cv2.boundingRect(cnt) for cnt in color_contours]
        
        
        texture_mask = self.texture_filter(frame)
        self.detection_evidence['texture'] = texture_mask  
        
        
        texture_contours, _ = cv2.findContours(texture_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        texture_contours = [cnt for cnt in texture_contours if cv2.contourArea(cnt) > self.min_detection_area]
        
        if texture_contours:
            
            texture_area = sum(cv2.contourArea(cnt) for cnt in texture_contours)
            texture_area_ratio = texture_area / frame_area
            
            
            texture_bboxes = [cv2.boundingRect(cnt) for cnt in texture_contours]
        
        
        contour_mask = self.contour_analysis(frame)
        self.detection_evidence['contour'] = contour_mask  
        
        
        contour_analysis_contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_analysis_contours = [cnt for cnt in contour_analysis_contours if cv2.contourArea(cnt) > self.min_detection_area]
        
        if contour_analysis_contours:
            
            contour_area = sum(cv2.contourArea(cnt) for cnt in contour_analysis_contours)
            contour_area_ratio = contour_area / frame_area
            
            
            contour_bboxes = [cv2.boundingRect(cnt) for cnt in contour_analysis_contours]
        
        
        combined_mask = np.zeros_like(color_mask)
        
        
        if color_mask.max() > 0:
            combined_mask = cv2.addWeighted(combined_mask, 1.0, color_mask, self.weights['color'], 0)
        
        
        if texture_mask.max() > 0:
            combined_mask = cv2.addWeighted(combined_mask, 1.0, texture_mask, self.weights['texture'], 0)
        
        
        if contour_mask.max() > 0:
            combined_mask = cv2.addWeighted(combined_mask, 1.0, contour_mask, self.weights['contour'], 0)
        
        
        _, combined_mask = cv2.threshold(combined_mask, 50, 255, cv2.THRESH_BINARY)
        self.detection_evidence['combined'] = combined_mask  
        
        
        combined_contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        combined_contours = [cnt for cnt in combined_contours if cv2.contourArea(cnt) > self.min_detection_area]
        
        
        final_bbox = None
        confidence = 0.0
        
        if combined_contours:
            
            largest_contour = max(combined_contours, key=cv2.contourArea)
            final_bbox = cv2.boundingRect(largest_contour)
            
            
            confidence = self.compute_confidence(
                color_area_ratio, 
                texture_area_ratio,
                contour_area_ratio
            )
        elif color_bboxes and contour_bboxes:
            
            for color_box in color_bboxes:
                for contour_box in contour_bboxes:
                    cx1, cy1, cw, ch = color_box
                    sx1, sy1, sw, sh = contour_box
                    
                    
                    if (cx1 < sx1 + sw and cx1 + cw > sx1 and
                        cy1 < sy1 + sh and cy1 + ch > sy1):
                        
                        final_bbox = color_box
                        confidence = (color_area_ratio * 2 + contour_area_ratio) / 3  
                        break
                if final_bbox is not None:
                    break
            
            
            if final_bbox is None and color_bboxes:
                largest_color_contour = max(color_contours, key=cv2.contourArea)
                final_bbox = cv2.boundingRect(largest_color_contour)
                confidence = color_area_ratio * 2  
        elif color_bboxes:
            
            largest_color_contour = max(color_contours, key=cv2.contourArea)
            final_bbox = cv2.boundingRect(largest_color_contour)
            confidence = color_area_ratio * 2  
        elif contour_bboxes:
            
            largest_contour_analysis_contour = max(contour_analysis_contours, key=cv2.contourArea)
            final_bbox = cv2.boundingRect(largest_contour_analysis_contour)
            confidence = contour_area_ratio * 2  
        
        
        smoothed_bbox = self.smooth_bbox(final_bbox)
        
        if smoothed_bbox is not None:
            final_bbox = smoothed_bbox
        
        return final_bbox, confidence
    
    
    def visualize(self, frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]], 
                  confidence: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Visualize detection results and create debug frame
        
        Args:
            frame: Input BGR frame
            bbox: Detected bounding box
            confidence: Detection confidence
            
        Returns:
            Tuple of (display frame, debug frame)
        """
        try:
            
            display_frame = frame.copy()
            
            
            frame_h, frame_w = frame.shape[:2]
            
            
            debug_frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
            
            
            panel_h, panel_w = frame_h // 2, frame_w // 2
            
            
            
            positions = {
                'TL': (0, 0, panel_h, panel_w),
                'TR': (0, panel_w, panel_h, panel_w),
                'BL': (panel_h, 0, panel_h, panel_w),
                'BR': (panel_h, panel_w, panel_h, panel_w)
            }
            
            
            if self.detection_evidence['color'] is not None:
                try:
                    color_mask = self.detection_evidence['color']
                    color_panel = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
                    color_panel = cv2.putText(color_panel, "Color", (10, 30), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    
                    resized_panel = cv2.resize(color_panel, (panel_w, panel_h))
                    
                    
                    y, x, h, w = positions['TL']
                    debug_frame[y:y+h, x:x+w] = resized_panel
                except Exception as e:
                    print(f"Error creating color panel: {e}")
            
            
            if self.detection_evidence['texture'] is not None:
                try:
                    texture_mask = self.detection_evidence['texture']
                    texture_panel = cv2.cvtColor(texture_mask, cv2.COLOR_GRAY2BGR)
                    texture_panel = cv2.putText(texture_panel, "Texture", (10, 30), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    
                    resized_panel = cv2.resize(texture_panel, (panel_w, panel_h))
                    
                    
                    y, x, h, w = positions['TR']
                    debug_frame[y:y+h, x:x+w] = resized_panel
                except Exception as e:
                    print(f"Error creating texture panel: {e}")
            
            
            if self.detection_evidence['contour'] is not None:
                try:
                    contour_mask = self.detection_evidence['contour']
                    contour_panel = cv2.cvtColor(contour_mask, cv2.COLOR_GRAY2BGR)
                    contour_panel = cv2.putText(contour_panel, "Contour", (10, 30), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    
                    resized_panel = cv2.resize(contour_panel, (panel_w, panel_h))
                    
                    
                    y, x, h, w = positions['BL']
                    debug_frame[y:y+h, x:x+w] = resized_panel
                except Exception as e:
                    print(f"Error creating contour panel: {e}")
            
            
            try:
                overlay_panel = frame.copy()
                if bbox is not None:
                    x, y, w, h = bbox
                    cv2.rectangle(overlay_panel, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                overlay_panel = cv2.putText(overlay_panel, "Detection", (10, 30), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                
                resized_panel = cv2.resize(overlay_panel, (panel_w, panel_h))
                
                
                y, x, h, w = positions['BR']
                debug_frame[y:y+h, x:x+w] = resized_panel
            except Exception as e:
                print(f"Error creating overlay panel: {e}")
                
        except Exception as e:
            print(f"Error in visualization: {e}")
            
            debug_frame = frame.copy()
            if bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        
        if bbox is not None:
            x, y, w, h = bbox
            
            
            if confidence > 0.7:
                color = (0, 255, 0)  
            elif confidence > 0.4:
                color = (0, 255, 255)  
            else:
                color = (0, 165, 255)  
                
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            
            
            text = f"Plant: {confidence:.2f}"
            cv2.putText(display_frame, text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return display_frame, debug_frame