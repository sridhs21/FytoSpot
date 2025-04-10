import json
import time
import numpy as np
import cv2
import torch
from typing import Dict, Tuple, List, Optional, Any
from pathlib import Path


class PlantIdentifier:
    """
    Plant identification module that integrates with the ObjectTracker.
    Identifies plants in images and provides visualization capabilities.
    
    Args:
        model_path: Path to the trained model
        class_mapping_path: Path to the class mapping file
        knowledge_base_path: Path to the plant knowledge base file
        device: Device to run the model on ('cuda' or 'cpu')
        confidence_threshold: Minimum confidence for a valid identification
        visualization_alpha: Transparency factor for attention visualization
    """
    def __init__(
        self,
        model_path: Optional[str],
        class_mapping_path: str,
        knowledge_base_path: Optional[str] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.7,
        visualization_alpha: float = 0.6,
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.visualization_alpha = visualization_alpha
        
        
        self.class_mapping = self._load_class_mapping(class_mapping_path)
        
        
        self.knowledge_base = None
        if knowledge_base_path is not None:
            self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
        
        
        
        self.model = None
        
        
        if model_path is not None:
            try:
                
                from .transformer import PlantIdentificationModel
                self.model = PlantIdentificationModel(
                    model_path=model_path,
                    num_classes=len(self.class_mapping),
                    device=self.device,
                    confidence_threshold=self.confidence_threshold,
                )
            except Exception as e:
                print(f"Error loading model: {e}")
                print("The model will need to be set manually later")
        
        
        self.current_predictions = None
        self.current_attention_map = None
        self.current_identification = None
        self.identification_history = []
        self.last_identification_time = 0
        self.identification_cooldown = 1.0  
    
    def _load_class_mapping(self, class_mapping_path: str) -> Dict[int, str]:
        """Load class mapping from file."""
        try:
            with open(class_mapping_path, 'r') as f:
                class_to_idx = json.load(f)
            
            
            idx_to_class = {int(v): k for k, v in class_to_idx.items()}
            return idx_to_class
        except Exception as e:
            print(f"Error loading class mapping: {e}")
            
            return {i: f"Class_{i}" for i in range(1000)}
    
    def _load_knowledge_base(self, knowledge_base_path: str) -> Dict[str, Dict]:
        """Load plant knowledge base from file."""
        try:
            with open(knowledge_base_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            return {}
    
    def identify_plant(self, image: np.ndarray) -> Dict:
        """
        Identify plant in the image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with identification results
        """
        
        current_time = time.time()
        if current_time - self.last_identification_time < self.identification_cooldown:
            
            if self.current_identification is not None:
                return self.current_identification
            
            
            return {
                'status': 'cooldown',
                'message': 'Identification on cooldown',
            }
        
        try:
            
            if self.model is None:
                print("Model not available for identification")
                return {
                    'status': 'error',
                    'message': 'Model not available for identification',
                }
            
            
            if len(image.shape) == 3 and image.shape[2] == 3:
                if image[0, 0, 0] > image[0, 0, 2]:  
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            
            class_id, confidence, class_probs = self.model.predict(image)
            print(f"Raw prediction: class_id={class_id}, confidence={confidence}")
            print(f"Class probabilities: {class_probs}")
            
            
            attention_map = self.model.get_attention_visualization(image)
            
            
            self.current_predictions = class_probs
            self.current_attention_map = attention_map
            
            
            if confidence < self.confidence_threshold:
                result = {
                    'status': 'low_confidence',
                    'message': f'Low confidence identification ({confidence:.2f})',
                    'class_id': class_id,
                    'confidence': float(confidence),
                    'class_name': self.class_mapping.get(class_id, f"Unknown_{class_id}"),
                    'top_predictions': self._format_predictions(class_probs),
                }
                print(f"Low confidence result: {result}")
                self.current_identification = result
                return result
            
            
            class_name = self.class_mapping.get(class_id, f"Unknown_{class_id}")
            
            
            plant_info = None
            if self.knowledge_base is not None and class_name in self.knowledge_base:
                plant_info = self.knowledge_base[class_name]
            
            
            result = {
                'status': 'success',
                'message': f'Plant identified with {confidence:.2f} confidence',
                'class_id': class_id,
                'confidence': float(confidence),
                'class_name': class_name,
                'top_predictions': self._format_predictions(class_probs),
                'plant_info': plant_info,
            }
            
            print(f"Success result: {result}")
            self.current_identification = result
            
            
            self.identification_history.append({
                'timestamp': current_time,
                'class_name': class_name,
                'confidence': float(confidence),
            })
            
            
            self.last_identification_time = current_time
            
            return result
        
        except Exception as e:
            print(f"Error identifying plant: {e}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'error',
                'message': f'Error identifying plant: {str(e)}',
            }
    
    def _format_predictions(self, class_probs: Dict[int, float]) -> List[Dict]:
        """Format class probabilities for display."""
        sorted_probs = sorted(
            [(k, v) for k, v in class_probs.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        top_probs = sorted_probs[:5]        
        result = []
        for class_id, prob in top_probs:
            result.append({
                'class_id': class_id,
                'class_name': self.class_mapping.get(class_id, f"Unknown_{class_id}"),
                'probability': float(prob),
            })
        
        print(f"Formatted predictions: {result}")  
        return result
    
    def get_visualization(self, image: np.ndarray) -> np.ndarray:
        """
        Get visualization of the plant identification with attention overlay.
        
        Args:
            image: Input image
            
        Returns:
            Visualization image with attention overlay
        """
        if self.current_attention_map is None:
            
            return image
        
        
        return self.visualize_attention(
            image,
            self.current_attention_map,
            alpha=self.visualization_alpha
        )
    
    def visualize_attention(self, image, attention_map, alpha=0.6):
        """
        Create a visualization of the attention map overlaid on the image.
        
        Args:
            image: Input image as numpy array
            attention_map: Attention map as numpy array
            alpha: Transparency factor
            
        Returns:
            Visualization image
        """
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:, :, :3]
        
        
        if attention_map is None:
            return image
        
        attention_map = attention_map - attention_map.min()
        if attention_map.max() > 0:
            attention_map = attention_map / attention_map.max()
        
        
        if attention_map.shape[:2] != image.shape[:2]:
            attention_map = cv2.resize(
                attention_map,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        
        
        heatmap = cv2.applyColorMap(
            (attention_map * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        
        
        if heatmap.shape[2] == 3 and heatmap[0, 0, 0] > heatmap[0, 0, 2]:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        
        visualization = (
            (1 - alpha) * image + alpha * heatmap
        ).astype(np.uint8)
        
        return visualization
    
    def get_recent_history(self, limit: int = 10) -> List[Dict]:
        """
        Get recent identification history.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of recent identifications
        """
        
        sorted_history = sorted(
            self.identification_history,
            key=lambda x: x['timestamp'],
            reverse=True
        )
        
        return sorted_history[:limit]
    
    def clear_history(self) -> None:
        self.identification_history = []