import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from typing import Dict, Tuple, Optional, Union

class TrainedModelWrapper:
    """
    Wrapper for the Vision Transformer model to maintain compatibility with 
    the original PlantIdentifier interface.
    
    This class adapts the Vision Transformer model to work with the existing codebase
    that expects a ResNet-based model.
    """
    
    def __init__(self, model_path, num_classes, device):
        """
        Initialize the model wrapper.
        
        Args:
            model_path: Path to the trained model weights
            num_classes: Number of plant classes
            device: Device to run the model on (cpu or cuda)
        """
        from .transformer import PlantIdentificationModel
        
        self.device = device
        
        # Initialize the Vision Transformer model
        self.model = PlantIdentificationModel(
            model_path=model_path,
            num_classes=num_classes,
            device=device,
            confidence_threshold=0.7
        )
        
        print(f"Transformer model loaded successfully from {model_path}")
    
    def predict(self, image):
        """
        Predict plant class from image.
        
        Args:
            image: Input image as numpy array, PIL Image, or tensor
            
        Returns:
            Tuple of (class_id, confidence, class_probabilities)
        """
        return self.model.predict(image)
    
    def get_attention_visualization(self, image):
        """
        Generate attention visualization for the image.
        
        Args:
            image: Input image
            
        Returns:
            attention_map: Visualization of attention
        """
        return self.model.get_attention_visualization(image)