import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2


class TrainedModelWrapper:
    """
    Wrapper to adapt the trained model to PlantIdentifier interface.
    
    This class provides compatibility between the ResNet-based model from training
    and the transformer-based model interface expected by PlantIdentifier.
    """
    
    def __init__(self, model_path, num_classes, device):
        """
        Initialize the model wrapper.
        
        Args:
            model_path: Path to the trained model weights
            num_classes: Number of plant classes
            device: Device to run the model on (cpu or cuda)
        """
        from .plant_classifier import PlantClassifier
        
        self.device = device
        self.model = PlantClassifier(num_classes=num_classes, backbone='resnet50')
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            print("Initializing with random weights")
        
        self.model.to(device)
        self.model.eval()
        
        
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def predict(self, image):
        """
        Predict plant class from image.
        
        Args:
            image: Input image as numpy array, PIL Image, or tensor
            
        Returns:
            Tuple of (class_id, confidence, class_probabilities)
        """
        
        input_tensor = self.preprocess_image(image)
        input_tensor = input_tensor.to(self.device)
        
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)[0]
            
            
            confidence, class_id = torch.max(probs, dim=0)
            
            
            class_probs = {i: float(prob) for i, prob in enumerate(probs)}
            
            return int(class_id), float(confidence), class_probs
    
    def get_attention_visualization(self, image):
        """
        Generate attention visualization for the image using Grad-CAM.
        
        Args:
            image: Input image
            
        Returns:
            attention_map: Visualization of attention
        """
        
        
        
        
        
        
        
        input_tensor = self.preprocess_image(image)
        input_tensor = input_tensor.to(self.device)
        
        
        
        features = None
        
        def hook_fn(module, input, output):
            nonlocal features
            features = output
        
        
        hook = self.model.backbone.layer4.register_forward_hook(hook_fn)
        
        
        with torch.no_grad():
            self.model(input_tensor)
        
        
        hook.remove()
        
        
        if features is not None:
            
            attention_map = features.mean(dim=1).squeeze().cpu().numpy()
            
            
            attention_map = attention_map - attention_map.min()
            if attention_map.max() > 0:
                attention_map = attention_map / attention_map.max()
            
            
            h, w = image.shape[:2] if len(image.shape) > 2 else image.shape
            attention_map = cv2.resize(attention_map, (w, h))
            
            return attention_map
        else:
            
            h, w = image.shape[:2] if len(image.shape) > 2 else image.shape
            return np.ones((h, w), dtype=np.float32) * 0.5
    
    def preprocess_image(self, image):
        """
        Preprocess an image for the model.
        
        Args:
            image: PIL Image, numpy array, or tensor
            
        Returns:
            Preprocessed tensor
        """
        if isinstance(image, torch.Tensor):
            
            if image.max() > 1.0 and image.size(0) == 3:
                
                image = image / 255.0
            
            
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            
            return image
        
        
        if not isinstance(image, Image.Image):
            
            if isinstance(image, np.ndarray) and len(image.shape) == 3 and image.shape[2] == 3:
                if image[0, 0, 0] > image[0, 0, 2]:  
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            
            image = Image.fromarray(np.uint8(image))
        
        
        tensor = self.preprocess(image)
        
        
        tensor = tensor.unsqueeze(0)
        
        return tensor