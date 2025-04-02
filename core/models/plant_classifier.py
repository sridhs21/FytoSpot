import torch
import torch.nn as nn
from torchvision import models


class PlantClassifier(nn.Module):
    """
    Plant classification model using a ResNet backbone.
    
    Args:
        num_classes: Number of plant classes
        backbone: Backbone model type ('resnet18', 'resnet34', 'resnet50', 'resnet101')
        pretrained: Whether to use pretrained weights (only used when online)
    """
    def __init__(self, num_classes=1000, backbone='resnet50', pretrained=False):
        super().__init__()
        
        
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            feature_dim = 2048
        else:
            
            print(f"Unrecognized backbone '{backbone}', defaulting to ResNet50")
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        
        
        self.backbone.fc = nn.Identity()
        
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape [batch_size, 3, height, width]
            
        Returns:
            logits: Output tensor of shape [batch_size, num_classes]
        """
        features = self.backbone(x)
        return self.classifier(features)
    
    def extract_features(self, x):
        """
        Extract features from the model.
        
        Args:
            x: Input tensor of shape [batch_size, 3, height, width]
            
        Returns:
            features: Feature tensor
        """
        return self.backbone(x)
        
    def get_attention_maps(self, x):
        """
        Get attention maps for visualization.
        This is a placeholder for compatibility with the transformer model.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (None, logits)
        """
        logits = self.forward(x)
        return None, logits