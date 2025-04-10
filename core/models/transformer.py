import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.hub import load_state_dict_from_url
import timm
from PIL import Image
import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List, Union


class PlantIdentificationModel:
    """
    Plant identification model using a Vision Transformer architecture.
    
    Args:
        model_path: Path to the trained model weights
        num_classes: Number of plant classes
        device: Device to run the model on ('cuda' or 'cpu')
        confidence_threshold: Minimum confidence for a valid identification
        model_name: Vision transformer model name from timm library
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        num_classes: int = 1000,
        device: Optional[torch.device] = None,
        confidence_threshold: float = 0.7,
        model_name: str = "vit_base_patch16_224",
    ):
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.confidence_threshold = confidence_threshold

        # Initialize the transformer model
        try:
            self.model = timm.create_model(
                model_name, 
                pretrained=False, 
                num_classes=num_classes
            )
            print(f"Created {model_name} with {num_classes} output classes")
        except Exception as e:
            print(f"Error creating timm model: {e}")
            print("Falling back to ViT implementation from scratch...")
            # Fallback to our own Vision Transformer implementation
            self.model = VisionTransformer(
                img_size=224,
                patch_size=16,
                in_channels=3,
                num_classes=num_classes,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4.0,
                dropout_rate=0.1,
                attn_dropout_rate=0.0
            )
        
        # Load model weights if provided
        if model_path is not None:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Model weights loaded from {model_path}")
            except Exception as e:
                print(f"Error loading model weights: {e}")
                print("Initializing with random weights")
        
        # Move model to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()
        
        # Set up preprocessing transformations
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        # Store last attention weights for visualization
        self.last_attn = None
        
        # Register hook to capture attention weights
        self._register_attention_hooks()
    
    def _register_attention_hooks(self):
        """Register hooks to capture attention weights from transformer blocks."""
        def get_attention_hook(name):
            def hook(module, input, output):
                # Store attention weights for visualization
                self.last_attn = output[1] if isinstance(output, tuple) and len(output) > 1 else None
            return hook
        
        # Try to register hooks based on model type
        try:
            if hasattr(self.model, 'blocks'):
                # ViT models from timm typically have blocks
                for i, block in enumerate(self.model.blocks):
                    if hasattr(block, 'attn'):
                        block.attn.register_forward_hook(get_attention_hook(f"block{i}"))
            
            # If using our custom model, register to the last transformer block
            elif hasattr(self.model, 'transformer_encoder'):
                last_block = self.model.transformer_encoder.layers[-1]
                if hasattr(last_block, 'self_attn'):
                    last_block.self_attn.register_forward_hook(get_attention_hook("last_layer"))
        except Exception as e:
            print(f"Warning: Failed to register attention hooks: {e}")
    
    def predict(self, image: Union[np.ndarray, Image.Image]) -> Tuple[int, float, Dict[int, float]]:
        """
        Predict plant class from image.
        
        Args:
            image: Input image as numpy array or PIL Image
            
        Returns:
            Tuple of (class_id, confidence, class_probabilities)
        """
        # Preprocess image
        input_tensor = self._preprocess_image(image)
        input_tensor = input_tensor.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)[0]
            
            # Get class with highest probability
            confidence, class_id = torch.max(probs, dim=0)
            
            # Create dictionary of class probabilities
            class_probs = {i: float(prob) for i, prob in enumerate(probs)}
            
            return int(class_id), float(confidence), class_probs
    
    def get_attention_visualization(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Generate attention visualization for the image.
        
        Args:
            image: Input image
            
        Returns:
            attention_map: Visualization of attention
        """
        # Ensure we have a fresh forward pass to get attention
        input_tensor = self._preprocess_image(image)
        input_tensor = input_tensor.to(self.device)
        
        # Forward pass to capture attention through hooks
        with torch.no_grad():
            self.model(input_tensor)
        
        # Process the attention map
        attention_map = self._process_attention_map(image)
        
        return attention_map
    
    def _process_attention_map(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Process the raw attention weights into a visualization."""
        # Get image dimensions
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:  # PIL Image
            w, h = image.size
        
        # If we don't have attention weights, return a default heatmap
        if self.last_attn is None:
            print("Warning: No attention weights captured")
            return np.ones((h, w), dtype=np.float32) * 0.5
        
        try:
            # Process attention weights
            # The attention shape is typically [batch_size, num_heads, seq_len, seq_len]
            # We want to extract the attention from the CLS token to all patch tokens
            attn = self.last_attn.detach().cpu()
            
            # Average across heads
            attn = attn.mean(dim=1)
            
            # Extract attention from CLS token (first token) to patch tokens
            # Skip the CLS token itself (starting from index 1)
            patch_attn = attn[0, 0, 1:]
            
            # Reshape to spatial dimensions
            patch_size = 16  # Assuming patch size of 16
            num_patches_per_side = int((224 // patch_size))
            
            attention_map = patch_attn.reshape(num_patches_per_side, num_patches_per_side).numpy()
            
            # Resize to original image dimensions
            attention_map = cv2.resize(attention_map, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Normalize
            attention_map = attention_map - attention_map.min()
            if attention_map.max() > 0:
                attention_map = attention_map / attention_map.max()
            
            return attention_map
            
        except Exception as e:
            print(f"Error processing attention map: {e}")
            import traceback
            traceback.print_exc()
            
            # Return a default heatmap if processing fails
            return np.ones((h, w), dtype=np.float32) * 0.5
    
    def _preprocess_image(self, image: Union[np.ndarray, Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess an image for the model.
        
        Args:
            image: PIL Image, numpy array, or tensor
            
        Returns:
            Preprocessed tensor
        """
        if isinstance(image, torch.Tensor):
            # If already a tensor, ensure it's in the right format
            if image.max() > 1.0 and image.dim() > 2:
                # Assume values are in [0, 255], normalize to [0, 1]
                image = image / 255.0
            
            # Add batch dimension if needed
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            
            return image
        
        # Convert numpy array to PIL Image
        if not isinstance(image, Image.Image):
            # Convert BGR to RGB if from OpenCV
            if isinstance(image, np.ndarray) and len(image.shape) == 3 and image.shape[2] == 3:
                if image[0, 0, 0] > image[0, 0, 2]:  # Likely BGR
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            image = Image.fromarray(np.uint8(image))
        
        # Apply preprocessing
        tensor = self.preprocess(image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor


class PatchEmbedding(nn.Module):
    """
    Converts image into patches and embeds them.
    
    Args:
        img_size: Size of input image
        patch_size: Size of each patch
        in_channels: Number of input channels
        embed_dim: Dimension of linear projection
    """
    def __init__(
        self, 
        img_size: int = 224, 
        patch_size: int = 16, 
        in_channels: int = 3, 
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        
        # Number of patches along each dimension
        self.n_patches = (img_size // patch_size) ** 2
        
        # Linear projection of flattened patches
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, in_channels, height, width]
            
        Returns:
            patch_embeddings: Tensor of shape [batch_size, n_patches, embed_dim]
        """
        # Project and flatten
        x = self.proj(x)  # [B, embed_dim, h', w']
        x = x.flatten(2)  # [B, embed_dim, n_patches]
        x = x.transpose(1, 2)  # [B, n_patches, embed_dim]
        
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    
    Args:
        embed_dim: Dimension of embedding
        num_heads: Number of attention heads
        dropout_rate: Dropout rate
    """
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int = 8, 
        dropout_rate: float = 0.0
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Stored attention weights for visualization
        self.attention_weights = None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            
        Returns:
            output: Tensor of shape [batch_size, seq_len, embed_dim]
            attention_weights: Attention weights for visualization
        """
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, values
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        self.attention_weights = attention_weights  # Store for visualization
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights)
        
        # Compute weighted sum
        output = torch.matmul(attention_weights, v)
        
        # Reshape and project
        output = output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)
        
        return output, attention_weights


class TransformerBlock(nn.Module):
    """
    Transformer block with multi-head attention and feed-forward network.
    
    Args:
        embed_dim: Dimension of embedding
        num_heads: Number of attention heads
        mlp_ratio: Ratio of mlp hidden dim to embedding dim
        dropout_rate: Dropout rate
        attn_dropout_rate: Attention dropout rate
    """
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int = 8, 
        mlp_ratio: float = 4.0, 
        dropout_rate: float = 0.0, 
        attn_dropout_rate: float = 0.0
    ):
        super().__init__()
        
        # Layer Normalization 1
        self.ln1 = nn.LayerNorm(embed_dim)
        
        # Multi-head Attention
        self.self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout_rate=attn_dropout_rate
        )
        
        # Layer Normalization 2
        self.ln2 = nn.LayerNorm(embed_dim)
        
        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            
        Returns:
            output: Tensor of shape [batch_size, seq_len, embed_dim]
            attention_weights: Attention weights for visualization
        """
        # Multi-head Attention with residual connection
        attn_output, attention_weights = self.self_attn(self.ln1(x))
        x = x + attn_output
        
        # MLP with residual connection
        x = x + self.mlp(self.ln2(x))
        
        return x, attention_weights


class VisionTransformer(nn.Module):
    """
    Vision Transformer model for image classification.
    
    Args:
        img_size: Input image size
        patch_size: Patch size
        in_channels: Number of input channels
        num_classes: Number of classes for classification
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: Ratio of mlp hidden dim to embedding dim
        dropout_rate: Dropout rate
        attn_dropout_rate: Attention dropout rate
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.0,
        attn_dropout_rate: float = 0.0,
    ):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Number of patches
        num_patches = self.patch_embed.n_patches
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Transformer encoder
        self.transformer_encoder = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout_rate=dropout_rate,
                attn_dropout_rate=attn_dropout_rate
            )
            for _ in range(depth)
        ])
        
        # Layer Normalization
        self.ln = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        # Initialize patch embedding like a linear layer
        nn.init.normal_(self.patch_embed.proj.weight, std=0.02)
        
        # Initialize position embeddings
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Initialize class token
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Initialize transformer blocks
        self.apply(self._init_transformer_weights)
    
    def _init_transformer_weights(self, m):
        """Initialize transformer weights."""
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            logits: Output tensor of shape [batch_size, num_classes]
        """
        batch_size = x.shape[0]
        
        # Patch embedding [B, n_patches, embed_dim]
        x = self.patch_embed(x)
        
        # Prepend class token
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Apply dropout
        x = self.dropout(x)
        
        # Transformer encoder
        attention_weights = None
        for block in self.transformer_encoder:
            x, attn = block(x)
            attention_weights = attn  # Keep the last one for visualization
        
        # Take class token output
        x = self.ln(x[:, 0])
        
        # Classification head
        logits = self.head(x)
        
        return logits