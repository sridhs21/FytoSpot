#import models for easier access
from .plant_classifier import PlantClassifier
from .plant_identifier import PlantIdentifier
from .wrapper import TrainedModelWrapper

#import transformer models
try:
    from .transformer import (
        PlantIdentificationModel,
        VisionTransformer,
        MultiHeadAttention,
        TransformerBlock,
        PatchEmbedding
    )
    transformer_available = True
except ImportError as e:
    print(f"Warning: Transformer modules not available: {e}")
    transformer_available = False

#define available models
__all__ = [
    'PlantClassifier',
    'PlantIdentifier',
    'TrainedModelWrapper'
]

#add transformer models if available
if transformer_available:
    __all__.extend([
        'PlantIdentificationModel',
        'VisionTransformer',
        'MultiHeadAttention',
        'TransformerBlock',
        'PatchEmbedding'
    ])