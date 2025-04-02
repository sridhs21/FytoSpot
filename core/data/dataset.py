import os
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, Callable


class PlantDataset(Dataset):
    """
    Dataset class for plant identification.
    
    Args:
        data_dir: Directory containing the dataset
        split: Train, val, or test split
        transform: Image transformations
        target_transform: Target transformations
        metadata_file: JSON file containing dataset metadata
    """
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        metadata_file: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        
        if metadata_file is not None and os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = None
        
        
        self.samples = []
        self.targets = []
        self.class_to_idx = {}
        self.classes = []
        
        
        split_dir = self.data_dir / split
        
        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} does not exist")
        
        
        class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        self.classes = [d.name for d in class_dirs]
        
        
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(self.classes))}
        
        
        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]
            
            
            img_extensions = ('.jpg', '.jpeg', '.png')
            image_files = [
                f for f in class_dir.iterdir() 
                if f.is_file() and f.suffix.lower() in img_extensions
            ]
            
            
            for img_file in image_files:
                self.samples.append((str(img_file), class_idx))
                self.targets.append(class_idx)
        
        print(f"Loaded {len(self.samples)} images for {split} split across {len(self.classes)} classes")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        img_path, target = self.samples[idx]   
        try:
            with open(img_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            
            img = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        
        if self.transform is not None:
            img = self.transform(img)
        
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target
    
    def get_class_name(self, idx: int) -> str:
        for cls_name, cls_idx in self.class_to_idx.items():
            if cls_idx == idx:
                return cls_name
        return "Unknown"