import os
import sys
import json
import torch
from pathlib import Path


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)


from core.detection.plant_detector import PlantDetector
from core.models.plant_identifier import PlantIdentifier
from core.data.knowledge_base import prepare_knowledge_base
from core.models.wrapper import TrainedModelWrapper


from platforms.desktop.utils.desktop_camera import DesktopCamera
from platforms.desktop.gui.splash import show_splash
from platforms.desktop.gui.plant_app import PlantTrackerApp
from core.detection.object_tracker import ObjectTracker


def create_directories():
    """Create necessary directories for the application."""
    os.makedirs('recordings', exist_ok=True)
    os.makedirs('assets/images', exist_ok=True)
    os.makedirs('models/trained/common', exist_ok=True)
    os.makedirs('models/configs', exist_ok=True)
    os.makedirs('data', exist_ok=True)


def check_and_copy_model_files():
    """
    Check for model files in the old location and copy them to the new location
    if necessary.
    
    Returns:
        Tuple of (model_path, class_mapping_path)
    """
    
    new_model_path = os.path.join(ROOT_DIR, "models/trained/final_model.pt")
    new_mapping_path = os.path.join(ROOT_DIR, "models/configs/class_mapping.json")
    
    
    if os.path.exists(new_model_path) and os.path.exists(new_mapping_path):
        print(f"Using existing model files at {new_model_path}")
        return new_model_path, new_mapping_path
    
    
    old_model_paths = [
        "output/offline/final_model.pt",
        "../output/offline/final_model.pt",
        "models/final_model.pt"
    ]
    
    old_mapping_paths = [
        "output/offline/class_mapping.json",
        "../output/offline/class_mapping.json",
        "models/class_mapping.json"
    ]
    
    
    model_path = None
    for path in old_model_paths:
        if os.path.exists(path):
            print(f"Found model at old location: {path}")
            
            os.makedirs(os.path.dirname(new_model_path), exist_ok=True)
            
            
            import shutil
            shutil.copy2(path, new_model_path)
            print(f"Copied model to new location: {new_model_path}")
            model_path = new_model_path
            break
    
    
    mapping_path = None
    for path in old_mapping_paths:
        if os.path.exists(path):
            print(f"Found class mapping at old location: {path}")
            
            os.makedirs(os.path.dirname(new_mapping_path), exist_ok=True)
            
            
            import shutil
            shutil.copy2(path, new_mapping_path)
            print(f"Copied class mapping to new location: {new_mapping_path}")
            mapping_path = new_mapping_path
            break
    
    
    if model_path is None:
        model_path = new_model_path
    if mapping_path is None:
        mapping_path = new_mapping_path
    
    return model_path, mapping_path


def load_class_mapping(mapping_path):
    """Load class mapping from JSON file."""
    try:
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        return mapping
    except Exception as e:
        print(f"Error loading class mapping: {e}")
        
        default_mapping = {"class_0": 0, "class_1": 1, "rose": 2, "tomato": 3, "lavender": 4}
        
        
        os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
        
        
        with open(mapping_path, 'w') as f:
            json.dump(default_mapping, f, indent=2)
            
        return default_mapping


def load_species_names(species_names_path):
    """Load species names from JSON file."""
    try:
        with open(species_names_path, 'r') as f:
            species_names = json.load(f)
        return species_names
    except Exception as e:
        print(f"Error loading species names: {e}")
        return {}


def main():
    """Main entry point for the desktop application."""
    
    create_directories()
    
    
    model_path, class_mapping_path = check_and_copy_model_files()
    
    
    species_names_path = os.path.join(ROOT_DIR, "data/plantnet_300K/plantnet300K_species_names.json")
    
    
    show_splash(duration=2.0)
    
    
    model_exists = os.path.exists(model_path)
    if not model_exists:
        print(f"Warning: Trained model not found at {model_path}")
        print("Plant identification will not be available")
    
    
    class_mapping = load_class_mapping(class_mapping_path)
    num_classes = len(class_mapping)
    
    
    species_names = load_species_names(species_names_path)
    print(f"Loaded {len(species_names)} species names")
    
    
    knowledge_base_path = prepare_knowledge_base(os.path.join(ROOT_DIR, "models/knowledge_base.json"))
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    
    camera = DesktopCamera()
    
    
    plant_detector = PlantDetector()
    
    
    tracker = ObjectTracker(camera, plant_detector)
    
    
    plant_identifier = None
    if model_exists:
        try:
            
            model_wrapper = TrainedModelWrapper(model_path, num_classes, device)
            
            
            plant_identifier = PlantIdentifier(
                model_path=None,  
                class_mapping_path=class_mapping_path,
                knowledge_base_path=knowledge_base_path,
                confidence_threshold=0.5
            )
            
            
            plant_identifier.model = model_wrapper
            
            print("Plant identifier initialized successfully with trained model")
        except Exception as e:
            print(f"Error initializing plant identifier: {e}")
            plant_identifier = None
    
    
    app = PlantTrackerApp(tracker, plant_identifier, species_names)
    
    
    app.detection_method.set("multi")
    app.btn_multi.select()  
    app.on_multi()          
    
    
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    
    app.mainloop()


if __name__ == "__main__":
    main()