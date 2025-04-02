import os
import json
from pathlib import Path
from typing import Dict, List, Optional


class KnowledgeBase:
    """
    Plant knowledge base containing detailed information about plants.
    
    Args:
        base_path: Path to the knowledge base file
        usda_data_path: Path to USDA plants data (optional)
        plantnet_data_path: Path to PlantNet data (optional)
    """
    def __init__(
        self,
        base_path: str,
        usda_data_path: Optional[str] = None,
        plantnet_data_path: Optional[str] = None,
    ):
        self.base_path = Path(base_path)
        self.knowledge_base = {}
        
        
        self._load_base_knowledge()
        
        
        if usda_data_path is not None:
            self._enrich_with_usda(usda_data_path)
        
        
        if plantnet_data_path is not None:
            self._enrich_with_plantnet(plantnet_data_path)
        
        
        self._save_knowledge_base()
    
    def _load_base_knowledge(self) -> None:
        
        if not self.base_path.exists():
            self.knowledge_base = {}
            return
        
        
        try:
            with open(self.base_path, 'r') as f:
                self.knowledge_base = json.load(f)
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            self.knowledge_base = {}
    
    def _save_knowledge_base(self) -> None:
        try:
            
            self.base_path.parent.mkdir(parents=True, exist_ok=True)
            
            
            with open(self.base_path, 'w') as f:
                json.dump(self.knowledge_base, f, indent=2)
        except Exception as e:
            print(f"Error saving knowledge base: {e}")
    
    def _enrich_with_usda(self, usda_data_path: str) -> None:
        """
        This enriches knowledge base with USDA plants data.
        
        Args:
            usda_data_path: Path to USDA plants data
        """
        try:
            
            usda_data = {}
            with open(usda_data_path, 'r') as f:
                import csv
                reader = csv.DictReader(f)
                for row in reader:
                    
                    if 'Scientific_Name' in row:
                        usda_data[row['Scientific_Name']] = row
            
            
            for plant_name, plant_info in self.knowledge_base.items():
                
                scientific_name = plant_info.get('scientific_name', '')
                if scientific_name in usda_data:
                    usda_info = usda_data[scientific_name]
                    
                    
                    plant_info['usda'] = {
                        'symbol': usda_info.get('Symbol', ''),
                        'family': usda_info.get('Family', ''),
                        'duration': usda_info.get('Duration', ''),
                        'growth_habit': usda_info.get('Growth_Habit', ''),
                        'native_status': usda_info.get('Native_Status', ''),
                    }
                    
                    
                    self.knowledge_base[plant_name] = plant_info
        except Exception as e:
            print(f"Error enriching with USDA data: {e}")
    
    def _enrich_with_plantnet(self, plantnet_data_path: str) -> None:
        """
        Enrich knowledge base with PlantNet data.
        
        Args:
            plantnet_data_path: Path to PlantNet data
        """
        try:
            
            with open(plantnet_data_path, 'r') as f:
                plantnet_data = json.load(f)
            
            
            for plant_name, plant_info in self.knowledge_base.items():
                
                for plantnet_plant in plantnet_data:
                    if (
                        plant_name == plantnet_plant.get('name', '') or
                        plant_info.get('scientific_name', '') == plantnet_plant.get('scientific_name', '')
                    ):
                        
                        plant_info['plantnet'] = {
                            'family': plantnet_plant.get('family', ''),
                            'genus': plantnet_plant.get('genus', ''),
                            'common_names': plantnet_plant.get('common_names', []),
                            'description': plantnet_plant.get('description', ''),
                            'habitat': plantnet_plant.get('habitat', ''),
                            'uses': plantnet_plant.get('uses', ''),
                        }
                        
                        
                        self.knowledge_base[plant_name] = plant_info
                        break
        except Exception as e:
            print(f"Error enriching with PlantNet data: {e}")
    
    def get_plant_info(self, plant_name: str) -> Dict:
        """
        Get information about a plant.
        
        Args:
            plant_name: Name of the plant
            
        Returns:
            Plant information
        """
        return self.knowledge_base.get(plant_name, {})
    
    def add_plant_info(self, plant_name: str, plant_info: Dict) -> None:
        """
        Add information about a plant.
        
        Args:
            plant_name: Name of the plant
            plant_info: Plant information
        """
        self.knowledge_base[plant_name] = plant_info
        self._save_knowledge_base()
    
    def get_all_plants(self) -> List[str]:
        """
        Get all plant names in the knowledge base.
        
        Returns:
            List of plant names
        """
        return sorted(list(self.knowledge_base.keys()))


def prepare_knowledge_base(output_path: str = 'models/knowledge_base.json') -> str:
    """
    Prepare plant knowledge base with sample data.
    
    Args:
        output_path: Path to save the knowledge base
        
    Returns:
        Path to the created knowledge base
    """
    
    base_path = Path(output_path)
    
    
    base_path.parent.mkdir(parents=True, exist_ok=True)
    
    
    if base_path.exists():
        return str(base_path)
    
    
    knowledge_base = {}
    
    
    sample_plants = [
        {
            "name": "rose",
            "scientific_name": "Rosa",
            "family": "Rosaceae",
            "description": "Roses are perennial flowering plants known for their beauty and fragrance.",
            "care": {
                "watering": "Regular watering, especially during dry periods",
                "sunlight": "At least 6 hours of direct sunlight daily",
                "soil": "Well-draining, rich soil with pH 6.0-6.5",
                "pruning": "Prune in early spring before new growth starts"
            },
            "common_varieties": ["Hybrid Tea", "Floribunda", "Grandiflora", "Climbing"]
        },
        {
            "name": "tomato",
            "scientific_name": "Solanum lycopersicum",
            "family": "Solanaceae",
            "description": "Tomatoes are annual plants in the nightshade family, grown for their edible fruits.",
            "care": {
                "watering": "Regular, consistent watering",
                "sunlight": "Full sun, at least 8 hours daily",
                "soil": "Well-draining, rich soil with pH 6.0-6.8",
                "support": "Most varieties benefit from staking or caging"
            },
            "common_varieties": ["Cherry", "Beefsteak", "Roma", "Grape", "Heirloom"]
        },
        {
            "name": "lavender",
            "scientific_name": "Lavandula",
            "family": "Lamiaceae",
            "description": "Lavender is a perennial flowering plant known for its fragrant flowers and aromatic leaves.",
            "care": {
                "watering": "Drought-tolerant once established",
                "sunlight": "Full sun, at least 6 hours daily",
                "soil": "Well-draining, alkaline soil with pH 6.5-8.0",
                "pruning": "Prune after flowering or in early spring"
            },
            "common_varieties": ["English", "French", "Spanish", "Portuguese"]
        }
    ]
    
    
    for plant in sample_plants:
        knowledge_base[plant["name"]] = plant
    
    
    with open(base_path, 'w') as f:
        json.dump(knowledge_base, f, indent=2)
    
    print(f"Created knowledge base at {base_path}")
    return str(base_path)