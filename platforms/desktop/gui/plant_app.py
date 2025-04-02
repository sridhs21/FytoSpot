import tkinter as tk
from tkinter import filedialog
import os
import cv2
import threading
import numpy as np
from PIL import Image
import customtkinter as ctk

from platforms.desktop.gui.app import ObjectTrackerApp
from core.models.plant_identifier import PlantIdentifier
from assets.styles.colors import COLORS


class PlantTrackerApp(ObjectTrackerApp):
    """
    Enhanced version of ObjectTrackerApp with plant identification capabilities.
    """
    def __init__(self, tracker, plant_identifier=None, species_names=None):
        """
        Initialize the plant tracker application.
        
        Args:
            tracker: ObjectTracker instance
            plant_identifier: PlantIdentifier instance for plant recognition
            species_names: Dictionary mapping species IDs to names
        """
        
        super().__init__(tracker)
        
        
        self.plant_identifier = plant_identifier
        
        
        self.species_names = species_names or {}
        
        
        if self.plant_identifier is not None:
            self._add_plant_identification_panel()
        
        
        self.title("FytóSpot - Plant Identification and Tracking System")
    
    def _add_plant_identification_panel(self):
        """Add modern plant identification panel to the UI."""
        
        self.plant_panel = ctk.CTkFrame(self.main_frame, corner_radius=10)
        self.plant_panel.grid(row=0, column=1, padx=8, pady=8, sticky="nsew")
        
        
        self.plant_title_frame = ctk.CTkFrame(
            self.plant_panel,
            fg_color=COLORS["medium_grey"], 
            corner_radius=8,
            height=36
        )
        self.plant_title_frame.pack(fill="x", padx=4, pady=4)
        self.plant_title_frame.pack_propagate(False)  
        
        
        self.plant_icon = ctk.CTkLabel(
            self.plant_title_frame,
            text="🌿",  
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=COLORS["medium_green"]
        )
        self.plant_icon.pack(side="left", padx=(10, 5))
        
        
        self.plant_title = ctk.CTkLabel(
            self.plant_title_frame,
            text="Plant Identification",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLORS["off_white"]
        )
        self.plant_title.pack(side="left")
        
        
        if self.plant_identifier is None:
            
            self.model_missing_frame = ctk.CTkFrame(
                self.plant_panel,
                fg_color=COLORS["light_grey"],
                corner_radius=8
            )
            self.model_missing_frame.pack(padx=10, pady=10, fill="x")
            
            
            self.warning_icon = ctk.CTkLabel(
                self.model_missing_frame,
                text="⚠️",  
                font=ctk.CTkFont(size=24),
                text_color=COLORS["warning"]
            )
            self.warning_icon.pack(padx=10, pady=(10, 5))
            
            
            self.model_missing_label = ctk.CTkLabel(
                self.model_missing_frame,
                text="Plant identification model not found",
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=COLORS["warning"]
            )
            self.model_missing_label.pack(padx=10, pady=2)
            
            
            self.model_missing_info = ctk.CTkLabel(
                self.model_missing_frame,
                text="Only tracking features are available",
                font=ctk.CTkFont(size=12),
                text_color=COLORS["off_white"]
            )
            self.model_missing_info.pack(padx=10, pady=(0, 5))
            
            
            self.check_model_button = ctk.CTkButton(
                self.plant_panel,
                text="Check Model Path",
                font=ctk.CTkFont(size=13, weight="bold"),
                fg_color=COLORS["teal"],
                hover_color=COLORS["dark_teal"],
                command=self.on_check_model,
                height=36,
                corner_radius=8,
                border_spacing=10,
                image=self.create_image_from_emoji("🔍")  
            )
            self.check_model_button.pack(padx=10, pady=(5, 10), fill="x")
            
            
            self.info_frame = ctk.CTkFrame(
                self.plant_panel,
                fg_color=COLORS["medium_grey"],
                corner_radius=8
            )
            self.info_frame.pack(padx=10, pady=5, fill="both", expand=True)
            
            
            self.info_header = ctk.CTkFrame(
                self.info_frame,
                fg_color=COLORS["dark_grey"],
                corner_radius=8,
                height=30
            )
            self.info_header.pack(fill="x", padx=5, pady=5)
            self.info_header.pack_propagate(False)
            
            
            self.info_icon = ctk.CTkLabel(
                self.info_header,
                text="ℹ️",  
                font=ctk.CTkFont(size=14),
                text_color=COLORS["info"]
            )
            self.info_icon.pack(side="left", padx=(10, 5))
            
            
            self.info_title = ctk.CTkLabel(
                self.info_header,
                text="Model Information",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=COLORS["off_white"]
            )
            self.info_title.pack(side="left")
            
            
            self.info_text = ctk.CTkTextbox(
                self.info_frame,
                font=ctk.CTkFont(size=12),
                text_color=COLORS["off_white"],
                fg_color=COLORS["light_grey"],
                corner_radius=5,
                wrap="word"
            )
            self.info_text.pack(padx=5, pady=5, fill="both", expand=True)
            self.info_text.insert("1.0", "Expected model path:\n- models/trained/final_model.pt\n\n"
                                "Expected class mapping path:\n- models/configs/class_mapping.json\n\n"
                                "Make sure these files exist.")
            self.info_text.configure(state="disabled")  
            return

        
        self.test_image_button = ctk.CTkButton(
            self.plant_panel,
            text="Test with Image",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=COLORS["teal"],
            hover_color=COLORS["dark_teal"],
            command=self.on_test_image,
            height=36,
            corner_radius=8,
            border_spacing=10,
            image=self.create_image_from_emoji("🖼️")  
        )
        self.test_image_button.pack(padx=10, pady=(5, 10), fill="x")
        
        
        self.prediction_frame = ctk.CTkFrame(
            self.plant_panel,
            fg_color=COLORS["light_grey"],
            corner_radius=8
        )
        self.prediction_frame.pack(padx=10, pady=5, fill="x")
        
        
        self.plant_result_icon = ctk.CTkLabel(
            self.prediction_frame,
            text="🌱",  
            font=ctk.CTkFont(size=36),
            text_color=COLORS["medium_green"]
        )
        self.plant_result_icon.pack(padx=10, pady=(10, 0))
        
        
        self.plant_name = ctk.CTkLabel(
            self.prediction_frame,
            text="No plant detected",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=COLORS["off_white"]
        )
        self.plant_name.pack(padx=10, pady=5)
        
        
        self.confidence_frame = ctk.CTkFrame(
            self.prediction_frame,
            fg_color="transparent",
        )
        self.confidence_frame.pack(padx=15, pady=5, fill="x")
        
        
        self.confidence_frame.grid_columnconfigure(0, weight=0)  
        self.confidence_frame.grid_columnconfigure(1, weight=1)  
        
        
        self.confidence_label = ctk.CTkLabel(
            self.confidence_frame,
            text="Confidence:",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["off_white"]
        )
        self.confidence_label.grid(row=0, column=0, padx=(0, 5), sticky="w")
        
        
        self.confidence_value = ctk.CTkLabel(
            self.confidence_frame,
            text="0%",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["off_white"],
            width=40  
        )
        self.confidence_value.grid(row=0, column=2, padx=(5, 0), sticky="e")
        
        
        self.confidence_bar = ctk.CTkProgressBar(
            self.confidence_frame,
            mode="determinate",
            height=8,
            corner_radius=4,
            fg_color=COLORS["dark_grey"],
            progress_color=COLORS["medium_green"]
        )
        self.confidence_bar.grid(row=0, column=1, padx=5, sticky="ew")
        self.confidence_bar.set(0)  
        
        
        self.top_predictions_frame = ctk.CTkFrame(
            self.plant_panel,
            fg_color=COLORS["medium_grey"],
            corner_radius=8
        )
        self.top_predictions_frame.pack(padx=10, pady=(10, 5), fill="x")
        
        
        self.predictions_header = ctk.CTkFrame(
            self.top_predictions_frame,
            fg_color=COLORS["dark_grey"],
            corner_radius=8,
            height=30
        )
        self.predictions_header.pack(fill="x", padx=5, pady=5)
        self.predictions_header.pack_propagate(False)
        
        
        self.alternative_icon = ctk.CTkLabel(
            self.predictions_header,
            text="🔄",  
            font=ctk.CTkFont(size=14),
            text_color=COLORS["medium_green"]
        )
        self.alternative_icon.pack(side="left", padx=(10, 5))
        
        
        self.top_predictions_title = ctk.CTkLabel(
            self.predictions_header,
            text="Alternative Matches",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["off_white"]
        )
        self.top_predictions_title.pack(side="left")
        
        
        self.alternatives_list = ctk.CTkFrame(
            self.top_predictions_frame,
            fg_color="transparent"
        )
        self.alternatives_list.pack(padx=5, pady=5, fill="x")
        
        
        self.top_predictions_labels = []
        
        
        for i in range(4):  
            alt_card = ctk.CTkFrame(
                self.alternatives_list,
                fg_color=COLORS["light_grey"],
                corner_radius=6,
                height=30
            )
            alt_card.pack(fill="x", padx=5, pady=2)
            alt_card.pack_propagate(False)
            
            
            alt_card.grid_columnconfigure(0, weight=0)  
            alt_card.grid_columnconfigure(1, weight=1)  
            alt_card.grid_columnconfigure(2, weight=0)  
            
            
            alt_num = ctk.CTkLabel(
                alt_card,
                text=f"{i+1}.",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=COLORS["medium_green"],
                width=20
            )
            alt_num.grid(row=0, column=0, padx=(8, 4), pady=0, sticky="w")
            
            
            alt_name = ctk.CTkLabel(
                alt_card,
                text="--",
                font=ctk.CTkFont(size=12),
                text_color=COLORS["off_white"]
            )
            alt_name.grid(row=0, column=1, padx=0, pady=0, sticky="w")
            
            
            alt_pct = ctk.CTkLabel(
                alt_card,
                text="--",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=COLORS["teal"],
                width=50
            )
            alt_pct.grid(row=0, column=2, padx=(0, 8), pady=0, sticky="e")
            
            
            self.top_predictions_labels.append((alt_name, alt_pct))
        
        
        self.info_frame = ctk.CTkFrame(
            self.plant_panel,
            fg_color=COLORS["medium_grey"],
            corner_radius=8
        )
        self.info_frame.pack(padx=10, pady=5, fill="both", expand=True)
        
        
        self.plant_info_header = ctk.CTkFrame(
            self.info_frame,
            fg_color=COLORS["dark_grey"],
            corner_radius=8,
            height=30
        )
        self.plant_info_header.pack(fill="x", padx=5, pady=5)
        self.plant_info_header.pack_propagate(False)
        
        
        self.plant_info_icon = ctk.CTkLabel(
            self.plant_info_header,
            text="📋",  
            font=ctk.CTkFont(size=14),
            text_color=COLORS["medium_green"]
        )
        self.plant_info_icon.pack(side="left", padx=(10, 5))
        
        
        self.info_title = ctk.CTkLabel(
            self.plant_info_header,
            text="Plant Information",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["off_white"]
        )
        self.info_title.pack(side="left")
        
        
        self.info_text = ctk.CTkTextbox(
            self.info_frame,
            font=ctk.CTkFont(size=12),
            text_color=COLORS["off_white"],
            fg_color=COLORS["light_grey"],
            corner_radius=5,
            wrap="word"
        )
        self.info_text.pack(padx=5, pady=5, fill="both", expand=True)
        self.info_text.insert("1.0", "Select a plant to view information.")
        self.info_text.configure(state="disabled")  
        
        
        self.identify_button = ctk.CTkButton(
            self.plant_panel,
            text="Identify Plant",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=COLORS["medium_green"],
            hover_color=COLORS["dark_green"],
            command=self.on_identify_plant,
            height=36,
            corner_radius=8,
            border_spacing=10,
            image=self.create_image_from_emoji("🔍")  
        )
        self.identify_button.pack(padx=10, pady=10, fill="x")
        
        
        self.plant_panel.lift()


    
    def _update_identification_ui(self, identification):
        """Update UI with plant identification results in a modern style."""
        
        if identification['status'] == 'success':
            self.plant_result_icon.configure(
                text="🌿",  
                text_color=COLORS["success"]
            )
        elif identification['status'] == 'low_confidence':
            self.plant_result_icon.configure(
                text="🌱",  
                text_color=COLORS["warning"]
            )
        else:
            self.plant_result_icon.configure(
                text="❓",  
                text_color=COLORS["info"]
            )
        
        
        if identification['status'] == 'success':
            
            class_id = identification.get('class_id')
            species_name = self.get_species_name(identification.get('class_name', class_id))
            
            self.plant_name.configure(
                text=species_name,
                text_color=COLORS["success"]  
            )
            
            
            confidence_percent = identification['confidence'] * 100
            self.confidence_value.configure(
                text=f"{confidence_percent:.1f}%"
            )
            
            
            self.confidence_bar.set(identification['confidence'])
            
            
            if confidence_percent >= 80:
                self.confidence_bar.configure(progress_color=COLORS["success"])
            elif confidence_percent >= 60:
                self.confidence_bar.configure(progress_color=COLORS["medium_green"])
            elif confidence_percent >= 40:
                self.confidence_bar.configure(progress_color=COLORS["teal"])
            else:
                self.confidence_bar.configure(progress_color=COLORS["warning"])
            
            
            top_predictions = identification.get('top_predictions', [])
            for i, (name_label, pct_label) in enumerate(self.top_predictions_labels):
                if i < len(top_predictions):
                    pred = top_predictions[i+1] if i+1 < len(top_predictions) else top_predictions[i]
                    prob_percent = pred['probability'] * 100
                    species_name = self.get_species_name(pred.get('class_name', pred.get('class_id', '')))
                    
                    name_label.configure(text=species_name)
                    pct_label.configure(text=f"{prob_percent:.1f}%")
                else:
                    name_label.configure(text="--")
                    pct_label.configure(text="--")
            
            
            plant_info = identification.get('plant_info', None)
            self.info_text.configure(state="normal")
            self.info_text.delete("1.0", "end")
            
            if plant_info is not None:
                
                info_text = f"Scientific Name: {plant_info.get('scientific_name', 'Unknown')}\n"
                info_text += f"Family: {plant_info.get('family', 'Unknown')}\n\n"
                
                if 'description' in plant_info:
                    info_text += "DESCRIPTION\n"
                    info_text += "=" * 40 + "\n"
                    info_text += f"{plant_info['description']}\n\n"
                
                if 'care' in plant_info:
                    info_text += "CARE INFORMATION\n"
                    info_text += "=" * 40 + "\n"
                    for care_type, care_info in plant_info['care'].items():
                        info_text += f"• {care_type.capitalize()}: {care_info}\n"
                    info_text += "\n"
                
                if 'common_varieties' in plant_info:
                    info_text += "COMMON VARIETIES\n"
                    info_text += "=" * 40 + "\n"
                    info_text += "• " + "\n• ".join(plant_info['common_varieties'])
                    info_text += "\n"
                
                self.info_text.insert("1.0", info_text)
            else:
                
                species_parts = species_name.split()
                if len(species_parts) >= 2:
                    genus = species_parts[0]
                    species = ' '.join(species_parts[1:])
                    
                    info_text = "BASIC INFORMATION\n"
                    info_text += "=" * 40 + "\n"
                    info_text += f"Scientific Name: {species_name}\n"
                    info_text += f"Genus: {genus}\n"
                    info_text += f"Species: {species}\n\n"
                    
                    info_text += "NOTE\n"
                    info_text += "=" * 40 + "\n"
                    info_text += "No detailed information available for this plant species.\n"
                    info_text += "Consider researching more about it online."
                else:
                    info_text = "No detailed information available for this plant."
                
                self.info_text.insert("1.0", info_text)
            
            self.info_text.configure(state="disabled")
            
        elif identification['status'] == 'low_confidence':
            
            class_id = identification.get('class_id')
            species_name = self.get_species_name(identification.get('class_name', class_id))
            
            self.plant_name.configure(
                text=f"{species_name}",
                text_color=COLORS["warning"]  
            )
            
            
            confidence_percent = identification['confidence'] * 100
            self.confidence_value.configure(
                text=f"{confidence_percent:.1f}%"
            )
            
            
            self.confidence_bar.set(identification['confidence'])
            self.confidence_bar.configure(progress_color=COLORS["warning"])
            
            
            top_predictions = identification.get('top_predictions', [])
            for i, (name_label, pct_label) in enumerate(self.top_predictions_labels):
                if i < len(top_predictions):
                    pred = top_predictions[i+1] if i+1 < len(top_predictions) else top_predictions[i]
                    prob_percent = pred['probability'] * 100
                    species_name = self.get_species_name(pred.get('class_name', pred.get('class_id', '')))
                    
                    name_label.configure(text=species_name)
                    pct_label.configure(text=f"{prob_percent:.1f}%")
                else:
                    name_label.configure(text="--")
                    pct_label.configure(text="--")
            
            
            self.info_text.configure(state="normal")
            self.info_text.delete("1.0", "end")
            
            
            warning_text = "LOW CONFIDENCE WARNING\n"
            warning_text += "=" * 40 + "\n"
            warning_text += "The identification system has low confidence in this match.\n\n"
            warning_text += "Possible reasons:\n"
            warning_text += "• Image quality is too low\n"
            warning_text += "• Plant is not fully visible\n"
            warning_text += "• Species is not in the training dataset\n"
            warning_text += "• Similar looking plants causing confusion\n\n"
            warning_text += "Suggestions:\n"
            warning_text += "• Try a clearer image\n"
            warning_text += "• Adjust lighting conditions\n"
            warning_text += "• Include more of the plant in the frame\n"
            
            self.info_text.insert("1.0", warning_text)
            self.info_text.configure(state="disabled")
            
        else:
            self.plant_name.configure(
                text="No plant detected",
                text_color=COLORS["off_white"]  
            )
            
            
            self.confidence_value.configure(text="0%")
            self.confidence_bar.set(0)
            
            
            for name_label, pct_label in self.top_predictions_labels:
                name_label.configure(text="--")
                pct_label.configure(text="--")
            
            
            self.info_text.configure(state="normal")
            self.info_text.delete("1.0", "end")
            
            
            if identification['status'] == 'error':
                self.info_text.insert("1.0", f"Error: {identification.get('message', 'Unknown error')}\n\n"
                                    "Please try again or check the system configuration.")
            else:
                self.info_text.insert("1.0", "No plant detected in the current frame.\n\n"
                                    "Try the following:\n"
                                    "• Make sure a plant is visible in the camera\n"
                                    "• Use the tracking feature to select a plant\n"
                                    "• Try different lighting conditions\n"
                                    "• Use 'Test with Image' to try a saved image")
                                    
            self.info_text.configure(state="disabled")
    
    def _process_frame(self):
        """Override _process_frame to include plant identification."""
        
        super()._process_frame()
        
        
        if self.plant_identifier is not None and self.tracker is not None:
            
            if self.tracker.tracking:
                
                ret, frame = self.tracker.capture.read()
                if ret:
                    
                    if self.tracker.detection_bbox is not None:
                        x, y, w, h = self.tracker.detection_bbox
                        
                        if x >= 0 and y >= 0 and w > 0 and h > 0:
                            
                            crop = frame[y:y+h, x:x+w]
                            if crop.size > 0:
                                
                                self._schedule_identification(crop)
    
    def _schedule_identification(self, image):
        """Schedule plant identification in background thread."""
        
        if not hasattr(self, '_identifying') or not self._identifying:
            self._identifying = True
            threading.Thread(target=self._identify_plant_thread, args=(image,), daemon=True).start()
    
    def _identify_plant_thread(self, image):
        """Identify plant in background thread."""
        try:
            
            identification = self.plant_identifier.identify_plant(image)
            
            
            self.after(10, lambda: self._update_identification_ui(identification))
        except Exception as e:
            print(f"Error identifying plant: {e}")
            
            import traceback
            traceback.print_exc()
            
            
            self.after(10, lambda: self._update_identification_ui({
                'status': 'error',
                'message': f'Error identifying plant: {str(e)}',
            }))
        finally:
            self._identifying = False
    
    def get_species_name(self, class_id_or_name):
        """Get readable species name from class ID or raw name."""
        
        if isinstance(class_id_or_name, str) and not class_id_or_name.isdigit():
            
            name = class_id_or_name.replace('_', ' ')
            return name.title()
        
        
        if str(class_id_or_name) in self.species_names:
            raw_name = self.species_names[str(class_id_or_name)]
            return raw_name.replace('_', ' ').title()
        
        
        if isinstance(class_id_or_name, str) and '_' in class_id_or_name:
            return class_id_or_name.replace('_', ' ').title()
        
        
        return f"Plant {class_id_or_name}"
    
    def _update_identification_ui(self, identification):
        """Update UI with plant identification results in a modern style."""
        print(f"Updating UI with identification: {identification}")  
        
        
        if identification['status'] == 'success':
            self.plant_result_icon.configure(
                text="🌿",  
                text_color=COLORS["success"]
            )
        elif identification['status'] == 'low_confidence':
            self.plant_result_icon.configure(
                text="🌱",  
                text_color=COLORS["warning"]
            )
        else:
            self.plant_result_icon.configure(
                text="❓",  
                text_color=COLORS["info"]
            )
        
        
        if identification['status'] == 'success' or identification['status'] == 'low_confidence':
            
            class_id = identification.get('class_id')
            species_name = self.get_species_name(identification.get('class_name', class_id))
            
            display_text = species_name
            if identification['status'] == 'low_confidence':
                display_text = f"{species_name} (low confidence)"
                text_color = COLORS["warning"]
            else:
                text_color = COLORS["success"]
                
            self.plant_name.configure(
                text=display_text,
                text_color=text_color
            )
            
            
            confidence_percent = identification['confidence'] * 100
            print(f"Confidence: {confidence_percent}%")  
            
            
            
            if hasattr(self, 'confidence_value'):
                self.confidence_value.configure(text=f"{confidence_percent:.1f}%")
            
            
            if hasattr(self, 'confidence_bar'):
                self.confidence_bar.set(identification['confidence'])
                
                
                if confidence_percent >= 80:
                    self.confidence_bar.configure(progress_color=COLORS["success"])
                elif confidence_percent >= 60:
                    self.confidence_bar.configure(progress_color=COLORS["medium_green"])
                elif confidence_percent >= 40:
                    self.confidence_bar.configure(progress_color=COLORS["teal"])
                else:
                    self.confidence_bar.configure(progress_color=COLORS["warning"])
            
            
            top_predictions = identification.get('top_predictions', [])
            print(f"Top predictions: {top_predictions}")  
            
            
            
            if hasattr(self, 'top_predictions_labels'):
                if isinstance(self.top_predictions_labels, list):
                    if isinstance(self.top_predictions_labels[0], tuple):
                        
                        for i, (name_label, pct_label) in enumerate(self.top_predictions_labels):
                            if i < len(top_predictions):
                                pred = top_predictions[i]
                                prob_percent = pred['probability'] * 100
                                species_name = self.get_species_name(pred.get('class_name', pred.get('class_id', '')))
                                
                                name_label.configure(text=species_name)
                                pct_label.configure(text=f"{prob_percent:.1f}%")
                            else:
                                name_label.configure(text="--")
                                pct_label.configure(text="--")
                    else:
                        
                        for i, label in enumerate(self.top_predictions_labels):
                            if i < len(top_predictions):
                                pred = top_predictions[i]
                                prob_percent = pred['probability'] * 100
                                species_name = self.get_species_name(pred.get('class_name', pred.get('class_id', '')))
                                label.configure(text=f"{i+1}. {species_name} ({prob_percent:.1f}%)")
                            else:
                                label.configure(text=f"{i+1}. --")
            
            
            plant_info = identification.get('plant_info', None)
            if hasattr(self, 'info_text'):
                self.info_text.configure(state="normal")
                self.info_text.delete("1.0", "end")
                
                if plant_info is not None:
                    
                    info_text = f"Scientific Name: {plant_info.get('scientific_name', 'Unknown')}\n"
                    info_text += f"Family: {plant_info.get('family', 'Unknown')}\n\n"
                    
                    if 'description' in plant_info:
                        info_text += "DESCRIPTION\n"
                        info_text += "=" * 40 + "\n"
                        info_text += f"{plant_info['description']}\n\n"
                    
                    if 'care' in plant_info:
                        info_text += "CARE INFORMATION\n"
                        info_text += "=" * 40 + "\n"
                        for care_type, care_info in plant_info['care'].items():
                            info_text += f"• {care_type.capitalize()}: {care_info}\n"
                        info_text += "\n"
                    
                    if 'common_varieties' in plant_info:
                        info_text += "COMMON VARIETIES\n"
                        info_text += "=" * 40 + "\n"
                        info_text += "• " + "\n• ".join(plant_info['common_varieties'])
                        info_text += "\n"
                    
                    self.info_text.insert("1.0", info_text)
                else:
                    
                    species_parts = species_name.split()
                    if len(species_parts) >= 2:
                        genus = species_parts[0]
                        species = ' '.join(species_parts[1:])
                        
                        info_text = "BASIC INFORMATION\n"
                        info_text += "=" * 40 + "\n"
                        info_text += f"Scientific Name: {species_name}\n"
                        info_text += f"Genus: {genus}\n"
                        info_text += f"Species: {species}\n\n"
                        
                        if identification['status'] == 'low_confidence':
                            info_text += "LOW CONFIDENCE WARNING\n"
                            info_text += "=" * 40 + "\n"
                            info_text += "The identification system has low confidence in this match.\n\n"
                            info_text += "Possible reasons:\n"
                            info_text += "• Image quality is too low\n"
                            info_text += "• Plant is not fully visible\n"
                            info_text += "• Species is not in the training dataset\n"
                            info_text += "• Similar looking plants causing confusion\n\n"
                        else:
                            info_text += "NOTE\n"
                            info_text += "=" * 40 + "\n"
                            info_text += "No detailed information available for this plant species.\n"
                            info_text += "Consider researching more about it online."
                    else:
                        info_text = "No detailed information available for this plant."
                    
                    self.info_text.insert("1.0", info_text)
                
                self.info_text.configure(state="disabled")
            
        else:
            
            self.plant_name.configure(
                text="No plant detected",
                text_color=COLORS["off_white"]  
            )
            
            
            if hasattr(self, 'confidence_value'):
                self.confidence_value.configure(text="0%")
            if hasattr(self, 'confidence_bar'):
                self.confidence_bar.set(0)
            
            
            if hasattr(self, 'top_predictions_labels'):
                if isinstance(self.top_predictions_labels, list):
                    if isinstance(self.top_predictions_labels[0], tuple):
                        for name_label, pct_label in self.top_predictions_labels:
                            name_label.configure(text="--")
                            pct_label.configure(text="--")
                    else:
                        for i, label in enumerate(self.top_predictions_labels):
                            label.configure(text=f"{i+1}. --")
            
            
            if hasattr(self, 'info_text'):
                self.info_text.configure(state="normal")
                self.info_text.delete("1.0", "end")
                
                
                if identification['status'] == 'error':
                    self.info_text.insert("1.0", f"Error: {identification.get('message', 'Unknown error')}\n\n"
                                        "Please try again or check the system configuration.")
                else:
                    self.info_text.insert("1.0", "No plant detected in the current frame.\n\n"
                                        "Try the following:\n"
                                        "• Make sure a plant is visible in the camera\n"
                                        "• Use the tracking feature to select a plant\n"
                                        "• Try different lighting conditions\n"
                                        "• Use 'Test with Image' to try a saved image")
                                        
                self.info_text.configure(state="disabled")
    
    def on_identify_plant(self):
        """Manually trigger plant identification."""
        
        if self.tracker is not None and self.tracker.tracking:
            
            ret, frame = self.tracker.capture.read()
            if ret:
                
                if self.tracker.detection_bbox is not None:
                    x, y, w, h = self.tracker.detection_bbox
                    
                    if x >= 0 and y >= 0 and w > 0 and h > 0:
                        
                        crop = frame[y:y+h, x:x+w]
                        if crop.size > 0:
                            
                            self._schedule_identification(crop)
                            return
                
                
                self._update_identification_ui({
                    'status': 'error',
                    'message': 'No plant detected in the tracking area.'
                })
            else:
                
                self._update_identification_ui({
                    'status': 'error',
                    'message': 'Please start tracking a plant first.'
                })
    
    def on_test_image(self):
        """Handle test image button click"""
        
        file_path = filedialog.askopenfilename(
            title="Select Test Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.load_test_image(file_path)
    
    def load_test_image(self, image_path):
        """
        Load a test image and run identification on it
        
        Args:
            image_path: Path to the image file
        """
        try:
            
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error: Could not load image from {image_path}")
                return
                    
            
            if self.plant_identifier is not None:
                self._schedule_identification(img)
                    
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img_rgb.shape[:2]
                    
                
                max_dim = 400
                if h > max_dim or w > max_dim:
                    scale = max_dim / max(h, w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    img_rgb = cv2.resize(img_rgb, (new_w, new_h))
                    
                
                pil_img = Image.fromarray(img_rgb)
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, 
                                    size=(pil_img.width, pil_img.height))
                    
                
                self.debug_view.configure(image=ctk_img)
                self.debug_view.image = ctk_img
                    
                
                h, w = img.shape[:2]
                self.tracker.detection_bbox = (0, 0, w, h)
                    
                
                self.status.configure(text=f"Status: Testing with image: {os.path.basename(image_path)}")
        except Exception as e:
            print(f"Error testing image: {e}")
    
    def restore_debug_view(self):
        """Restore normal debug view after testing"""
        self.debug_view.configure(image=None)
        self.debug_view.image = None
    
    def on_reset(self):
        """Override on_reset to also reset plant identification."""
        
        super().on_reset()
        
        
        if self.plant_identifier is not None:
            
            self.plant_identifier.clear_history()
            
            
            self.restore_debug_view()
            
            
            self._update_identification_ui({
                'status': 'error',
                'message': 'Reset completed.'
            })