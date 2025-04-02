import tkinter
import customtkinter as ctk
import cv2
import numpy as np
import PIL.Image, PIL.ImageTk
from typing import Tuple, Optional, Dict, Any, Callable
import threading
import time
import os
from pathlib import Path

# Import from core modules
from core.detection.object_tracker import ObjectTracker
from assets.styles.colors import COLORS
from assets.styles.themes import MODERN_THEME, apply_theme

class ObjectTrackerApp(ctk.CTk):
    """
    Modern desktop GUI application for the FytóSpot plant tracker.
    """
    def __init__(self, tracker: ObjectTracker):
        """
        Initialize the desktop application.
        
        Args:
            tracker: ObjectTracker instance
        """
        super().__init__()
        
        # Store the tracker object
        self.tracker = tracker
        
        # Set appearance mode and theme
        ctk.set_appearance_mode("Dark")
        apply_theme(ctk.ThemeManager, MODERN_THEME)
        
        # Configure the window
        self.title("FytóSpot")
        self.geometry("1280x720")  # Larger default size
        self.minsize(1024, 650)    # Slightly larger minimum size
        
        # Set the background color
        self.configure(fg_color=COLORS["dark_grey"])
        
        # Create the main grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Create a title bar at the top
        self.create_title_bar()
        
        # Create the main frame with subtle rounded corners - Adjust position for title bar
        self.main_frame = ctk.CTkFrame(self, corner_radius=12, fg_color=COLORS["dark_grey"])
        self.main_frame.pack(fill="both", expand=True, padx=15, pady=(15, 15))
        
        # Configure the main frame grid
        self.main_frame.grid_columnconfigure(0, weight=7)  # Camera view
        self.main_frame.grid_columnconfigure(1, weight=3)  # Plant identification panel
        self.main_frame.grid_columnconfigure(2, weight=3)  # Debug view
        self.main_frame.grid_rowconfigure(0, weight=1)     # Views row
        self.main_frame.grid_rowconfigure(1, weight=0)     # Status bar row
        self.main_frame.grid_rowconfigure(2, weight=0)     # Controls row
        
        # Camera view frame - place in column 0
        self.camera_frame = ctk.CTkFrame(self.main_frame, corner_radius=10)
        self.camera_frame.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")
        
        # Camera view title with icon
        self.camera_title_frame = ctk.CTkFrame(
            self.camera_frame, 
            fg_color=COLORS["medium_grey"], 
            corner_radius=8, 
            height=36
        )
        self.camera_title_frame.pack(fill="x", padx=4, pady=4)
        self.camera_title_frame.pack_propagate(False)  # Don't shrink to fit content
        
        # Camera icon (placeholder, you'd add a real icon)
        self.camera_icon = ctk.CTkLabel(
            self.camera_title_frame,
            text="📹",  # Unicode camera icon
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=COLORS["medium_green"]
        )
        self.camera_icon.pack(side="left", padx=(10, 5))
        
        # Camera title
        self.camera_title = ctk.CTkLabel(
            self.camera_title_frame, 
            text="Camera Feed", 
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLORS["off_white"]
        )
        self.camera_title.pack(side="left")
        
        # Camera view frame with shadow effect
        self.camera_view_frame = ctk.CTkFrame(
            self.camera_frame, 
            corner_radius=8,
            fg_color=COLORS["dark_grey"],
            border_width=1,
            border_color=COLORS["very_light_grey"]
        )
        self.camera_view_frame.pack(fill="both", expand=True, padx=8, pady=8)
        
        # Camera view label
        self.camera_view = ctk.CTkLabel(self.camera_view_frame, text="")
        self.camera_view.pack(fill="both", expand=True)
        
        # Debug view frame - place in column 2
        self.debug_frame = ctk.CTkFrame(self.main_frame, corner_radius=10)
        self.debug_frame.grid(row=0, column=2, padx=8, pady=8, sticky="nsew")
        
        # Debug view title with icon
        self.debug_title_frame = ctk.CTkFrame(
            self.debug_frame, 
            fg_color=COLORS["medium_grey"], 
            corner_radius=8, 
            height=36
        )
        self.debug_title_frame.pack(fill="x", padx=4, pady=4)
        self.debug_title_frame.pack_propagate(False)  # Don't shrink to fit content
        
        # Debug icon (placeholder)
        self.debug_icon = ctk.CTkLabel(
            self.debug_title_frame,
            text="🔍",  # Unicode magnifying glass icon
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=COLORS["medium_green"]
        )
        self.debug_icon.pack(side="left", padx=(10, 5))
        
        # Debug title
        self.debug_label = ctk.CTkLabel(
            self.debug_title_frame, 
            text="Debug View", 
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLORS["off_white"]
        )
        self.debug_label.pack(side="left")
        
        # Debug view frame with shadow effect
        self.debug_view_frame = ctk.CTkFrame(
            self.debug_frame, 
            corner_radius=8,
            fg_color=COLORS["dark_grey"],
            border_width=1,
            border_color=COLORS["very_light_grey"]
        )
        self.debug_view_frame.pack(fill="both", expand=True, padx=8, pady=8)
        
        # Debug view
        self.debug_view = ctk.CTkLabel(self.debug_view_frame, text="")
        self.debug_view.pack(fill="both", expand=True)
        
        # Status bar frame
        self.status_frame = ctk.CTkFrame(
            self.main_frame, 
            fg_color=COLORS["medium_grey"],
            corner_radius=8,
            height=32
        )
        self.status_frame.grid(row=1, column=0, columnspan=3, padx=8, pady=(0, 8), sticky="ew")
        self.status_frame.grid_propagate(False)  # Don't shrink to fit content
        
        # Configure status frame grid
        self.status_frame.grid_columnconfigure(0, weight=1)  # Status text
        self.status_frame.grid_columnconfigure(1, weight=0)  # Status indicators
        
        # Status icon
        self.status_icon = ctk.CTkLabel(
            self.status_frame,
            text="ℹ️",  # Info icon
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLORS["info"]
        )
        self.status_icon.grid(row=0, column=0, padx=(10, 5), pady=4, sticky="w")
        
        # Status label with medium green color
        self.status = ctk.CTkLabel(
            self.status_frame, 
            text="Ready to detect plants", 
            font=ctk.CTkFont(size=13),
            padx=0, pady=4,
            text_color=COLORS["off_white"]
        )
        self.status.grid(row=0, column=0, padx=(30, 8), pady=4, sticky="w")
        
        # Right-aligned recording indicator (initially hidden)
        self.recording_indicator = ctk.CTkLabel(
            self.status_frame,
            text="● RECORDING",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLORS["error"],
        )
        self.recording_indicator.grid(row=0, column=1, padx=10, pady=4, sticky="e")
        self.recording_indicator.grid_remove()  # Hide initially
        
        # Controls frame
        self.controls_frame = ctk.CTkFrame(self.main_frame, corner_radius=10)
        self.controls_frame.grid(row=2, column=0, columnspan=3, padx=8, pady=(0, 8), sticky="ew")
        
        # Detection method section
        self.detection_frame = ctk.CTkFrame(
            self.controls_frame, 
            corner_radius=8,
            fg_color=COLORS["medium_grey"]
        )
        self.detection_frame.pack(fill="x", padx=8, pady=8)
        
        # Detection method label with icon
        self.detection_icon = ctk.CTkLabel(
            self.detection_frame,
            text="🔎",  # Search icon
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLORS["medium_green"]
        )
        self.detection_icon.grid(row=0, column=0, padx=(10, 0), pady=4, sticky="w")
        
        # Detection method label
        self.detection_label = ctk.CTkLabel(
            self.detection_frame, 
            text="Detection Method:", 
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLORS["off_white"]
        )
        self.detection_label.grid(row=0, column=1, padx=(5, 8), pady=4, sticky="w")
        
        # Detection method radio buttons
        self.detection_method = tkinter.StringVar(value="multi")
        
        self.btn_multi = ctk.CTkRadioButton(
            self.detection_frame, 
            text="Multi Detection", 
            variable=self.detection_method, 
            value="multi",
            command=self.on_multi,
            fg_color=COLORS["medium_green"],
            hover_color=COLORS["dark_green"],
            text_color=COLORS["off_white"]
        )
        self.btn_multi.grid(row=0, column=2, padx=15, pady=4)

        self.btn_color = ctk.CTkRadioButton(
            self.detection_frame, 
            text="Color Detection", 
            variable=self.detection_method, 
            value="color",
            command=self.on_color,
            fg_color=COLORS["medium_green"],
            hover_color=COLORS["dark_green"],
            text_color=COLORS["off_white"]
        )
        self.btn_color.grid(row=0, column=3, padx=15, pady=4)
        
        self.btn_contour = ctk.CTkRadioButton(
            self.detection_frame, 
            text="Contour Analysis", 
            variable=self.detection_method, 
            value="contour",
            command=self.on_contour,
            fg_color=COLORS["medium_green"],
            hover_color=COLORS["dark_green"],
            text_color=COLORS["off_white"]
        )
        self.btn_contour.grid(row=0, column=4, padx=15, pady=4)
        
        self.btn_texture = ctk.CTkRadioButton(
            self.detection_frame, 
            text="Texture Detection", 
            variable=self.detection_method, 
            value="texture",
            command=self.on_texture,
            fg_color=COLORS["medium_green"],
            hover_color=COLORS["dark_green"],
            text_color=COLORS["off_white"]
        )
        self.btn_texture.grid(row=0, column=5, padx=15, pady=4)
        
        # Action buttons section
        self.action_frame = ctk.CTkFrame(
            self.controls_frame,
            corner_radius=8,
            fg_color=COLORS["medium_grey"]
        )
        self.action_frame.pack(fill="x", padx=8, pady=(0, 8))
        
        # Configure the action frame grid - one column for reset button
        self.action_frame.grid_columnconfigure(0, weight=1)
        
        # Reset button only - modern style
        button_height = 36  # Taller button for better touch target
        
        self.btn_reset = ctk.CTkButton(
            self.action_frame, 
            text="Reset", 
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=COLORS["teal"], 
            hover_color=COLORS["dark_teal"],
            command=self.on_reset,
            height=button_height,
            corner_radius=8,
            border_spacing=10,
            image=self.create_image_from_emoji("🔄")  # Reset icon
        )
        self.btn_reset.grid(row=0, column=0, padx=8, pady=8, sticky="ew")
        
        # Initialize video update
        self.update_pending = False
        self.update_interval = 33  # ~30 FPS in milliseconds
        
        # Auto-start tracking
        self.after(1000, self.auto_start_tracking)
        
        # Start the video update loop
        self.update_frame()

    def auto_start_tracking(self):
        """Automatically start tracking when the app loads"""
        # Only start if there's a detection
        if self.tracker.detection_bbox is not None:
            self.tracker.start_tracking()
            self.update_status("Tracking Active", "✅", "success")
        else:
            # Try again in 1 second if no detection yet
            self.after(1000, self.auto_start_tracking)
    
    def create_title_bar(self):
        """Create a modern title bar with app logo and name"""
        # Title bar frame
        self.title_bar = ctk.CTkFrame(
            self, 
            height=50, 
            corner_radius=0, 
            fg_color=COLORS["medium_grey"]
        )
        # Change this positioning to be ABOVE the main layout
        self.title_bar.pack(side="top", fill="x", padx=0, pady=0)
        self.title_bar.pack_propagate(False)  # Don't resize
        
        # Configure title bar as horizontal layout
        title_frame = ctk.CTkFrame(
            self.title_bar,
            fg_color="transparent"
        )
        title_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # App logo (placeholder - replace with your actual logo)
        self.logo_label = ctk.CTkLabel(
            title_frame,
            text="🌱",  # Plant emoji as placeholder
            font=ctk.CTkFont(size=24),
            text_color=COLORS["medium_green"]
        )
        self.logo_label.pack(side="left", padx=(5, 5))
        
        # App title
        self.app_title = ctk.CTkLabel(
            title_frame,
            text="FytóSpot",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=COLORS["off_white"]
        )
        self.app_title.pack(side="left", padx=5)
        
        # App version
        self.app_version = ctk.CTkLabel(
            title_frame,
            text="v1.0.0",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["light_grey"]
        )
        self.app_version.pack(side="right", padx=5)
    
    def create_image_from_emoji(self, emoji, size=(20, 20)):
        """Create a CTkImage from a Unicode emoji for button icons"""
        # Create a PIL image with transparent background
        img = PIL.Image.new('RGBA', (30, 30), (0, 0, 0, 0))
        
        # Add text to the image
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        
        # Try to use a font that supports emoji
        try:
            # Try different font options
            font_options = [
                ("Segoe UI Emoji", 20),
                ("Apple Color Emoji", 20),
                ("Noto Color Emoji", 20),
                ("Arial", 18)
            ]
            
            font = None
            for font_name, font_size in font_options:
                try:
                    font = ImageFont.truetype(font_name, font_size)
                    break
                except:
                    continue
            
            # If no font worked, use default
            if font is None:
                font = ImageFont.load_default()
            
            # Draw the emoji
            draw.text((15, 15), emoji, font=font, fill=(255, 255, 255, 255), anchor="mm")
            
        except Exception as e:
            print(f"Error creating emoji image: {e}")
            # Return None if failed
            return None
        
        # Convert to CTkImage
        return ctk.CTkImage(light_image=img, dark_image=img, size=size)
        
    def update_frame(self):
        """Update the video frames"""
        if not self.update_pending:
            self.update_pending = True
            self.after(self.update_interval, self._process_frame)
    
    def _process_frame(self):
        """Process and display frames"""
        self.update_pending = False
        
        # Get frames from tracker
        frame, debug_frame = self.tracker.process_frame()
        
        if frame is not None:
            # Convert to PIL.Image format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = PIL.Image.fromarray(frame_rgb)
            
            # Get original dimensions
            orig_width, orig_height = pil_img.width, pil_img.height
            
            # Calculate the size of the square (using the smaller dimension)
            square_size = min(orig_width, orig_height)
            
            # Calculate center crop coordinates
            left = (orig_width - square_size) // 2
            top = (orig_height - square_size) // 2
            right = left + square_size
            bottom = top + square_size
            
            # Crop to square
            pil_img = pil_img.crop((left, top, right, bottom))
            
            # Scale the image to fit the container better
            scale_factor = 0.85  # Scale down to 85% of original size
            img_size = int(square_size * scale_factor)
            
            # Create CTkImage with square dimensions
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(img_size, img_size))
            self.camera_view.configure(image=ctk_img)
            self.camera_view.image = ctk_img  # Keep a reference
        
        if debug_frame is not None:
            # Ensure debug_frame is in correct format (BGR)
            if len(debug_frame.shape) == 2:  # If grayscale
                debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_GRAY2BGR)
            
            # Convert to RGB for display
            debug_rgb = cv2.cvtColor(debug_frame, cv2.COLOR_BGR2RGB)
            debug_pil = PIL.Image.fromarray(debug_rgb)
            
            # Get original dimensions
            orig_width, orig_height = debug_pil.width, debug_pil.height
            
            # Calculate the size of the square (using the smaller dimension)
            square_size = min(orig_width, orig_height)
            
            # Calculate center crop coordinates
            left = (orig_width - square_size) // 2
            top = (orig_height - square_size) // 2
            right = left + square_size
            bottom = top + square_size
            
            # Crop to square
            debug_pil = debug_pil.crop((left, top, right, bottom))
            
            # Scale the debug image to fit the container better
            scale_factor = 0.85  # Scale down to 85% of original size
            debug_size = int(square_size * scale_factor)
            
            # Create CTkImage with square dimensions
            debug_ctk = ctk.CTkImage(light_image=debug_pil, dark_image=debug_pil, size=(debug_size, debug_size))
            self.debug_view.configure(image=debug_ctk)
            self.debug_view.image = debug_ctk  # Keep a reference
        
        # Schedule the next update
        self.update_frame()
    
    def on_multi(self):
        """Switch to multi detection method"""
        self.tracker.set_detection_method('multi')
        self.update_status("Multi Detection Mode", "🔍")

    def on_color(self):
        """Switch to color detection method"""
        self.tracker.set_detection_method('color')
        self.update_status("Color Detection Mode", "🎨")

    def on_contour(self):
        """Switch to contour analysis detection method"""
        self.tracker.set_detection_method('contour')
        self.update_status("Contour Analysis Mode", "📊")
        
    def on_texture(self):
        """Switch to texture detection method"""
        self.tracker.set_detection_method('texture')
        self.update_status("Texture Detection Mode", "🧩")
    
    def update_status(self, message, icon="ℹ️", status_type="info"):
        """Update status bar with message and icon"""
        # Update status text
        self.status.configure(text=message)
        
        # Update status icon
        self.status_icon.configure(text=icon)
        
        # Update icon color based on status type
        if status_type == "success":
            self.status_icon.configure(text_color=COLORS["success"])
        elif status_type == "warning":
            self.status_icon.configure(text_color=COLORS["warning"])
        elif status_type == "error":
            self.status_icon.configure(text_color=COLORS["error"])
        else:  # info
            self.status_icon.configure(text_color=COLORS["info"])
    
    def on_track(self):
        """Start/stop tracking"""
        if not self.tracker.tracking:
            success = self.tracker.start_tracking()
            if success:
                self.btn_track.configure(
                    text="Stop Tracking", 
                    fg_color=COLORS["teal"], 
                    hover_color=COLORS["dark_teal"],
                    image=self.create_image_from_emoji("⏹️")  # Stop icon
                )
                self.update_status("Tracking Active", "✅", "success")
            else:
                self.update_status("Failed to start tracking", "⚠️", "warning")
        else:
            self.tracker.stop_tracking()
            self.btn_track.configure(
                text="Start Tracking", 
                fg_color=COLORS["medium_green"], 
                hover_color=COLORS["dark_green"],
                image=self.create_image_from_emoji("🔄")  # Track icon
            )
            self.update_status("Tracking Stopped", "🛑", "info")
    
    def on_record(self):
        """Start/stop recording"""
        if not self.tracker.recording:
            # Create a unique filename with timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"recordings/fytospot_recording_{timestamp}.mp4"
            
            # Create recordings directory if it doesn't exist
            os.makedirs("recordings", exist_ok=True)
            
            self.tracker.start_recording(filename)
            self.btn_record.configure(
                text="Stop Recording", 
                fg_color=COLORS["error"], 
                hover_color="#d32f2f",  # Darker red
                image=self.create_image_from_emoji("⏹️")  # Stop icon
            )
            
            # Show recording indicator
            self.recording_indicator.grid()
            
            # Update status
            self.update_status(f"Recording to {filename}", "⏺️", "error")
            
            # Make recording indicator blink
            self.blink_recording_indicator()
        else:
            self.tracker.stop_recording()
            self.btn_record.configure(
                text="Start Recording", 
                fg_color=COLORS["medium_green"], 
                hover_color=COLORS["dark_green"],
                image=self.create_image_from_emoji("⏺️")  # Record icon
            )
            
            # Hide recording indicator
            self.recording_indicator.grid_remove()
            
            # Cancel any pending blink
            if hasattr(self, 'blink_job') and self.blink_job is not None:
                self.after_cancel(self.blink_job)
                self.blink_job = None
            
            # Update status
            self.update_status("Recording Stopped", "🛑", "info")
    
    def blink_recording_indicator(self):
        """Make recording indicator blink on and off"""
        if not self.tracker.recording:
            return
            
        # Toggle visibility
        if self.recording_indicator.winfo_viewable():
            self.recording_indicator.grid_remove()
        else:
            self.recording_indicator.grid()
        
        # Schedule next blink
        self.blink_job = self.after(500, self.blink_recording_indicator)
    
    def on_reset(self):
        """Reset the tracker"""
        self.tracker.stop_tracking()
        self.tracker.stop_recording()
        self.tracker.detection_bbox = None
        
        self.update_status("System Reset", "🔄", "info")
        self.after(1000, self.auto_start_tracking)
    
    def on_closing(self):
        """Called when the app is closing"""
        if self.tracker.recording:
            self.tracker.stop_recording()
        
        if self.tracker.tracking:
            self.tracker.stop_tracking()
        
        self.tracker.cleanup()
        
        self.destroy()