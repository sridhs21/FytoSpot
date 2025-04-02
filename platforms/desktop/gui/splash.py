import tkinter as tk
from PIL import Image, ImageTk
import os
import time
from pathlib import Path
from assets.styles.colors import COLORS

def show_splash(duration=3.0, logo_path=None):
    """
    Show a splash screen using the FytóSpot logo image as background
    with a simple loading bar.
    
    Args:
        duration: Time in seconds to display the splash screen
        logo_path: Path to the logo image (optional)
    """
    
    root = tk.Tk()
    root.title("FytóSpot")
    root.overrideredirect(True)  
    
    
    window_width = 800
    window_height = 500
    root.geometry(f"{window_width}x{window_height}")
    
    
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    
    root.configure(bg=COLORS["dark_grey"])
    
    
    main_canvas = tk.Canvas(
        root, 
        width=window_width, 
        height=window_height,
        highlightthickness=0,
        bd=0
    )
    main_canvas.pack(fill="both", expand=True)
    
    # Try to load logo
    if logo_path is None:
        # Search for logo in standard locations
        logo_paths = [
            "assets/images/logo.png",
            "logo.png",
            "../logo.png",
            "GUI/logo.png"
        ]
    else:
        logo_paths = [logo_path]
    
    background_image = None
    
    for path in logo_paths:
        if os.path.exists(path):
            try:
                # Create a gradient background that matches the logo colors
                img = Image.open(path)
                
                # Calculate the image size to cover the entire window while maintaining aspect ratio
                img_width, img_height = img.size
                img_ratio = img_width / img_height
                window_ratio = window_width / window_height
                
                if window_ratio > img_ratio:
                    # Window is wider than image
                    new_width = window_width
                    new_height = int(window_width / img_ratio)
                else:
                    # Window is taller than image
                    new_height = window_height
                    new_width = int(window_height * img_ratio)
                
                # Resize image to cover the window
                img = img.resize((new_width, new_height), Image.LANCZOS)
                
                # Convert to PhotoImage for Tkinter
                background_image = ImageTk.PhotoImage(img)
                
                # Add the background image to the canvas
                main_canvas.create_image(
                    window_width // 2,  # Center x
                    window_height // 2,  # Center y
                    image=background_image,
                    anchor="center"
                )
                
                # Calculate loading bar position and dimensions
                bar_width = int(window_width * 0.7)
                bar_height = 10
                bar_x = (window_width - bar_width) // 2
                bar_y = int(window_height * 0.85)
                
                # Create loading bar background
                loading_bg = main_canvas.create_rectangle(
                    bar_x, bar_y, 
                    bar_x + bar_width, bar_y + bar_height,
                    fill="#333333",
                    outline=""
                )
                
                # Create initial loading progress (starts at 0)
                loading_progress = main_canvas.create_rectangle(
                    bar_x, bar_y, 
                    bar_x, bar_y + bar_height,
                    fill=COLORS["medium_green"],
                    outline=""
                )
                
                # Update progress bar function
                def update_progress_bar(current_width, step):
                    if current_width < bar_width:
                        next_width = min(current_width + step, bar_width)
                        
                        main_canvas.coords(
                            loading_progress, 
                            bar_x, bar_y, 
                            bar_x + next_width, bar_y + bar_height
                        )
                        
                        root.after(50, update_progress_bar, next_width, step)
                    else:
                        root.after(500, close_splash)
                
                def close_splash():
                    root.destroy()
                
                step_size = bar_width / (duration * 20)  
                update_progress_bar(0, step_size)
                
                break
                
            except Exception as e:
                print(f"Error loading logo from {path}: {e}")
                background_image = None
    
    # If no logo loaded, then create a default gradient background
    if background_image is None:
        # Create a gradient from light green to teal
        for i in range(window_width):
            # Calculate color based on position
            r = int(180 - (180 - 74) * i / window_width)
            g = int(233 - (233 - 155) * i / window_width)
            b = int(157 - (157 - 136) * i / window_width)
            color = f'#{r:02x}{g:02x}{b:02x}'
            
            main_canvas.create_line(
                i, 0, i, window_height, 
                fill=color, 
                width=1
            )
        
        main_canvas.create_text(
            window_width // 2, 
            window_height // 2 - 50,
            text="FytóSpot",
            font=("Arial", 48, "bold"),
            fill="#ffffff"
        )
        
        bar_width = int(window_width * 0.7)
        bar_height = 10
        bar_x = (window_width - bar_width) // 2
        bar_y = int(window_height * 0.7)
        
        loading_bg = main_canvas.create_rectangle(
            bar_x, bar_y, 
            bar_x + bar_width, bar_y + bar_height,
            fill="#333333",
            outline=""
        )
        
        loading_progress = main_canvas.create_rectangle(
            bar_x, bar_y, 
            bar_x, bar_y + bar_height,
            fill=COLORS["medium_green"],
            outline=""
        )
        
        # Update progress bar function
        def update_progress_bar(current_width, step):
            if current_width < bar_width:
                # Update progress bar width
                next_width = min(current_width + step, bar_width)
                
                # Update progress bar rectangle
                main_canvas.coords(
                    loading_progress, 
                    bar_x, bar_y, 
                    bar_x + next_width, bar_y + bar_height
                )
                
                # Schedule next update
                root.after(50, update_progress_bar, next_width, step)
            else:
                # Schedule the closing of the splash screen
                root.after(500, close_splash)
        
        def close_splash():
            # Destroy the root window completely
            root.destroy()
        
        # Start progress bar animation
        step_size = bar_width / (duration * 20)  # 20 updates per second
        update_progress_bar(0, step_size)
    
    # Force update to draw everything
    root.update()
    
    # Start the main loop
    root.mainloop()

# For testing
if __name__ == "__main__":
    show_splash(3.0)
    print("Splash screen closed, main app would start now")