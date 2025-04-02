# FytóSpot Color Palette
# Consistent colors for use across the application

# Main color palette based on the FytóSpot logo
COLORS = {
    # Primary greens (slightly adjusted for more modern feel)
    "light_green": "#9be89d",    # Lighter, more vibrant green
    "medium_green": "#50c985",   # More saturated medium green
    "dark_green": "#3a9b6f",     # Richer dark green
    
    # Secondary colors
    "teal": "#3a9b9b",           # More saturated teal
    "dark_teal": "#2d7f88",      # Deeper teal
    
    # Accent colors (new)
    "accent_purple": "#8c54a1",  # Purple accent for variety
    "accent_orange": "#ff7f50",  # Coral/orange for warnings/actions
    
    # Neutrals (improved contrast and feel)
    "dark_grey": "#1e1e1e",      # Darker background
    "medium_grey": "#2a2a2a",    # Improved medium tone
    "light_grey": "#3a3a3a",     # Better light grey
    "very_light_grey": "#4a4a4a", # Added another shade
    "white": "#ffffff",
    "off_white": "#f0f0f0",      # Softer white for text
    
    # Status colors (more vibrant)
    "success": "#4caf50",        # Material design green
    "warning": "#ff9800",        # Material design orange
    "error": "#f44336",          # Material design red
    "info": "#2196f3",           # Material design blue
}

# Color usage recommendations
SEMANTIC_COLORS = {
    # Main colors
    "primary": COLORS["medium_green"],
    "primary_hover": COLORS["dark_green"],
    "secondary": COLORS["teal"],
    "secondary_hover": COLORS["dark_teal"],
    
    # Background colors
    "bg_primary": COLORS["dark_grey"],
    "bg_secondary": COLORS["medium_grey"],
    "bg_tertiary": COLORS["light_grey"],
    
    # Text colors
    "text_primary": COLORS["white"],
    "text_secondary": COLORS["light_green"],
    "text_tertiary": COLORS["teal"],
    
    # Status colors
    "success": COLORS["success"],
    "warning": COLORS["warning"],
    "error": COLORS["error"],
    "info": COLORS["info"],
}