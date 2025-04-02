from .colors import COLORS

MODERN_THEME = {
    "CTk": {
        "fg_color": [COLORS["dark_grey"], COLORS["dark_grey"]]
    },
    "CTkFrame": {
        "fg_color": [COLORS["medium_grey"], COLORS["medium_grey"]],
        "corner_radius": 10,     
        "border_width": 0        
    },
    "CTkButton": {
        "fg_color": [COLORS["medium_green"], COLORS["medium_green"]],
        "hover_color": [COLORS["dark_green"], COLORS["dark_green"]],
        "border_color": [COLORS["dark_grey"], COLORS["dark_grey"]],
        "text_color": [COLORS["white"], COLORS["white"]],
        "text_color_disabled": [COLORS["light_grey"], COLORS["light_grey"]],
        "corner_radius": 8,      
        "border_width": 0,       
        "font": ("Helvetica", 12, "bold") 
    },
    "CTkLabel": {
        "fg_color": "transparent",
        "text_color": [COLORS["off_white"], COLORS["off_white"]],
        "font": ("Helvetica", 12) 
    },
    "CTkRadioButton": {
        "fg_color": [COLORS["medium_green"], COLORS["medium_green"]],
        "hover_color": [COLORS["dark_green"], COLORS["dark_green"]],
        "text_color": [COLORS["off_white"], COLORS["off_white"]],
        "border_width": 2,       
        "corner_radius": 1000, 
        "font": ("Helvetica", 12) 
    },
    "CTkEntry": {
        "fg_color": [COLORS["light_grey"], COLORS["light_grey"]],
        "border_color": [COLORS["medium_green"], COLORS["medium_green"]],
        "text_color": [COLORS["white"], COLORS["white"]],
        "placeholder_text_color": [COLORS["very_light_grey"], COLORS["very_light_grey"]],
        "corner_radius": 8      
    },
    "CTkCheckBox": {
        "fg_color": [COLORS["medium_green"], COLORS["medium_green"]],
        "hover_color": [COLORS["dark_green"], COLORS["dark_green"]],
        "text_color": [COLORS["off_white"], COLORS["off_white"]],
        "corner_radius": 6    
    },
    "CTkTextbox": {
        "fg_color": [COLORS["light_grey"], COLORS["light_grey"]],
        "border_color": [COLORS["medium_green"], COLORS["medium_green"]],
        "text_color": [COLORS["off_white"], COLORS["off_white"]],
        "corner_radius": 8,      
        "border_width": 0        
    },
    "CTkScrollbar": {
        "fg_color": [COLORS["medium_grey"], COLORS["medium_grey"]],
        "button_color": [COLORS["very_light_grey"], COLORS["very_light_grey"]],
        "button_hover_color": [COLORS["medium_green"], COLORS["medium_green"]],
        "corner_radius": 8,      
    },
    "CTkProgressBar": {
        "fg_color": [COLORS["light_grey"], COLORS["light_grey"]],
        "progress_color": [COLORS["medium_green"], COLORS["medium_green"]],
        "corner_radius": 8       
    },
    "CTkSegmentedButton": {
        "fg_color": [COLORS["light_grey"], COLORS["light_grey"]],
        "selected_color": [COLORS["medium_green"], COLORS["medium_green"]],
        "selected_hover_color": [COLORS["dark_green"], COLORS["dark_green"]],
        "unselected_color": [COLORS["medium_grey"], COLORS["medium_grey"]],
        "unselected_hover_color": [COLORS["light_grey"], COLORS["light_grey"]],
        "text_color": [COLORS["off_white"], COLORS["off_white"]],
        "text_color_disabled": [COLORS["light_grey"], COLORS["light_grey"]],
        "corner_radius": 8       
    }
}

def apply_theme(theme_manager, theme_dict):
    for widget, properties in theme_dict.items():
        theme_manager.theme[widget].update(properties)