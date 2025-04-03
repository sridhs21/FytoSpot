# FytóSpot - Plant Identification and Tracking System

<div align="center">
  <img src="assets/images/logo_enhanced.jpg" alt="FytóSpot Logo" width="700px" />
  <br>
  <h3>Computer Vision-Powered Plant Identification System</h3>
</div>

## 📋 Overview

FytóSpot is a computer vision-based plant identification and tracking system. It uses machine learning techniques to detect and identify plants from images and video streams, with support for multiple detection methods, real-time tracking, and detailed information about identified plant species.

## ✨ Features

- **Multi-Method Plant Detection:**
  - Color-based Detection
  - Texture Analysis
  - Contour Detection
  - Combined Multi-detection approach

- **Real-time Plant Tracking:**
  - Automatic bounding box creation
  - Smooth tracking with temporal filtering
  - Video recording capabilities

- **Plant Identification:**
  - Species classification using ResNet-based model
  - Attention visualization for model explainability
  - Detailed plant information database

- **Modern User Interfaces:**
  - Web Interface: Responsive web app with PWA support
  - Desktop Application: Native experience with CustomTkinter
  
- **User-Friendly Experience:**
  - Intuitive UI for both web and desktop
  - Visualization of detection and debug information
  - Real-time confidence metrics

## 🔧 Technology Stack

### Computer Vision
- [OpenCV](https://opencv.org/) - Core image processing and computer vision functions
- NumPy - Numerical processing for image data

### Machine Learning
- [PyTorch](https://pytorch.org/) - Deep learning framework
- ResNet - Convolutional neural network architecture for classification
- torchvision - Computer vision utilities for PyTorch

### Backend
- Flask - Lightweight web server
- JSON - Data interchange format for plant information

### Frontend
- HTML/CSS - Responsive web interface
- JavaScript - Interactive UI elements
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) - Modern desktop UI components

## 📦 Installation

### Prerequisites

- Python 3.8+
- Pip package manager
- Git

### Web Application

```bash
# Clone the repository
git clone https://github.com/yourusername/fytospot.git
cd fytospot

# Install dependencies
pip install -r requirements.txt

# Start the server
python platforms/web/app.py
```

The web interface will be available at `http://localhost:5000`.

### Desktop Application

```bash
# Clone the repository
git clone https://github.com/yourusername/fytospot.git
cd fytospot

# Install desktop dependencies
pip install -r requirements-desktop.txt

# Launch the desktop app
python platforms/desktop/main.py
```

## 🖥️ Usage

### Web Interface

1. Navigate to `http://localhost:5000` in your browser
2. Upload an image or take a photo of a plant
3. Select a detection method (Multi, Color, Texture, or Contour)
4. Process the image to detect plants
5. Identify the plant to get detailed species information

### Desktop Application

1. Launch the application
2. The camera will automatically start and begin detecting plants
3. Use the radio buttons to select different detection methods
4. When a plant is detected, it will be automatically tracked

## 🛠️ Development

### Project Structure

```
fytospot/
├── core/                         # Core functionality
│   ├── detection/                # Plant detection algorithms
│   │   ├── object_tracker.py     # Plant tracking implementation
│   │   └── plant_detector.py     # Plant detection implementations
│   ├── models/                   # Machine learning models
│   │   ├── plant_classifier.py   # ResNet-based classifier
│   │   ├── plant_identifier.py   # Plant identification logic
│   │   └── wrapper.py            # Model wrapper classes
│   └── data/                     # Data management
│       └── knowledge_base.py     # Plant information database
├── platforms/                    # Platform-specific code
│   ├── web/                      # Web application
│   │   ├── static/               # CSS, JS, and images
│   │   └── templates/            # HTML templates
│   └── desktop/                  # Desktop application
│       ├── gui/                  # Desktop UI components
│       └── utils/                # Desktop utilities
├── models/                       # Trained model files
│   ├── trained/                  # Trained models
│   └── configs/                  # Model configurations
└── assets/                       # Application assets
    ├── styles/                   # Style definitions
    └── images/                   # Images and icons
```

### Adding New Detection Methods

To add a new detection method, modify the `PlantDetector` class in `core/detection/plant_detector.py`:

```python
def your_new_method(self, frame):
    # Implement your detection algorithm
    # Return a binary mask of detected plants
    return mask
```

Then update the `detect_object` method to include your new method.

## 📊 Performance

- **Detection Speed:** 20-30 FPS on modern hardware
- **Tracking Stability:** Smooth tracking with temporal filtering
- **Identification Accuracy:** ~80% for common plant species


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
