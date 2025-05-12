"""
Configuration settings for the object detection system
"""

# Video capture settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# YOLOv5 settings
CONFIDENCE_THRESHOLD = 0.45
NMS_THRESHOLD = 0.45
YOLO_WEIGHTS = "yolov5s.pt"  # Will be downloaded automatically

# Display settings
WINDOW_NAME = "Real-time Object Detection"
FONT_SCALE = 0.6
FONT_THICKNESS = 2
BOX_THICKNESS = 2

# Colors for visualization (BGR format)
COLORS = {
    'box': (0, 255, 0),      # Green
    'text': (255, 255, 255), # White
    'fps': (0, 0, 255)       # Red
}
