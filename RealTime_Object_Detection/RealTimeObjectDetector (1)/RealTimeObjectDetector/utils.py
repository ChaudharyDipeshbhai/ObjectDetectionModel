"""
Utility functions for object detection system
"""
import cv2
import numpy as np
import time
from config import COLORS, FONT_SCALE, FONT_THICKNESS, BOX_THICKNESS

class FPSMonitor:
    def __init__(self):
        self.prev_time = 0
        self.curr_time = 0
        self.fps_filter = 0

    def update(self):
        self.curr_time = time.time()
        fps = 1 / (self.curr_time - self.prev_time) if self.prev_time > 0 else 0
        self.fps_filter = 0.95 * self.fps_filter + 0.05 * fps
        self.prev_time = self.curr_time
        return int(self.fps_filter)

def draw_detections(frame, results, class_names):
    """
    Draw bounding boxes and labels for detected objects
    """
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Get class and confidence
            conf = float(box.conf)
            cls = int(box.cls)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS['box'], BOX_THICKNESS)

            # Draw label
            label = f'{class_names[cls]} {conf:.2f}'
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)

            # Draw label background
            cv2.rectangle(frame, (x1, y1-label_height-10), 
                        (x1+label_width, y1), COLORS['box'], -1)

            # Draw label text
            cv2.putText(frame, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                       COLORS['text'], FONT_THICKNESS)

    return frame

def preprocess_frame(frame):
    """
    Preprocess frame for YOLO model
    """
    return cv2.resize(frame, (640, 640))