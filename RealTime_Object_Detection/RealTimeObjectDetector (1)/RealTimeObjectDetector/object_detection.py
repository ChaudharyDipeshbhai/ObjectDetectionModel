import cv2
import torch
from ultralytics import YOLO
import numpy as np
from utils import FPSMonitor, draw_detections, preprocess_frame
from config import *
import sys
# from yolov8 import YOLOv8

class ObjectDetector:
    def __init__(self):
        self.setup_camera()
        self.setup_model()
        self.fps_monitor = FPSMonitor()

    def setup_camera(self):
        """Initialize video capture"""
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    def setup_model(self):
        """Initialize YOLOv5 model"""
        try:
            self.model = YOLO('yolov8n.pt') 
            self.model.conf = CONFIDENCE_THRESHOLD
            self.model.iou = NMS_THRESHOLD
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLOv8 model: {str(e)}")

    def process_frame(self, frame):
        """Process a single frame"""
        # Run detection
        results = self.model(frame, stream=True, classes = [0,1,2,3,5,7,9,39,40,41]) #wineGlass, Cup, Bottle, trafficLight,Truck, bus, motorcycle, car, bicycle, person

        # Draw detections
        frame = draw_detections(frame, results, self.model.names)

        # Add FPS counter
        fps = self.fps_monitor.update()
        cv2.putText(frame, f"FPS: {fps}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                    COLORS['fps'], FONT_THICKNESS)

        return frame

    def run(self):
        """Main detection loop"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Process frame
                processed_frame = self.process_frame(frame)

                # Display result
                cv2.imshow(WINDOW_NAME, processed_frame)

                # Break on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("Interrupted by user")
        except Exception as e:
            print(f"Error occurred: {str(e)}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Release resources"""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        detector = ObjectDetector()
        detector.run()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)