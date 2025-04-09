import cv2
import numpy as np
import time

# Camera Configuration
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
FPS = 30

class Camera:
    def __init__(self, camera_id=0, width=None, height=None, fps=None):
        self.camera_id = camera_id
        self.width = width or FRAME_WIDTH
        self.height = height or FRAME_HEIGHT
        self.fps = fps or FPS
        self.cap = None
        
    def initialize(self):
        """Initialize webcam for laptop use"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                print("Error: Could not open webcam.")
                return None
                
            # Set properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Verify settings
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"Camera initialized successfully.")
            print(f"Resolution: {actual_width}x{actual_height}, FPS: {actual_fps}")
            return self.cap
        except Exception as e:
            print(f"Error: Could not initialize camera - {e}")
            return None
            
    def get_frame(self):
        """Capture a frame from the camera"""
        if self.cap is None:
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame.")
            return None
            
        # Convert BGR (OpenCV format) to RGB (which was PiCamera's format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb
        
    def release(self):
        """Release the camera"""
        if self.cap is not None:
            self.cap.release()
            
    def create_frame_container(self):
        """Create a wrapper class to mimic PiCamera's array interface"""
        return FrameContainer()
            
class FrameContainer:
    """Mimics PiCamera's PiRGBArray for compatibility"""
    def __init__(self):
        self.array = None
        
    def update(self, frame):
        self.array = frame
        
    def truncate(self, size):
        pass  # No need to implement for OpenCV




# import cv2

# def initialize_camera():
#     cap = cv2.VideoCapture(0)  # 0 for built-in webcam, change for USB cam
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
#     cap.set(cv2.CAP_PROP_FPS, 30)
#     print("Camera initialized successfully.")
#     return cap

# def capture_frame(camera):
#     ret, frame = camera.read()
#     if not ret:
#         print("Failed to capture frame")
#         return None
#     return frame






# import picamera

# FRAME_WIDTH = 320
# FRAME_HEIGHT = 240
# FPS = 30

# def initialize_camera():
#     try:
#         camera = picamera.PiCamera()
#         camera.resolution = (FRAME_WIDTH, FRAME_HEIGHT)
#         camera.framerate = FPS
#         print("Camera initialized successfully.")
#         return camera
#     except picamera.PiCameraError as e:
#         print(f"Error: Could not initialize camera - {e}")
#         sys.exit(1)
