import numpy as np
import picamera
import picamera.array
import serial
import time
import os
import logging
from collections import deque
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("object_tracker.log"), logging.StreamHandler()]
)
logger = logging.getLogger("ObjectTracker")

# Configuration settings
CONFIG = {
    "serial_port": "/dev/ttyUSB0",
    "baud_rate": 9600,
    "frame_width": 320,
    "frame_height": 240,
    "fps": 30,
    "camera_rotation": 0,
    "lower_color": np.array([200, 50, 50]),
    "upper_color": np.array([255, 150, 150]),
    "min_object_pixels": 100,
    "section_count": 5,
    "center_zone_width": 0.4,
    "stop_distance_threshold": 8000,
    "forward_distance_threshold": 4000,
    "turn_speed_factor": 1.5,
    "smoothing_buffer_size": 5,
    "position_smoothing_factor": 0.7,
    "area_smoothing_factor": 0.5,
    "lost_object_timeout": 1.0,
    "debug_mode": False,
    "save_debug_frames": False,
}

class ObjectTracker:
    def __init__(self, config):
        self.config = config
        self.arduino = None
        self.camera = None
        self.object_positions_x = deque(maxlen=config["smoothing_buffer_size"])
        self.object_areas = deque(maxlen=config["smoothing_buffer_size"])
        self.previous_command = None
        self.last_object_detection_time = 0
        self.frame_count = 0

        # Calculate section boundaries
        self.width = config["frame_width"]
        center_width = int(self.width * config["center_zone_width"])
        self.center_start = (self.width - center_width) // 2
        self.center_end = self.center_start + center_width

        # Initialize hardware
        self.initialize_serial()
        self.initialize_camera()

    def initialize_serial(self):
        """Initialize serial connection to Arduino"""
        try:
            self.arduino = serial.Serial(self.config["serial_port"], self.config["baud_rate"], timeout=1)
            time.sleep(2)
            self.send_command("STOP")
            logger.info("Serial connection established")
        except serial.SerialException as e:
            logger.error(f"Serial connection failed: {e}")

    def initialize_camera(self):
        """Initialize PiCamera"""
        try:
            self.camera = picamera.PiCamera()
            self.camera.resolution = (self.config["frame_width"], self.config["frame_height"])
            self.camera.framerate = self.config["fps"]
            self.camera.rotation = self.config["camera_rotation"]
            time.sleep(2)
            logger.info("Camera initialized successfully")
        except picamera.PiCameraError as e:
            logger.critical(f"Camera initialization failed: {e}")

    def send_command(self, command):
        """Send movement command to Arduino"""
        if command != self.previous_command:
            try:
                if self.arduino and self.arduino.is_open:
                    self.arduino.write((command + "\n").encode())
                    self.previous_command = command
                    logger.info(f"Sent command: {command}")
            except serial.SerialException as e:
                logger.error(f"Serial write error: {e}")

    def smooth_value(self, value, data_queue, factor):
        """Apply exponential smoothing"""
        data_queue.append(value)
        if len(data_queue) > 1:
            return int(data_queue[-2] * (1 - factor) + value * factor)
        return value

    def detect_object(self, image):
        """Detect an object based on color filtering (without OpenCV)"""
        try:
            mask = np.logical_and(
                np.all(image >= self.config["lower_color"], axis=-1),
                np.all(image <= self.config["upper_color"], axis=-1)
            )

            # Basic dilation (merge nearby pixels)
            kernel = np.ones((3, 3), dtype=bool)
            dilated_mask = np.zeros_like(mask)
            for i in range(1, mask.shape[0] - 1):
                for j in range(1, mask.shape[1] - 1):
                    if np.any(mask[i - 1:i + 2, j - 1:j + 2]):
                        dilated_mask[i, j] = True

            indices = np.where(dilated_mask)

            if len(indices[0]) >= self.config["min_object_pixels"]:
                y_min, y_max = np.min(indices[0]), np.max(indices[0])
                x_min, x_max = np.min(indices[1]), np.max(indices[1])
                center_x = (x_min + x_max) // 2
                object_area = len(indices[0])

                self.last_object_detection_time = time.time()
                return center_x, object_area, True

            return None, 0, False
        except Exception as e:
            logger.error(f"Object detection error: {e}")
            return None, 0, False

    def calculate_movement(self, center_x, area):
        """Determine movement command based on position and size"""
        center_x = self.smooth_value(center_x, self.object_positions_x, self.config["position_smoothing_factor"])
        area = self.smooth_value(area, self.object_areas, self.config["area_smoothing_factor"])

        if area > self.config["stop_distance_threshold"]:
            return "STOP"
        elif self.center_start <= center_x <= self.center_end:
            return "MOVE FORWARD" if area > self.config["forward_distance_threshold"] else "MOVE FORWARD FAST"
        else:
            return "TURN LEFT" if center_x < self.center_start else "TURN RIGHT"

    def check_object_lost(self):
        """Check if object has been lost for too long"""
        return time.time() - self.last_object_detection_time > self.config["lost_object_timeout"]

    def process_frame(self, frame):
        """Process a frame to detect object and control movement"""
        try:
            image = np.array(frame.array, dtype=np.uint8)
            center_x, area, detected = self.detect_object(image)

            if detected:
                command = self.calculate_movement(center_x, area)
                self.send_command(command)
            elif self.check_object_lost():
                self.send_command("STOP")

        except Exception as e:
            logger.error(f"Frame processing error: {e}")

    def run(self):
        """Main loop to capture and process frames"""
        with picamera.array.PiRGBArray(self.camera, size=(self.config["frame_width"], self.config["frame_height"])) as stream:
            while True:
                self.camera.capture(stream, format="rgb", use_video_port=True)
                self.process_frame(stream)
                stream.truncate(0)

if __name__ == "__main__":
    tracker = ObjectTracker(CONFIG)
    tracker.run()
