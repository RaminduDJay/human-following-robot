import numpy as np
import picamera
import picamera.array
import serial
import time
import sys
import os
import argparse
import json
import logging
from collections import deque
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("object_tracker.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ObjectTracker")

# Default configuration
DEFAULT_CONFIG = {
    # Serial settings
    "serial_port": "/dev/ttyUSB0",
    "baud_rate": 9600,
    "reconnect_attempts": 3,
    "reconnect_delay": 2,
    
    # Camera settings
    "frame_width": 320,
    "frame_height": 240,
    "fps": 30,
    "camera_rotation": 0,
    
    # Object detection settings
    "lower_color": [200, 50, 50],
    "upper_color": [255, 150, 150],
    "min_object_pixels": 100,  # Minimum number of pixels to consider as an object
    
    # Movement control settings
    "section_count": 5,  # How many horizontal sections to divide the frame into
    "center_zone_width": 0.4,  # Width of center zone as a fraction of total width
    "stop_distance_threshold": 8000,  # Object area threshold to stop
    "forward_distance_threshold": 4000,  # Object area threshold to move forward
    "turn_speed_factor": 1.5,  # Factor to adjust turn speed based on distance from center
    
    # Tracking settings
    "smoothing_buffer_size": 5,
    "position_smoothing_factor": 0.7,
    "area_smoothing_factor": 0.5,
    "lost_object_timeout": 1.0,  # Seconds before considering object lost
    
    # Performance settings
    "target_processing_time": 0.03,  # Target time per frame in seconds
    
    # Debugging settings
    "debug_mode": False,
    "save_debug_frames": False,
    "debug_frame_interval": 30,  # Save every Nth frame
}

class ObjectTracker:
    def __init__(self, config=None):
        # Load configuration
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
            
        # Initialize variables
        self.arduino = None
        self.camera = None
        self.object_positions_x = deque(maxlen=self.config["smoothing_buffer_size"])
        self.object_areas = deque(maxlen=self.config["smoothing_buffer_size"])
        self.previous_command = None
        self.previous_command_time = 0
        self.last_object_detection_time = 0
        self.frame_count = 0
        self.processing_times = deque(maxlen=30)  # Store last 30 processing times
        self.debug_frames_dir = "debug_frames"
        
        # Convert color thresholds to numpy arrays
        self.lower_color = np.array(self.config["lower_color"])
        self.upper_color = np.array(self.config["upper_color"])
        
        # Calculate section boundaries
        self.calculate_sections()
        
        # Create debug directory if needed
        if self.config["save_debug_frames"]:
            os.makedirs(self.debug_frames_dir, exist_ok=True)
        
        # Initialize hardware
        self.initialize_serial()
        self.initialize_camera()
    
    def calculate_sections(self):
        """Calculate frame sections for movement decisions"""
        width = self.config["frame_width"]
        self.section_width = width // self.config["section_count"]
        center_width = int(width * self.config["center_zone_width"])
        self.center_start = (width - center_width) // 2
        self.center_end = self.center_start + center_width
        
        logger.debug(f"Frame sections: width={width}, center_zone={self.center_start}-{self.center_end}")
    
    def initialize_serial(self):
        """Initialize serial connection with retry logic"""
        attempts = 0
        while attempts < self.config["reconnect_attempts"]:
            try:
                self.arduino = serial.Serial(
                    self.config["serial_port"], 
                    self.config["baud_rate"], 
                    timeout=1
                )
                time.sleep(2)  # Allow Arduino to initialize
                self.send_command("STOP")  # Initialize Arduino state
                logger.info("Serial connection established")
                return True
            except serial.SerialException as e:
                attempts += 1
                logger.error(f"Serial connection attempt {attempts} failed: {e}")
                if attempts >= self.config["reconnect_attempts"]:
                    logger.critical("Could not connect to Arduino after multiple attempts")
                    return False
                time.sleep(self.config["reconnect_delay"])
        return False
    
    def reconnect_serial(self):
        """Attempt to reconnect to serial port"""
        logger.warning("Attempting to reconnect to Arduino...")
        try:
            if self.arduino:
                self.arduino.close()
            return self.initialize_serial()
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            return False
    
    def initialize_camera(self):
        """Initialize PiCamera with settings from config"""
        try:
            self.camera = picamera.PiCamera()
            self.camera.resolution = (self.config["frame_width"], self.config["frame_height"])
            self.camera.framerate = self.config["fps"]
            self.camera.rotation = self.config["camera_rotation"]
            
            # Allow camera to warm up
            time.sleep(2)
            logger.info("Camera initialized successfully")
            return True
        except picamera.PiCameraError as e:
            logger.critical(f"Camera initialization failed: {e}")
            return False
    
    def send_command(self, command, force=False):
        """Send command to Arduino with error handling"""
        current_time = time.time()
        
        # Only send command if it's different from previous or forced
        if force or command != self.previous_command:
            try:
                if self.arduino and self.arduino.is_open:
                    logger.info(f"Sending command: {command}")
                    self.arduino.write((command + "\n").encode())
                    self.previous_command = command
                    self.previous_command_time = current_time
                    return True
                else:
                    logger.warning("Cannot send command - serial connection closed")
                    if self.reconnect_serial():
                        # Try again after reconnection
                        self.arduino.write((command + "\n").encode())
                        self.previous_command = command
                        self.previous_command_time = current_time
                        return True
                    return False
            except serial.SerialException as e:
                logger.error(f"Serial write error: {e}")
                self.reconnect_serial()
                return False
        return True
    
    def smooth_value(self, value, data_queue, smoothing_factor):
        """Apply exponential smoothing to a value"""
        data_queue.append(value)
        if len(data_queue) > 1:
            # Use exponential smoothing
            avg = data_queue[-2] * (1 - smoothing_factor) + value * smoothing_factor
            return int(avg)
        return value
    
    def detect_object(self, image):
        """Detect object using color thresholds and return position and size"""
        try:
            # Create color mask using NumPy operations
            mask = np.logical_and(
                np.all(image >= self.lower_color, axis=-1),
                np.all(image <= self.upper_color, axis=-1)
            )
            
            # Apply simple noise reduction
            # Apply 3x3 dilation using NumPy to help merge nearby pixels
            kernel = np.ones((3, 3), dtype=bool)
            # Manual dilation without scipy or OpenCV
            dilated_mask = np.zeros_like(mask)
            for i in range(1, mask.shape[0]-1):
                for j in range(1, mask.shape[1]-1):
                    if np.any(mask[i-1:i+2, j-1:j+2]):
                        dilated_mask[i, j] = True
            
            # Find object pixels
            indices = np.where(dilated_mask)
            
            # Only if we have enough pixels to consider an object
            if len(indices[0]) >= self.config["min_object_pixels"]:
                # Get bounding box
                y_min, y_max = np.min(indices[0]), np.max(indices[0])
                x_min, x_max = np.min(indices[1]), np.max(indices[1])
                center_x = (x_min + x_max) // 2
                object_area = len(indices[0])  # Area as pixel count
                
                # Update detection time
                self.last_object_detection_time = time.time()
                
                # Save debug frame if enabled
                if self.config["save_debug_frames"] and self.frame_count % self.config["debug_frame_interval"] == 0:
                    self.save_debug_frame(image, x_min, y_min, x_max, y_max, center_x, object_area)
                
                return center_x, object_area, True
            
            return None, 0, False
        
        except Exception as e:
            logger.error(f"Error in object detection: {e}")
            return None, 0, False
    
    def save_debug_frame(self, image, x_min, y_min, x_max, y_max, center_x, area):
        """Save a debug frame with object visualization"""
        try:
            # Create a copy of the image
            debug_image = image.copy()
            
            # Draw rectangle around object (simple version without OpenCV)
            # Top and bottom horizontal lines
            for x in range(max(0, x_min), min(debug_image.shape[1], x_max + 1)):
                debug_image[max(0, y_min), x] = [0, 255, 0]  # Top line
                debug_image[min(debug_image.shape[0] - 1, y_max), x] = [0, 255, 0]  # Bottom line
                
            # Left and right vertical lines
            for y in range(max(0, y_min), min(debug_image.shape[0], y_max + 1)):
                debug_image[y, max(0, x_min)] = [0, 255, 0]  # Left line
                debug_image[y, min(debug_image.shape[1] - 1, x_max)] = [0, 255, 0]  # Right line
            
            # Mark center with a cross
            cross_size = 5
            for offset in range(-cross_size, cross_size + 1):
                cx, cy = center_x, (y_min + y_max) // 2
                if 0 <= cx < debug_image.shape[1] and 0 <= cy + offset < debug_image.shape[0]:
                    debug_image[cy + offset, cx] = [0, 0, 255]  # Vertical line
                if 0 <= cx + offset < debug_image.shape[1] and 0 <= cy < debug_image.shape[0]:
                    debug_image[cy, cx + offset] = [0, 0, 255]  # Horizontal line
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            # Save as raw numpy array
            filename = f"{self.debug_frames_dir}/frame_{timestamp}.npy"
            np.save(filename, debug_image)
            
            logger.debug(f"Saved debug frame: {filename}")
            
            # Create a simple text description file
            with open(f"{self.debug_frames_dir}/frame_{timestamp}.txt", "w") as f:
                f.write(f"Object detected at: ({center_x}, {(y_min + y_max) // 2})\n")
                f.write(f"Bounding box: ({x_min}, {y_min}) to ({x_max}, {y_max})\n")
                f.write(f"Area: {area} pixels\n")
                f.write(f"Command: {self.previous_command}\n")
                
        except Exception as e:
            logger.error(f"Error saving debug frame: {e}")
    
    def calculate_movement(self, center_x, area):
        """Determine the movement command based on object position and size"""
        width = self.config["frame_width"]
        
        # Smooth position and area
        center_x = self.smooth_value(
            center_x, 
            self.object_positions_x, 
            self.config["position_smoothing_factor"]
        )
        
        area = self.smooth_value(
            area, 
            self.object_areas, 
            self.config["area_smoothing_factor"]
        )
        
        # Calculate distance from center as a percentage (-1.0 to 1.0)
        # where 0 is center, -1 is far left, 1 is far right
        center_offset = (center_x - (width / 2)) / (width / 2)
        
        # Determine command based on position and area
        if area > self.config["stop_distance_threshold"]:
            # Object is too close, stop
            return "STOP"
        
        elif self.center_start <= center_x <= self.center_end:
            # Object is in center zone
            if area > self.config["forward_distance_threshold"]:
                return "MOVE FORWARD"
            else:
                # Object is in center but too far, need to move faster
                return "MOVE FORWARD"
        
        else:
            # Object is not in center, need to turn
            # Calculate turn intensity based on distance from center
            turn_intensity = abs(center_offset) * self.config["turn_speed_factor"]
            
            if center_x < self.center_start:
                # Object is to the left
                return "TURN LEFT"
            else:
                # Object is to the right
                return "TURN RIGHT"
    
    def check_object_lost(self):
        """Check if the object has been lost for too long"""
        if time.time() - self.last_object_detection_time > self.config["lost_object_timeout"]:
            logger.warning("Object lost - stopping robot")
            return True
        return False
    
    def process_frame(self, frame):
        """Process a single frame to track object and control robot"""
        start_time = time.time()
        
        try:
            # Convert frame to numpy array
            image = np.array(frame.array, dtype=np.uint8)
            
            # Detect object
            center_x, area, detected = self.detect_object(image)
            
            if detected:
                # Calculate movement command
                command = self.calculate_movement(center_x, area)
                self.send_command(command)
            else:
                # Check if object has been lost for too long
                if self.check_object_lost() and self.previous_command != "STOP":
                    self.send_command("STOP")
            
            # Track performance
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Log performance data periodically
            if self.frame_count % 100 == 0:
                avg_time = sum(self.processing_times) / len(self.processing_times)
                fps = 1.0 / avg_time if avg_time > 0 else 0
                logger.info(f"Performance: {fps:.1f} FPS, {avg_time*1000:.1f}ms per frame")
            
            self.frame_count += 1
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
    
    def run(self):
        """Main run loop for object tracking"""
        try:
            with picamera.array.PiRGBArray(
                self.camera, 
                size=(self.config["frame_width"], self.config["frame_height"])
            ) as stream:
                logger.info("Starting object tracking... Press Ctrl+C to stop.")
                
                # Capture continuous frames
                for frame in self.camera.capture_continuous(
                    stream, format="rgb", use_video_port=True
                ):
                    self.process_frame(frame)
                    stream.truncate(0)  # Clear stream for next frame
                    
                    # Adaptively sleep to maintain desired framerate
                    if self.processing_times and len(self.processing_times) > 5:
                        avg_time = sum(self.processing_times) / len(self.processing_times)
                        sleep_time = max(0, self.config["target_processing_time"] - avg_time)
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            logger.info("User stopped the program")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources...")
        
        # Stop the robot
        self.send_command("STOP", force=True)
        
        # Close camera
        if self.camera:
            try:
                self.camera.close()
                logger.info("Camera closed")
            except Exception as e:
                logger.error(f"Error closing camera: {e}")
        
        # Close serial connection
        if self.arduino:
            try:
                self.arduino.close()
                logger.info("Serial connection closed")
            except Exception as e:
                logger.error(f"Error closing serial connection: {e}")
        
        logger.info("Cleanup complete")

def load_config(config_file):
    """Load configuration from JSON file"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_file}")
        return config
    except Exception as e:
        logger.warning(f"Could not load config file: {e}")
        logger.info("Using default configuration")
        return {}

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Object Tracking Robot Control")
    parser.add_argument(
        "--config", 
        help="Path to configuration JSON file",
        default="tracker_config.json"
    )
    parser.add_argument(
        "--debug", 
        help="Enable debug mode", 
        action="store_true"
    )
    parser.add_argument(
        "--port", 
        help="Serial port for Arduino",
        default=None
    )
    parser.add_argument(
        "--save-frames", 
        help="Save debug frames", 
        action="store_true"
    )
    
    return parser.parse_args()

def convert_npy_to_png():
    """Utility function to convert saved numpy arrays to PNG images (for later viewing)"""
    try:
        import matplotlib.pyplot as plt
        
        logger.info("Converting NPY debug frames to PNG images...")
        
        debug_dir = "debug_frames"
        if not os.path.exists(debug_dir):
            logger.warning("Debug frames directory not found")
            return
            
        npy_files = [f for f in os.listdir(debug_dir) if f.endswith('.npy')]
        
        for npy_file in npy_files:
            try:
                # Load the numpy array
                img_array = np.load(os.path.join(debug_dir, npy_file))
                
                # Convert to PNG using matplotlib
                png_path = os.path.join(debug_dir, npy_file.replace('.npy', '.png'))
                plt.imsave(png_path, img_array)
                
                logger.debug(f"Converted {npy_file} to PNG")
                
            except Exception as e:
                logger.error(f"Error converting {npy_file}: {e}")
                
        logger.info(f"Converted {len(npy_files)} NPY files to PNG")
        
    except ImportError:
        logger.warning("Matplotlib not available for image conversion")

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Set logging level based on debug flag
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.port:
        config["serial_port"] = args.port
    if args.debug:
        config["debug_mode"] = True
    if args.save_frames:
        config["save_debug_frames"] = True
    
    # Create and run tracker
    tracker = ObjectTracker(config)
    tracker.run()
    
    # Convert any saved debug frames to viewable images
    if config["save_debug_frames"]:
        convert_npy_to_png()




# import numpy as np
# import picamera
# import picamera.array
# import serial
# import time
# import sys
# import os
# import cv2
# import argparse
# import json
# import logging
# from collections import deque
# from datetime import datetime

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("object_tracker.log"),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger("ObjectTracker")

# # Default configuration
# DEFAULT_CONFIG = {
#     # Serial settings
#     "serial_port": "/dev/ttyUSB0",
#     "baud_rate": 9600,
#     "reconnect_attempts": 3,
#     "reconnect_delay": 2,
    
#     # Camera settings
#     "frame_width": 320,
#     "frame_height": 240,
#     "fps": 30,
#     "camera_rotation": 0,
    
#     # Object detection settings
#     "lower_color": [200, 50, 50],
#     "upper_color": [255, 150, 150],
#     "min_object_area": 500,
#     "max_object_area": 15000,
    
#     # Movement control settings
#     "section_count": 5,  # How many horizontal sections to divide the frame into
#     "center_zone_width": 0.4,  # Width of center zone as a fraction of total width
#     "stop_distance_threshold": 8000,  # Object area threshold to stop
#     "forward_distance_threshold": 4000,  # Object area threshold to move forward
#     "turn_speed_factor": 1.5,  # Factor to adjust turn speed based on distance from center
    
#     # Tracking settings
#     "smoothing_buffer_size": 5,
#     "position_smoothing_factor": 0.7,
#     "area_smoothing_factor": 0.5,
#     "lost_object_timeout": 1.0,  # Seconds before considering object lost
    
#     # Performance settings
#     "target_processing_time": 0.03,  # Target time per frame in seconds
    
#     # Debugging settings
#     "debug_mode": False,
#     "save_debug_frames": False,
#     "debug_frame_interval": 30,  # Save every Nth frame
# }

# class ObjectTracker:
#     def __init__(self, config=None):
#         # Load configuration
#         self.config = DEFAULT_CONFIG.copy()
#         if config:
#             self.config.update(config)
            
#         # Initialize variables
#         self.arduino = None
#         self.camera = None
#         self.object_positions_x = deque(maxlen=self.config["smoothing_buffer_size"])
#         self.object_areas = deque(maxlen=self.config["smoothing_buffer_size"])
#         self.previous_command = None
#         self.previous_command_time = 0
#         self.last_object_detection_time = 0
#         self.frame_count = 0
#         self.processing_times = deque(maxlen=30)  # Store last 30 processing times
#         self.debug_frames_dir = "debug_frames"
        
#         # Convert color thresholds to numpy arrays
#         self.lower_color = np.array(self.config["lower_color"])
#         self.upper_color = np.array(self.config["upper_color"])
        
#         # Calculate section boundaries
#         self.calculate_sections()
        
#         # Create debug directory if needed
#         if self.config["save_debug_frames"]:
#             os.makedirs(self.debug_frames_dir, exist_ok=True)
        
#         # Initialize hardware
#         self.initialize_serial()
#         self.initialize_camera()
    
#     def calculate_sections(self):
#         """Calculate frame sections for movement decisions"""
#         width = self.config["frame_width"]
#         self.section_width = width // self.config["section_count"]
#         center_width = int(width * self.config["center_zone_width"])
#         self.center_start = (width - center_width) // 2
#         self.center_end = self.center_start + center_width
        
#         logger.debug(f"Frame sections: width={width}, center_zone={self.center_start}-{self.center_end}")
    
#     def initialize_serial(self):
#         """Initialize serial connection with retry logic"""
#         attempts = 0
#         while attempts < self.config["reconnect_attempts"]:
#             try:
#                 self.arduino = serial.Serial(
#                     self.config["serial_port"], 
#                     self.config["baud_rate"], 
#                     timeout=1
#                 )
#                 time.sleep(2)  # Allow Arduino to initialize
#                 self.send_command("STOP")  # Initialize Arduino state
#                 logger.info("Serial connection established")
#                 return True
#             except serial.SerialException as e:
#                 attempts += 1
#                 logger.error(f"Serial connection attempt {attempts} failed: {e}")
#                 if attempts >= self.config["reconnect_attempts"]:
#                     logger.critical("Could not connect to Arduino after multiple attempts")
#                     return False
#                 time.sleep(self.config["reconnect_delay"])
#         return False
    
#     def reconnect_serial(self):
#         """Attempt to reconnect to serial port"""
#         logger.warning("Attempting to reconnect to Arduino...")
#         try:
#             if self.arduino:
#                 self.arduino.close()
#             return self.initialize_serial()
#         except Exception as e:
#             logger.error(f"Reconnection failed: {e}")
#             return False
    
#     def initialize_camera(self):
#         """Initialize PiCamera with settings from config"""
#         try:
#             self.camera = picamera.PiCamera()
#             self.camera.resolution = (self.config["frame_width"], self.config["frame_height"])
#             self.camera.framerate = self.config["fps"]
#             self.camera.rotation = self.config["camera_rotation"]
            
#             # Allow camera to warm up
#             time.sleep(2)
#             logger.info("Camera initialized successfully")
#             return True
#         except picamera.PiCameraError as e:
#             logger.critical(f"Camera initialization failed: {e}")
#             return False
    
#     def send_command(self, command, force=False):
#         """Send command to Arduino with error handling"""
#         current_time = time.time()
        
#         # Only send command if it's different from previous or forced
#         if force or command != self.previous_command:
#             try:
#                 if self.arduino and self.arduino.is_open:
#                     logger.info(f"Sending command: {command}")
#                     self.arduino.write((command + "\n").encode())
#                     self.previous_command = command
#                     self.previous_command_time = current_time
#                     return True
#                 else:
#                     logger.warning("Cannot send command - serial connection closed")
#                     if self.reconnect_serial():
#                         # Try again after reconnection
#                         self.arduino.write((command + "\n").encode())
#                         self.previous_command = command
#                         self.previous_command_time = current_time
#                         return True
#                     return False
#             except serial.SerialException as e:
#                 logger.error(f"Serial write error: {e}")
#                 self.reconnect_serial()
#                 return False
#         return True
    
#     def smooth_value(self, value, data_queue, smoothing_factor):
#         """Apply exponential smoothing to a value"""
#         data_queue.append(value)
#         if len(data_queue) > 1:
#             # Use exponential smoothing
#             avg = data_queue[-2] * (1 - smoothing_factor) + value * smoothing_factor
#             return int(avg)
#         return value
    
#     def adaptive_color_threshold(self, frame):
#         """Dynamically adjust color thresholds based on lighting conditions"""
#         # This is a placeholder for future implementation
#         # For now, just use the static thresholds from config
#         return self.lower_color, self.upper_color
    
#     def detect_object(self, image):
#         """Detect object using color thresholds and return position and size"""
#         try:
#             # Adjust thresholds based on current frame if needed
#             lower_color, upper_color = self.adaptive_color_threshold(image)
            
#             # Apply color masking
#             mask = cv2.inRange(image, lower_color, upper_color)
            
#             # Apply morphological operations to reduce noise
#             kernel = np.ones((5, 5), np.uint8)
#             mask = cv2.erode(mask, kernel, iterations=1)
#             mask = cv2.dilate(mask, kernel, iterations=2)
            
#             # Find contours
#             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
#             if contours:
#                 # Find the largest contour
#                 largest_contour = max(contours, key=cv2.contourArea)
#                 area = cv2.contourArea(largest_contour)
                
#                 # Only process contours above minimum size
#                 if area >= self.config["min_object_area"]:
#                     # Get bounding box
#                     x, y, w, h = cv2.boundingRect(largest_contour)
#                     center_x = x + w // 2
                    
#                     # Update detection time
#                     self.last_object_detection_time = time.time()
                    
#                     # Save debugging image if enabled
#                     if self.config["save_debug_frames"] and self.frame_count % self.config["debug_frame_interval"] == 0:
#                         debug_image = image.copy()
#                         cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
#                         cv2.circle(debug_image, (center_x, y+h//2), 5, (0, 0, 255), -1)
#                         cv2.putText(debug_image, f"Area: {area}", (x, y-10), 
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#                         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
#                         cv2.imwrite(f"{self.debug_frames_dir}/frame_{timestamp}.jpg", debug_image)
                    
#                     return center_x, area, True
            
#             return None, 0, False
            
#         except Exception as e:
#             logger.error(f"Error in object detection: {e}")
#             return None, 0, False
    
#     def calculate_movement(self, center_x, area):
#         """Determine the movement command based on object position and size"""
#         width = self.config["frame_width"]
        
#         # Smooth position and area
#         center_x = self.smooth_value(
#             center_x, 
#             self.object_positions_x, 
#             self.config["position_smoothing_factor"]
#         )
        
#         area = self.smooth_value(
#             area, 
#             self.object_areas, 
#             self.config["area_smoothing_factor"]
#         )
        
#         # Calculate distance from center as a percentage (-1.0 to 1.0)
#         # where 0 is center, -1 is far left, 1 is far right
#         center_offset = (center_x - (width / 2)) / (width / 2)
        
#         # Determine command based on position and area
#         if area > self.config["stop_distance_threshold"]:
#             # Object is too close, stop
#             return "STOP"
        
#         elif self.center_start <= center_x <= self.center_end:
#             # Object is in center zone
#             if area > self.config["forward_distance_threshold"]:
#                 return "MOVE FORWARD"
#             else:
#                 # Object is in center but too far, need to move faster
#                 return "MOVE FORWARD"
        
#         else:
#             # Object is not in center, need to turn
#             # Calculate turn intensity based on distance from center
#             turn_intensity = abs(center_offset) * self.config["turn_speed_factor"]
            
#             if center_x < self.center_start:
#                 # Object is to the left
#                 if turn_intensity > 0.7:
#                     return "TURN LEFT FAST"
#                 else:
#                     return "TURN LEFT"
#             else:
#                 # Object is to the right
#                 if turn_intensity > 0.7:
#                     return "TURN RIGHT FAST"
#                 else:
#                     return "TURN RIGHT"
    
#     def check_object_lost(self):
#         """Check if the object has been lost for too long"""
#         if time.time() - self.last_object_detection_time > self.config["lost_object_timeout"]:
#             logger.warning("Object lost - stopping robot")
#             return True
#         return False
    
#     def process_frame(self, frame):
#         """Process a single frame to track object and control robot"""
#         start_time = time.time()
        
#         try:
#             # Convert frame to numpy array
#             image = np.array(frame.array, dtype=np.uint8)
            
#             # Detect object
#             center_x, area, detected = self.detect_object(image)
            
#             if detected:
#                 # Calculate movement command
#                 command = self.calculate_movement(center_x, area)
#                 self.send_command(command)
#             else:
#                 # Check if object has been lost for too long
#                 if self.check_object_lost() and self.previous_command != "STOP":
#                     self.send_command("STOP")
            
#             # Track performance
#             processing_time = time.time() - start_time
#             self.processing_times.append(processing_time)
            
#             # Log performance data periodically
#             if self.frame_count % 100 == 0:
#                 avg_time = sum(self.processing_times) / len(self.processing_times)
#                 fps = 1.0 / avg_time if avg_time > 0 else 0
#                 logger.info(f"Performance: {fps:.1f} FPS, {avg_time*1000:.1f}ms per frame")
            
#             self.frame_count += 1
            
#         except Exception as e:
#             logger.error(f"Error processing frame: {e}")
    
#     def run(self):
#         """Main run loop for object tracking"""
#         try:
#             with picamera.array.PiRGBArray(
#                 self.camera, 
#                 size=(self.config["frame_width"], self.config["frame_height"])
#             ) as stream:
#                 logger.info("Starting object tracking... Press Ctrl+C to stop.")
                
#                 # Capture continuous frames
#                 for frame in self.camera.capture_continuous(
#                     stream, format="bgr", use_video_port=True
#                 ):
#                     self.process_frame(frame)
#                     stream.truncate(0)  # Clear stream for next frame
                    
#                     # Adaptively sleep to maintain desired framerate
#                     if self.processing_times and len(self.processing_times) > 5:
#                         avg_time = sum(self.processing_times) / len(self.processing_times)
#                         sleep_time = max(0, self.config["target_processing_time"] - avg_time)
#                         if sleep_time > 0:
#                             time.sleep(sleep_time)
                    
#         except KeyboardInterrupt:
#             logger.info("User stopped the program")
#         except Exception as e:
#             logger.error(f"Unexpected error: {e}")
#         finally:
#             self.cleanup()
    
#     def cleanup(self):
#         """Clean up resources"""
#         logger.info("Cleaning up resources...")
        
#         # Stop the robot
#         self.send_command("STOP", force=True)
        
#         # Close camera
#         if self.camera:
#             try:
#                 self.camera.close()
#                 logger.info("Camera closed")
#             except Exception as e:
#                 logger.error(f"Error closing camera: {e}")
        
#         # Close serial connection
#         if self.arduino:
#             try:
#                 self.arduino.close()
#                 logger.info("Serial connection closed")
#             except Exception as e:
#                 logger.error(f"Error closing serial connection: {e}")
        
#         logger.info("Cleanup complete")

# def load_config(config_file):
#     """Load configuration from JSON file"""
#     try:
#         with open(config_file, 'r') as f:
#             config = json.load(f)
#         logger.info(f"Loaded configuration from {config_file}")
#         return config
#     except Exception as e:
#         logger.warning(f"Could not load config file: {e}")
#         logger.info("Using default configuration")
#         return {}

# def parse_arguments():
#     """Parse command line arguments"""
#     parser = argparse.ArgumentParser(description="Object Tracking Robot Control")
#     parser.add_argument(
#         "--config", 
#         help="Path to configuration JSON file",
#         default="tracker_config.json"
#     )
#     parser.add_argument(
#         "--debug", 
#         help="Enable debug mode", 
#         action="store_true"
#     )
#     parser.add_argument(
#         "--port", 
#         help="Serial port for Arduino",
#         default=None
#     )
#     parser.add_argument(
#         "--save-frames", 
#         help="Save debug frames", 
#         action="store_true"
#     )
    
#     return parser.parse_args()

# if __name__ == "__main__":
#     # Parse command line arguments
#     args = parse_arguments()
    
#     # Set logging level based on debug flag
#     if args.debug:
#         logger.setLevel(logging.DEBUG)
#         logger.debug("Debug mode enabled")
    
#     # Load configuration
#     config = load_config(args.config)
    
#     # Override config with command line arguments
#     if args.port:
#         config["serial_port"] = args.port
#     if args.debug:
#         config["debug_mode"] = True
#     if args.save_frames:
#         config["save_debug_frames"] = True
    
#     # Create and run tracker
#     tracker = ObjectTracker(config)
#     tracker.run()