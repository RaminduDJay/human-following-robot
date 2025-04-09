import numpy as np
import cv2
import picamera
import picamera.array
import serial
import time
import sys
from collections import deque

# Configuration Constants
SERIAL_PORT = '/dev/ttyUSB0'  # Adjust based on your system
BAUD_RATE = 9600

# Camera Configuration
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
FPS = 30

# Color Detection Parameters
LOWER_HSV = np.array([0, 100, 100])   # Red color lower bound
UPPER_HSV = np.array([10, 255, 255])  # Red color upper bound

# Movement Configuration
BASE_SPEED = 150
MAX_SPEED = 255
MIN_SPEED = 50
TURN_STRENGTH = 50

# Screen Segmentation
SECTION_WIDTH = FRAME_WIDTH // 4
MIDDLE_START = SECTION_WIDTH
MIDDLE_END = 3 * SECTION_WIDTH

# Smoothing Parameters
SMOOTHING_BUFFER_SIZE = 5
POSITION_SMOOTHING_FACTOR = 0.3

class SmoothTracker:
    def __init__(self, buffer_size=5, smoothing_factor=0.3):
        self.positions = deque(maxlen=buffer_size)
        self.smoothing_factor = smoothing_factor

    def update(self, new_position):
        if len(self.positions) == 0:
            self.positions.append(new_position)
            return new_position

        last_pos = self.positions[-1]
        smoothed_pos = (self.smoothing_factor * new_position + 
                        (1 - self.smoothing_factor) * last_pos)
        self.positions.append(smoothed_pos)
        return smoothed_pos

class MovementController:
    def __init__(self, base_speed=BASE_SPEED, turn_strength=TURN_STRENGTH):
        self.base_speed = base_speed
        self.turn_strength = turn_strength
        
    def get_turn_command(self, direction):
        if direction == 'LEFT':
            left_speed = max(0, self.base_speed - self.turn_strength)
            right_speed = min(255, self.base_speed + self.turn_strength)
            return f"TURN LEFT {left_speed} {right_speed}"
        elif direction == 'RIGHT':
            left_speed = min(255, self.base_speed + self.turn_strength)
            right_speed = max(0, self.base_speed - self.turn_strength)
            return f"TURN RIGHT {left_speed} {right_speed}"
        return "STOP"

class RobotVision:
    def __init__(self):
        self.position_tracker = SmoothTracker()
        self.movement_controller = MovementController()
        self.previous_command = None

    def process_image(self, frame):
        # Convert to HSV color space (PiCamera provides BGR format)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Apply adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        hsv[:,:,2] = clahe.apply(hsv[:,:,2])
        
        # Noise reduction
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        
        # Create color mask
        mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
        
        # Morphological operations
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask

    def detect_object(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) < 100:
            return None
        
        x, y, w, h = cv2.boundingRect(largest_contour)
        object_center_x = x + w // 2
        object_area = cv2.contourArea(largest_contour)
        
        return {
            'center_x': object_center_x, 
            'area': object_area
        }

    def determine_movement(self, object_info):
        if not object_info:
            return "STOP"
        
        center_x = self.position_tracker.update(object_info['center_x'])
        
        if center_x < SECTION_WIDTH:
            return self.movement_controller.get_turn_command('LEFT')
        elif center_x > 3 * SECTION_WIDTH:
            return self.movement_controller.get_turn_command('RIGHT')
        elif MIDDLE_START <= center_x <= MIDDLE_END:
            speed = int(np.interp(object_info['area'], [100, 5000], [MIN_SPEED, MAX_SPEED]))
            return f"MOVE {speed}"
        
        return "STOP"

class RobotController:
    def __init__(self, serial_port, baud_rate):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.arduino = None
        self.camera = None
        self.robot_vision = RobotVision()
        
    def initialize_serial(self):
        try:
            self.arduino = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
            time.sleep(2)
            print("Serial connection established.")
        except serial.SerialException as e:
            print(f"Serial connection error: {e}")
            sys.exit(1)
    
    def initialize_camera(self):
        try:
            self.camera = picamera.PiCamera()
            self.camera.resolution = (FRAME_WIDTH, FRAME_HEIGHT)
            self.camera.framerate = FPS
            print("Camera initialized successfully.")
        except picamera.PiCameraError as e:
            print(f"Camera initialization error: {e}")
            sys.exit(1)
    
    def send_command(self, command):
        try:
            if command != self.robot_vision.previous_command:
                print(f"Sending Command: {command}")
                self.arduino.write((command + "\n").encode())
                self.robot_vision.previous_command = command
        except serial.SerialException as e:
            print(f"Serial transmission error: {e}")
    
    def run(self):
        self.initialize_serial()
        self.initialize_camera()
        
        try:
            with picamera.array.PiRGBArray(self.camera, size=(FRAME_WIDTH, FRAME_HEIGHT)) as stream:
                print("Starting object tracking... Press Ctrl+C to stop.")
                
                for frame in self.camera.capture_continuous(stream, format="bgr", use_video_port=True):
                    image = np.array(frame.array)
                    
                    mask = self.robot_vision.process_image(image)
                    object_info = self.robot_vision.detect_object(mask)
                    
                    movement_command = self.robot_vision.determine_movement(object_info)
                    self.send_command(movement_command)
                    
                    # Optional debug display
                    cv2.imshow("Tracking", mask)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
                    stream.truncate(0)
        
        except KeyboardInterrupt:
            print("\nTracking stopped by user.")
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            if self.arduino:
                self.arduino.close()
            if self.camera:
                self.camera.close()
            cv2.destroyAllWindows()

def main():
    controller = RobotController(SERIAL_PORT, BAUD_RATE)
    controller.run()

if __name__ == "__main__":
    main()