import numpy as np
import cv2
import picamera
import picamera.array
import serial
import time
import sys
from collections import deque

# Serial Port Configuration
SERIAL_PORT = '/dev/ttyUSB0'  # Adjust based on your system
BAUD_RATE = 9600

# Camera Configuration
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
FPS = 30

# Movement Configuration
BASE_SPEED = 150
MAX_SPEED = 255
MIN_SPEED = 50
TURN_STRENGTH = 50

# Object Color HSV Range (Initial Guess - Will Auto-Adjust)
LOWER_HSV = np.array([0, 100, 100])
UPPER_HSV = np.array([10, 255, 255])

# Define screen sections for movement decisions
SECTION_WIDTH = FRAME_WIDTH // 4
MIDDLE_START = SECTION_WIDTH
MIDDLE_END = 3 * SECTION_WIDTH

# Moving average filter for smoother tracking
SMOOTHING_BUFFER_SIZE = 5
object_positions = deque(maxlen=SMOOTHING_BUFFER_SIZE)

# PID Controller with Improved Smoothing
class SmoothPID:
    def _init_(self, Kp, Ki, Kd, setpoint=0, integral_limit=1000, smoothing_factor=0.2):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.last_error = 0
        self.integral = 0
        self.last_time = time.time()
        self.integral_limit = integral_limit
        self.smoothing_factor = smoothing_factor
        self.smoothed_output = 0

    def update(self, measured_value):
        now = time.time()
        dt = now - self.last_time
        if dt <= 0:
            dt = 1e-16  # Prevent division by zero
        
        error = self.setpoint - measured_value
        self.integral += error * dt
        
        # Anti-windup
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        
        derivative = (error - self.last_error) / dt
        
        # Calculate raw output
        raw_output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        
        # Apply exponential smoothing
        self.smoothed_output = (self.smoothing_factor * raw_output + 
                                (1 - self.smoothing_factor) * self.smoothed_output)
        
        self.last_error = error
        self.last_time = now
        
        return self.smoothed_output

# Smooth Movement Controller
class SmoothMovementController:
    def _init_(self, base_speed=BASE_SPEED, turn_strength=TURN_STRENGTH):
        self.base_speed = base_speed
        self.turn_strength = turn_strength
        self.current_left_speed = 0
        self.current_right_speed = 0
        self.acceleration_step = 5
        
    def calculate_turn_speeds(self, direction):
        if direction == 'LEFT':
            left_speed = max(0, self.base_speed - self.turn_strength)
            right_speed = min(MAX_SPEED, self.base_speed + self.turn_strength)
            return f"LEFT_MOVE {left_speed} {right_speed}"
        elif direction == 'RIGHT':
            left_speed = min(MAX_SPEED, self.base_speed + self.turn_strength)
            right_speed = max(0, self.base_speed - self.turn_strength)
            return f"RIGHT_MOVE {left_speed} {right_speed}"
        return "STOP"

# Initialize smooth movement controller
smooth_movement = SmoothMovementController()

# PID controller for distance/area control
pid_controller = SmoothPID(Kp=0.1, Ki=0.01, Kd=0.05, setpoint=2500)

# Initialize Serial Connection
def initialize_serial():
    try:
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Allow Arduino to initialize
        print("Serial connection established.")
        return arduino
    except serial.SerialException as e:
        print(f"Error: Could not connect to Arduino - {e}")
        sys.exit(1)

# Process Frame to Detect Object and Control Motor Speed
def process_frame(frame, arduino):
    movement = "STOP"

    try:
        # Convert frame to NumPy array and process image
        image = np.array(frame.array, dtype=np.uint8)
        mask = process_image(image)
        
        # Find object position and area
        indices = np.where(mask > 0)
        if len(indices[0]) > 0:
            x_min, x_max = min(indices[1]), max(indices[1])
            object_center_x = (x_min + x_max) // 2
            object_area = len(indices[0])

            # Determine turning and movement commands
            if object_center_x < SECTION_WIDTH:
                movement = smooth_movement.calculate_turn_speeds('LEFT')
            elif object_center_x > 3 * SECTION_WIDTH:
                movement = smooth_movement.calculate_turn_speeds('RIGHT')
            elif MIDDLE_START <= object_center_x <= MIDDLE_END:
                # PID-based speed control
                speed_output = pid_controller.update(object_area)
                speed = max(MIN_SPEED, min(int(abs(speed_output)), MAX_SPEED))
                movement = f"MOVE {speed}"

        # Send movement command to Arduino
        send_command(arduino, movement)

    except Exception as e:
        print(f"Error processing frame: {e}")

# Improved command sending with less frequent updates
def send_command(arduino, new_command):
    global previous_command
    try:
        # Only send if command has significantly changed
        if _command_changed(new_command, previous_command):
            print(f"Sending Command: {new_command}")
            arduino.write((new_command + "\n").encode())
            previous_command = new_command
    except serial.SerialException as e:
        print(f"Serial Write Error: {e}")

def _command_changed(new_command, old_command, threshold=10):
    if not old_command:
        return True
    
    # More sophisticated change detection for different command types
    if new_command.startswith(("MOVE", "LEFT_MOVE", "RIGHT_MOVE")):
        parts = new_command.split()
        old_parts = old_command.split()
        
        if len(parts) != len(old_parts):
            return True
        
        # Check if speeds have changed significantly
        for i in range(1, len(parts)):
            try:
                if abs(int(parts[i]) - int(old_parts[i])) > threshold:
                    return True
            except ValueError:
                return True
        return False
    
    return new_command != old_command

# Rest of the functions like process_image(), initialize_camera(), etc. 
# remain largely the same as in the previous implementation

# Initialize PiCamera
def initialize_camera():
    try:
        camera = picamera.PiCamera()
        camera.resolution = (FRAME_WIDTH, FRAME_HEIGHT)
        camera.framerate = FPS
        print("Camera initialized successfully.")
        return camera
    except picamera.PiCameraError as e:
        print(f"Error: Could not initialize camera - {e}")
        sys.exit(1)

# Apply smoothing filter to object position
def smooth_position(new_position):
    object_positions.append(new_position)
    return int(np.mean(object_positions))

# ----------------- Updated: Enhanced Adaptive Color Thresholding -----------------
def process_image(image):
    try:
        # Convert RGB to HSV for more robust color detection under varying lighting
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # Create a CLAHE object for adaptive histogram equalization on the V channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
        # Apply Gaussian Blur to reduce noise
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        # Create a binary mask based on the HSV range
        mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
        # Clean up the mask using morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask
    except Exception as e:
        print(f"Error in process_image: {e}")
        return None


# Main Program Execution
if _name_ == "_main_":
    try:
        arduino = initialize_serial()
        camera = initialize_camera()
        previous_command = None
        
        with picamera.array.PiRGBArray(camera, size=(FRAME_WIDTH, FRAME_HEIGHT)) as stream:
            print("Starting smooth object tracking... Press 'q' to quit.")
            for frame in camera.capture_continuous(stream, format="rgb", use_video_port=True):
                process_frame(frame, arduino)
                
                # Display camera view
                image = np.array(frame.array, dtype=np.uint8)
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imshow("Camera", image_bgr)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                stream.truncate(0)  # Clear stream for next frame
    
    except KeyboardInterrupt:
        print("\nUser stopped the program.")
    except Exception as e:
        print(f"Unexpected Error: {e}")
    finally:
        # Cleanup resources
        cv2.destroyAllWindows()
        arduino.close()
        camera.close()