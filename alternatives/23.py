import numpy as np
import cv2
import picamera
import picamera.array
import serial
import time
import sys
from collections import deque

# Serial Configuration
SERIAL_PORT = "/dev/ttyUSB0"  # Change as needed
BAUD_RATE = 9600

# Camera Configuration
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
FPS = 30

# HSV Color Range for Object Tracking
LOWER_HSV = np.array([0, 100, 100])
UPPER_HSV = np.array([10, 255, 255])

# Screen Section Divisions for Movement
SECTION_WIDTH = FRAME_WIDTH // 4
MIDDLE_START = SECTION_WIDTH
MIDDLE_END = 3 * SECTION_WIDTH

# Smoothing Buffer for Position Stability
SMOOTHING_BUFFER_SIZE = 5
object_positions = deque(maxlen=SMOOTHING_BUFFER_SIZE)

# PID Constants and Speed Control
DESIRED_AREA = 2500  # Target object area
MAX_SPEED = 255
MIN_SPEED = 50

previous_command = None

# PID Controller Class
class PID:
    def __init__(self, Kp, Ki, Kd, setpoint=0, integral_limit=1000):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.last_error = 0
        self.integral = 0
        self.last_time = time.time()
        self.integral_limit = integral_limit

    def update(self, measured_value):
        now = time.time()
        dt = now - self.last_time
        dt = max(dt, 1e-16)  # Avoid division by zero

        error = self.setpoint - measured_value
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)

        derivative = (error - self.last_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        self.last_error = error
        self.last_time = now
        return output

pid_controller = PID(Kp=0.1, Ki=0.01, Kd=0.05, setpoint=DESIRED_AREA)

# Serial Connection
def initialize_serial():
    try:
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print("Serial connection established.")
        return arduino
    except serial.SerialException as e:
        print(f"Error: Could not connect to Arduino - {e}")
        sys.exit(1)

# PiCamera Initialization
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

# Process Image for Object Tracking
def process_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# Smooth Position Tracking
def smooth_position(new_position):
    object_positions.append(new_position)
    return int(np.mean(object_positions))

# Process Camera Frames
def process_frame(frame, arduino):
    global previous_command
    movement = "STOP"

    image = np.array(frame.array, dtype=np.uint8)
    mask = process_image(image)

    indices = np.where(mask > 0)
    if len(indices[0]) > 0:
        x_min, x_max = min(indices[1]), max(indices[1])
        object_center_x = (x_min + x_max) // 2
        object_area = len(indices[0])

        object_center_x = smooth_position(object_center_x)

        turn_strength = 50
        base_speed = pid_controller.update(object_area)
        base_speed = max(MIN_SPEED, min(base_speed, MAX_SPEED))

        if object_center_x < SECTION_WIDTH:
            movement = f"CURVE LEFT {int(base_speed)} {turn_strength}"
        elif object_center_x > 3 * SECTION_WIDTH:
            movement = f"CURVE RIGHT {int(base_speed)} {turn_strength}"
        else:
            movement = f"MOVE {int(base_speed)}"

    if movement != previous_command:
        print(f"Sending Command: {movement}")
        arduino.write((movement + "\n").encode())
        previous_command = movement

# Main Program Execution
if __name__ == "__main__":
    try:
        arduino = initialize_serial()
        camera = initialize_camera()
        with picamera.array.PiRGBArray(camera, size=(FRAME_WIDTH, FRAME_HEIGHT)) as stream:
            print("Starting object tracking... Press 'q' to quit.")
            for frame in camera.capture_continuous(stream, format="rgb", use_video_port=True):
                process_frame(frame, arduino)
                stream.truncate(0)
    except KeyboardInterrupt:
        print("\nUser stopped the program.")
        sys.exit(0)
