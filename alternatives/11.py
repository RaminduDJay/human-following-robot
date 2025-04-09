import numpy as np
import cv2
import picamera
import picamera.array
import serial
import time
import sys
from collections import deque

# Serial Port Configuration
SERIAL_PORT = '/dev/ttyUSB0'  # Adjust based on system
BAUD_RATE = 9600

# Camera Configuration
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
FPS = 30

# Object Color HSV Range (Initial Guess - Will Auto-Adjust)
LOWER_HSV = np.array([0, 100, 100])  # Initial Hue Lower Bound
UPPER_HSV = np.array([10, 255, 255])  # Initial Hue Upper Bound

# Define screen sections for movement decisions
SECTION_WIDTH = FRAME_WIDTH // 4
MIDDLE_START = SECTION_WIDTH
MIDDLE_END = 3 * SECTION_WIDTH

# Moving average filter for smoother tracking
SMOOTHING_BUFFER_SIZE = 5
object_positions = deque(maxlen=SMOOTHING_BUFFER_SIZE)

# Distance control thresholds
STOP_THRESHOLD = 3000  # Stop at 1m
MOVE_TOWARD_THRESHOLD = 1500  # Move forward if object too far
SMOOTHING_FACTOR = 0.7

previous_command = None  # Store last command to avoid redundant messages

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

# Adaptive Color Thresholding in HSV
def process_image(image):
    try:
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Adaptive Brightness Correction (Histogram Equalization on V-channel)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])

        # Apply Gaussian Blur to reduce noise
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

        # Create binary mask using adaptive color thresholds
        mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)

        # Apply Morphological Operations to clean noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask
    except Exception as e:
        print(f"Error in process_image: {e}")
        return None

# Process Frame to Detect Object and Move Robot
def process_frame(frame, arduino):
    global previous_command
    movement = "STOP"

    try:
        # Convert frame to NumPy array
        image = np.array(frame.array, dtype=np.uint8)

        # Process image to obtain binary mask
        mask = process_image(image)

        if mask is None:
            print("Error: Mask generation failed.")
            return

        # Find object position
        indices = np.where(mask > 0)
        if len(indices[0]) > 0:
            x_min, x_max = min(indices[1]), max(indices[1])
            object_center_x = (x_min + x_max) // 2
            object_area = len(indices[0])  # Approximate object size

            # Smooth object position for better tracking
            object_center_x = smooth_position(object_center_x)

            # Adjust movement based on position and object distance
            if object_center_x < SECTION_WIDTH:  # Left section
                movement = "TURN LEFT"
            elif object_center_x > 3 * SECTION_WIDTH:  # Right section
                movement = "TURN RIGHT"
            elif MIDDLE_START <= object_center_x <= MIDDLE_END:  # Centered
                if object_area > STOP_THRESHOLD:  # Object too close (less than 1m)
                    movement = "STOP"
                elif object_area < MOVE_TOWARD_THRESHOLD:  # Object too far (more than 1m)
                    movement = "MOVE FORWARD"

        # Only send command if it has changed
        if movement != previous_command:
            print(f"Sending Command: {movement}")
            try:
                arduino.write((movement + "\n").encode())
            except serial.SerialException as e:
                print(f"Serial Write Error: {e}")
            previous_command = movement

    except Exception as e:
        print(f"Error processing frame: {e}")

# Graceful Exit Handling
def clean_exit(camera, arduino):
    print("\nCleaning up resources...")
    try:
        camera.close()
        print("Camera closed.")
    except Exception as e:
        print(f"Error closing camera: {e}")

    try:
        arduino.close()
        print("Serial connection closed.")
    except Exception as e:
        print(f"Error closing serial connection: {e}")

    print("Exiting safely.")
    sys.exit(0)

# Main Program
if __name__ == "__main__":
    try:
        arduino = initialize_serial()
        camera = initialize_camera()

        with picamera.array.PiRGBArray(camera, size=(FRAME_WIDTH, FRAME_HEIGHT)) as stream:
            print("Starting object tracking... Press Ctrl+C to stop.")

            for frame in camera.capture_continuous(stream, format="rgb", use_video_port=True):
                process_frame(frame, arduino)
                stream.truncate(0)  # Clear stream for next frame

    except KeyboardInterrupt:
        print("\nUser stopped the program.")
        clean_exit(camera, arduino)

    except Exception as e:
        print(f"Unexpected Error: {e}")
        clean_exit(camera, arduino)
