import numpy as np
import picamera
import picamera.array
import serial
import time
import sys
from collections import deque
import cv2

# Serial Port Configuration
SERIAL_PORT = '/dev/ttyUSB0'  # Adjust for your system
BAUD_RATE = 9600

# Initial HSV Range for Orange Object (dynamically adjustable)
lower_hsv = np.array([0, 100, 100], dtype=np.uint8)  # H, S, V
upper_hsv = np.array([12, 255, 255], dtype=np.uint8)  # H, S, V

# Camera Configuration
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
FPS = 30

# Define Screen Sections for Movement Decisions
SECTION_WIDTH = FRAME_WIDTH // 4
MIDDLE_START = SECTION_WIDTH
MIDDLE_END = 3 * SECTION_WIDTH

# Smoothing Filter for Object Position
SMOOTHING_BUFFER_SIZE = 5
object_positions = deque(maxlen=SMOOTHING_BUFFER_SIZE)

# Movement Thresholds (in pixels, adjust as needed)
STOP_THRESHOLD = 8000      # Stop if object is too close
FORWARD_THRESHOLD = 4000   # Move forward if object is at medium distance
SMOOTHING_FACTOR = 0.7     # Smoothing factor (0.0 slow, 1.0 fast)

# Global Variables
previous_command = None    # Track last command to avoid redundancy
frame_counter = 0          # Count frames for periodic HSV updates

# Initialize Serial Connection
def initialize_serial():
    """Set up serial communication with Arduino."""
    try:
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Wait for Arduino to initialize
        print("Serial connection established.")
        return arduino
    except serial.SerialException as e:
        print(f"Error: Could not connect to Arduino - {e}")
        sys.exit(1)

# Initialize PiCamera
def initialize_camera():
    """Configure and initialize the PiCamera."""
    try:
        camera = picamera.PiCamera()
        camera.resolution = (FRAME_WIDTH, FRAME_HEIGHT)
        camera.framerate = FPS
        print("Camera initialized successfully.")
        return camera
    except picamera.PiCameraError as e:
        print(f"Error: Could not initialize camera - {e}")
        sys.exit(1)

# Smooth Object Position
def smooth_position(new_position):
    """Apply moving average to object center position."""
    object_positions.append(new_position)
    return int(np.mean(object_positions))

# Process Frame and Control Robot
def process_frame(frame, arduino):
    """Process each frame to detect object and send movement commands."""
    global previous_command, frame_counter, lower_hsv, upper_hsv
    movement = "STOP"
    frame_counter += 1

    try:
        image = frame.array
        # Convert to HSV once per frame
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Update HSV bounds every 100 frames (~3.3s at 30 FPS)
        if frame_counter % 100 == 0:
            v_channel = hsv_image[:, :, 2]  # Extract Value channel
            average_v = np.mean(v_channel)  # Calculate average brightness
            if average_v < 100:  # Dim lighting
                lower_hsv[2] = 50  # Lower V bound for darker pixels
                lower_hsv[1] = 80  # Lower S bound for less saturated colors
            else:  # Normal or bright lighting
                lower_hsv[2] = 100  # Default V bound
                lower_hsv[1] = 100  # Default S bound
            print(f"Updated HSV bounds: lower_hsv = {lower_hsv}, average_v = {average_v:.2f}")

        # Create mask with current HSV bounds
        mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
        indices = np.where(mask > 0)

        # Detect object position and size
        if len(indices[0]) > 0:
            x_min, x_max = min(indices[1]), max(indices[1])
            object_center_x = (x_min + x_max) // 2
            object_area = len(indices[0])  # Approximate area in pixels

            # Smooth the object’s x-position
            object_center_x = smooth_position(object_center_x)

            # Determine movement based on position and size
            if object_center_x < SECTION_WIDTH:  # Left section
                movement = "TURN LEFT"
            elif object_center_x > 3 * SECTION_WIDTH:  # Right section
                movement = "TURN RIGHT"
            elif MIDDLE_START <= object_center_x <= MIDDLE_END:  # Center
                if object_area > STOP_THRESHOLD:  # Object too close
                    movement = "STOP"
                elif object_area > FORWARD_THRESHOLD:  # Object at medium distance
                    movement = "MOVE FORWARD"

        # Send command only if it differs from the previous one
        if movement != previous_command:
            print(f"Sending Command: {movement}")
            try:
                arduino.write((movement + "\n").encode())
            except serial.SerialException as e:
                print(f"Serial Write Error: {e}")
            previous_command = movement

    except Exception as e:
        print(f"Error processing frame: {e}")

# Clean Up Resources
def clean_exit(camera, arduino):
    """Safely close camera and serial connections."""
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