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

# Distance control thresholds using PID control
DESIRED_AREA = 2500  # Target object area corresponding to 1 m distance
MAX_SPEED = 255      # Maximum motor speed value
MIN_SPEED = 50       # Minimum motor speed value for forward movement

previous_command = None  # Store last command to avoid redundant messages

# ----------------- Updated: PID Controller Class with Anti-Windup -----------------
class PID:
    def init(self, Kp, Ki, Kd, setpoint=0, integral_limit=1000):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.last_error = 0
        self.integral = 0
        self.last_time = time.time()
        self.integral_limit = integral_limit  # Limit for the integral term

    def update(self, measured_value):
        now = time.time()
        dt = now - self.last_time
        if dt <= 0:
            dt = 1e-16  # Prevent division by zero
        error = self.setpoint - measured_value
        self.integral += error * dt
        # Anti-windup: Clamp the integral term
        if self.integral > self.integral_limit:
            self.integral = self.integral_limit
        elif self.integral < -self.integral_limit:
            self.integral = -self.integral_limit
        derivative = (error - self.last_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.last_error = error
        self.last_time = now
        return output

# ----------------- Updated: Create PID controller instance -----------------
pid_controller = PID(Kp=0.1, Ki=0.01, Kd=0.05, setpoint=DESIRED_AREA)

# ----------------- Updated: Deadband helper for serial command updates -----------------
def command_changed(new_command, old_command, threshold=5):
    """
    Compare MOVE commands and update only if the speed difference exceeds the threshold.
    For non-MOVE commands, use standard string comparison.
    """
    if new_command.startswith("MOVE") and old_command and old_command.startswith("MOVE"):
        try:
            new_speed = int(new_command.split()[1])
            old_speed = int(old_command.split()[1])
            if abs(new_speed - old_speed) < threshold:
                return False
        except Exception as e:
            # If parsing fails, assume the command has changed.
            return True
    return new_command != old_command

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

# Process Frame to Detect Object and Control Motor Speed with PID
def process_frame(frame, arduino):
    global previous_command
    movement = "STOP"

    try:
        # Convert frame to NumPy array
        image = np.array(frame.array, dtype=np.uint8)
        # Process image to obtain binary mask (robust under changing lighting)
        mask = process_image(image)
        if mask is None:
            print("Error: Mask generation failed.")
            return

        # Find object position from mask
        indices = np.where(mask > 0)
        if len(indices[0]) > 0:
            x_min, x_max = min(indices[1]), max(indices[1])
            object_center_x = (x_min + x_max) // 2
            object_area = len(indices[0])  # Approximate object size based on pixel count

            # Smooth object center for stability
            object_center_x = smooth_position(object_center_x)

            # Determine turning commands based on horizontal position of the object
            if object_center_x < SECTION_WIDTH:
                movement = "TURN LEFT"
            elif object_center_x > 3 * SECTION_WIDTH:
                movement = "TURN RIGHT"
            elif MIDDLE_START <= object_center_x <= MIDDLE_END:
                # ----------------- Modified: Use PID controller to set speed smoothly -----------------
                speed_output = pid_controller.update(object_area)
                # Clamp the speed to within allowed limits
                if speed_output < 0:
                    speed_output = 0
                if speed_output > MAX_SPEED:
                    speed_output = MAX_SPEED
                # Ensure a minimum forward speed if nonzero
                if 0 < speed_output < MIN_SPEED:
                    speed_output = MIN_SPEED
                movement = f"MOVE {int(speed_output)}"

        # Only send command if it has changed (with deadband filtering for MOVE commands)
        if command_changed(movement, previous_command):
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
    cv2.destroyAllWindows()  # Destroy any OpenCV windows
    print("Exiting safely.")
    sys.exit(0)

# Main Program Execution
if name == "main":
    try:
        arduino = initialize_serial()
        camera = initialize_camera()
        with picamera.array.PiRGBArray(camera, size=(FRAME_WIDTH, FRAME_HEIGHT)) as stream:
            print("Starting object tracking... Press 'q' to quit or Ctrl+C to stop.")
            for frame in camera.capture_continuous(stream, format="rgb", use_video_port=True):
                process_frame(frame, arduino)
                # ----------------- Display the camera view -----------------
                image = np.array(frame.array, dtype=np.uint8)
                # Convert from RGB to BGR for OpenCV display
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imshow("Camera", image_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                stream.truncate(0)  # Clear stream for next frame
    except KeyboardInterrupt:
        print("\nUser stopped the program.")
        clean_exit(camera, arduino)
    except Exception as e:
        print(f"Unexpected Error: {e}")
        clean_exit(camera, arduino)