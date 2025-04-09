import numpy as np
import cv2
import picamera
import picamera.array
import serial
import time
import sys
from collections import deque

# ----------------- Serial Port and Camera Configuration -----------------
SERIAL_PORT = '/dev/ttyUSB0'  # Adjust as needed for your system.
BAUD_RATE = 9600

FRAME_WIDTH = 320             # Width of the camera frame.
FRAME_HEIGHT = 240            # Height of the camera frame.
FPS = 30                      # Frames per second.

# ----------------- Object Color HSV Range -----------------
LOWER_HSV = np.array([0, 100, 100])
UPPER_HSV = np.array([10, 255, 255])

# ----------------- Screen Sections -----------------
SECTION_WIDTH = FRAME_WIDTH // 4
MIDDLE_START = SECTION_WIDTH
MIDDLE_END = 3 * SECTION_WIDTH
FRAME_CENTER = FRAME_WIDTH // 2

# ----------------- Smoothing Filter -----------------
SMOOTHING_BUFFER_SIZE = 5
object_positions = deque(maxlen=SMOOTHING_BUFFER_SIZE)

# ----------------- PID Controller Parameters -----------------
DESIRED_AREA = 2500  # Target object area corresponding to the ideal distance.
MAX_SPEED = 255      # Maximum motor PWM value.
MIN_SPEED = 50       # Minimum forward speed.
previous_command = None

# ----------------- PID Controller Class with Anti-Windup -----------------
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
        if dt <= 0:
            dt = 1e-16
        error = self.setpoint - measured_value
        self.integral += error * dt
        # Clamp the integral term to avoid windup.
        if self.integral > self.integral_limit:
            self.integral = self.integral_limit
        elif self.integral < -self.integral_limit:
            self.integral = -self.integral_limit
        derivative = (error - self.last_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.last_error = error
        self.last_time = now
        return output

pid_controller = PID(Kp=0.1, Ki=0.01, Kd=0.05, setpoint=DESIRED_AREA)

# ----------------- Deadband Helper Function -----------------
def command_changed(new_command, old_command, threshold=5):
    """
    For MOVE commands, update only if the speed difference exceeds threshold.
    For others, use simple string comparison.
    """
    if new_command.startswith("MOVE") and old_command and old_command.startswith("MOVE"):
        try:
            new_speed = int(new_command.split()[1])
            old_speed = int(old_command.split()[1])
            if abs(new_speed - old_speed) < threshold:
                return False
        except Exception:
            return True
    return new_command != old_command

# ----------------- Serial and Camera Initialization -----------------
def initialize_serial():
    try:
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print("Serial connection established.")
        return arduino
    except serial.SerialException as e:
        print(f"Error: Could not connect to Arduino - {e}")
        sys.exit(1)

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

# ----------------- Smoothing Filter for Object Position -----------------
def smooth_position(new_position):
    object_positions.append(new_position)
    return int(np.mean(object_positions))

# ----------------- Enhanced Adaptive Color Thresholding -----------------
def process_image(image):
    try:
        # Convert the image from RGB to HSV.
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # Apply CLAHE on the V channel for adaptive histogram equalization.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
        # Apply Gaussian blur to reduce noise.
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        # Create a binary mask based on the HSV range.
        mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
        # Use morphological operations to clean up the mask.
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask
    except Exception as e:
        print(f"Error in process_image: {e}")
        return None

# ----------------- Process Frame: Compute Forward Speed and Turn Offset -----------------
def process_frame(frame, arduino):
    global previous_command
    movement = "STOP"  # Default command.
    
    try:
        # Convert frame to a NumPy array.
        image = np.array(frame.array, dtype=np.uint8)
        # Process image to obtain a binary mask.
        mask = process_image(image)
        if mask is None:
            print("Error: Mask generation failed.")
            return

        # Find indices where the mask is nonzero.
        indices = np.where(mask > 0)
        if len(indices[0]) > 0:
            # Calculate the object's horizontal bounds.
            x_min, x_max = min(indices[1]), max(indices[1])
            object_center_x = (x_min + x_max) // 2
            object_area = len(indices[0])
            # Smooth the object center position.
            object_center_x = smooth_position(object_center_x)
            
            # Use the PID controller to determine forward speed based on object area.
            speed_output = pid_controller.update(object_area)
            # Clamp and adjust speed.
            speed_output = max(0, min(speed_output, MAX_SPEED))
            if 0 < speed_output < MIN_SPEED:
                speed_output = MIN_SPEED
            
            # Compute turning error: positive if object is to the left, negative if to the right.
            turning_error = FRAME_CENTER - object_center_x
            # Scale the turning error to produce a turn offset.
            K_turn = 0.2  # Adjust this gain for sensitivity.
            turn_offset = int(K_turn * turning_error)
            
            # Form the movement command as "MOVE <forward_speed> <turn_offset>"
            movement = f"MOVE {int(speed_output)} {turn_offset}"
        
        # Send command only if it has changed significantly.
        if command_changed(movement, previous_command):
            print(f"Sending Command: {movement}")
            try:
                arduino.write((movement + "\n").encode())
            except serial.SerialException as e:
                print(f"Serial Write Error: {e}")
            previous_command = movement

    except Exception as e:
        print(f"Error processing frame: {e}")

# ----------------- Graceful Exit Handling -----------------
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
    cv2.destroyAllWindows()
    print("Exiting safely.")
    sys.exit(0)

# ----------------- Main Program -----------------
if __name__ == "__main__":
    try:
        arduino = initialize_serial()
        camera = initialize_camera()
        with picamera.array.PiRGBArray(camera, size=(FRAME_WIDTH, FRAME_HEIGHT)) as stream:
            print("Starting object tracking... Press 'q' to quit or Ctrl+C to stop.")
            for frame in camera.capture_continuous(stream, format="rgb", use_video_port=True):
                process_frame(frame, arduino)
                # Display the camera view for debugging.
                image = np.array(frame.array, dtype=np.uint8)
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imshow("Camera", image_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                stream.truncate(0)
    except KeyboardInterrupt:
        print("\nUser stopped the program.")
        clean_exit(camera, arduino)
    except Exception as e:
        print(f"Unexpected Error: {e}")
        clean_exit(camera, arduino)
