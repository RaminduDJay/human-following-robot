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

# Color Detection HSV Range (Adjust based on your target)
LOWER_HSV = np.array([0, 100, 100])
UPPER_HSV = np.array([10, 255, 255])

# PID Control Parameters
DESIRED_AREA = 2500        # Target object area for distance control
MAX_SPEED = 255            # Maximum motor speed
MIN_SPEED = 50             # Minimum operational speed
MAX_ACCEL = 20             # Maximum acceleration per update
CENTER_DEADZONE = 20       # Pixels around center to ignore small adjustments

# PID Controllers
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
        error = self.setpoint - measured_value
        
        # Apply deadzone to center position errors
        if self.setpoint == FRAME_WIDTH//2 and abs(error) < CENTER_DEADZONE:
            error = 0
        
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        derivative = (error - self.last_error) / dt if dt > 0 else 0
        
        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        self.last_error = error
        self.last_time = now
        return output

# Initialize PID controllers
pid_speed = PID(Kp=0.15, Ki=0.02, Kd=0.1, setpoint=DESIRED_AREA)
pid_steering = PID(Kp=0.4, Ki=0.005, Kd=0.2, setpoint=FRAME_WIDTH//2)

# Motor control state
current_left_speed = 0
current_right_speed = 0
previous_command = None

# System initialization
object_positions = deque(maxlen=5)
arduino = None
camera = None

def initialize_serial():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        return ser
    except serial.SerialException as e:
        print(f"Serial Error: {e}")
        sys.exit(1)

def initialize_camera():
    try:
        cam = picamera.PiCamera()
        cam.resolution = (FRAME_WIDTH, FRAME_HEIGHT)
        cam.framerate = FPS
        return cam
    except picamera.PiCameraError as e:
        print(f"Camera Error: {e}")
        sys.exit(1)

def ramp_speed(current, target):
    """Smooth acceleration control with minimum speed enforcement"""
    if abs(target) > 0 and abs(target) < MIN_SPEED:
        target = np.sign(target) * MIN_SPEED
    if target > current + MAX_ACCEL:
        return current + MAX_ACCEL
    if target < current - MAX_ACCEL:
        return current - MAX_ACCEL
    return target

def process_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(hsv[:, :, 2])
    mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

def process_frame(frame, arduino):
    global current_left_speed, current_right_speed, previous_command

    try:
        image = np.array(frame.array, dtype=np.uint8)
        mask = process_image(image)
        indices = np.where(mask > 0)
        
        if len(indices[0]) > 0:
            x_min, x_max = min(indices[1]), max(indices[1])
            obj_center = (x_min + x_max) // 2
            obj_area = len(indices[0])
            
            # Update PID controllers
            base_speed = pid_speed.update(obj_area)
            steering = pid_steering.update(obj_center)
            
            # Apply dynamic speed dampening
            steering *= (1 - (base_speed/MAX_SPEED)**2)
            
            # Calculate motor speeds
            target_left = base_speed - steering
            target_right = base_speed + steering
            
            # Apply speed limits and ramping
            target_left = np.clip(target_left, -MAX_SPEED, MAX_SPEED)
            target_right = np.clip(target_right, -MAX_SPEED, MAX_SPEED)
            
            current_left_speed = ramp_speed(current_left_speed, target_left)
            current_right_speed = ramp_speed(current_right_speed, target_right)
            
            movement = f"MOVE {int(current_left_speed)} {int(current_right_speed)}"
        else:
            movement = "STOP"
            current_left_speed = current_right_speed = 0

        # Send command only if changed significantly
        if movement != previous_command:
            if movement.startswith("MOVE"):
                l_prev = int(previous_command.split()[1]) if previous_command and previous_command.startswith("MOVE") else 0
                r_prev = int(previous_command.split()[2]) if previous_command and previous_command.startswith("MOVE") else 0
                if abs(current_left_speed - l_prev) < 5 and abs(current_right_speed - r_prev) < 5:
                    return
            
            arduino.write(f"{movement}\n".encode())
            previous_command = movement

    except Exception as e:
        print(f"Processing error: {e}")

def clean_exit():
    global arduino, camera
    arduino.write(b"STOP\n")
    camera.close()
    arduino.close()
    cv2.destroyAllWindows()
    sys.exit(0)

if __name__ == "__main__":
    try:
        arduino = initialize_serial()
        camera = initialize_camera()
        
        with picamera.array.PiRGBArray(camera) as stream:
            print("Tracking started. Press q to quit.")
            for frame in camera.capture_continuous(stream, format="rgb", use_video_port=True):
                process_frame(frame, arduino)
                
                # Display debug view
                image = cv2.cvtColor(np.array(frame.array), cv2.COLOR_RGB2BGR)
                cv2.putText(image, f"L: {current_left_speed} R: {current_right_speed}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                cv2.imshow("Tracking", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                stream.truncate(0)
                
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        clean_exit()