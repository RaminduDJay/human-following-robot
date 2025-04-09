import numpy as np
import cv2
import picamera
import picamera.array
import serial
import time
import sys
from collections import deque

# Serial Port Configuration
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 9600

# Camera Configuration
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
FPS = 30

# Object Color HSV Range (for the specified object)
LOWER_HSV = np.array([0, 100, 100])
UPPER_HSV = np.array([10, 255, 255])

# Moving average filter for smoothing
SMOOTHING_BUFFER_SIZE = 5
object_positions = deque(maxlen=SMOOTHING_BUFFER_SIZE)

# Object Real-World Size
REAL_WIDTH_CM = 14
REAL_HEIGHT_CM = 6

# Minimum and Maximum Object Pixel Size Thresholds (to filter false positives)
MIN_OBJECT_AREA = 500   # Adjust based on camera distance
MAX_OBJECT_AREA = 5000  # Upper bound to remove large noise

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
        dt = max(dt, 1e-16)  # Prevent division by zero
        error = self.setpoint - measured_value
        self.integral = np.clip(self.integral + error * dt, -self.integral_limit, self.integral_limit)
        derivative = (error - self.last_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.last_error = error
        self.last_time = now
        return output

# Initialize Serial Connection
def initialize_serial():
    try:
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
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

# Process Image to Detect Object
def process_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# Process Frame and Detect Object
def process_frame(frame, arduino):
    image = np.array(frame.array, dtype=np.uint8)
    mask = process_image(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if MIN_OBJECT_AREA < area < MAX_OBJECT_AREA and area > max_area:
            max_area = area
            largest_contour = contour

    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h
        
        # Ensure it matches the expected aspect ratio (14cm/6cm ≈ 2.33)
        if 2.0 < aspect_ratio < 2.6:  # Allow slight variance
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, "Object Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"Object at ({x}, {y}), Size: {w}x{h} pixels")
            
            # Calculate distance based on size (placeholder formula, needs calibration)
            distance_cm = 5000 / max_area  # Placeholder calibration formula
            print(f"Estimated Distance: {distance_cm:.2f} cm")

            # Send movement commands (modify as needed)
            if distance_cm > 30:
                command = "MOVE FORWARD"
            elif distance_cm < 10:
                command = "MOVE BACKWARD"
            else:
                command = "STOP"
            
            print(f"Sending Command: {command}")
            arduino.write((command + "\n").encode())
    
    # Display Image
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Camera", image_bgr)

# Graceful Exit
def clean_exit(camera, arduino):
    print("\nCleaning up resources...")
    camera.close()
    arduino.close()
    cv2.destroyAllWindows()
    sys.exit(0)

# Main Program Execution
if __name__ == "__main__":
    try:
        arduino = initialize_serial()
        camera = initialize_camera()
        with picamera.array.PiRGBArray(camera, size=(FRAME_WIDTH, FRAME_HEIGHT)) as stream:
            print("Starting object tracking... Press 'q' to quit or Ctrl+C to stop.")
            for frame in camera.capture_continuous(stream, format="rgb", use_video_port=True):
                process_frame(frame, arduino)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                stream.truncate(0)
    except KeyboardInterrupt:
        clean_exit(camera, arduino)
    except Exception as e:
        print(f"Unexpected Error: {e}")
        clean_exit(camera, arduino)
