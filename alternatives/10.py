import numpy as np
import cv2
import serial
import time

# Constants
FRAME_WIDTH = 320
KP_SPEED, KI_SPEED, KD_SPEED = 0.1, 0.01, 0.05  # PID for speed
KP_TURN = 0.5  # Proportional for turning
DESIRED_AREA = 6000  # Target object size in pixels

# Serial setup
arduino = serial.Serial('/dev/ttyUSB0', 9600)
time.sleep(2)

# PID variables
integral_error = 0
previous_error = 0

def pid_speed_control(error, dt):
    global integral_error, previous_error
    integral_error += error * dt
    derivative_error = (error - previous_error) / dt
    output = KP_SPEED * error + KI_SPEED * integral_error + KD_SPEED * derivative_error
    previous_error = error
    return output

def process_frame(image):
    # Convert to HSV and detect object (e.g., orange)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 100, 100), (12, 255, 255))
    indices = np.where(mask > 0)

    if len(indices[0]) > 0:
        x_min, x_max = min(indices[1]), max(indices[1])
        object_center_x = (x_min + x_max) // 2
        object_area = len(indices[0])

        # Errors
        distance_error = DESIRED_AREA - object_area
        position_error = object_center_x - (FRAME_WIDTH // 2)

        # PID for speed
        dt = 1 / 30  # Assuming 30 FPS
        speed_adjustment = pid_speed_control(distance_error, dt)

        # Proportional for turning
        turn_adjustment = KP_TURN * position_error

        # Movement logic
        if abs(position_error) > 10:  # Deadband
            if position_error < 0:
                command = f"TURN LEFT {abs(turn_adjustment):.1f}"
            else:
                command = f"TURN RIGHT {turn_adjustment:.1f}"
        else:
            if object_area > 8000:  # Too close
                command = "STOP"
            else:
                command = f"MOVE FORWARD {speed_adjustment:.1f}"

        # Send command
        arduino.write((command + "\n").encode())
        print(command)

# Main loop (simplified)
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        process_frame(frame)
    time.sleep(0.033)  # ~30 FPS