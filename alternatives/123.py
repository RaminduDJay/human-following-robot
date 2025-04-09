import numpy as np
import picamera
import picamera.array
import serial
import time
import sys

# Serial Port Configuration
SERIAL_PORT = '/dev/ttyUSB0'  # Adjust based on your system
BAUD_RATE = 9600

# Object Color Range (Adjust these values)
LOWER_COLOR = np.array([200, 50, 50])  
UPPER_COLOR = np.array([255, 150, 150])

# Camera Configuration
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
FPS = 30

# Define screen sections
SECTION_WIDTH = FRAME_WIDTH // 4
MIDDLE_START = SECTION_WIDTH
MIDDLE_END = 3 * SECTION_WIDTH

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

# Process Frame to Detect Object
def process_frame(frame, arduino):
    global previous_command
    movement = "STOP"  # Default movement

    try:
        # Convert frame to NumPy array
        image = np.array(frame.array, dtype=np.uint8)

        # Filter pixels within color range
        mask = np.logical_and(
            np.all(image >= LOWER_COLOR, axis=-1),
            np.all(image <= UPPER_COLOR, axis=-1)
        )

        # Find object position
        indices = np.where(mask)
        if len(indices[0]) > 0:
            x_min, x_max = min(indices[1]), max(indices[1])
            object_center_x = (x_min + x_max) // 2
            object_area = len(indices[0])  # Approximate object size

            # Decide movement based on object position
            if object_center_x < SECTION_WIDTH:
                movement = "TURN LEFT"
            elif object_center_x > 3 * SECTION_WIDTH:
                movement = "TURN RIGHT"
            elif MIDDLE_START <= object_center_x <= MIDDLE_END:
                if object_area > 8000:  # Object too close
                    movement = "STOP"
                elif object_area > 4000:  # Object at correct distance
                    movement = "MOVE FORWARD"

        # Send command to Arduino only if it has changed
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
            print("Starting object detection... Press Ctrl+C to stop.")

            for frame in camera.capture_continuous(stream, format="rgb", use_video_port=True):
                process_frame(frame, arduino)
                stream.truncate(0)  # Clear stream for next frame

    except KeyboardInterrupt:
        print("\nUser stopped the program.")
        clean_exit(camera, arduino)

    except Exception as e:
        print(f"Unexpected Error: {e}")
        clean_exit(camera, arduino)





# import numpy as np
# import picamera
# import picamera.array
# import serial
# import time

# # Initialize Serial Communication with Arduino (Adjust port)
# try:
#     arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)  # Change port if needed
#     time.sleep(2)  # Allow Arduino to initialize
# except Exception as e:
#     print(f"Error connecting to Arduino: {e}")
#     exit()

# # Color Range for Object Detection (Adjust for your object)
# LOWER_COLOR = np.array([200, 50, 50])   # Adjust these values based on object color
# UPPER_COLOR = np.array([255, 150, 150])

# # Camera Setup
# camera = picamera.PiCamera()
# camera.resolution = (320, 240)
# camera.framerate = 30
# frame_width, frame_height = camera.resolution

# # Define screen sections
# section_width = frame_width // 4
# middle_start = section_width
# middle_end = 3 * section_width

# previous_command = None  # To prevent duplicate commands

# def process_frame(frame):
#     global previous_command
#     movement = "STOP"  # Default

#     # Convert frame to NumPy array
#     image = np.array(frame.array, dtype=np.uint8)

#     # Filter pixels within color range
#     mask = np.logical_and(
#         np.all(image >= LOWER_COLOR, axis=-1),
#         np.all(image <= UPPER_COLOR, axis=-1)
#     )

#     # Find object position
#     indices = np.where(mask)
#     if len(indices[0]) > 0:
#         x_min, x_max = min(indices[1]), max(indices[1])
#         object_center_x = (x_min + x_max) // 2
#         object_area = len(indices[0])  # Approximate object size

#         # Decide movement based on object position
#         if object_center_x < section_width:
#             movement = "TURN LEFT"
#         elif object_center_x > 3 * section_width:
#             movement = "TURN RIGHT"
#         elif middle_start <= object_center_x <= middle_end:
#             if object_area > 8000:  # Object is too close
#                 movement = "STOP"
#             elif object_area > 4000:  # Object is in the correct range
#                 movement = "MOVE FORWARD"

#     # Send command to Arduino if it has changed
#     if movement != previous_command:
#         print(f"Sending Command: {movement}")
#         try:
#             arduino.write((movement + "\n").encode())
#         except Exception as e:
#             print(f"Serial Write Error: {e}")
#         previous_command = movement

# # Start camera processing
# with picamera.array.PiRGBArray(camera, size=(320, 240)) as stream:
#     for frame in camera.capture_continuous(stream, format="rgb", use_video_port=True):
#         process_frame(frame)
#         stream.truncate(0)  # Clear stream for next frame
