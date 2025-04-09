import numpy as np

# Camera Configuration
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
FPS = 30

# Object Color HSV Range
LOWER_HSV = np.array([0, 100, 100])
UPPER_HSV = np.array([10, 255, 255])

# Movement decision sections
SECTION_WIDTH = FRAME_WIDTH // 4
MIDDLE_START = SECTION_WIDTH
MIDDLE_END = 3 * SECTION_WIDTH

# PID Control
DESIRED_AREA = 2500
MAX_SPEED = 255
MIN_SPEED = 50

# Serial Port (For Raspberry Pi, change accordingly)
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 9600
