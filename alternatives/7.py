import numpy as np
import picamera
import picamera.array
import serial
import time
import sys
import pygame
from collections import deque

# Serial Port Configuration
SERIAL_PORT = '/dev/ttyUSB0'  # Adjust for your system
BAUD_RATE = 9600

# Camera Configuration
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
FPS = 30

# Display Configuration
DISPLAY_WIDTH = FRAME_WIDTH * 2  # Double width to show original and processed images side by side
DISPLAY_HEIGHT = FRAME_HEIGHT
DISPLAY_CAPTION = "Robot Camera Vision"

# Define screen sections for movement decisions
SECTION_WIDTH = FRAME_WIDTH // 4
MIDDLE_START = SECTION_WIDTH
MIDDLE_END = 3 * SECTION_WIDTH

# Moving average filter for smoother tracking
SMOOTHING_BUFFER_SIZE = 5
object_positions = deque(maxlen=SMOOTHING_BUFFER_SIZE)
object_areas = deque(maxlen=SMOOTHING_BUFFER_SIZE)

# Movement thresholds
STOP_THRESHOLD = 10000
FORWARD_THRESHOLD = 5000
MINIMUM_OBJECT_SIZE = 500  # Minimum pixel count to be considered an object

# Auto-calibration settings
AUTO_CALIBRATE = True
CALIBRATION_INTERVAL = 100  # Frames between calibration
CALIBRATION_SAMPLES = 5
calibration_counter = 0
color_samples = []

# Initial color range (will be updated during calibration)
target_color_hsv = np.array([2, 185, 251])  # H, S, V from the description
color_tolerance_h = 10
color_tolerance_s = 80
color_tolerance_v = 80

previous_command = None  # Store last command to avoid redundant messages

# RGB to HSV conversion (vectorized version)
def vectorized_rgb_to_hsv(rgb_image):
    rgb_normalized = rgb_image / 255.0
    r, g, b = rgb_normalized[:,:,0], rgb_normalized[:,:,1], rgb_normalized[:,:,2]
    
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc
    
    h = np.zeros_like(r)
    
    # Handle division by zero
    mask = delta != 0
    
    # When r is max
    rmask = mask & (maxc == r)
    h[rmask] = ((g[rmask] - b[rmask]) / delta[rmask]) % 6
    
    # When g is max
    gmask = mask & (maxc == g)
    h[gmask] = ((b[gmask] - r[gmask]) / delta[gmask]) + 2
    
    # When b is max
    bmask = mask & (maxc == b)
    h[bmask] = ((r[bmask] - g[bmask]) / delta[bmask]) + 4
    
    h = h * 60
    
    s = np.zeros_like(r)
    nonzero = maxc != 0
    s[nonzero] = delta[nonzero] / maxc[nonzero]
    
    v = maxc
    
    return np.stack([h, s * 255, v * 255], axis=2)

# Vectorized color matching
def vectorized_color_mask(hsv_image, target_hsv, h_tolerance, s_tolerance, v_tolerance):
    h_diff = np.minimum(
        np.abs(hsv_image[:,:,0] - target_hsv[0]), 
        360 - np.abs(hsv_image[:,:,0] - target_hsv[0])
    )
    s_diff = np.abs(hsv_image[:,:,1] - target_hsv[1])
    v_diff = np.abs(hsv_image[:,:,2] - target_hsv[2])
    
    return (h_diff <= h_tolerance) & (s_diff <= s_tolerance) & (v_diff <= v_tolerance)

# Initialize Pygame for display
def initialize_display():
    pygame.init()
    pygame.display.set_caption(DISPLAY_CAPTION)
    display = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
    font = pygame.font.SysFont('Arial', 14)
    return display, font

# Update display with current frame and processing
def update_display(display, font, rgb_image, mask, object_center_x=None, object_area=None, command=None):
    # Convert images to pygame surfaces
    orig_surface = pygame.surfarray.make_surface(np.transpose(rgb_image, (1, 0, 2)))
    
    # Create a colored mask display (show white where object is detected)
    mask_display = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    mask_display[mask] = [255, 255, 255]  # White for detected pixels
    mask_surface = pygame.surfarray.make_surface(np.transpose(mask_display, (1, 0, 2)))
    
    # Display original image on left
    display.blit(orig_surface, (0, 0))
    
    # Display mask on right
    display.blit(mask_surface, (FRAME_WIDTH, 0))
    
    # Draw section dividers on original image
    pygame.draw.line(display, (255, 0, 0), (SECTION_WIDTH, 0), (SECTION_WIDTH, FRAME_HEIGHT), 1)
    pygame.draw.line(display, (255, 0, 0), (3 * SECTION_WIDTH, 0), (3 * SECTION_WIDTH, FRAME_HEIGHT), 1)
    
    # If object detected, draw its center
    if object_center_x is not None:
        pygame.draw.line(display, (0, 255, 0), (object_center_x, 0), (object_center_x, FRAME_HEIGHT), 2)
    
    # Display information text
    info_lines = [
        f"Target HSV: ({target_color_hsv[0]:.1f}, {target_color_hsv[1]:.1f}, {target_color_hsv[2]:.1f})",
        f"Tolerances: H={color_tolerance_h:.1f}, S={color_tolerance_s:.1f}, V={color_tolerance_v:.1f}"
    ]
    
    if object_area is not None:
        info_lines.append(f"Object Area: {object_area}")
    
    if command is not None:
        info_lines.append(f"Command: {command}")
    
    for i, line in enumerate(info_lines):
        text_surface = font.render(line, True, (255, 255, 0))
        display.blit(text_surface, (10, 10 + i * 20))
    
    pygame.display.update()

# Auto-calibrate color based on the largest object in the frame
def calibrate_color(rgb_image):
    global target_color_hsv, color_tolerance_h, color_tolerance_s, color_tolerance_v
    
    # Use initial color range to find potential target
    hsv_image = vectorized_rgb_to_hsv(rgb_image)
    mask = vectorized_color_mask(hsv_image, target_color_hsv, 
                                color_tolerance_h, color_tolerance_s, color_tolerance_v)
    
    # Find connected components (simplified)
    visited = np.zeros_like(mask, dtype=bool)
    components = []
    
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x] and not visited[y, x]:
                # Simple flood fill to find connected component
                component = []
                stack = [(y, x)]
                visited[y, x] = True
                
                while stack:
                    cy, cx = stack.pop()
                    component.append((cy, cx))
                    
                    # Check 4-connected neighbors
                    for ny, nx in [(cy+1, cx), (cy-1, cx), (cy, cx+1), (cy, cx-1)]:
                        if (0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and 
                            mask[ny, nx] and not visited[ny, nx]):
                            stack.append((ny, nx))
                            visited[ny, nx] = True
                
                components.append(component)
    
    # Find largest component
    if components:
        largest_component = max(components, key=len)
        
        # If component is large enough to be our object
        if len(largest_component) > MINIMUM_OBJECT_SIZE:
            # Sample colors from this component
            color_sum = np.zeros(3)
            for y, x in largest_component:
                color_sum += hsv_image[y, x]
            
            avg_color = color_sum / len(largest_component)
            color_samples.append(avg_color)
            
            # If we have enough samples, update target color
            if len(color_samples) >= CALIBRATION_SAMPLES:
                avg_sample = np.mean(color_samples, axis=0)
                
                # Update target color gradually
                target_color_hsv = (0.7 * target_color_hsv + 0.3 * avg_sample)
                
                # Calculate color variances to adjust tolerances
                variances = np.var(color_samples, axis=0)
                color_tolerance_h = max(10, min(30, variances[0] * 2))
                color_tolerance_s = max(50, min(100, variances[1] * 0.5))
                color_tolerance_v = max(50, min(100, variances[2] * 0.5))
                
                # Reset samples for next calibration
                color_samples.clear()
                
                print(f"Calibrated color: HSV=({target_color_hsv[0]:.1f}, {target_color_hsv[1]:.1f}, {target_color_hsv[2]:.1f})")
                print(f"Tolerances: H={color_tolerance_h:.1f}, S={color_tolerance_s:.1f}, V={color_tolerance_v:.1f}")

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

# Apply smoothing filter to object position and area
def smooth_position(new_position):
    object_positions.append(new_position)
    return int(np.mean(object_positions))

def smooth_area(new_area):
    object_areas.append(new_area)
    return int(np.mean(object_areas))

# Simple noise reduction (remove isolated pixels)
def reduce_noise(mask):
    filtered_mask = np.zeros_like(mask)
    for y in range(1, mask.shape[0]-1):
        for x in range(1, mask.shape[1]-1):
            # Count neighbors
            neighbors = np.sum(mask[y-1:y+2, x-1:x+2])
            filtered_mask[y, x] = neighbors >= 5  # At least 5 neighbors (including self)
    return filtered_mask

# Process Frame to Detect Object and Move Robot
def process_frame(frame, arduino, display, font):
    global previous_command, calibration_counter
    movement = "STOP"
    object_center_x = None
    object_area = None
    
    try:
        # Convert frame to NumPy array
        rgb_image = np.array(frame.array, dtype=np.uint8)
        
        # Handle pygame events to keep the window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
        
        # Periodically auto-calibrate color if enabled
        if AUTO_CALIBRATE:
            calibration_counter += 1
            if calibration_counter >= CALIBRATION_INTERVAL:
                calibrate_color(rgb_image)
                calibration_counter = 0
        
        # Convert to HSV and create mask
        hsv_image = vectorized_rgb_to_hsv(rgb_image)
        mask = vectorized_color_mask(hsv_image, target_color_hsv, 
                                    color_tolerance_h, color_tolerance_s, color_tolerance_v)
        
        # Reduce noise in mask
        filtered_mask = reduce_noise(mask)
        
        # Find object position
        indices = np.where(filtered_mask)
        if len(indices[0]) > MINIMUM_OBJECT_SIZE:
            y_indices, x_indices = indices
            
            # Calculate center and area
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            
            object_center_x = (x_min + x_max) // 2
            object_width = x_max - x_min
            object_height = y_max - y_min
            object_area = len(indices[0])
            
            # Calculate aspect ratio
            aspect_ratio = float(object_width) / max(1, object_height)  # Avoid division by zero
            expected_ratio = 14.0 / 6.0  # Based on the 14cm x 6cm object
            
            # Check if aspect ratio is reasonable (with tolerance)
            if 0.5 * expected_ratio <= aspect_ratio <= 2.0 * expected_ratio:
                # Smooth values
                object_center_x = smooth_position(object_center_x)
                object_area = smooth_area(object_area)
                
                # Adjust movement based on position
                if object_center_x < SECTION_WIDTH:  # Left section
                    movement = "TURN LEFT"
                elif object_center_x > 3 * SECTION_WIDTH:  # Right section
                    movement = "TURN RIGHT"
                elif MIDDLE_START <= object_center_x <= MIDDLE_END:  # Center
                    if object_area > STOP_THRESHOLD:  # Object too close
                        movement = "STOP"
                    elif object_area > FORWARD_THRESHOLD:  # Move forward smoothly
                        movement = "MOVE FORWARD"
                    else:  # Object is far but centered
                        movement = "MOVE FORWARD"
            else:
                print(f"Object detected but aspect ratio ({aspect_ratio:.2f}) doesn't match expected ({expected_ratio:.2f})")
        
        # Update the display
        update_display(display, font, rgb_image, filtered_mask, object_center_x, object_area, movement)
        
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
    
    # Clean up pygame
    pygame.quit()
        
    print("Exiting safely.")
    sys.exit(0)

# Main Program
if __name__ == "__main__":
    try:
        arduino = initialize_serial()
        camera = initialize_camera()
        display, font = initialize_display()
        
        print("Starting with initial color target (HSV):", target_color_hsv)
        print("Display initialized. Close window or press Ctrl+C to exit.")

        with picamera.array.PiRGBArray(camera, size=(FRAME_WIDTH, FRAME_HEIGHT)) as stream:
            print("Starting object tracking...")
            print("Auto-calibration enabled, tracking will improve over time.")
            
            for frame in camera.capture_continuous(stream, format="rgb", use_video_port=True):
                process_frame(frame, arduino, display, font)
                stream.truncate(0)  # Clear stream for next frame

    except KeyboardInterrupt:
        print("\nUser stopped the program.")
        clean_exit(camera, arduino)

    except Exception as e:
        print(f"Unexpected Error: {e}")
        clean_exit(camera, arduino)