import numpy as np
import cv2
import picamera
import picamera.array
import serial
import time
import sys
import threading
from collections import deque

# Serial Port Configuration
SERIAL_PORT = '/dev/ttyUSB0'  # Adjust based on your system
BAUD_RATE = 9600
WATCHDOG_TIMEOUT = 1.0  # Seconds before safety stop if no communication

# Camera Configuration
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
FPS = 30

# Multi-Color Detection (configurable for different targets)
COLOR_PROFILES = {
    "red": {
        "lower_hsv": np.array([0, 100, 100]),
        "upper_hsv": np.array([10, 255, 255]),
        "secondary_lower": np.array([160, 100, 100]),  # Red wraps around HSV
        "secondary_upper": np.array([180, 255, 255])
    },
    "green": {
        "lower_hsv": np.array([40, 100, 100]),
        "upper_hsv": np.array([80, 255, 255])
    },
    "blue": {
        "lower_hsv": np.array([100, 100, 100]),
        "upper_hsv": np.array([140, 255, 255])
    }
}
ACTIVE_COLOR = "red"  # Default tracking color

# PID Control Parameters
DESIRED_AREA = 2500        # Target object area for distance control
MIN_AREA = 500             # Minimum area to track (reduces noise)
MAX_AREA = 6000            # Maximum area (too close)
MAX_SPEED = 255            # Maximum motor speed
MIN_SPEED = 40             # Minimum operational speed
MAX_ACCEL = 15             # Maximum acceleration per update
CENTER_DEADZONE = int(FRAME_WIDTH * 0.05)  # 5% of frame width
OBJECT_TOO_CLOSE = 5000    # Area threshold to trigger backward movement

# Movement modes
MODE_FOLLOW = "follow"     # Follow target object
MODE_ROTATE = "rotate"     # Rotate in place to find object
MODE_AVOID = "avoid"       # Back away from too-close object
MODE_STOP = "stop"         # Stop all movement

# Path memory
MAX_PATH_MEMORY = 10       # Number of positions to remember

# PID Controllers with adaptive parameters
class AdaptivePID:
    def __init__(self, Kp, Ki, Kd, setpoint=0, integral_limit=1000, name=""):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.last_error = 0
        self.integral = 0
        self.last_time = time.time()
        self.integral_limit = integral_limit
        self.last_output = 0
        self.name = name
        self.output_filter = 0.7  # Output smoothing factor (higher = more smoothing)
        
    def update(self, measured_value, dt_override=None):
        now = time.time()
        dt = dt_override if dt_override is not None else (now - self.last_time)
        
        if dt <= 0:
            dt = 0.01  # Avoid division by zero
        
        error = self.setpoint - measured_value
        
        # Apply deadzone for steering to avoid jitter
        if self.name == "steering" and abs(error) < CENTER_DEADZONE:
            error = 0
            self.integral = 0  # Reset integral term in deadzone
        
        # Adaptive integral term based on error
        # Reduce integral influence when error is large
        integral_factor = 1.0 if abs(error) < CENTER_DEADZONE*2 else 0.3
        
        self.integral += error * dt * integral_factor
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        
        # Calculate derivative with additional smoothing for noise reduction
        derivative = (error - self.last_error) / dt if dt > 0 else 0
        
        # Adaptive derivative gain based on speed of change
        derivative_factor = 1.0
        if abs(derivative) > 100:  # If change is very rapid
            derivative_factor = 0.5  # Reduce derivative influence
        
        raw_output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative * derivative_factor)
        
        # Apply exponential smoothing to output
        filtered_output = (1 - self.output_filter) * raw_output + self.output_filter * self.last_output
        
        self.last_error = error
        self.last_time = now
        self.last_output = filtered_output
        
        return filtered_output
    
    def reset(self):
        """Reset the controller state"""
        self.integral = 0
        self.last_error = 0
        self.last_output = 0

# Robot Controller
class RobotController:
    def __init__(self):
        # Motor control state
        self.current_left_speed = 0
        self.current_right_speed = 0
        self.previous_command = None
        self.last_command_time = time.time()
        
        # Movement state
        self.mode = MODE_STOP
        self.object_detected = False
        self.search_direction = 1  # Direction to rotate when searching (1=right, -1=left)
        self.last_object_position = None
        
        # Initialize PID controllers
        self.pid_distance = AdaptivePID(Kp=0.15, Ki=0.01, Kd=0.08, setpoint=DESIRED_AREA, name="distance")
        self.pid_steering = AdaptivePID(Kp=0.4, Ki=0.005, Kd=0.15, setpoint=FRAME_WIDTH//2, name="steering")
        
        # Path memory
        self.position_history = deque(maxlen=MAX_PATH_MEMORY)
        self.area_history = deque(maxlen=MAX_PATH_MEMORY)
        
        # System components
        self.arduino = None
        self.camera = None
        self.stream = None
        self.running = False
        
        # Obstacle detection
        self.obstacle_regions = [False, False, False]  # Left, Center, Right
        
        # Performance metrics
        self.fps = 0
        self.last_frame_time = time.time()
        self.frame_count = 0
        
    def initialize_serial(self):
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            time.sleep(2)  # Wait for Arduino to reset
            return ser
        except serial.SerialException as e:
            print(f"Serial Error: {e}")
            sys.exit(1)
    
    def initialize_camera(self):
        try:
            cam = picamera.PiCamera()
            cam.resolution = (FRAME_WIDTH, FRAME_HEIGHT)
            cam.framerate = FPS
            cam.brightness = 55
            cam.contrast = 60
            return cam
        except picamera.PiCameraError as e:
            print(f"Camera Error: {e}")
            sys.exit(1)
    
    def ramp_speed(self, current, target):
        """Smooth acceleration control with minimum speed enforcement"""
        if abs(target) < 5:  # Complete stop
            return 0
            
        # Apply minimum operational speed
        if abs(target) > 0 and abs(target) < MIN_SPEED:
            target = np.sign(target) * MIN_SPEED
            
        # Apply acceleration limit
        if target > current + MAX_ACCEL:
            return current + MAX_ACCEL
        if target < current - MAX_ACCEL:
            return current - MAX_ACCEL
            
        return target
    
    def process_image(self, image):
        # Apply adaptive brightness correction
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
        
        # Get active color profile
        color = COLOR_PROFILES[ACTIVE_COLOR]
        
        # Create mask based on HSV ranges
        mask = cv2.inRange(hsv, color["lower_hsv"], color["upper_hsv"])
        
        # For colors that wrap around (like red), combine two ranges
        if "secondary_lower" in color:
            mask2 = cv2.inRange(hsv, color["secondary_lower"], color["secondary_upper"])
            mask = cv2.bitwise_or(mask, mask2)
        
        # Noise reduction
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        
        return mask, hsv
    
    def detect_object(self, mask):
        """Detect and analyze object properties from mask"""
        # Find contours - FIXED LINE
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Find largest contour (main object)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < MIN_AREA:
            return None
            
        # Get bounding box and center point
        x, y, w, h = cv2.boundingRect(largest_contour)
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Calculate aspect ratio for shape detection
        aspect_ratio = float(w) / h if h > 0 else 1.0
        
        # Get rotated rectangle for orientation
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        return {
            "center_x": center_x,
            "center_y": center_y,
            "area": area,
            "width": w,
            "height": h,
            "aspect_ratio": aspect_ratio,
            "contour": largest_contour,
            "box": box,
            "orientation": rect[2]  # Angle of the object
        }
    
    def predict_movement(self):
        """Predict object movement based on history"""
        if len(self.position_history) < 3:
            return None, None
            
        # Calculate movement vector from last positions
        positions = list(self.position_history)
        x_movement = positions[-1]["x"] - positions[-3]["x"]
        y_movement = positions[-1]["y"] - positions[-3]["y"]
        
        # Predict next position
        predicted_x = positions[-1]["x"] + (x_movement / 2)
        predicted_y = positions[-1]["y"] + (y_movement / 2)
        
        # Calculate movement speed
        speed = np.sqrt(x_movement**2 + y_movement**2)
        
        return (predicted_x, predicted_y), speed
    
    def decide_movement_mode(self, obj_data=None):
        """Decide which movement mode to use based on object detection"""
        # If no object data and no recent detection, search
        if obj_data is None:
            if time.time() - self.last_command_time > 0.5:
                self.object_detected = False
                return MODE_ROTATE
            return self.mode  # Keep current mode briefly
            
        # Object detected
        self.object_detected = True
        self.last_command_time = time.time()
        
        # Object is too close - back up
        if obj_data["area"] > OBJECT_TOO_CLOSE:
            return MODE_AVOID
            
        # Normal following
        return MODE_FOLLOW
    
    def rotate_to_find(self):
        """Rotate in place to find the object"""
        # Alternate search direction if we've been searching for a while
        if time.time() - self.last_command_time > 5.0:
            self.search_direction *= -1
            self.last_command_time = time.time()
            
        # Create a slower rotation speed
        rotation_speed = 60 * self.search_direction
        self.set_motor_speeds(-rotation_speed, rotation_speed)
    
    def follow_object(self, obj_data):
        """PID-based object following"""
        # Get predicted position if available
        predicted_pos, movement_speed = self.predict_movement()
        
        # Use current position with predictive influence if available
        center_x = obj_data["center_x"]
        if predicted_pos:
            # Blend current and predicted positions (70% current, 30% prediction)
            center_x = int(0.7 * center_x + 0.3 * predicted_pos[0])
        
        # Calculate base forward/backward speed based on area
        base_speed = self.pid_distance.update(obj_data["area"])
        
        # Calculate steering adjustment
        steering = self.pid_steering.update(center_x)
        
        # Apply dynamic steering scaling based on speed
        # Less steering influence at higher speeds
        speed_factor = abs(base_speed) / MAX_SPEED
        dynamic_steering = steering * (1 - (speed_factor * 0.7))
        
        # Calculate left and right motor speeds
        left_speed = base_speed - dynamic_steering
        right_speed = base_speed + dynamic_steering
        
        # Apply speed limits
        left_speed = np.clip(left_speed, -MAX_SPEED, MAX_SPEED)
        right_speed = np.clip(right_speed, -MAX_SPEED, MAX_SPEED)
        
        # Apply ramping for smooth acceleration
        self.current_left_speed = self.ramp_speed(self.current_left_speed, left_speed)
        self.current_right_speed = self.ramp_speed(self.current_right_speed, right_speed)
        
        # Send command to motors
        self.set_motor_speeds(self.current_left_speed, self.current_right_speed)
    
    def avoid_close_object(self, obj_data):
        """Back away from too-close object"""
        # Calculate how much to back up based on how close we are
        backup_speed = -(obj_data["area"] - DESIRED_AREA) * 0.05
        backup_speed = np.clip(backup_speed, -MAX_SPEED, -MIN_SPEED)
        
        # Add slight steering to back away and to the side
        center_offset = (obj_data["center_x"] - FRAME_WIDTH//2) * 0.3
        
        # Apply speed limits and ramping
        left_speed = backup_speed - center_offset
        right_speed = backup_speed + center_offset
        
        self.current_left_speed = self.ramp_speed(self.current_left_speed, left_speed)
        self.current_right_speed = self.ramp_speed(self.current_right_speed, right_speed)
        
        # Send command to motors
        self.set_motor_speeds(self.current_left_speed, self.current_right_speed)
    
    def emergency_stop(self):
        """Immediate stop for safety"""
        self.current_left_speed = 0
        self.current_right_speed = 0
        self.set_motor_speeds(0, 0)
        self.pid_distance.reset()
        self.pid_steering.reset()
    
    def set_motor_speeds(self, left, right):
        """Send motor commands to Arduino"""
        left = int(left)
        right = int(right)
        
        # Only send command if changed significantly or periodically
        now = time.time()
        movement = f"MOVE {left} {right}"
        
        should_send = False
        if self.previous_command is None:
            should_send = True
        elif movement.startswith("MOVE") and self.previous_command.startswith("MOVE"):
            prev_left = int(self.previous_command.split()[1])
            prev_right = int(self.previous_command.split()[2])
            if abs(left - prev_left) > 5 or abs(right - prev_right) > 5:
                should_send = True
        elif movement != self.previous_command:
            should_send = True
        elif now - self.last_command_time > WATCHDOG_TIMEOUT / 2:
            # Send periodic keepalive commands
            should_send = True
            
        if should_send:
            try:
                self.arduino.write(f"{movement}\n".encode())
                self.previous_command = movement
                self.last_command_time = now
            except Exception as e:
                print(f"Serial communication error: {e}")
    
    def watchdog_thread(self):
        """Safety watchdog to stop motors if communication is lost"""
        while self.running:
            if time.time() - self.last_command_time > WATCHDOG_TIMEOUT:
                print("Watchdog: Communication timeout - stopping motors")
                try:
                    self.arduino.write(b"STOP\n")
                except:
                    pass  # Already disconnected
            time.sleep(0.1)
    
    def calculate_fps(self):
        """Calculate and update FPS counter"""
        self.frame_count += 1
        now = time.time()
        elapsed = now - self.last_frame_time
        
        if elapsed > 1.0:  # Update FPS every second
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_frame_time = now
    
    def process_frame(self, frame):
        """Process a video frame for object tracking and control"""
        self.calculate_fps()
        
        try:
            # Convert frame to numpy array
            image = np.array(frame.array, dtype=np.uint8)
            
            # Process image to detect object
            mask, hsv = self.process_image(image)
            obj_data = self.detect_object(mask)
            
            # Update movement mode
            self.mode = self.decide_movement_mode(obj_data)
            
            # Apply appropriate control based on mode
            if self.mode == MODE_FOLLOW and obj_data:
                # Save position history
                self.position_history.append({"x": obj_data["center_x"], "y": obj_data["center_y"]})
                self.area_history.append(obj_data["area"])
                self.follow_object(obj_data)
            elif self.mode == MODE_AVOID and obj_data:
                self.avoid_close_object(obj_data)
            elif self.mode == MODE_ROTATE:
                self.rotate_to_find()
            elif self.mode == MODE_STOP:
                self.emergency_stop()
            
            # Prepare debug visualization
            debug_image = self.create_debug_view(image, mask, obj_data)
            
            return debug_image
            
        except Exception as e:
            print(f"Processing error: {e}")
            import traceback
            traceback.print_exc()
            return image
    
    def create_debug_view(self, image, mask, obj_data):
        """Create debug visualization of tracking process"""
        # Convert to BGR for OpenCV display
        debug = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
        
        # Draw crosshair at center
        cv2.line(debug, (FRAME_WIDTH//2, 0), (FRAME_WIDTH//2, FRAME_HEIGHT), (0,255,0), 1)
        cv2.line(debug, (0, FRAME_HEIGHT//2), (FRAME_WIDTH, FRAME_HEIGHT//2), (0,255,0), 1)
        
        # Draw deadzone
        cv2.rectangle(debug, 
                     (FRAME_WIDTH//2 - CENTER_DEADZONE, 0),
                     (FRAME_WIDTH//2 + CENTER_DEADZONE, FRAME_HEIGHT),
                     (100,100,100), 1)
        
        # Draw object detection
        if obj_data:
            # Draw bounding box
            x = obj_data["center_x"] - obj_data["width"]//2
            y = obj_data["center_y"] - obj_data["height"]//2
            cv2.rectangle(debug, (x, y), 
                         (x + obj_data["width"], y + obj_data["height"]), 
                         (0,0,255), 2)
            
            # Draw center point
            cv2.circle(debug, (obj_data["center_x"], obj_data["center_y"]), 5, (0,255,255), -1)
            
            # Draw rotated rectangle
            cv2.drawContours(debug, [obj_data["box"]], 0, (255,0,0), 2)
            
            # Draw prediction if available
            predicted_pos, speed = self.predict_movement()
            if predicted_pos:
                px, py = int(predicted_pos[0]), int(predicted_pos[1])
                if 0 <= px < FRAME_WIDTH and 0 <= py < FRAME_HEIGHT:
                    cv2.circle(debug, (px, py), 5, (255,0,255), -1)
                    cv2.line(debug, (obj_data["center_x"], obj_data["center_y"]), (px, py), (255,0,255), 2)
        
        # Draw position history
        points = [(p["x"], p["y"]) for p in self.position_history]
        for i in range(1, len(points)):
            cv2.line(debug, points[i-1], points[i], (0,255,0), 2)
        
        # Add status text
        cv2.putText(debug, f"Mode: {self.mode}", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(debug, f"L: {self.current_left_speed} R: {self.current_right_speed}", 
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(debug, f"FPS: {self.fps:.1f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        if obj_data:
            cv2.putText(debug, f"Area: {obj_data['area']}", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(debug, f"Aspect: {obj_data['aspect_ratio']:.2f}", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Overlay mask in corner
        mask_small = cv2.resize(mask, (FRAME_WIDTH//4, FRAME_HEIGHT//4))
        mask_color = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
        debug[0:FRAME_HEIGHT//4, 0:FRAME_WIDTH//4] = mask_color
        
        return debug
    
    def start_tracking(self):
        """Start the tracking and control loop"""
        try:
            # Initialize hardware
            self.arduino = self.initialize_serial()
            self.camera = self.initialize_camera()
            
            # Start watchdog thread
            self.running = True
            watchdog = threading.Thread(target=self.watchdog_thread)
            watchdog.daemon = True
            watchdog.start()
            
            # Main processing loop
            with picamera.array.PiRGBArray(self.camera) as stream:
                self.stream = stream
                print("Tracking started. Press q to quit, c to change color tracking.")
                
                for frame in self.camera.capture_continuous(stream, format="rgb", use_video_port=True):
                    debug_view = self.process_frame(frame)
                    
                    # Display debug view
                    cv2.imshow("Tracking", debug_view)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('c'):
                        global ACTIVE_COLOR
                        # Cycle through available colors
                        colors = list(COLOR_PROFILES.keys())
                        current_idx = colors.index(ACTIVE_COLOR)
                        next_idx = (current_idx + 1) % len(colors)
                        ACTIVE_COLOR = colors[next_idx]
                        print(f"Switched to tracking color: {ACTIVE_COLOR}")
                    
                    # Clear the stream for next frame
                    stream.truncate(0)
                
        except KeyboardInterrupt:
            print("\nExiting...")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.clean_exit()
    
    def clean_exit(self):
        """Clean shutdown of all resources"""
        self.running = False
        print("Shutting down...")
        
        # Stop motors
        if self.arduino:
            try:
                self.arduino.write(b"STOP\n")
                time.sleep(0.1)
                self.arduino.close()
            except:
                pass
        
        # Close camera
        if self.camera:
            try:
                self.camera.close()
            except:
                pass
        
        # Close windows
        cv2.destroyAllWindows()
        print("Shutdown complete.")

if __name__ == "__main__":
    controller = RobotController()
    controller.start_tracking()