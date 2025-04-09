import time
from collections import deque
from pid_controller import PID
from utils import command_changed

# Distance control thresholds
DESIRED_AREA = 2500  # Target object area corresponding to 1 m distance
MAX_SPEED = 255      # Maximum motor speed value
MIN_SPEED = 50       # Minimum motor speed value for forward movement

# Movement state constants
STATE_IDLE = "IDLE"
STATE_TURNING = "TURNING"
STATE_MOVING = "MOVING"
STATE_CENTERED = "CENTERED"

class MovementController:
    def __init__(self, serial_connection, desired_area=None):
        self.serial = serial_connection
        self.previous_command = None
        self.desired_area = desired_area or DESIRED_AREA
        
        # Create PID controllers
        self.distance_pid = PID(
            Kp=0.1, 
            Ki=0.01, 
            Kd=0.05, 
            setpoint=self.desired_area
        )
        self.distance_pid.set_output_limits(0, MAX_SPEED)
        
        self.steering_pid = PID(
            Kp=0.5,
            Ki=0.0,  # No integral for steering to avoid oscillation
            Kd=0.1,
            setpoint=0  # Target is center (0 deviation)
        )
        self.steering_pid.set_output_limits(-100, 100)
        
        # State management for smoother transitions
        self.current_state = STATE_IDLE
        self.state_entry_time = time.time()
        self.turn_stable_count = 0
        self.centered_count = 0
        
        # Moving average for command smoothing
        self.speed_history = deque(maxlen=3)
        self.steering_history = deque(maxlen=3)
        
        # Timing and history
        self.last_command_time = time.time()
        self.min_command_interval = 0.05  # 50ms minimum between commands
        
        # Frame dimensions (will be set on first call)
        self.frame_width = None
        self.frame_height = None
        
    def is_centered(self, x_pos, frame_width):
        """Check if object is centered in frame"""
        center = frame_width // 2
        tolerance = frame_width // 10  # 10% tolerance
        return abs(x_pos - center) < tolerance
    
    def adjust_pid_parameters(self, deviation_percentage):
        """Dynamically adjust PID parameters based on position deviation"""
        if deviation_percentage > 0.7:  # Far from center
            self.steering_pid.set_tunings(0.7, 0.0, 0.15)  # More aggressive
        elif deviation_percentage < 0.3:  # Close to center
            self.steering_pid.set_tunings(0.3, 0.0, 0.05)  # Gentler control
        else:  # Medium distance
            self.steering_pid.set_tunings(0.5, 0.0, 0.1)  # Standard settings
    
    def get_smooth_average(self, value, history_deque):
        """Apply simple moving average to smooth values"""
        history_deque.append(value)
        return sum(history_deque) / len(history_deque)
    
    def determine_movement(self, object_data, frame_width=None, frame_height=None):
        """Determine what movement command to send based on object position and velocity"""
        # Store frame dimensions
        if frame_width is not None and frame_height is not None:
            self.frame_width = frame_width
            self.frame_height = frame_height
        
        # Default to stop if no object detected
        if object_data is None:
            # If we've been in the idle state too long, reset PIDs
            if self.current_state != STATE_IDLE or (time.time() - self.state_entry_time > 1.0):
                self.current_state = STATE_IDLE
                self.state_entry_time = time.time()
                self.distance_pid.reset()
                self.steering_pid.reset()
            return "STOP"
        
        # Use frame dimensions
        if self.frame_width is None or self.frame_height is None:
            print("Warning: Frame dimensions not set")
            return "STOP"
            
        # Calculate frame center and deviation
        frame_center = self.frame_width // 2
        
        # Prefer predicted position if available for more responsive control
        if 'predicted_x' in object_data:
            center_x = object_data['predicted_x']
        else:
            center_x = object_data['center_x']
            
        object_area = object_data['area']
        deviation = center_x - frame_center
        deviation_percentage = abs(deviation) / (self.frame_width // 2)
        
        # Adjust PID parameters based on deviation
        self.adjust_pid_parameters(deviation_percentage)
        
        # State machine for smoother transitions
        now = time.time()
        
        # Handle state transitions
        if self.current_state == STATE_IDLE:
            # Just detected an object, reset and start tracking
            self.current_state = STATE_TURNING if deviation_percentage > 0.3 else STATE_CENTERED
            self.state_entry_time = now
            self.turn_stable_count = 0
            self.centered_count = 0
            
        elif self.current_state == STATE_TURNING:
            # Check if we're getting aligned
            if self.is_centered(center_x, self.frame_width):
                self.turn_stable_count += 1
                if self.turn_stable_count >= 3:  # Require 3 consecutive centered frames
                    self.current_state = STATE_CENTERED
                    self.state_entry_time = now
                    self.centered_count = 0
            else:
                # Not centered anymore, reset counter
                self.turn_stable_count = 0
                
        elif self.current_state == STATE_CENTERED:
            # Check if we're still centered
            if not self.is_centered(center_x, self.frame_width):
                self.centered_count += 1
                if self.centered_count >= 3:  # Require 3 consecutive off-center frames
                    self.current_state = STATE_TURNING
                    self.state_entry_time = now
                    self.turn_stable_count = 0
            else:
                # Still centered, reset counter
                self.centered_count = 0
                
        # Calculate steering using PID (higher deviation = stronger turn)
        normalized_deviation = deviation / (self.frame_width / 2)  # -1 to 1
        steering_output = self.steering_pid.update(normalized_deviation)
        
        # Smooth steering output
        smooth_steering = self.get_smooth_average(steering_output, self.steering_history)
        
        # For turning state, prioritize alignment
        if self.current_state == STATE_TURNING:
            if smooth_steering < -40:  # Strong left turn needed
                return "TURN LEFT 100"  # Full power turn
            elif smooth_steering > 40:  # Strong right turn needed
                return "TURN RIGHT 100"  # Full power turn
            elif smooth_steering < -20:  # Moderate left turn needed
                return f"TURN LEFT {abs(int(smooth_steering * 2))}"
            elif smooth_steering > 20:  # Moderate right turn needed
                return f"TURN RIGHT {abs(int(smooth_steering * 2))}"
            else:
                # Almost centered, use proportional turning with forward movement
                # Use PID for distance control
                speed_output = self.distance_pid.update(object_area)
                speed_output = max(MIN_SPEED, min(MAX_SPEED, speed_output))
                
                # Apply smooth acceleration/deceleration
                smooth_speed = self.get_smooth_average(speed_output, self.speed_history)
                
                # Combine movement with slight steering correction
                turn_direction = "LEFT" if smooth_steering < 0 else "RIGHT"
                correction = abs(int(smooth_steering))
                return f"MOVE {int(smooth_speed)} {turn_direction} {correction}"
                
        # For centered state, focus on distance control
        elif self.current_state == STATE_CENTERED:
            # Use PID for distance control
            speed_output = self.distance_pid.update(object_area)
            speed_output = max(MIN_SPEED, min(MAX_SPEED, speed_output))
            
            # Apply smooth acceleration/deceleration
            smooth_speed = self.get_smooth_average(speed_output, self.speed_history)
            
            # If there's still some deviation, add a slight steering correction
            if abs(normalized_deviation) > 0.1:
                turn_direction = "LEFT" if normalized_deviation < 0 else "RIGHT"
                correction = min(30, abs(int(normalized_deviation * 50)))
                return f"MOVE {int(smooth_speed)} {turn_direction} {correction}"
            else:
                # Perfectly centered - straight ahead
                return f"MOVE {int(smooth_speed)}"
                
        # Default/fallback for any other state
        return "STOP"
    
    def send_movement_command(self, movement):
        """Send movement command to Arduino if changed or sufficient time has passed"""
        now = time.time()
        
        # Only send command if it's different or minimum interval has passed
        if (command_changed(movement, self.previous_command) or 
            (now - self.last_command_time) > self.min_command_interval):
            
            print(f"Sending Command: {movement}")
            try:
                self.serial.send_command(movement)
                self.previous_command = movement
                self.last_command_time = now
            except Exception as e:
                print(f"Error sending movement command: {e}")
    
    def process_frame(self, object_data, frame_width, frame_height):
        """Process a single frame and determine/send the appropriate movement command"""
        movement = self.determine_movement(object_data, frame_width, frame_height)
        self.send_movement_command(movement)
        return movement




# from pid_controller import PID
# from utils import command_changed

# # Distance control thresholds
# DESIRED_AREA = 2500  # Target object area corresponding to 1 m distance
# MAX_SPEED = 255      # Maximum motor speed value
# MIN_SPEED = 50       # Minimum motor speed value for forward movement

# class MovementController:
#     def __init__(self, serial_connection, desired_area=None):
#         self.serial = serial_connection
#         self.previous_command = None
#         self.desired_area = desired_area or DESIRED_AREA
        
#         # Create PID controller instance
#         self.pid_controller = PID(
#             Kp=0.1, 
#             Ki=0.01, 
#             Kd=0.05, 
#             setpoint=self.desired_area
#         )
        
#     # def determine_movement(self, object_data):
#     #     """Determine what movement command to send based on object position"""
#     #     if object_data is None:
#     #         return "STOP"
            
#     #     center_x = object_data['center_x']
#     #     object_area = object_data['area']
#     #     sections = object_data['screen_sections']
        
#     #     # Determine turning commands based on horizontal position of the object
#     #     if center_x < sections['section_width']:
#     #         movement = "TURN LEFT"
#     #     elif center_x > sections['middle_end']:
#     #         movement = "TURN RIGHT"
#     #     elif sections['middle_start'] <= center_x <= sections['middle_end']:
#     #         # Use PID controller to set speed smoothly
#     #         speed_output = self.pid_controller.update(object_area)
            
#     #         # Clamp the speed to within allowed limits
#     #         if speed_output < 0:
#     #             speed_output = 0
#     #         if speed_output > MAX_SPEED:
#     #             speed_output = MAX_SPEED
                
#     #         # Ensure a minimum forward speed if nonzero
#     #         if 0 < speed_output < MIN_SPEED:
#     #             speed_output = MIN_SPEED
                
#     #         movement = f"MOVE {int(speed_output)}"
#     #     else:
#     #         movement = "STOP"
            
#     #     return movement
    
#     def determine_movement(self, object_data, frame_width=None, frame_height=None):
#         """Determine what movement command to send based on object position"""
#         if object_data is None:
#             return "STOP"
            
#         center_x = object_data['center_x']
#         object_area = object_data['area']
        
#         # Calculate screen sections within the method
#         section_width = frame_width // 3
#         middle_start = section_width
#         middle_end = section_width * 2
        
#         # Determine turning commands based on horizontal position of the object
#         if center_x < middle_start:
#             movement = "TURN LEFT"
#         elif center_x > middle_end:
#             movement = "TURN RIGHT"
#         elif middle_start <= center_x <= middle_end:
#             # Use PID controller to set speed smoothly
#             speed_output = self.pid_controller.update(object_area)
            
#             # Clamp the speed to within allowed limits
#             if speed_output < 0:
#                 speed_output = 0
#             if speed_output > MAX_SPEED:
#                 speed_output = MAX_SPEED
                
#             # Ensure a minimum forward speed if nonzero
#             if 0 < speed_output < MIN_SPEED:
#                 speed_output = MIN_SPEED
                
#             movement = f"MOVE {int(speed_output)}"
#         else:
#             movement = "STOP"
            
#         return movement
    
        
#     def send_movement_command(self, movement):
#         """Send movement command to Arduino if changed"""
#         # Only send command if it has changed (with deadband filtering for MOVE commands)
#         if command_changed(movement, self.previous_command):
#             print(f"Sending Command: {movement}")
#             self.serial.send_command(movement)
#             self.previous_command = movement




# from pid_controller import PID
# from utils import command_changed

# # Distance control thresholds
# DESIRED_AREA = 2500  # Target object area corresponding to 1 m distance
# MAX_SPEED = 255      # Maximum motor speed value
# MIN_SPEED = 50       # Minimum motor speed value for forward movement

# class MovementController:
#     def __init__(self, serial_connection, desired_area=None):
#         self.serial = serial_connection
#         self.previous_command = None
#         self.desired_area = desired_area or DESIRED_AREA
        
#         # Create PID controller instance
#         self.pid_controller = PID(
#             Kp=0.1, 
#             Ki=0.01, 
#             Kd=0.05, 
#             setpoint=self.desired_area
#         )
        
#     def determine_movement(self, object_data):
#         """Determine what movement command to send based on object position"""
#         if object_data is None:
#             return "STOP"
            
#         center_x = object_data['center_x']
#         object_area = object_data['area']
#         sections = object_data['screen_sections']
        
#         # Determine turning commands based on horizontal position of the object
#         if center_x < sections['section_width']:
#             movement = "TURN LEFT"
#         elif center_x > sections['middle_end']:
#             movement = "TURN RIGHT"
#         elif sections['middle_start'] <= center_x <= sections['middle_end']:
#             # Use PID controller to set speed smoothly
#             speed_output = self.pid_controller.update(object_area)
            
#             # Clamp the speed to within allowed limits
#             if speed_output < 0:
#                 speed_output = 0
#             if speed_output > MAX_SPEED:
#                 speed_output = MAX_SPEED
                
#             # Ensure a minimum forward speed if nonzero
#             if 0 < speed_output < MIN_SPEED:
#                 speed_output = MIN_SPEED
                
#             movement = f"MOVE {int(speed_output)}"
#         else:
#             movement = "STOP"
            
#         return movement
        
#     def send_movement_command(self, movement):
#         """Send movement command to Arduino if changed"""
#         # Only send command if it has changed (with deadband filtering for MOVE commands)
#         if command_changed(movement, self.previous_command):
#             print(f"Sending Command: {movement}")
#             self.serial.send_command(movement)
#             self.previous_command = movement




# def decide_movement(object_center_x, object_area, pid_controller):
#     FRAME_WIDTH = 320
#     SECTION_WIDTH = FRAME_WIDTH // 4
#     MIDDLE_START = SECTION_WIDTH
#     MIDDLE_END = 3 * SECTION_WIDTH

#     if object_center_x is None:
#         return "STOP"

#     if object_center_x < SECTION_WIDTH:
#         return "TURN LEFT"
#     elif object_center_x > 3 * SECTION_WIDTH:
#         return "TURN RIGHT"
#     elif MIDDLE_START <= object_center_x <= MIDDLE_END:
#         speed = pid_controller.update(object_area)
#         speed = max(50, min(255, int(speed)))  # Clamp speed
#         return f"MOVE {speed}"
    
#     return "STOP"





# import numpy as np
# from collections import deque
# from pid_controller import PID
# from serial_comm import send_command

# # Define screen sections
# FRAME_WIDTH = 320
# SECTION_WIDTH = FRAME_WIDTH // 4
# MIDDLE_START = SECTION_WIDTH
# MIDDLE_END = 3 * SECTION_WIDTH

# # PID Controller for distance control
# DESIRED_AREA = 2500
# MAX_SPEED = 255
# MIN_SPEED = 50
# pid_controller = PID(Kp=0.1, Ki=0.01, Kd=0.05, setpoint=DESIRED_AREA)

# object_positions = deque(maxlen=5)
# previous_command = None

# def smooth_position(new_position):
#     object_positions.append(new_position)
#     return int(np.mean(object_positions))

# def command_changed(new_command, old_command, threshold=5):
#     if new_command.startswith("MOVE") and old_command and old_command.startswith("MOVE"):
#         try:
#             new_speed = int(new_command.split()[1])
#             old_speed = int(old_command.split()[1])
#             return abs(new_speed - old_speed) >= threshold
#         except:
#             return True
#     return new_command != old_command

# def process_frame(frame, arduino):
#     global previous_command
#     movement = "STOP"

#     image = np.array(frame.array, dtype=np.uint8)
#     mask = process_image(image)
#     if mask is None:
#         print("Error: Mask generation failed.")
#         return

#     indices = np.where(mask > 0)
#     if len(indices[0]) > 0:
#         x_min, x_max = min(indices[1]), max(indices[1])
#         object_center_x = (x_min + x_max) // 2
#         object_area = len(indices[0])

#         object_center_x = smooth_position(object_center_x)

#         if object_center_x < SECTION_WIDTH:
#             movement = "TURN LEFT"
#         elif object_center_x > 3 * SECTION_WIDTH:
#             movement = "TURN RIGHT"
#         elif MIDDLE_START <= object_center_x <= MIDDLE_END:
#             speed_output = pid_controller.update(object_area)
#             speed_output = max(min(speed_output, MAX_SPEED), 0)
#             if 0 < speed_output < MIN_SPEED:
#                 speed_output = MIN_SPEED
#             movement = f"MOVE {int(speed_output)}"

#     if command_changed(movement, previous_command):
#         print(f"Sending Command: {movement}")
#         send_command(arduino, movement)
#         previous_command = movement
