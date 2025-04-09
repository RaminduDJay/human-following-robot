import cv2
import numpy as np
from collections import deque
import time

# Object Color HSV Range (Initial Guess - Will Auto-Adjust)
LOWER_HSV = np.array([0, 100, 100])
UPPER_HSV = np.array([10, 255, 255])

# Moving average filter for smoother tracking
SMOOTHING_BUFFER_SIZE = 5
VELOCITY_BUFFER_SIZE = 5  # For velocity calculation

# Grid configuration
GRID_ROWS = 10
GRID_COLS = 10

# Target rectangle aspect ratio (width/height = 14cm/6cm ≈ 2.33)
TARGET_ASPECT_RATIO = 14 / 6
ASPECT_RATIO_TOLERANCE = 0.5  # Allow some deviation from the exact ratio

class ObjectDetector:
    def __init__(self, lower_hsv=None, upper_hsv=None):
        self.lower_hsv = lower_hsv or LOWER_HSV
        self.upper_hsv = upper_hsv or UPPER_HSV
        self.grid_lines = True  # Enable/disable grid lines in visualization
        
        # Enhanced tracking with position and velocity
        self.object_positions_x = deque(maxlen=SMOOTHING_BUFFER_SIZE)
        self.object_positions_y = deque(maxlen=SMOOTHING_BUFFER_SIZE)
        self.position_timestamps = deque(maxlen=VELOCITY_BUFFER_SIZE)
        self.velocity_x = deque(maxlen=VELOCITY_BUFFER_SIZE)
        self.velocity_y = deque(maxlen=VELOCITY_BUFFER_SIZE)
        
        # Adaptive thresholding parameters
        self.auto_adjust_hsv = True
        self.last_successful_detection = None
        self.frames_since_detection = 0

    def process_image(self, image):
        """Enhanced Adaptive Color Thresholding"""
        try:
            # Convert RGB to HSV for more robust color detection under varying lighting
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Create a CLAHE object for adaptive histogram equalization on the V channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
            
            # Apply Gaussian Blur to reduce noise
            hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
            
            # Create a binary mask based on the HSV range
            mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
            
            # Clean up the mask using morphological operations
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Auto-adjust HSV range if enabled
            if self.auto_adjust_hsv and self.last_successful_detection is not None:
                # Only adjust if it's been a while since last detection
                if self.frames_since_detection > 5:
                    # Expand the HSV range slightly
                    self.lower_hsv = np.maximum(0, self.lower_hsv - np.array([1, 5, 5]))
                    self.upper_hsv = np.minimum(255, self.upper_hsv + np.array([1, 5, 5]))
                    self.frames_since_detection = 0
            
            return mask
        except Exception as e:
            print(f"Error in process_image: {e}")
            return None
            
    def detect_object(self, mask, frame_width, frame_height):
        """Find rectangular object with the target aspect ratio from the mask"""
        try:
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Return None if no contours found
            if not contours:
                self.frames_since_detection += 1
                return None
                
            # Filter for rectangle-like contours with aspect ratio close to target
            rectangles = []
            
            for contour in contours:
                # Skip small contours (noise)
                if cv2.contourArea(contour) < 500:  # Minimum area threshold
                    continue
                
                # Get the rotated rectangle (handles tilted rectangles)
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # Get rectangle width and height
                width = rect[1][0]
                height = rect[1][1]
                
                # Calculate aspect ratio (ensure width > height)
                if width < height:
                    width, height = height, width
                
                if height > 0:  # Avoid division by zero
                    aspect_ratio = width / height
                    
                    # Check if aspect ratio is close to the target
                    if abs(aspect_ratio - TARGET_ASPECT_RATIO) < ASPECT_RATIO_TOLERANCE:
                        # Calculate rectangle quality (closeness to target aspect ratio)
                        quality = 1 / (abs(aspect_ratio - TARGET_ASPECT_RATIO) + 0.1)
                        rectangles.append((box, rect, quality, contour))
            
            # If no suitable rectangles found
            if not rectangles:
                self.frames_since_detection += 1
                return None
                
            # Sort by quality (best match first)
            rectangles.sort(key=lambda x: x[2], reverse=True)
            
            # Use the best match
            best_rect = rectangles[0]
            box = best_rect[0]
            rect = best_rect[1]
            contour = best_rect[3]
            
            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                object_center_x = int(M["m10"] / M["m00"])
                object_center_y = int(M["m01"] / M["m00"])
            else:
                # Fallback to the center of the rect
                object_center_x = int(rect[0][0])
                object_center_y = int(rect[0][1])
            
            # Calculate area
            object_area = cv2.contourArea(contour)
            
            # Capture timestamp for velocity calculation
            current_time = time.time()
            
            # Update position history
            self.object_positions_x.append(object_center_x)
            self.object_positions_y.append(object_center_y)
            self.position_timestamps.append(current_time)
            
            # Calculate velocity if we have enough history
            vx, vy = 0, 0
            if len(self.object_positions_x) >= 2 and len(self.position_timestamps) >= 2:
                time_diff = self.position_timestamps[-1] - self.position_timestamps[-2]
                if time_diff > 0:
                    vx = (self.object_positions_x[-1] - self.object_positions_x[-2]) / time_diff
                    vy = (self.object_positions_y[-1] - self.object_positions_y[-2]) / time_diff
                    self.velocity_x.append(vx)
                    self.velocity_y.append(vy)
            
            # Calculate average velocity
            avg_vx = np.mean(self.velocity_x) if self.velocity_x else 0
            avg_vy = np.mean(self.velocity_y) if self.velocity_y else 0
            
            # Predict future position (for more responsive control)
            prediction_time = 0.2  # Look ahead 200ms
            predicted_x = object_center_x + int(avg_vx * prediction_time)
            predicted_y = object_center_y + int(avg_vy * prediction_time)
            
            # Smooth object position for stability
            smoothed_x = int(np.mean(self.object_positions_x))
            smoothed_y = int(np.mean(self.object_positions_y))
            
            # Map to grid
            grid_x, grid_y = self.map_to_grid(smoothed_x, smoothed_y, frame_width, frame_height)
            
            # Get the bounding box coordinates
            x_coords = [point[0] for point in box]
            y_coords = [point[1] for point in box]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Create rotated rectangle data
            angle = rect[2]
            width = max(rect[1])
            height = min(rect[1])
            aspect_ratio = width / height if height > 0 else 0
            
            # Reset frames since detection counter
            self.frames_since_detection = 0
            self.last_successful_detection = current_time
            
            # Refined HSV range if detection was successful
            if self.auto_adjust_hsv:
                # Get average color of the detected object
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [contour], 0, 255, -1)
                mean_color = cv2.mean(hsv, mask=mask)[:3]
                
                # Adjust HSV range based on the detected color
                h_range = 10
                s_range = 50
                v_range = 50
                
                # Create a new HSV range centered around the detected color
                lower_bound = np.array([max(0, mean_color[0] - h_range), 
                                       max(0, mean_color[1] - s_range),
                                       max(0, mean_color[2] - v_range)])
                upper_bound = np.array([min(179, mean_color[0] + h_range),
                                       min(255, mean_color[1] + s_range),
                                       min(255, mean_color[2] + v_range)])
                
                # Gradual adjustment (blend old and new)
                self.lower_hsv = (0.9 * self.lower_hsv + 0.1 * lower_bound).astype(np.uint8)
                self.upper_hsv = (0.9 * self.upper_hsv + 0.1 * upper_bound).astype(np.uint8)
            
            return {
                'center_x': object_center_x,
                'center_y': object_center_y,
                'smoothed_x': smoothed_x,
                'smoothed_y': smoothed_y,
                'predicted_x': predicted_x,
                'predicted_y': predicted_y,
                'velocity_x': avg_vx,
                'velocity_y': avg_vy,
                'area': object_area,
                'width': width,
                'height': height,
                'aspect_ratio': aspect_ratio,
                'angle': angle,
                'grid_position': (grid_x, grid_y),
                'bounding_box': (x_min, y_min, x_max, y_max),
                'rotated_box': box,
                'timestamp': current_time
            }
            
        except Exception as e:
            print(f"Error in detect_object: {e}")
            self.frames_since_detection += 1
            return None
    
    def map_to_grid(self, x, y, frame_width, frame_height):
        """Map pixel coordinates to grid coordinates"""
        grid_x = int((x / frame_width) * GRID_COLS)
        grid_y = int((y / frame_height) * GRID_ROWS)
        
        # Ensure within bounds
        grid_x = max(0, min(grid_x, GRID_COLS - 1))
        grid_y = max(0, min(grid_y, GRID_ROWS - 1))
        
        return grid_x, grid_y
        
    def draw_visualization(self, image, object_data):
        """Draw visualization on the image for debugging"""
        if image is None:
            return None
            
        # Create a copy to avoid modifying the original
        viz_image = image.copy()
        
        # Draw grid if enabled
        if self.grid_lines:
            self.draw_grid(viz_image)
        
        if object_data is None:
            return viz_image
            
        # Draw rotated rectangle if available
        if 'rotated_box' in object_data:
            box = object_data['rotated_box']
            cv2.drawContours(viz_image, [box], 0, (0, 255, 0), 2)
        
        # Draw standard bounding box as fallback
        elif 'bounding_box' in object_data:
           x_min, y_min, x_max, y_max = object_data['bounding_box']
           cv2.rectangle(viz_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Draw center point
        center_x = object_data['center_x']
        center_y = object_data['center_y']
        cv2.circle(viz_image, (center_x, center_y), 5, (255, 0, 0), -1)
        
        # Draw predicted position if available
        if 'predicted_x' in object_data and 'predicted_y' in object_data:
            pred_x = object_data['predicted_x']
            pred_y = object_data['predicted_y']
            cv2.circle(viz_image, (pred_x, pred_y), 5, (0, 0, 255), -1)
            # Draw line from current to predicted position
            cv2.line(viz_image, (center_x, center_y), (pred_x, pred_y), (255, 0, 255), 2)
        
        # Draw screen divisions (30% and 70% for left/right boundaries)
        h, w = viz_image.shape[:2]
        left_boundary = int(w * 0.3)
        right_boundary = int(w * 0.7)
        
        cv2.line(viz_image, (left_boundary, 0), (left_boundary, h), (255, 255, 0), 2)
        cv2.line(viz_image, (right_boundary, 0), (right_boundary, h), (255, 255, 0), 2)
                 
        # Draw text with object information
        area_text = f"Area: {object_data['area']}"
        pos_text = f"Pos: ({center_x}, {center_y})"
        grid_text = f"Grid: {object_data['grid_position']}"
        
        if 'velocity_x' in object_data and 'velocity_y' in object_data:
            vel_x = object_data['velocity_x']
            vel_y = object_data['velocity_y']
            vel_text = f"Vel: ({vel_x:.1f}, {vel_y:.1f})"
            cv2.putText(viz_image, vel_text, (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if 'aspect_ratio' in object_data:
            ratio_text = f"Ratio: {object_data['aspect_ratio']:.2f}"
            cv2.putText(viz_image, ratio_text, (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(viz_image, area_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(viz_image, pos_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(viz_image, grid_text, (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
        return viz_image
    
    def draw_grid(self, image):
        """Draw grid lines on the image for position reference"""
        h, w = image.shape[:2]
        
        # Draw vertical lines
        for i in range(1, GRID_COLS):
            x = int((i / GRID_COLS) * w)
            cv2.line(image, (x, 0), (x, h), (100, 100, 100), 1)
            
        # Draw horizontal lines
        for i in range(1, GRID_ROWS):
            y = int((i / GRID_ROWS) * h)
            cv2.line(image, (0, y), (w, y), (100, 100, 100), 1)




# import cv2
# import numpy as np
# from collections import deque

# # Object Color HSV Range (Initial Guess - Will Auto-Adjust)
# LOWER_HSV = np.array([0, 100, 100])
# UPPER_HSV = np.array([10, 255, 255])

# # Moving average filter for smoother tracking
# SMOOTHING_BUFFER_SIZE = 5
# object_positions_x = deque(maxlen=SMOOTHING_BUFFER_SIZE)
# object_positions_y = deque(maxlen=SMOOTHING_BUFFER_SIZE)

# # Grid configuration
# GRID_ROWS = 10
# GRID_COLS = 10

# # Target rectangle aspect ratio (width/height = 14cm/6cm ≈ 2.33)
# TARGET_ASPECT_RATIO = 14 / 6
# ASPECT_RATIO_TOLERANCE = 0.5  # Allow some deviation from the exact ratio

# class ObjectDetector:
#     def __init__(self, lower_hsv=None, upper_hsv=None):
#         self.lower_hsv = lower_hsv or LOWER_HSV
#         self.upper_hsv = upper_hsv or UPPER_HSV
#         self.grid_lines = True  # Enable/disable grid lines in visualization

#     def process_image(self, image):
#         """Enhanced Adaptive Color Thresholding"""
#         try:
#             # Convert RGB to HSV for more robust color detection under varying lighting
#             hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
#             # Create a CLAHE object for adaptive histogram equalization on the V channel
#             clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#             hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
            
#             # Apply Gaussian Blur to reduce noise
#             hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
            
#             # Create a binary mask based on the HSV range
#             mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
            
#             # Clean up the mask using morphological operations
#             kernel = np.ones((5, 5), np.uint8)
#             mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#             mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
#             return mask
#         except Exception as e:
#             print(f"Error in process_image: {e}")
#             return None
            
#     def detect_object(self, mask, frame_width, frame_height):
#         """Find rectangular object with the target aspect ratio from the mask"""
#         try:
#             # Find contours in the mask
#             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
#             # Return None if no contours found
#             if not contours:
#                 return None
                
#             # Filter for rectangle-like contours with aspect ratio close to target
#             rectangles = []
            
#             for contour in contours:
#                 # Skip small contours (noise)
#                 if cv2.contourArea(contour) < 500:  # Minimum area threshold
#                     continue
                
#                 # Get the rotated rectangle (handles tilted rectangles)
#                 rect = cv2.minAreaRect(contour)
#                 box = cv2.boxPoints(rect)
#                 box = np.int0(box)
                
#                 # Get rectangle width and height
#                 width = rect[1][0]
#                 height = rect[1][1]
                
#                 # Calculate aspect ratio (ensure width > height)
#                 if width < height:
#                     width, height = height, width
                
#                 if height > 0:  # Avoid division by zero
#                     aspect_ratio = width / height
                    
#                     # Check if aspect ratio is close to the target
#                     if abs(aspect_ratio - TARGET_ASPECT_RATIO) < ASPECT_RATIO_TOLERANCE:
#                         # Calculate rectangle quality (closeness to target aspect ratio)
#                         quality = 1 / (abs(aspect_ratio - TARGET_ASPECT_RATIO) + 0.1)
#                         rectangles.append((box, rect, quality, contour))
            
#             # If no suitable rectangles found
#             if not rectangles:
#                 return None
                
#             # Sort by quality (best match first)
#             rectangles.sort(key=lambda x: x[2], reverse=True)
            
#             # Use the best match
#             best_rect = rectangles[0]
#             box = best_rect[0]
#             rect = best_rect[1]
#             contour = best_rect[3]
            
#             # Calculate centroid
#             M = cv2.moments(contour)
#             if M["m00"] != 0:
#                 object_center_x = int(M["m10"] / M["m00"])
#                 object_center_y = int(M["m01"] / M["m00"])
#             else:
#                 # Fallback to the center of the rect
#                 object_center_x = int(rect[0][0])
#                 object_center_y = int(rect[0][1])
            
#             # Calculate area
#             object_area = cv2.contourArea(contour)
            
#             # Smooth object position for stability
#             object_center_x = self.smooth_position_x(object_center_x)
#             object_center_y = self.smooth_position_y(object_center_y)
            
#             # Map to grid
#             grid_x, grid_y = self.map_to_grid(object_center_x, object_center_y, frame_width, frame_height)
            
#             # Get the bounding box coordinates
#             x_coords = [point[0] for point in box]
#             y_coords = [point[1] for point in box]
#             x_min, x_max = min(x_coords), max(x_coords)
#             y_min, y_max = min(y_coords), max(y_coords)
            
#             # Create rotated rectangle data
#             angle = rect[2]
#             width = max(rect[1])
#             height = min(rect[1])
#             aspect_ratio = width / height if height > 0 else 0
            
#             return {
#                 'center_x': object_center_x,
#                 'center_y': object_center_y,
#                 'area': object_area,
#                 'width': width,
#                 'height': height,
#                 'aspect_ratio': aspect_ratio,
#                 'angle': angle,
#                 'grid_position': (grid_x, grid_y),
#                 'bounding_box': (x_min, y_min, x_max, y_max),
#                 'rotated_box': box
#             }
            
#         except Exception as e:
#             print(f"Error in detect_object: {e}")
#             return None
            
#     def smooth_position_x(self, new_position):
#         """Apply smoothing filter to object x position"""
#         object_positions_x.append(new_position)
#         return int(np.mean(object_positions_x))
        
#     def smooth_position_y(self, new_position):
#         """Apply smoothing filter to object y position"""
#         object_positions_y.append(new_position)
#         return int(np.mean(object_positions_y))
    
#     def map_to_grid(self, x, y, frame_width, frame_height):
#         """Map pixel coordinates to grid coordinates"""
#         grid_x = int((x / frame_width) * GRID_COLS)
#         grid_y = int((y / frame_height) * GRID_ROWS)
        
#         # Ensure within bounds
#         grid_x = max(0, min(grid_x, GRID_COLS - 1))
#         grid_y = max(0, min(grid_y, GRID_ROWS - 1))
        
#         return grid_x, grid_y
        
#     def draw_visualization(self, image, object_data):
#         """Draw visualization on the image for debugging"""
#         if image is None:
#             return None
            
#         # Create a copy to avoid modifying the original
#         viz_image = image.copy()
        
#         # Draw grid if enabled
#         if self.grid_lines:
#             self.draw_grid(viz_image)
        
#         if object_data is None:
#             return viz_image
            
#         # Draw rotated rectangle if available
#         if 'rotated_box' in object_data:
#             box = object_data['rotated_box']
#             cv2.drawContours(viz_image, [box], 0, (0, 255, 0), 2)
        
#         # Draw standard bounding box as fallback
#         elif 'bounding_box' in object_data:
#             x_min, y_min, x_max, y_max = object_data['bounding_box']
#             cv2.rectangle(viz_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
#         # Draw center point
#         center_x = object_data['center_x']
#         center_y = object_data['center_y']
#         cv2.circle(viz_image, (center_x, center_y), 5, (255, 0, 0), -1)
        
#         # Draw screen divisions (30% and 70% for left/right boundaries)
#         h, w = viz_image.shape[:2]
#         left_boundary = int(w * 0.3)
#         right_boundary = int(w * 0.7)
        
#         cv2.line(viz_image, (left_boundary, 0), (left_boundary, h), (255, 255, 0), 2)
#         cv2.line(viz_image, (right_boundary, 0), (right_boundary, h), (255, 255, 0), 2)
                 
#         # Draw text with object information
#         area_text = f"Area: {object_data['area']}"
#         pos_text = f"Pos: ({center_x}, {center_y})"
#         grid_text = f"Grid: {object_data['grid_position']}"
        
#         if 'aspect_ratio' in object_data:
#             ratio_text = f"Ratio: {object_data['aspect_ratio']:.2f}"
#             cv2.putText(viz_image, ratio_text, (10, 120), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
#         cv2.putText(viz_image, area_text, (10, 30), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#         cv2.putText(viz_image, pos_text, (10, 60), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#         cv2.putText(viz_image, grid_text, (10, 90), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
#         return viz_image
    
#     def draw_grid(self, image):
#         """Draw grid lines on the image for position reference"""
#         h, w = image.shape[:2]
        
#         # Draw vertical lines
#         for i in range(1, GRID_COLS):
#             x = int((i / GRID_COLS) * w)
#             cv2.line(image, (x, 0), (x, h), (100, 100, 100), 1)
            
#         # Draw horizontal lines
#         for i in range(1, GRID_ROWS):
#             y = int((i / GRID_ROWS) * h)
#             cv2.line(image, (0, y), (w, y), (100, 100, 100), 1)




# import cv2
# import numpy as np
# from collections import deque

# # Object Color HSV Range (Initial Guess - Will Auto-Adjust)
# LOWER_HSV = np.array([109, 115, 63])
# UPPER_HSV = np.array([117, 150, 88])

# # Moving average filter for smoother tracking
# SMOOTHING_BUFFER_SIZE = 5
# object_positions_x = deque(maxlen=SMOOTHING_BUFFER_SIZE)
# object_positions_y = deque(maxlen=SMOOTHING_BUFFER_SIZE)

# # Grid configuration
# GRID_ROWS = 10
# GRID_COLS = 10

# # Rectangle detection parameters
# TARGET_ASPECT_RATIO = 14/6  # 14cm / 6cm = 2.33
# ASPECT_RATIO_TOLERANCE = 0.3  # Allow some deviation from exact ratio

# class ObjectDetector:
#     def __init__(self, lower_hsv=None, upper_hsv=None):
#         self.lower_hsv = lower_hsv or LOWER_HSV
#         self.upper_hsv = upper_hsv or UPPER_HSV
#         self.grid_lines = True  # Enable/disable grid lines in visualization

#     def process_image(self, image):
#         """Enhanced Adaptive Color Thresholding"""
#         try:
#             # Convert RGB to HSV for more robust color detection under varying lighting
#             hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
#             # Create a CLAHE object for adaptive histogram equalization on the V channel
#             clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#             hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
            
#             # Apply Gaussian Blur to reduce noise
#             hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
            
#             # Create a binary mask based on the HSV range
#             mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
            
#             # Clean up the mask using morphological operations
#             kernel = np.ones((5, 5), np.uint8)
#             mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#             mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
#             return mask
#         except Exception as e:
#             print(f"Error in process_image: {e}")
#             return None
            
#     def detect_object(self, mask, frame_width, frame_height):
#         """Find rectangle with specific aspect ratio from the mask"""
#         try:
#             # Find contours in the mask
#             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
#             # If no contours found, return None
#             if not contours:
#                 return None
                
#             # Filter contours to find rectangles with appropriate aspect ratio
#             best_match = None
#             best_score = float('inf')
            
#             for contour in contours:
#                 # Skip small contours (noise)
#                 if cv2.contourArea(contour) < 100:
#                     continue
                    
#                 # Get rotated rectangle (handles rotation)
#                 rect = cv2.minAreaRect(contour)
#                 box = cv2.boxPoints(rect)
#                 box = np.int0(box)
                
#                 # Get width and height of the rectangle
#                 width, height = rect[1]
                
#                 # Ensure width is the longer side
#                 if width < height:
#                     width, height = height, width
                    
#                 # Skip if height is zero (prevents division by zero)
#                 if height == 0:
#                     continue
                    
#                 # Calculate aspect ratio
#                 aspect_ratio = width / height
                
#                 # Check how close this aspect ratio is to our target
#                 ratio_diff = abs(aspect_ratio - TARGET_ASPECT_RATIO)
                
#                 # If within tolerance and better than previous match, update best match
#                 if ratio_diff < ASPECT_RATIO_TOLERANCE and ratio_diff < best_score:
#                     best_score = ratio_diff
                    
#                     # Get the rectangle information
#                     x_coords = [pt[0] for pt in box]
#                     y_coords = [pt[1] for pt in box]
                    
#                     x_min, x_max = min(x_coords), max(x_coords)
#                     y_min, y_max = min(y_coords), max(y_coords)
                    
#                     # Calculate center of rectangle
#                     object_center_x = int((x_min + x_max) / 2)
#                     object_center_y = int((y_min + y_max) / 2)
                    
#                     # Smooth positions
#                     object_center_x = self.smooth_position_x(object_center_x)
#                     object_center_y = self.smooth_position_y(object_center_y)
                    
#                     # Area - use contour area for more accurate measurement
#                     object_area = cv2.contourArea(contour)
                    
#                     # Store rectangle information
#                     best_match = {
#                         'center_x': object_center_x,
#                         'center_y': object_center_y,
#                         'area': object_area,
#                         'width': int(width),
#                         'height': int(height),
#                         'aspect_ratio': aspect_ratio,
#                         'rotated_box': box,
#                         'grid_position': self.map_to_grid(object_center_x, object_center_y, frame_width, frame_height),
#                         'bounding_box': (x_min, y_min, x_max, y_max)
#                     }
            
#             return best_match
#         except Exception as e:
#             print(f"Error in detect_object: {e}")
#             return None
            
#     def smooth_position_x(self, new_position):
#         """Apply smoothing filter to object x position"""
#         object_positions_x.append(new_position)
#         return int(np.mean(object_positions_x))
        
#     def smooth_position_y(self, new_position):
#         """Apply smoothing filter to object y position"""
#         object_positions_y.append(new_position)
#         return int(np.mean(object_positions_y))
    
#     def map_to_grid(self, x, y, frame_width, frame_height):
#         """Map pixel coordinates to grid coordinates"""
#         grid_x = int((x / frame_width) * GRID_COLS)
#         grid_y = int((y / frame_height) * GRID_ROWS)
        
#         # Ensure within bounds
#         grid_x = max(0, min(grid_x, GRID_COLS - 1))
#         grid_y = max(0, min(grid_y, GRID_ROWS - 1))
        
#         return grid_x, grid_y
        
#     def draw_visualization(self, image, object_data):
#         """Draw enhanced visualization on the image for debugging"""
#         if image is None:
#             return None
            
#         # Create a copy to avoid modifying the original
#         viz_image = image.copy()
        
#         # Draw grid if enabled
#         if self.grid_lines:
#             self.draw_grid(viz_image)
        
#         if object_data is None:
#             return viz_image
            
#         # Draw rotated bounding box if available (better for rectangles)
#         if 'rotated_box' in object_data:
#             box = object_data['rotated_box']
#             cv2.drawContours(viz_image, [box], 0, (0, 255, 0), 2)
#         # Fall back to regular bounding box if needed
#         elif 'bounding_box' in object_data:
#             x_min, y_min, x_max, y_max = object_data['bounding_box']
#             cv2.rectangle(viz_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
#         # Draw center point
#         center_x = object_data['center_x']
#         center_y = object_data['center_y']
#         cv2.circle(viz_image, (center_x, center_y), 5, (255, 0, 0), -1)
        
#         # Draw screen divisions (30% and 70% for left/right boundaries)
#         h, w = viz_image.shape[:2]
#         left_boundary = int(w * 0.3)
#         right_boundary = int(w * 0.7)
        
#         cv2.line(viz_image, (left_boundary, 0), (left_boundary, h), (255, 255, 0), 2)
#         cv2.line(viz_image, (right_boundary, 0), (right_boundary, h), (255, 255, 0), 2)
                 
#         # Draw text with object information
#         area_text = f"Area: {object_data['area']:.0f}"
#         pos_text = f"Pos: ({center_x}, {center_y})"
#         grid_text = f"Grid: {object_data['grid_position']}"
        
#         # Add aspect ratio information
#         if 'aspect_ratio' in object_data:
#             ratio_text = f"Ratio: {object_data['aspect_ratio']:.2f}"
#             cv2.putText(viz_image, ratio_text, (10, 120), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
#         cv2.putText(viz_image, area_text, (10, 30), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#         cv2.putText(viz_image, pos_text, (10, 60), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#         cv2.putText(viz_image, grid_text, (10, 90), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
#         return viz_image
    
#     def draw_grid(self, image):
#         """Draw grid lines on the image for position reference"""
#         h, w = image.shape[:2]
        
#         # Draw vertical lines
#         for i in range(1, GRID_COLS):
#             x = int((i / GRID_COLS) * w)
#             cv2.line(image, (x, 0), (x, h), (100, 100, 100), 1)
            
#         # Draw horizontal lines
#         for i in range(1, GRID_ROWS):
#             y = int((i / GRID_ROWS) * h)
#             cv2.line(image, (0, y), (w, y), (100, 100, 100), 1)




# import cv2
# import numpy as np
# from collections import deque

# # Object Color HSV Range (Initial Guess - Will Auto-Adjust)
# LOWER_HSV = np.array([0, 100, 100])
# UPPER_HSV = np.array([10, 255, 255])

# # Moving average filter for smoother tracking
# SMOOTHING_BUFFER_SIZE = 5
# object_positions_x = deque(maxlen=SMOOTHING_BUFFER_SIZE)
# object_positions_y = deque(maxlen=SMOOTHING_BUFFER_SIZE)

# # Grid configuration
# GRID_ROWS = 10
# GRID_COLS = 10

# class ObjectDetector:
#     def __init__(self, lower_hsv=None, upper_hsv=None):
#         self.lower_hsv = lower_hsv or LOWER_HSV
#         self.upper_hsv = upper_hsv or UPPER_HSV
#         self.grid_lines = True  # Enable/disable grid lines in visualization

#     def process_image(self, image):
#         """Enhanced Adaptive Color Thresholding"""
#         try:
#             # Convert RGB to HSV for more robust color detection under varying lighting
#             hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
#             # Create a CLAHE object for adaptive histogram equalization on the V channel
#             clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#             hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
            
#             # Apply Gaussian Blur to reduce noise
#             hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
            
#             # Create a binary mask based on the HSV range
#             mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
            
#             # Clean up the mask using morphological operations
#             kernel = np.ones((5, 5), np.uint8)
#             mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#             mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
#             return mask
#         except Exception as e:
#             print(f"Error in process_image: {e}")
#             return None
            
#     def detect_object(self, mask, frame_width, frame_height):
#         """Find object center and area from the mask"""
#         try:
#             # Find object position from mask
#             indices = np.where(mask > 0)
#             if len(indices[0]) > 0:
#                 y_indices, x_indices = indices[0], indices[1]
                
#                 # Calculate centroid
#                 x_min, x_max = min(x_indices), max(x_indices)
#                 y_min, y_max = min(y_indices), max(y_indices)
                
#                 object_center_x = (x_min + x_max) // 2
#                 object_center_y = (y_min + y_max) // 2
                
#                 # Calculate bounding box dimensions
#                 width = x_max - x_min
#                 height = y_max - y_min
                
#                 # Calculate area
#                 object_area = len(indices[0])  # Approximate object size based on pixel count
                
#                 # Smooth object position for stability
#                 object_center_x = self.smooth_position_x(object_center_x)
#                 object_center_y = self.smooth_position_y(object_center_y)
                
#                 # Map to grid
#                 grid_x, grid_y = self.map_to_grid(object_center_x, object_center_y, frame_width, frame_height)
                
#                 return {
#                     'center_x': object_center_x,
#                     'center_y': object_center_y,
#                     'area': object_area,
#                     'width': width,
#                     'height': height,
#                     'grid_position': (grid_x, grid_y),
#                     'bounding_box': (x_min, y_min, x_max, y_max)
#                 }
#             return None
#         except Exception as e:
#             print(f"Error in detect_object: {e}")
#             return None
            
#     def smooth_position_x(self, new_position):
#         """Apply smoothing filter to object x position"""
#         object_positions_x.append(new_position)
#         return int(np.mean(object_positions_x))
        
#     def smooth_position_y(self, new_position):
#         """Apply smoothing filter to object y position"""
#         object_positions_y.append(new_position)
#         return int(np.mean(object_positions_y))
    
#     def map_to_grid(self, x, y, frame_width, frame_height):
#         """Map pixel coordinates to grid coordinates"""
#         grid_x = int((x / frame_width) * GRID_COLS)
#         grid_y = int((y / frame_height) * GRID_ROWS)
        
#         # Ensure within bounds
#         grid_x = max(0, min(grid_x, GRID_COLS - 1))
#         grid_y = max(0, min(grid_y, GRID_ROWS - 1))
        
#         return grid_x, grid_y
        
#     def draw_visualization(self, image, object_data):
#         """Draw visualization on the image for debugging"""
#         if image is None:
#             return None
            
#         # Create a copy to avoid modifying the original
#         viz_image = image.copy()
        
#         # Draw grid if enabled
#         if self.grid_lines:
#             self.draw_grid(viz_image)
        
#         if object_data is None:
#             return viz_image
            
#         # Draw bounding box
#         if 'bounding_box' in object_data:
#             x_min, y_min, x_max, y_max = object_data['bounding_box']
#             cv2.rectangle(viz_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
#         # Draw center point
#         center_x = object_data['center_x']
#         center_y = object_data['center_y']
#         cv2.circle(viz_image, (center_x, center_y), 5, (255, 0, 0), -1)
        
#         # Draw screen divisions (30% and 70% for left/right boundaries)
#         h, w = viz_image.shape[:2]
#         left_boundary = int(w * 0.3)
#         right_boundary = int(w * 0.7)
        
#         cv2.line(viz_image, (left_boundary, 0), (left_boundary, h), (255, 255, 0), 2)
#         cv2.line(viz_image, (right_boundary, 0), (right_boundary, h), (255, 255, 0), 2)
                 
#         # Draw text with object information
#         area_text = f"Area: {object_data['area']}"
#         pos_text = f"Pos: ({center_x}, {center_y})"
#         grid_text = f"Grid: {object_data['grid_position']}"
        
#         cv2.putText(viz_image, area_text, (10, 30), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#         cv2.putText(viz_image, pos_text, (10, 60), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#         cv2.putText(viz_image, grid_text, (10, 90), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
#         return viz_image
    
#     def draw_grid(self, image):
#         """Draw grid lines on the image for position reference"""
#         h, w = image.shape[:2]
        
#         # Draw vertical lines
#         for i in range(1, GRID_COLS):
#             x = int((i / GRID_COLS) * w)
#             cv2.line(image, (x, 0), (x, h), (100, 100, 100), 1)
            
#         # Draw horizontal lines
#         for i in range(1, GRID_ROWS):
#             y = int((i / GRID_ROWS) * h)
#             cv2.line(image, (0, y), (w, y), (100, 100, 100), 1)




# import cv2
# import numpy as np
# from collections import deque

# # Object Color HSV Range (Initial Guess - Will Auto-Adjust)
# LOWER_HSV = np.array([0, 100, 100])
# UPPER_HSV = np.array([10, 255, 255])

# # Moving average filter for smoother tracking
# SMOOTHING_BUFFER_SIZE = 5
# object_positions = deque(maxlen=SMOOTHING_BUFFER_SIZE)

# class ObjectDetector:
#     def __init__(self, lower_hsv=None, upper_hsv=None):
#         self.lower_hsv = lower_hsv or LOWER_HSV
#         self.upper_hsv = upper_hsv or UPPER_HSV

#     def process_image(self, image):
#         """Enhanced Adaptive Color Thresholding"""
#         try:
#             # Convert RGB to HSV for more robust color detection under varying lighting
#             hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
#             # Create a CLAHE object for adaptive histogram equalization on the V channel
#             clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#             hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
            
#             # Apply Gaussian Blur to reduce noise
#             hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
            
#             # Create a binary mask based on the HSV range
#             mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
            
#             # Clean up the mask using morphological operations
#             kernel = np.ones((5, 5), np.uint8)
#             mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#             mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
#             return mask
#         except Exception as e:
#             print(f"Error in process_image: {e}")
#             return None
            
#     def detect_object(self, mask, frame_width):
#         """Find object center and area from the mask"""
#         # Define screen sections for movement decisions
#         section_width = frame_width // 4
#         middle_start = section_width
#         middle_end = 3 * section_width
        
#         try:
#             # Find object position from mask
#             indices = np.where(mask > 0)
#             if len(indices[0]) > 0:
#                 x_min, x_max = min(indices[1]), max(indices[1])
#                 object_center_x = (x_min + x_max) // 2
#                 object_area = len(indices[0])  # Approximate object size based on pixel count
                
#                 # Smooth object center for stability
#                 object_center_x = self.smooth_position(object_center_x)
                
#                 return {
#                     'center_x': object_center_x,
#                     'area': object_area,
#                     'screen_sections': {
#                         'section_width': section_width,
#                         'middle_start': middle_start,
#                         'middle_end': middle_end
#                     }
#                 }
#             return None
#         except Exception as e:
#             print(f"Error in detect_object: {e}")
#             return None
            
#     def smooth_position(self, new_position):
#         """Apply smoothing filter to object position"""
#         object_positions.append(new_position)
#         return int(np.mean(object_positions))
        
#     def draw_visualization(self, image, object_data):
#         """Draw visualization on the image for debugging"""
#         if object_data is None:
#             return image
            
#         # Create a copy to avoid modifying the original
#         viz_image = image.copy()
#         center_x = object_data['center_x']
#         sections = object_data['screen_sections']
        
#         # Draw center line
#         cv2.circle(viz_image, (center_x, viz_image.shape[0]//2), 10, (0, 255, 0), -1)
        
#         # Draw section lines
#         cv2.line(viz_image, (sections['section_width'], 0), 
#                  (sections['section_width'], viz_image.shape[0]), (255, 0, 0), 2)
#         cv2.line(viz_image, (sections['middle_end'], 0), 
#                  (sections['middle_end'], viz_image.shape[0]), (255, 0, 0), 2)
                 
#         # Draw text with object area
#         area_text = f"Area: {object_data['area']}"
#         cv2.putText(viz_image, area_text, (10, 30), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
#         return viz_image




# import cv2
# import numpy as np

# # Define object color HSV range
# LOWER_HSV = np.array([0, 100, 100])
# UPPER_HSV = np.array([10, 255, 255])

# def process_image(frame):
#     try:
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
#         return mask
#     except Exception as e:
#         print(f"Error in process_image: {e}")
#         return None

# def find_object_position(mask):
#     indices = np.where(mask > 0)
#     if len(indices[0]) > 0:
#         x_min, x_max = min(indices[1]), max(indices[1])
#         object_center_x = (x_min + x_max) // 2
#         object_area = len(indices[0])  # Approximate object size
#         return object_center_x, object_area
#     return None, None





# import cv2
# import numpy as np

# LOWER_HSV = np.array([0, 100, 100])
# UPPER_HSV = np.array([10, 255, 255])

# def process_image(image):
#     try:
#         hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#         hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
#         hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
#         mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
#         kernel = np.ones((5, 5), np.uint8)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#         return mask
#     except Exception as e:
#         print(f"Error in process_image: {e}")
#         return None
