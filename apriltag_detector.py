import cv2
import numpy as np
from pupil_apriltags import Detector
from collections import deque

# Grid configuration (same as original)
GRID_ROWS = 10
GRID_COLS = 10

# Target tag parameters
TARGET_TAG_FAMILY = 'tag25h9'
TARGET_TAG_ID = 12

# Exponential smoothing: last known positions and smoothing factor
last_position_x = None
last_position_y = None

class AprilTagDetector:
    def __init__(self):
        # Initialize AprilTag detector
        self.detector = Detector(
            families=TARGET_TAG_FAMILY,
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )
        
        self.grid_lines = True  # Enable grid overlay in visualization
        
        # Improved adaptive smoothing with velocity-based adjustment
        self.smoothing_alpha = 0.5  # Starting value
        self.min_alpha = 0.3  # More aggressive smoothing
        self.max_alpha = 0.8  # Less smoothing (more responsive)
        
        # Enhanced position tracking with velocity estimation
        self.position_history = deque(maxlen=10)  # Increased history for better velocity estimation
        self.velocity_x_filtered = 0
        self.velocity_y_filtered = 0
        self.velocity_filter_alpha = 0.2  # Low-pass filter for velocity
        
        # Tag tracking
        self.target_tag_id = TARGET_TAG_ID

    def detect_tags(self, image, rgb_input=True):
        """
        Detect AprilTags in the image.
        If rgb_input is True, assume input is in RGB (e.g., from PiCamera).
        """
        try:
            # Convert to grayscale for AprilTag detection
            if rgb_input:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect AprilTags
            results = self.detector.detect(gray, estimate_tag_pose=False, camera_params=None, tag_size=None)
            
            return results
        except Exception as e:
            print(f"Error in detect_tags: {e}")
            return []

    def find_target_tag(self, tags, frame_width, frame_height):
        """
        Find the target tag with ID=12 in the detected tags.
        Returns a dictionary with details about the tag.
        """
        target_tag = None
        
        for tag in tags:
            if tag.tag_id == self.target_tag_id:
                target_tag = tag
                break
        
        if target_tag is None:
            return None
        
        try:
            # Calculate the center of the tag
            center_x = int(sum(corner[0] for corner in target_tag.corners) / 4)
            center_y = int(sum(corner[1] for corner in target_tag.corners) / 4)
            
            # Calculate the area using cv2.contourArea
            corners_int32 = np.array(target_tag.corners, dtype=np.int32)
            area = cv2.contourArea(corners_int32)
            
            # Apply adaptive smoothing
            center_x, center_y = self.adaptive_smooth_position(center_x, center_y, area)
            
            # Calculate velocity
            velocity_x, velocity_y = self.calculate_velocity(center_x, center_y)
            
            # Map to grid
            grid_x, grid_y = self.map_to_grid(center_x, center_y, frame_width, frame_height)
            
            # Calculate bounding box
            x_coords = [corner[0] for corner in target_tag.corners]
            y_coords = [corner[1] for corner in target_tag.corners]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Calculate width and height
            width = x_max - x_min
            height = y_max - y_min
            
            # Calculate aspect ratio
            aspect_ratio = width / height if height > 0 else 0
            
            return {
                'tag_id': target_tag.tag_id,
                'center_x': center_x,
                'center_y': center_y,
                'area': area,
                'width': width,
                'height': height,
                'aspect_ratio': aspect_ratio,
                'grid_position': (grid_x, grid_y),
                'bounding_box': (x_min, y_min, x_max, y_max),
                'corners': target_tag.corners,
                'velocity_x': velocity_x,
                'velocity_y': velocity_y
            }
        except Exception as e:
            print(f"Error in find_target_tag: {e}")
            return None

    def adaptive_smooth_position(self, new_x, new_y, area):
        """Enhanced adaptive smoothing with velocity-based alpha adjustment"""
        global last_position_x, last_position_y
        
        # Calculate distance from last position if available
        if last_position_x is not None and last_position_y is not None:
            distance = np.sqrt((new_x - last_position_x)**2 + (new_y - last_position_y)**2)
            
            # Dynamic alpha based on movement speed and object size
            size_factor = min(1.0, area / 3000)  # Larger objects get more smoothing
            
            if distance > 30:
                # Fast movement - less smoothing (more responsive)
                self.smoothing_alpha = min(self.max_alpha, self.smoothing_alpha + 0.05)
            elif distance < 10:
                # Slow movement - more smoothing
                self.smoothing_alpha = max(self.min_alpha, self.smoothing_alpha - 0.02)
            
            # Adjust alpha based on object size
            adjusted_alpha = self.smoothing_alpha * (1 - size_factor * 0.5)
        else:
            # First position - no smoothing
            adjusted_alpha = 1.0
        
        # Apply smoothing
        if last_position_x is None:
            last_position_x = new_x
            last_position_y = new_y
        else:
            last_position_x = int(adjusted_alpha * new_x + (1 - adjusted_alpha) * last_position_x)
            last_position_y = int(adjusted_alpha * new_y + (1 - adjusted_alpha) * last_position_y)
        
        return last_position_x, last_position_y

    def calculate_velocity(self, x, y):
        """Calculate and filter velocity based on position history"""
        # Store position for velocity calculation
        self.position_history.append((x, y))
        
        # Calculate velocity with improved filtering
        if len(self.position_history) >= 2:
            # Use positions 5 frames apart for better velocity estimation
            span = min(5, len(self.position_history) - 1)
            prev_x, prev_y = self.position_history[0]
            curr_x, curr_y = self.position_history[-1]
            
            # Raw velocity calculation
            raw_vel_x = (curr_x - prev_x) / span
            raw_vel_y = (curr_y - prev_y) / span
            
            # Apply low-pass filter to smooth velocity
            self.velocity_x_filtered = self.velocity_filter_alpha * raw_vel_x + (1 - self.velocity_filter_alpha) * self.velocity_x_filtered
            self.velocity_y_filtered = self.velocity_filter_alpha * raw_vel_y + (1 - self.velocity_filter_alpha) * self.velocity_y_filtered
            
            return self.velocity_x_filtered, self.velocity_y_filtered
        else:
            return 0, 0
    
    def map_to_grid(self, x, y, frame_width, frame_height):
        """Map pixel coordinates to grid positions"""
        grid_x = int((x / frame_width) * GRID_COLS)
        grid_y = int((y / frame_height) * GRID_ROWS)
        grid_x = max(0, min(grid_x, GRID_COLS - 1))
        grid_y = max(0, min(grid_y, GRID_ROWS - 1))
        return grid_x, grid_y
    
    def draw_visualization(self, image, tag_data):
        """Enhanced visualization with AprilTag information"""
        if image is None:
            return None
        
        viz_image = image.copy()
        
        # Draw grid if enabled
        if self.grid_lines:
            self.draw_grid(viz_image)
        
        # Draw center area for aiming
        h, w = viz_image.shape[:2]
        left_boundary = int(w * 0.4)  # Narrower center zone (40% - 60%)
        right_boundary = int(w * 0.6)
        center_zone_color = (30, 150, 30)  # Dark green for center zone
        cv2.rectangle(viz_image, (left_boundary, 0), (right_boundary, h), center_zone_color, 2)
        
        # Fill center zone with semi-transparent overlay
        overlay = viz_image.copy()
        cv2.rectangle(overlay, (left_boundary, 0), (right_boundary, h), center_zone_color, -1)
        cv2.addWeighted(overlay, 0.1, viz_image, 0.9, 0, viz_image)
        
        # If no tag detected, show basic info
        if tag_data is None:
            cv2.putText(viz_image, "No AprilTag detected", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return viz_image
        
        # Draw tag outline and corners
        corners = np.array(tag_data['corners'], dtype=np.int32)
        cv2.polylines(viz_image, [corners], True, (0, 255, 0), 2)
        
        # Draw center point
        center_x = tag_data['center_x']
        center_y = tag_data['center_y']
        cv2.circle(viz_image, (center_x, center_y), 5, (255, 0, 0), -1)
        
        # Draw directional arrow for velocity
        if 'velocity_x' in tag_data and 'velocity_y' in tag_data:
            vel_x = tag_data['velocity_x']
            vel_y = tag_data['velocity_y']
            
            # Only draw arrow if velocity is significant
            if abs(vel_x) > 0.5 or abs(vel_y) > 0.5:
                # Scale arrow length based on velocity magnitude
                arrow_scale = 3
                arrow_end_x = int(center_x + vel_x * arrow_scale)
                arrow_end_y = int(center_y + vel_y * arrow_scale)
                cv2.arrowedLine(viz_image, (center_x, center_y), 
                               (arrow_end_x, arrow_end_y), (0, 255, 255), 2)
        
        # Display detailed information
        tag_id_text = f"Tag ID: {tag_data['tag_id']}"
        area_text = f"Area: {tag_data['area']:.0f}"
        pos_text = f"Pos: ({center_x}, {center_y})"
        grid_text = f"Grid: {tag_data['grid_position']}"
        
        # Add velocity information
        vel_text = f"Vel: ({tag_data['velocity_x']:.1f}, {tag_data['velocity_y']:.1f})"
        
        # Add aspect ratio information
        ratio_text = f"Ratio: {tag_data['aspect_ratio']:.2f}"
        
        # Add smoothing info
        smooth_text = f"Smooth: {self.smoothing_alpha:.2f}"
        
        # Draw text
        cv2.putText(viz_image, tag_id_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(viz_image, area_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(viz_image, pos_text, (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(viz_image, grid_text, (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(viz_image, vel_text, (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(viz_image, ratio_text, (10, 180), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(viz_image, smooth_text, (10, 210), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return viz_image
    
    def draw_grid(self, image):
        """Draw grid lines for reference."""
        h, w = image.shape[:2]
        for i in range(1, GRID_COLS):
            x = int((i / GRID_COLS) * w)
            cv2.line(image, (x, 0), (x, h), (100, 100, 100), 1)
        for i in range(1, GRID_ROWS):
            y = int((i / GRID_ROWS) * h)
            cv2.line(image, (0, y), (w, y), (100, 100, 100), 1)