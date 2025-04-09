import cv2
import numpy as np
import time
import sys

from camera import Camera
from serial_comm import ArduinoConnection
from apriltag_detector import AprilTagDetector
from movement_control import MovementController
from utils import clean_exit

# Performance monitoring
FPS_HISTORY_SIZE = 10

def main():
    # Initialize components
    camera = Camera()
    arduino = ArduinoConnection()
    
    cap = camera.initialize()
    ser = arduino.initialize()
    
    # Create AprilTag detector
    detector = AprilTagDetector()
    
    # Create movement controller
    controller = MovementController(arduino)
    
    # Performance tracking
    fps_history = []
    last_time = time.time()
    frame_count = 0
    
    # Tag tracking state with improved transition handling
    tracking_lost_time = None
    tracking_active = False
    lost_frames_count = 0
    max_lost_frames = 30  # About 1 second at 30 fps
    
    # Movement dampening for lost tags
    last_movement_command = "STOP"
    dampening_factor = 1.0  # Start at full power
    
    try:
        print("Starting AprilTag tracking... Press 'q' to quit or Ctrl+C to stop.")
        print("Tracking Tag Family: tag25h9 Tag ID: 12")
        
        while True:
            frame_start_time = time.time()
            
            # Get frame from camera
            frame = camera.get_frame()
            if frame is None:
                print("Failed to get frame, retrying...")
                time.sleep(0.1)
                continue
                
            # Create frame container for compatibility
            frame_container = camera.create_frame_container()
            frame_container.update(frame)
                
            # Detect AprilTags
            tags = detector.detect_tags(frame)
            
            # Find our target tag (ID: 12)
            frame_height, frame_width = frame.shape[:2]
            tag_data = detector.find_target_tag(tags, frame_width, frame_height)
            
            # Improved tracking state management with gradual transitions
            if tag_data:
                if not tracking_active:
                    print("Tag acquired - ID:", tag_data['tag_id'])
                tracking_active = True
                tracking_lost_time = None
                lost_frames_count = 0
                dampening_factor = 1.0  # Full power when tag is visible
            else:
                # Tag not detected
                lost_frames_count += 1
                
                # Just lost the tag
                if tracking_active and tracking_lost_time is None:
                    tracking_lost_time = time.time()
                    print("Tag lost")
                
                # Gradually reduce power as tag remains lost
                if lost_frames_count <= max_lost_frames:
                    # Linear decrease from 1.0 to 0.0
                    dampening_factor = 1.0 - (lost_frames_count / max_lost_frames)
                else:
                    # More than max_lost_frames without detection
                    if tracking_active:
                        tracking_active = False
                        print("Tracking inactive")
                    dampening_factor = 0.0  # Stop completely
            
            # Determine and send movement command with improved transitions
            if tag_data or (tracking_lost_time and lost_frames_count < max_lost_frames):
                # Use tag data or continue last movement briefly if just lost
                movement = controller.determine_movement(tag_data, frame_width, frame_height)
                
                # Apply dampening to the movement command when tag is lost
                if not tag_data and dampening_factor < 1.0:
                    # Extract command components
                    parts = movement.split()
                    command_type = parts[0]
                    
                    if command_type in ["TURN_LEFT", "TURN_RIGHT"] and len(parts) > 1:
                        # Reduce turn speed by dampening factor
                        speed = int(float(parts[1]) * dampening_factor)
                        if speed < 50:  # Minimum effective speed
                            movement = "STOP"
                        else:
                            movement = f"{command_type} {speed}"
                    elif command_type == "MOVE" and len(parts) > 1:
                        if len(parts) == 2:  # Single speed
                            speed = int(float(parts[1]) * dampening_factor)
                            if speed < 50:  # Minimum effective speed
                                movement = "STOP"
                            else:
                                movement = f"MOVE {speed}"
                        elif len(parts) == 3:  # Differential drive
                            left = int(float(parts[1]) * dampening_factor)
                            right = int(float(parts[2]) * dampening_factor)
                            if left < 50 and right < 50:
                                movement = "STOP"
                            else:
                                movement = f"MOVE {left} {right}"
                
                # Send command and update last movement
                controller.send_movement_command(movement)
                last_movement_command = movement
                
                # Prepare visualization
                image_with_viz = detector.draw_visualization(frame, tag_data)
                    
                # Display current movement command on screen
                cv2.putText(image_with_viz, f"Command: {movement}", (10, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Show dampening factor when tag is lost
                if not tag_data:
                    cv2.putText(image_with_viz, f"Dampening: {dampening_factor:.2f}", (10, 270), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                # No tag detected for a while
                controller.send_movement_command("STOP")
                last_movement_command = "STOP"
                
                # Show frame with grid only
                image_with_viz = detector.draw_visualization(frame, None)
                cv2.putText(image_with_viz, "Command: STOP (No tag detected)", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Calculate FPS
            frame_count += 1
            current_time = time.time()
            elapsed = current_time - last_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                fps_history.append(fps)
                if len(fps_history) > FPS_HISTORY_SIZE:
                    fps_history.pop(0)
                avg_fps = sum(fps_history) / len(fps_history)
                
                cv2.putText(image_with_viz, f"FPS: {avg_fps:.1f}", (10, 300), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                frame_count = 0
                last_time = current_time
                
            # Add processing time info
            processing_time = (time.time() - frame_start_time) * 1000
            cv2.putText(image_with_viz, f"Process: {processing_time:.1f}ms", (10, 330), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
            # Convert RGB back to BGR for OpenCV display
            image_bgr = cv2.cvtColor(image_with_viz, cv2.COLOR_RGB2BGR)
            cv2.imshow("AprilTag Tracking", image_bgr)
                
            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nUser stopped the program.")
    except Exception as e:
        print(f"Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        clean_exit(camera, arduino)

if __name__ == "__main__":
    main()