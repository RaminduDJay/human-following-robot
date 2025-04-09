import cv2
import numpy as np
import argparse
from apriltag_detector import AprilTagDetector

def test_apriltag_detection():
    """
    Test AprilTag detection using camera feed.
    This is useful for verifying that the camera can see and identify the AprilTag.
    """
    print("AprilTag Detection Test - Press 'q' to quit, 's' to save image")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)  # Use default camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Create AprilTag detector
    detector = AprilTagDetector()
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break
        
        # Convert to RGB (for consistency with main program)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect AprilTags
        tags = detector.detect_tags(rgb_frame)
        
        # Get image dimensions
        height, width = frame.shape[:2]
        
        # Find target tag
        tag_data = detector.find_target_tag(tags, width, height)
        
        # Create visualization
        viz_image = detector.draw_visualization(rgb_frame, tag_data)
        
        # Display help text
        cv2.putText(viz_image, "Press 's' to save image, 'q' to quit", (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display all detected tags
        cv2.putText(viz_image, f"Detected tags: {len(tags)}", (10, height - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Convert back to BGR for display
        display_image = cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR)
        
        # Show the image
        cv2.imshow("AprilTag Test", display_image)
        
        # Check for keypresses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save the frame
            cv2.imwrite("apriltag_test.jpg", display_image)
            print("Image saved as 'apriltag_test.jpg'")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_apriltag_detection()