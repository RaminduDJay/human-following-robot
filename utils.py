import re

def command_changed(new_command, previous_command):
    """
    Determine if a new command is meaningfully different from the previous command.
    Includes special handling for MOVE commands with deadband filtering.
    """
    # If either command is None, they're different
    if previous_command is None or new_command is None:
        return True
        
    # If they're exactly the same string, they're not different
    if new_command == previous_command:
        return False
        
    # If they're not both MOVE commands, they're different
    if not new_command.startswith("MOVE") or not previous_command.startswith("MOVE"):
        return True
        
    # Special handling for MOVE commands
    try:
        # Extract speed values
        new_speed_match = re.search(r"MOVE (\d+)", new_command)
        prev_speed_match = re.search(r"MOVE (\d+)", previous_command)
        
        if new_speed_match and prev_speed_match:
            new_speed = int(new_speed_match.group(1))
            prev_speed = int(prev_speed_match.group(1))
            
            # Calculate speed difference
            speed_diff = abs(new_speed - prev_speed)
            
            # If speed difference is small, consider commands the same
            # (unless one has turning and the other doesn't)
            if speed_diff < 10:
                # Check if turning directives are different
                new_has_turn = "LEFT" in new_command or "RIGHT" in new_command
                prev_has_turn = "LEFT" in previous_command or "RIGHT" in previous_command
                
                if new_has_turn != prev_has_turn:
                    return True  # Different turning behavior
                    
                # If both have turning, check if directions are different
                if new_has_turn and prev_has_turn:
                    new_direction = "LEFT" if "LEFT" in new_command else "RIGHT"
                    prev_direction = "LEFT" if "LEFT" in previous_command else "RIGHT"
                    
                    if new_direction != prev_direction:
                        return True  # Different turning directions
                        
                    # Extract turning intensity
                    new_intensity_match = re.search(r"(LEFT|RIGHT) (\d+)", new_command)
                    prev_intensity_match = re.search(r"(LEFT|RIGHT) (\d+)", previous_command)
                    
                    if new_intensity_match and prev_intensity_match:
                        new_intensity = int(new_intensity_match.group(2))
                        prev_intensity = int(prev_intensity_match.group(2))
                        
                        # If turning intensity difference is significant
                        if abs(new_intensity - prev_intensity) > 10:
                            return True
                
                # If we got here, the commands are similar enough
                return False
    except Exception as e:
        print(f"Error in command_changed: {e}")
        
    # Default to considering commands different if we can't determine similarity
    return True

def parse_serial_data(data):
    """Parse data received from serial connection"""
    if not data:
        return None
        
    try:
        # Remove any whitespace and convert to uppercase
        data = data.strip().upper()
        
        # Parse different types of data based on prefix
        if data.startswith("SENSOR:"):
            # Parse sensor data (format: "SENSOR:key1=value1;key2=value2")
            parts = data[7:].split(';')
            sensor_data = {}
            
            for part in parts:
                if '=' in part:
                    key, value = part.split('=', 1)
                    try:
                        # Try to convert to number if possible
                        sensor_data[key] = float(value)
                    except ValueError:
                        sensor_data[key] = value
                        
            return {'type': 'sensor', 'data': sensor_data}
            
        elif data.startswith("STATUS:"):
            # Parse status update (format: "STATUS:message")
            return {'type': 'status', 'message': data[7:]}
            
        elif data.startswith("ERROR:"):
            # Parse error message (format: "ERROR:message")
            return {'type': 'error', 'message': data[6:]}
            
        else:
            # Unknown format, return as raw data
            return {'type': 'unknown', 'raw': data}
            
    except Exception as e:
        print(f"Error parsing serial data: {e}")
        return {'type': 'error', 'message': f"Parse error: {e}", 'raw': data}

def calculate_distance_from_area(area, calibration_area=2500, calibration_distance=1.0):
    """
    Calculate approximate distance from object area.
    Uses the inverse square law: area ∝ 1/distance²
    
    Args:
        area: Current object area in pixels
        calibration_area: Area at known distance (default 2500 px at 1m)
        calibration_distance: Known distance in meters (default 1.0m)
        
    Returns:
        Estimated distance in meters
    """
    if area <= 0:
        return float('inf')  # Avoid division by zero
        
    # Apply inverse square law
    distance = calibration_distance * (calibration_area / area) ** 0.5
    return distance





# def command_changed(new_command, old_command, threshold=5):
#     """
#     Compare MOVE commands and update only if the speed difference exceeds the threshold.
#     For non-MOVE commands, use standard string comparison.
#     """
#     if new_command.startswith("MOVE") and old_command and old_command.startswith("MOVE"):
#         try:
#             new_speed = int(new_command.split()[1])
#             old_speed = int(old_command.split()[1])
#             if abs(new_speed - old_speed) < threshold:
#                 return False
#         except Exception as e:
#             # If parsing fails, assume the command has changed.
#             return True
#     return new_command != old_command

# def clean_exit(camera, arduino):
#     """
#     Gracefully close all resources before exiting
#     """
#     import cv2
#     import sys
    
#     print("\nCleaning up resources...")
#     try:
#         if camera is not None:
#             camera.release()
#             print("Camera closed.")
#     except Exception as e:
#         print(f"Error closing camera: {e}")
#     try:
#         if arduino is not None:
#             arduino.close()
#             print("Serial connection closed.")
#     except Exception as e:
#         print(f"Error closing serial connection: {e}")
#     cv2.destroyAllWindows()  # Destroy any OpenCV windows
#     print("Exiting safely.")
#     sys.exit(0)




# import cv2
# import sys

# def clean_exit(camera, arduino):
#     print("\nCleaning up resources...")
#     camera.release()
#     arduino.close()
#     cv2.destroyAllWindows()
#     print("Exiting safely.")
#     sys.exit(0)
