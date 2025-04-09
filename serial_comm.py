import serial
import time
import threading
from queue import Queue, Empty

class SerialHandler:
    def __init__(self, port, baud_rate, timeout=1):
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.serial = None
        self.is_connected = False
        self.send_queue = Queue()
        self.receive_queue = Queue()
        self.running = False
        self.thread = None
        
        # Try to connect to serial port
        self.connect()
        
        # Start processing thread
        if self.is_connected:
            self.running = True
            self.thread = threading.Thread(target=self._process_thread)
            self.thread.daemon = True
            self.thread.start()
    
    def connect(self):
        """Attempt to connect to the serial port"""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.timeout
            )
            self.is_connected = True
            print(f"Connected to {self.port} at {self.baud_rate} baud")
            return True
        except Exception as e:
            print(f"Error connecting to serial port: {e}")
            self.is_connected = False
            return False
    
    def send_command(self, command):
        """Queue a command to be sent to the serial port"""
        if not self.is_connected:
            print("Error: Not connected to serial port")
            return False
        
        # Add newline if not present
        if not command.endswith('\n'):
            command += '\n'
        
        # Add to send queue
        self.send_queue.put(command)
        return True
    
    def get_next_message(self, timeout=0.1):
        """Get the next message from the receive queue if available"""
        try:
            return self.receive_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def _process_thread(self):
        """Background thread to handle serial communication"""
        while self.running:
            try:
                # Send any queued commands
                while not self.send_queue.empty():
                    command = self.send_queue.get()
                    self.serial.write(command.encode())
                    self.send_queue.task_done()
                
                # Check for incoming data
                if self.serial.in_waiting > 0:
                    data = self.serial.readline().decode('utf-8', errors='replace').strip()
                    if data:
                        self.receive_queue.put(data)
                
                # Small delay to prevent CPU hogging
                time.sleep(0.01)
            except Exception as e:
                print(f"Serial thread error: {e}")
                # Try to reconnect
                self.is_connected = False
                self.connect()
                time.sleep(1)
    
    def close(self):
        """Close the serial connection and stop the processing thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.serial:
            self.serial.close()
        self.is_connected = False




# import serial
# import time
# import sys

# # Serial Port Configuration - Update these for your laptop
# # Windows example: 'COM3'
# # Linux/Mac example: '/dev/ttyUSB0' or '/dev/cu.usbserial'
# SERIAL_PORT = 'COM3'  # Default for Windows, adjust as needed
# BAUD_RATE = 9600

# class ArduinoConnection:
#     def __init__(self, port=None, baud_rate=None):
#         self.port = port or SERIAL_PORT
#         self.baud_rate = baud_rate or BAUD_RATE
#         self.arduino = None
        
#     def initialize(self):
#         """Initialize Serial Connection"""
#         try:
#             self.arduino = serial.Serial(self.port, self.baud_rate, timeout=1)
#             time.sleep(2)  # Allow Arduino to initialize
#             print(f"Serial connection established on {self.port} at {self.baud_rate} baud.")
#             return self.arduino
#         except serial.SerialException as e:
#             print(f"Error: Could not connect to Arduino - {e}")
#             print("If testing without Arduino, you can comment out this exit.")
#             return None  # Return None instead of exiting for testing without hardware
            
#     def send_command(self, command):
#         """Send command to Arduino"""
#         if self.arduino is None:
#             print(f"[MOCK] Sending command: {command}")
#             return True
            
#         try:
#             self.arduino.write((command + "\n").encode())
#             return True
#         except serial.SerialException as e:
#             print(f"Serial Write Error: {e}")
#             return False
            
#     def close(self):
#         """Close the serial connection"""
#         if self.arduino is not None:
#             self.arduino.close()




# import serial
# import time
# import sys

# SERIAL_PORT = "COM3"  # Change based on your system (Linux: /dev/ttyUSB0, Windows: COM3)
# BAUD_RATE = 9600

# def initialize_serial():
#     try:
#         arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
#         time.sleep(2)  # Allow Arduino to initialize
#         print("Serial connection established.")
#         return arduino
#     except serial.SerialException as e:
#         print(f"Error: Could not connect to Arduino - {e}")
#         sys.exit(1)

# def send_command(arduino, command):
#     try:
#         arduino.write((command + "\n").encode())
#         print(f"Sent: {command}")
#     except serial.SerialException as e:
#         print(f"Serial Write Error: {e}")





# import serial
# import time
# import sys

# SERIAL_PORT = '/dev/ttyUSB0'
# BAUD_RATE = 9600

# def initialize_serial():
#     try:
#         arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
#         time.sleep(2)
#         print("Serial connection established.")
#         return arduino
#     except serial.SerialException as e:
#         print(f"Error: Could not connect to Arduino - {e}")
#         sys.exit(1)

# def send_command(arduino, command):
#     try:
#         arduino.write((command + "\n").encode())
#     except serial.SerialException as e:
#         print(f"Serial Write Error: {e}")
