import time

class PID:
    def __init__(self, Kp, Ki, Kd, setpoint=0, integral_limit=1000):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.last_error = 0
        self.integral = 0
        self.last_time = time.time()
        self.integral_limit = integral_limit  # Limit for the integral term
        self.output_min = None
        self.output_max = None
        
    def set_output_limits(self, min_output, max_output):
        """Set the minimum and maximum output limits"""
        self.output_min = min_output
        self.output_max = max_output
    
    def reset(self):
        """Reset the controller state"""
        self.last_error = 0
        self.integral = 0
        self.last_time = time.time()
    
    def set_tunings(self, Kp, Ki, Kd):
        """Update PID parameters dynamically"""
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
    
    def update(self, measured_value):
        now = time.time()
        dt = now - self.last_time
        if dt <= 0:
            dt = 1e-16  # Prevent division by zero
            
        error = self.setpoint - measured_value
        
        # Proportional term
        p_term = self.Kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        if self.integral > self.integral_limit:
            self.integral = self.integral_limit
        elif self.integral < -self.integral_limit:
            self.integral = -self.integral_limit
        i_term = self.Ki * self.integral
        
        # Derivative term on measurement to avoid derivative kick
        d_term = 0
        if dt > 0:  # Extra check to avoid division by zero
            d_term = -self.Kd * (measured_value - (self.setpoint - self.last_error)) / dt
        
        # Calculate output
        output = p_term + i_term + d_term
        
        # Apply output limits if set
        if self.output_min is not None and output < self.output_min:
            output = self.output_min
        if self.output_max is not None and output > self.output_max:
            output = self.output_max
            
        self.last_error = error
        self.last_time = now
        
        return output




# import time

# class PID:
#     def __init__(self, Kp, Ki, Kd, setpoint=0, integral_limit=1000):
#         self.Kp = Kp
#         self.Ki = Ki
#         self.Kd = Kd
#         self.setpoint = setpoint
#         self.last_error = 0
#         self.integral = 0
#         self.last_time = time.time()
#         self.integral_limit = integral_limit  # Limit for the integral term

#     def update(self, measured_value):
#         now = time.time()
#         dt = now - self.last_time
#         if dt <= 0:
#             dt = 1e-16  # Prevent division by zero
#         error = self.setpoint - measured_value
#         self.integral += error * dt
#         # Anti-windup: Clamp the integral term
#         if self.integral > self.integral_limit:
#             self.integral = self.integral_limit
#         elif self.integral < -self.integral_limit:
#             self.integral = -self.integral_limit
#         derivative = (error - self.last_error) / dt
#         output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
#         self.last_error = error
#         self.last_time = now
#         return output




# import time

# class PID:
#     def __init__(self, Kp, Ki, Kd, setpoint=0, integral_limit=1000):
#         self.Kp = Kp
#         self.Ki = Ki
#         self.Kd = Kd
#         self.setpoint = setpoint
#         self.last_error = 0
#         self.integral = 0
#         self.last_time = time.time()
#         self.integral_limit = integral_limit

#     def update(self, measured_value):
#         now = time.time()
#         dt = now - self.last_time
#         if dt <= 0:
#             dt = 1e-16  # Prevent division by zero
#         error = self.setpoint - measured_value
#         self.integral += error * dt

#         # Anti-windup: Limit the integral term
#         self.integral = max(min(self.integral, self.integral_limit), -self.integral_limit)

#         derivative = (error - self.last_error) / dt
#         output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
#         self.last_error = error
#         self.last_time = now
#         return output





# import time

# class PID:
#     def __init__(self, Kp, Ki, Kd, setpoint=0, integral_limit=1000):
#         self.Kp = Kp
#         self.Ki = Ki
#         self.Kd = Kd
#         self.setpoint = setpoint
#         self.last_error = 0
#         self.integral = 0
#         self.last_time = time.time()
#         self.integral_limit = integral_limit  

#     def update(self, measured_value):
#         now = time.time()
#         dt = max(now - self.last_time, 1e-16)  # Avoid division by zero
#         error = self.setpoint - measured_value
#         self.integral = max(min(self.integral + error * dt, self.integral_limit), -self.integral_limit)
#         derivative = (error - self.last_error) / dt
#         output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
#         self.last_error = error
#         self.last_time = now
#         return output
