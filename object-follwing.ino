#include <AFMotor.h>
#include <NewPing.h>
#include <Servo.h>

// Pin Definitions
#define TRIG_PIN A5
#define ECHO_PIN A4
#define SERVO_PIN 10
#define MAX_DISTANCE 200

// Movement Parameters
#define BASE_SPEED 150
#define TURN_STRENGTH 50
#define ACCELERATION_STEP 5
#define MAX_SPEED 255
#define MIN_SPEED 50
#define SAFE_DISTANCE 20
#define RAMP_DELAY 20

// Motor Configuration
AF_DCMotor motorLeft1(1);
AF_DCMotor motorLeft2(1);
AF_DCMotor motorRight1(3);
AF_DCMotor motorRight2(3);

// Ultrasonic and Servo Setup
NewPing sonar(TRIG_PIN, ECHO_PIN, MAX_DISTANCE);
Servo scanServo;

// Movement State Variables
int currentLeftSpeed = 0;
int currentRightSpeed = 0;
int targetLeftSpeed = 0;
int targetRightSpeed = 0;
char currentDirection = 'S';
unsigned long movementTimeout = 0;
bool isMoving = false;

// Watchdog timer for safety
unsigned long lastCommandTime = 0;
#define COMMAND_TIMEOUT 3000 // Stop if no commands received for 3 seconds

// Function Prototypes
void rampMotorSpeeds();
int checkDistance();
void smoothTurn(char direction);
void smoothForward(int speed);
void stop();
void processCommand(String cmd);

void setup() {
  Serial.begin(9600);
  
  // Initialize Servo
  scanServo.attach(SERVO_PIN);
  scanServo.write(90);  // Center position
  
  // Initialize Motors
  stop();
  
  Serial.println("Object Tracking Robot Initialized");
  Serial.println("Commands:");
  Serial.println("- MOVE <speed>: Forward movement with speed control");
  Serial.println("- TURN LEFT/RIGHT: Turning based on object position");
  Serial.println("- STOP: Stop all movement");
}

void loop() {
  // Continuously ramp motor speeds for smooth acceleration/deceleration
  rampMotorSpeeds();
  
  // Safety: Stop if no commands received for a while
  if (isMoving && (millis() - lastCommandTime > COMMAND_TIMEOUT)) {
    stop();
    Serial.println("Command timeout - stopping");
  }
  
  // Obstacle detection during forward movement
  if (currentDirection == 'F') {
    int distance = checkDistance();
    if (distance > 0 && distance <= SAFE_DISTANCE) {
      stop();
      Serial.print("Obstacle detected at ");
      Serial.print(distance);
      Serial.println(" cm. Stopping.");
    }
  }
  
  // Process serial commands from Python program
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    if (command.length() > 0) {
      processCommand(command);
      lastCommandTime = millis(); // Reset timeout timer
    }
  }
  
  delay(RAMP_DELAY);
}

// Smoothly ramp motor speeds for gradual acceleration/deceleration
void rampMotorSpeeds() {
  // Left Motors
  if (currentLeftSpeed < targetLeftSpeed) {
    currentLeftSpeed = min(currentLeftSpeed + ACCELERATION_STEP, targetLeftSpeed);
  } else if (currentLeftSpeed > targetLeftSpeed) {
    currentLeftSpeed = max(currentLeftSpeed - ACCELERATION_STEP, targetLeftSpeed);
  }
  
  // Right Motors
  if (currentRightSpeed < targetRightSpeed) {
    currentRightSpeed = min(currentRightSpeed + ACCELERATION_STEP, targetRightSpeed);
  } else if (currentRightSpeed > targetRightSpeed) {
    currentRightSpeed = max(currentRightSpeed - ACCELERATION_STEP, targetRightSpeed);
  }
  
  // Apply speeds to motors
  motorLeft1.setSpeed(currentLeftSpeed);
  motorLeft2.setSpeed(currentLeftSpeed);
  motorRight1.setSpeed(currentRightSpeed);
  motorRight2.setSpeed(currentRightSpeed);
}

// Check distance using ultrasonic sensor
int checkDistance() {
  delay(10);  // Stability delay
  int distance = sonar.ping_cm();
  
  // Filter out invalid readings
  if (distance == 0) {
    delay(10);
    distance = sonar.ping_cm();
  }
  
  return distance;
}

// Smooth turning function with different turning rates based on object position
void smoothTurn(char direction) {
  if (direction == 'L') {
    // Left turn: reduce left motor speed, increase right motor speed
    targetLeftSpeed = max(0, BASE_SPEED - TURN_STRENGTH);
    targetRightSpeed = min(MAX_SPEED, BASE_SPEED + TURN_STRENGTH);
    
    motorLeft1.run(BACKWARD);
    motorLeft2.run(BACKWARD);
    motorRight1.run(FORWARD);
    motorRight2.run(FORWARD);
  } else if (direction == 'R') {
    // Right turn: reduce right motor speed, increase left motor speed
    targetLeftSpeed = min(MAX_SPEED, BASE_SPEED + TURN_STRENGTH);
    targetRightSpeed = max(0, BASE_SPEED - TURN_STRENGTH);
    
    motorLeft1.run(FORWARD);
    motorLeft2.run(FORWARD);
    motorRight1.run(BACKWARD);
    motorRight2.run(BACKWARD);
  }
  
  currentDirection = direction;
  isMoving = true;
  movementTimeout = millis() + 1000;  // 1 second timeout
}

// Smooth forward movement with PID-controlled speed from Python
void smoothForward(int speed = BASE_SPEED) {
  int distance = checkDistance();
  
  if (distance > SAFE_DISTANCE || distance == 0) {
    targetLeftSpeed = constrain(speed, MIN_SPEED, MAX_SPEED);
    targetRightSpeed = constrain(speed, MIN_SPEED, MAX_SPEED);
    
    motorLeft1.run(FORWARD);
    motorLeft2.run(FORWARD);
    motorRight1.run(FORWARD);
    motorRight2.run(FORWARD);
    
    currentDirection = 'F';
    isMoving = true;
    movementTimeout = millis() + 1000;  // 1 second timeout
  } else {
    stop();
    Serial.println("Obstacle detected. Cannot move forward.");
  }
}

// Stop all motors
void stop() {
  targetLeftSpeed = 0;
  targetRightSpeed = 0;
  
  motorLeft1.run(RELEASE);
  motorLeft2.run(RELEASE);
  motorRight1.run(RELEASE);
  motorRight2.run(RELEASE);
  
  currentDirection = 'S';
  isMoving = false;
}

// Command processing function
void processCommand(String cmd) {
  Serial.println("Received: " + cmd);
  
  if (cmd.equalsIgnoreCase("STOP")) {
    stop();
  }
  else if (cmd.startsWith("MOVE")) {
    int spaceIndex = cmd.indexOf(' ');
    if (spaceIndex > 0) {
      String speedStr = cmd.substring(spaceIndex + 1);
      int speed = speedStr.toInt();
      smoothForward(speed);
      Serial.print("Moving forward at speed: ");
      Serial.println(speed);
    } else {
      smoothForward();
      Serial.print("Moving forward at default speed: ");
      Serial.println(BASE_SPEED);
    }
  }
  else if (cmd.equalsIgnoreCase("TURN LEFT")) {
    smoothTurn('L');
    Serial.println("Turning left");
  }
  else if (cmd.equalsIgnoreCase("TURN RIGHT")) {
    smoothTurn('R');
    Serial.println("Turning right");
  }
  else {
    Serial.println("Unknown command: " + cmd);
  }
}