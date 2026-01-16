"""ROS Config for the JetBot"""

# ROS TOPICS
# Input topics (subscribed)
CAMERA_TOPIC = "/csi_cam_0/image_raw"
JOY_TOPIC = "/joy"                    # sensor_msgs/Joy
LIDAR_TOPIC = "/scan"                 # sensor_msgs/LaserScan
IMU_TOPIC = "/imu/data"               # sensor_msgs/Imu

# Output topics (published)
CMD_VEL_TOPIC = "/cmd_vel"            # geometry_msgs/Twist
STEERING_TOPIC = "/steering_cmd"      # std_msgs/Float32
SPEED_TOPIC = "/speed_multiplier"     # std_msgs/Float32

# IMAGE SETTINGS
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

# DATA COLLECTION
DATA_DIR = "./data/training"

# Gamepad axis mapping
JOY_STEERING_AXIS = 0   # Left stick horizontal
JOY_THROTTLE_AXIS = 3   # Right trigger
JOY_DEADZONE = 0.1

# TRAINING PARAMETERS
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
VAL_SPLIT = 0.2
MODEL_SAVE_PATH = "./models/checkpoints"

# ROBOT CONTROL
BASE_SPEED = 0.25        # m/s forward velocity
MAX_ANGULAR_VEL = 2.0    # rad/s maximum rotation

# LiDAR Safety Parameters
LIDAR_TOPIC = "/scan"
LIDAR_MIN_DISTANCE = 0.3  # meters - emergency stop
LIDAR_SAFE_DISTANCE = 0.5

# Speed multipliers
SPEED_STOP = 0.0
SPEED_SLOW = 0.3
SPEED_MODERATE = 0.7
SPEED_FULL = 1.0

# CNN MODEL
# Predict steering angle or angular velocity? We use only if we implement IMU
PREDICT_ANGULAR_VELOCITY = False  # Set True to use omega instead of steering

# Model selection
USE_LIGHTWEIGHT_MODEL = False  # True to use the lighter model

# ROS NODE RATES
DATA_COLLECTION_RATE = 10  # Hz
CNN_INFERENCE_RATE = 10    # Hz
SAFETY_CHECK_RATE = 10     # Hz unused but keep for now
MOTOR_CONTROL_RATE = 10    # Hz

# LOGGING
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARN, ERROR

LAPTOP_IP = "192.168.137.1" #TODO LAPTOP_IP
JETSON_IP = "192.168.137.112" #TODO JETSON_IP