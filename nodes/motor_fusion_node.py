#!/usr/bin/env python3
"""
Motor Fusion Node
Subscribes to: /steering_cmd, /speed_multiplier
Publishes to: /cmd_vel (Twist)
Combines CNN steering with LiDAR safety
"""

import rospy
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
import math
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import BASE_SPEED, MAX_ANGULAR_VEL


class MotorFusionNode:
    """Fuses vision steering with LiDAR override and IMU drift correction"""
    
    def __init__(self):
        rospy.init_node('motor_fusion', anonymous=True)
        
        # State
        self.vision_steering = 0.0
        self.speed_multiplier = 1.0
        self.lidar_override = False
        self.lidar_cmd_vel = Twist()
        
        # IMU drift correction
        self.use_imu_correction = rospy.get_param('~use_imu', False)
        self.target_yaw = None
        self.current_yaw = 0.0
        self.drift_correction_gain = 0.1  # How aggressively to correct drift
        
        # Subscribers
        self.vision_sub = rospy.Subscriber(
            '/steering_cmd', Float32, self.vision_callback, queue_size=1
        )
        self.speed_sub = rospy.Subscriber(
            '/speed_multiplier', Float32, self.speed_callback, queue_size=1
        )
        self.lidar_override_sub = rospy.Subscriber(
            '/lidar_override', Bool, self.lidar_override_callback, queue_size=1
        )
        self.lidar_cmd_sub = rospy.Subscriber(
            '/lidar_cmd_vel', Twist, self.lidar_cmd_callback, queue_size=1
        )
        
        if self.use_imu_correction:
            self.imu_sub = rospy.Subscriber(
                '/kalman_imu', Imu, self.imu_callback, queue_size=1
            )
        
        self.cnn_ready = False
        # Subscribe to CNN ready signal
        self.cnn_ready_sub = rospy.Subscriber(
            '/cnn_ready', Bool, self.cnn_ready_callback, queue_size=1
        )
        
        # Publisher
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        rospy.loginfo("Motor Fusion Node started")
        rospy.loginfo(f"Base speed: {BASE_SPEED} m/s")
        rospy.loginfo(f"IMU drift correction: {self.use_imu_correction}")
    
    def cnn_ready_callback(self, msg):
        if msg.data and not self.cnn_ready:
            self.cnn_ready = True
            rospy.loginfo("✓ CNN is ready! Motor fusion starting...")
    
    def vision_callback(self, msg):
        self.vision_steering = msg.data
    
    def speed_callback(self, msg):
        self.speed_multiplier = msg.data
    
    def lidar_override_callback(self, msg):
        self.lidar_override = msg.data
        
        if self.lidar_override:
            rospy.logwarn("LiDAR OVERRIDE ACTIVE")
            # Reset target yaw when override starts
            self.target_yaw = None
        else:
            rospy.loginfo("Vision control resumed")
    
    def lidar_cmd_callback(self, msg):
        self.lidar_cmd_vel = msg
    
    @staticmethod
    def euler_from_quaternion(quat):
        """Convert quaternion to euler angles (roll, pitch, yaw)"""
        x, y, z, w = quat
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def imu_callback(self, msg):
        """Extract yaw from filtered IMU"""
        quat = [
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ]
        _, _, self.current_yaw = self.euler_from_quaternion(quat)
        
        # Set target yaw on first reading (or after override)
        if self.target_yaw is None:
            self.target_yaw = self.current_yaw
    
    def compute_drift_correction(self):
        """Calculate steering correction based on yaw drift"""
        if not self.use_imu_correction or self.target_yaw is None:
            return 0.0
        
        # Calculate yaw error
        yaw_error = self.target_yaw - self.current_yaw
        
        # Normalize to [-pi, pi]
        while yaw_error > np.pi:
            yaw_error -= 2 * np.pi
        while yaw_error < -np.pi:
            yaw_error += 2 * np.pi
        
        # Small correction proportional to error
        correction = self.drift_correction_gain * yaw_error
        correction = np.clip(correction, -0.2, 0.2)  # Limit correction
        
        if abs(yaw_error) > 0.1:  # 5.7 degrees
            rospy.loginfo_throttle(2.0, 
                f"Drift correction: {correction:+.2f} (error: {np.degrees(yaw_error):.1f}°)")
        
        return correction
    
    def run(self):
        rate = rospy.Rate(20)  # 20 Hz
        
        while not rospy.is_shutdown():
            if not self.cnn_ready:
                # Stay stopped while waiting for CNN
                cmd = Twist()
                self.cmd_vel_pub.publish(cmd)
                rate.sleep()
                continue
            
            cmd = Twist()
            
            if self.lidar_override:
                # LiDAR in control (backup/adjust sequence)
                cmd = self.lidar_cmd_vel
            else:
                # Vision + IMU in control
                drift_correction = self.compute_drift_correction()
                steering = self.vision_steering + drift_correction
                
                cmd.linear.x = BASE_SPEED * self.speed_multiplier
                cmd.angular.z = steering * MAX_ANGULAR_VEL
            
            self.cmd_vel_pub.publish(cmd)
            rate.sleep()


if __name__ == '__main__':
    try:
        node = MotorFusionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass