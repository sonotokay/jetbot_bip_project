#!/usr/bin/env python3
"""
Motor Fusion Node
Subscribes to: /steering_cmd, /speed_multiplier
Publishes to: /cmd_vel (Twist)
Combines CNN steering with LiDAR safety
"""

import rospy
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import (
    STEERING_TOPIC, SPEED_TOPIC, CMD_VEL_TOPIC,
    BASE_SPEED, STEERING_GAIN, MAX_ANGULAR_VEL,
    WHEELBASE, PREDICT_ANGULAR_VELOCITY, MOTOR_CONTROL_RATE
)


class MotorFusionNode:
    """ROS node for fusing steering and safety into motor commands"""
    
    def __init__(self):
        rospy.init_node('motor_fusion', anonymous=True)
        
        # State
        self.latest_steering = 0.0
        self.latest_speed_multiplier = 1.0
        
        # Publisher
        self.cmd_vel_pub = rospy.Publisher(
            CMD_VEL_TOPIC, Twist, queue_size=1
        )
        
        # Subscribers
        self.steering_sub = rospy.Subscriber(
            STEERING_TOPIC, Float32, self.steering_callback, queue_size=1
        )
        self.speed_sub = rospy.Subscriber(
            SPEED_TOPIC, Float32, self.speed_callback, queue_size=1
        )
        
        # Stats
        self.cmd_count = 0
        
        rospy.loginfo("Motor Fusion Node started")
        mode = "angular velocity" if PREDICT_ANGULAR_VELOCITY else "steering angle"
        rospy.loginfo(f"Mode: {mode}")
    
    def steering_callback(self, msg):
        """Callback for steering commands"""
        self.latest_steering = msg.data
    
    def speed_callback(self, msg):
        """Callback for speed multiplier"""
        self.latest_speed_multiplier = msg.data
    
    def compute_cmd_vel(self):
        """Compute Twist command from steering and speed"""
        cmd = Twist()
        
        # Apply speed multiplier to base speed
        linear_vel = BASE_SPEED * self.latest_speed_multiplier
        
        if PREDICT_ANGULAR_VELOCITY:
            # Model predicts angular velocity directly (rad/s)
            angular_vel = self.latest_steering
            
            # Clamp
            angular_vel = max(-MAX_ANGULAR_VEL, min(MAX_ANGULAR_VEL, angular_vel))
        
        else:
            # Model predicts steering angle (-1 to 1)
            # Convert to angular velocity
            angular_vel = self.latest_steering * MAX_ANGULAR_VEL
        
        # Set command
        cmd.linear.x = linear_vel
        cmd.angular.z = angular_vel
        
        return cmd
    
    def run(self):
        """Main control loop"""
        rate = rospy.Rate(MOTOR_CONTROL_RATE)
        
        rospy.loginfo(f"Publishing cmd_vel at {MOTOR_CONTROL_RATE} Hz")
        
        while not rospy.is_shutdown():
            # Compute and publish command
            cmd = self.compute_cmd_vel()
            self.cmd_vel_pub.publish(cmd)
            
            self.cmd_count += 1
            
            # Log
            if self.cmd_count % 100 == 0:
                rospy.loginfo(
                    f"Commands: {self.cmd_count} | "
                    f"Linear: {cmd.linear.x:.2f} m/s | "
                    f"Angular: {cmd.angular.z:+.2f} rad/s | "
                    f"Speed mult: {self.latest_speed_multiplier:.1%}"
                )
            
            rate.sleep()


if __name__ == '__main__':
    try:
        node = MotorFusionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass