#!/usr/bin/env python3
"""
LiDAR Safety Node
Subscribes to: /scan (LaserScan)
Publishes to: /speed_multiplier (Float32)
"""

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import (
    LIDAR_TOPIC, SPEED_TOPIC,
    LIDAR_STOP_DISTANCE, LIDAR_SLOW_DISTANCE, LIDAR_NORMAL_DISTANCE,
    LIDAR_SECTOR_ANGLE, SPEED_STOP, SPEED_SLOW, SPEED_MODERATE, SPEED_FULL,
    SAFETY_CHECK_RATE
)


class LiDARSafetyNode:
    """ROS node for LiDAR-based safety control"""
    
    def __init__(self):
        rospy.init_node('lidar_safety', anonymous=True)
        
        # Publisher
        self.speed_pub = rospy.Publisher(
            SPEED_TOPIC, Float32, queue_size=1
        )
        
        # Subscriber
        self.scan_sub = rospy.Subscriber(
            LIDAR_TOPIC, LaserScan, self.scan_callback, queue_size=1
        )
        
        # State
        self.latest_distance = None
        self.check_count = 0
        
        rospy.loginfo("LiDAR Safety Node started")
        rospy.loginfo(f"Stop: <{LIDAR_STOP_DISTANCE}m, "
                    f"Slow: <{LIDAR_SLOW_DISTANCE}m, "
                    f"Normal: >{LIDAR_NORMAL_DISTANCE}m")
    
    def scan_callback(self, msg):
        """Callback for LiDAR scans"""
        try:
            # Get forward sector angles
            angle_min = msg.angle_min
            angle_max = msg.angle_max
            angle_increment = msg.angle_increment
            
            # Convert sector angle to radians
            sector_rad = np.deg2rad(LIDAR_SECTOR_ANGLE)
            
            # Find indices for forward sector
            # Assuming 0 radians is front
            forward_indices = []
            for i, distance in enumerate(msg.ranges):
                angle = angle_min + i * angle_increment
                
                # Normalize angle to [-pi, pi]
                angle = np.arctan2(np.sin(angle), np.cos(angle))
                
                # Check if in forward sector (±sector_angle from front)
                if abs(angle) <= sector_rad:
                    if not np.isinf(distance) and not np.isnan(distance):
                        if msg.range_min <= distance <= msg.range_max:
                            forward_indices.append(distance)
            
            # Get minimum distance in forward sector
            if forward_indices:
                min_distance = min(forward_indices)
                self.latest_distance = min_distance
            else:
                self.latest_distance = None
            
            # Calculate speed multiplier
            speed = self.calculate_speed_multiplier(self.latest_distance)
            
            # Publish
            speed_msg = Float32()
            speed_msg.data = speed
            self.speed_pub.publish(speed_msg)
            
            # Log
            self.check_count += 1
            if self.check_count % 50 == 0:
                dist_str = f"{self.latest_distance:.2f}m" if self.latest_distance else "N/A"
                rospy.loginfo(
                    f"LiDAR checks: {self.check_count} | "
                    f"Distance: {dist_str} | "
                    f"Speed: {speed:.1%}"
                )
        
        except Exception as e:
            rospy.logerr(f"Scan callback error: {e}")
    
    def calculate_speed_multiplier(self, distance):
        """Calculate speed multiplier based on obstacle distance"""
        if distance is None:
            return SPEED_FULL  # No obstacle detected
        
        if distance < LIDAR_STOP_DISTANCE:
            return SPEED_STOP  # EMERGENCY STOP
        elif distance < LIDAR_SLOW_DISTANCE:
            return SPEED_SLOW  # SLOW DOWN
        elif distance < LIDAR_NORMAL_DISTANCE:
            return SPEED_MODERATE  # MODERATE SPEED
        else:
            return SPEED_FULL  # FULL SPEED
    
    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = LiDARSafetyNode()
        node.run()
    except rospy.ROSInterruptException:
        pass