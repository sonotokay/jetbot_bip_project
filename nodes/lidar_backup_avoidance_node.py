#!/usr/bin/env python3
"""
LiDAR Backup and Avoidance Node
When obstacle too close: back up, adjust, try again
"""

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Bool, String
from geometry_msgs.msg import Twist
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import LIDAR_TOPIC, LIDAR_MIN_DISTANCE, BASE_SPEED

class State:
    NORMAL = "NORMAL"
    BACKING_UP = "BACKING_UP"
    ADJUSTING = "ADJUSTING"
    WAITING = "WAITING"

class LidarBackupAvoidanceNode:
    """LiDAR node with backup and micro-adjustment behavior"""
    
    def __init__(self):
        rospy.init_node('lidar_backup_avoidance', anonymous=True)
        
        # State machine
        self.state = State.NORMAL
        self.backup_start_time = None
        self.adjust_start_time = None
        self.wait_start_time = None
        
        # Behavior parameters
        self.BACKUP_DURATION = 1.0  # seconds to back up
        self.BACKUP_SPEED = -0.2    # m/s (negative = backward)
        self.ADJUST_DURATION = 0.5  # seconds to adjust
        self.ADJUST_ANGLE = 0.3     # steering adjustment (±0.3)
        self.WAIT_DURATION = 0.3    # seconds to wait before retrying
        
        # Obstacle detection
        self.last_obstacle_side = None  # 'left' or 'right'
        
        # Publishers
        self.override_pub = rospy.Publisher('/lidar_override', Bool, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher('/lidar_cmd_vel', Twist, queue_size=1)
        self.state_pub = rospy.Publisher('/lidar_state', String, queue_size=1)
        self.speed_pub = rospy.Publisher('/speed_multiplier', Float32, queue_size=1)
        
        # Subscriber
        self.system_ready = False
        self.ready_sub = rospy.Subscriber(
            '/cnn_ready', Bool, self.ready_callback, queue_size=1
        )
        
        rospy.loginfo("⏸LiDAR waiting for system ready...")
        self.lidar_sub = rospy.Subscriber(
            LIDAR_TOPIC, LaserScan, self.lidar_callback, queue_size=1
        )        
        
        rospy.loginfo("LiDAR Backup Avoidance Node started")
        rospy.loginfo(f"Critical distance: {LIDAR_MIN_DISTANCE}m")
        rospy.loginfo(f"Backup: {self.BACKUP_DURATION}s at {self.BACKUP_SPEED} m/s")
        rospy.loginfo(f"Adjust: {self.ADJUST_DURATION}s at ±{self.ADJUST_ANGLE}")
    
    def ready_callback(self, msg):
        if msg.data:
            self.system_ready = True
            rospy.loginfo("LiDAR active!")
    
    def analyze_obstacle(self, ranges, angle_min, angle_increment):
        """Determine if obstacle is present and which side"""
        
        # Split forward view into left and right
        left_distances = []
        right_distances = []
        center_distances = []
        
        for i, distance in enumerate(ranges):
            angle = angle_min + i * angle_increment
            
            # Forward 60° arc (±30°)
            if -np.pi/6 <= angle <= np.pi/6:
                if not np.isinf(distance):
                    center_distances.append(distance)
                    
                    if angle > 0:  # Left side
                        left_distances.append(distance)
                    else:  # Right side
                        right_distances.append(distance)
        
        if not center_distances:
            return False, None, float('inf')
        
        min_distance = min(center_distances)
        
        # Determine which side is more obstructed
        left_avg = np.mean(left_distances) if left_distances else float('inf')
        right_avg = np.mean(right_distances) if right_distances else float('inf')
        
        obstacle_side = 'left' if left_avg < right_avg else 'right'
        
        return True, obstacle_side, min_distance
    
    def lidar_callback(self, msg):
        """Process LiDAR and manage state machine"""
        if not self.system_ready:
            return  # Don't process until CNN ready
        
        ranges = np.array(msg.ranges)
        ranges = np.nan_to_num(ranges, nan=float('inf'))
        
        has_obstacle, obstacle_side, min_distance = self.analyze_obstacle(
            ranges, msg.angle_min, msg.angle_increment
        )
        
        # State machine
        current_time = rospy.get_time()
        
        if self.state == State.NORMAL:
            # Check for obstacle
            if has_obstacle and min_distance < LIDAR_MIN_DISTANCE:
                rospy.logwarn(f"OBSTACLE DETECTED at {min_distance:.2f}m on {obstacle_side} side!")
                rospy.logwarn("Initiating BACKUP sequence...")
                
                self.state = State.BACKING_UP
                self.backup_start_time = current_time
                self.last_obstacle_side = obstacle_side
                
                # Override vision control
                self.override_pub.publish(Bool(data=True))
                self.state_pub.publish(String(data=self.state))
            else:
                # Normal operation - let vision control
                self.override_pub.publish(Bool(data=False))
                self.speed_pub.publish(Float32(data=1.0))
        
        elif self.state == State.BACKING_UP:
            # Back up for set duration
            elapsed = current_time - self.backup_start_time
            
            if elapsed < self.BACKUP_DURATION:
                # Send backup command
                cmd = Twist()
                cmd.linear.x = self.BACKUP_SPEED
                cmd.angular.z = 0.0
                self.cmd_vel_pub.publish(cmd)
                
                rospy.loginfo_throttle(0.5, f"Backing up... {elapsed:.1f}s")
            else:
                # Done backing up, start adjustment
                rospy.loginfo("Backup complete. Starting ADJUSTMENT...")
                self.state = State.ADJUSTING
                self.adjust_start_time = current_time
                self.state_pub.publish(String(data=self.state))
        
        elif self.state == State.ADJUSTING:
            # Make micro adjustment
            elapsed = current_time - self.adjust_start_time
            
            if elapsed < self.ADJUST_DURATION:
                # Adjust away from obstacle side
                cmd = Twist()
                cmd.linear.x = 0.0
                
                # If obstacle on left, turn right (negative)
                # If obstacle on right, turn left (positive)
                adjust_direction = -1 if self.last_obstacle_side == 'left' else 1
                cmd.angular.z = adjust_direction * self.ADJUST_ANGLE * 2.0  # Angular velocity
                
                self.cmd_vel_pub.publish(cmd)
                
                rospy.loginfo_throttle(0.5, 
                    f"Adjusting {self.last_obstacle_side} → "
                    f"{'RIGHT' if adjust_direction < 0 else 'LEFT'}... {elapsed:.1f}s")
            else:
                # Done adjusting, wait briefly
                rospy.loginfo("Adjustment complete. Waiting before retry...")
                self.state = State.WAITING
                self.wait_start_time = current_time
                self.state_pub.publish(String(data=self.state))
        
        elif self.state == State.WAITING:
            # Brief pause
            elapsed = current_time - self.wait_start_time
            
            # Stop moving during wait
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd)
            
            if elapsed >= self.WAIT_DURATION:
                # Resume normal operation
                rospy.loginfo("Resuming NORMAL operation")
                self.state = State.NORMAL
                self.override_pub.publish(Bool(data=False))
                self.state_pub.publish(String(data=self.state))
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = LidarBackupAvoidanceNode()
        node.run()
    except rospy.ROSInterruptException:
        pass