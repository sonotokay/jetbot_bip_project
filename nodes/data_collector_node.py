#!/usr/bin/env python3
"""
Data Collection ROS Node
Collects training data (no web viewer - use web_video_server instead)
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, Joy, Imu
import os
import json
from datetime import datetime
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import (
    CAMERA_TOPIC, JOY_TOPIC, IMU_TOPIC, DATA_DIR, IMAGE_WIDTH, IMAGE_HEIGHT,
    JOY_STEERING_AXIS, JOY_THROTTLE_AXIS, JOY_DEADZONE,
    DATA_COLLECTION_RATE
)


class DataCollectorNode:
    """ROS node for collecting training data"""
    
    def __init__(self):
        rospy.init_node('data_collector', anonymous=True)
        
        os.makedirs(DATA_DIR, exist_ok=True)
        
        self.latest_image = None
        self.latest_joy = None
        self.latest_imu = None
        self.metadata = []
        self.frame_count = 0
        
        self.image_sub = rospy.Subscriber(
            CAMERA_TOPIC, Image, self.image_callback, queue_size=1
        )
        self.joy_sub = rospy.Subscriber(
            JOY_TOPIC, Joy, self.joy_callback, queue_size=1
        )
        self.imu_sub = rospy.Subscriber(
            IMU_TOPIC, Imu, self.imu_callback, queue_size=1
        )
        
        rospy.loginfo("Data Collector started")
        rospy.loginfo(f"Saving to: {DATA_DIR}")
        rospy.loginfo("View camera at: http://JETBOT_IP:8080/stream_viewer?topic=/csi_cam_0/image_raw")
    
    def apply_deadzone(self, value):
        if abs(value) < JOY_DEADZONE:
            return 0.0
        return value
    
    def image_callback(self, msg):
        """Convert ROS Image to numpy array without cv_bridge"""
        try:
            # Manual conversion: ROS Image -> numpy array
            if msg.encoding == "bgr8":
                img_array = np.frombuffer(msg.data, dtype=np.uint8)
                img_array = img_array.reshape((msg.height, msg.width, 3))
                self.latest_image = img_array
            elif msg.encoding == "rgb8":
                img_array = np.frombuffer(msg.data, dtype=np.uint8)
                img_array = img_array.reshape((msg.height, msg.width, 3))
                self.latest_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                rospy.logwarn_throttle(10, f"Unsupported encoding: {msg.encoding}")
        except Exception as e:
            rospy.logerr_throttle(10, f"Image callback error: {e}")
    
    def joy_callback(self, msg):
        self.latest_joy = msg
    
    def imu_callback(self, msg):
        self.latest_imu = msg
    
    def collect_sample(self):
        if self.latest_image is None:
            rospy.logwarn_throttle(5, "No image received")
            return
        
        if self.latest_joy is None:
            rospy.logwarn_throttle(5, "No joystick data received")
            return
        
        if len(self.latest_joy.axes) <= JOY_STEERING_AXIS:
            rospy.logwarn("Joystick axes insufficient")
            return
        
        steering = self.apply_deadzone(self.latest_joy.axes[JOY_STEERING_AXIS])
        
        throttle = 0.0
        if len(self.latest_joy.axes) > JOY_THROTTLE_AXIS:
            throttle = self.apply_deadzone(self.latest_joy.axes[JOY_THROTTLE_AXIS])
            throttle = (throttle + 1.0) / 2.0
        
        omega = 0.0
        if self.latest_imu is not None:
            omega = self.latest_imu.angular_velocity.z
        
        image = cv2.resize(self.latest_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_filename = f"img_{timestamp}.jpg"
        image_path = os.path.join(DATA_DIR, image_filename)
        cv2.imwrite(image_path, image)
        
        self.metadata.append({
            'image': image_filename,
            'steering': float(steering),
            'throttle': float(throttle),
            'omega': float(omega),
            'timestamp': timestamp
        })
        
        self.frame_count += 1
        
        if self.frame_count % 100 == 0:
            rospy.loginfo(
                f"Collected {self.frame_count} frames | "
                f"Steering: {steering:+.2f} | "
                f"Omega: {omega:+.2f} rad/s"
            )
    
    def run(self):
        rate = rospy.Rate(DATA_COLLECTION_RATE)
        
        rospy.loginfo(f"Collecting at {DATA_COLLECTION_RATE} Hz")
        rospy.loginfo("Drive with gamepad")
        
        try:
            while not rospy.is_shutdown():
                self.collect_sample()
                rate.sleep()
        except rospy.ROSInterruptException:
            pass
        finally:
            self.save_metadata()
    
    def save_metadata(self):
        metadata_path = os.path.join(DATA_DIR, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        rospy.loginfo(f"Total frames: {self.frame_count}")
        rospy.loginfo(f"Saved to: {DATA_DIR}")


if __name__ == '__main__':
    try:
        node = DataCollectorNode()
        node.run()
    except rospy.ROSInterruptException:
        pass