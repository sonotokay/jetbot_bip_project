#!/usr/bin/env python3
"""
Data Collection ROS Node with Live Web Stream
Collects training data and serves live camera feed
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, Joy, Imu
from cv_bridge import CvBridge
import os
import json
from datetime import datetime
import sys
import threading
from flask import Flask, Response

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import (
    CAMERA_TOPIC, JOY_TOPIC, IMU_TOPIC, DATA_DIR, IMAGE_WIDTH, IMAGE_HEIGHT,
    JOY_STEERING_AXIS, JOY_THROTTLE_AXIS, JOY_DEADZONE,
    DATA_COLLECTION_RATE
)

app = Flask(__name__)
data_collector = None


class DataCollectorNode:
    """ROS node for collecting training data with live web preview"""
    
    def __init__(self):
        global data_collector
        data_collector = self
        
        rospy.init_node('data_collector', anonymous=True)
        
        os.makedirs(DATA_DIR, exist_ok=True)
        
        self.bridge = CvBridge()
        self.current_frame = None
        self.latest_image = None
        self.frame_lock = threading.Lock()
        
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
        
        self.flask_thread = threading.Thread(target=self.start_flask, daemon=True)
        self.flask_thread.start()
        
        rospy.loginfo("Data Collector started")
        rospy.loginfo(f"Saving to: {DATA_DIR}")
        rospy.loginfo("Web viewer: http://JETBOT_IP:5000")
    
    def start_flask(self):
        app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    
    def get_current_frame(self):
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def apply_deadzone(self, value):
        if abs(value) < JOY_DEADZONE:
            return 0.0
        return value
    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.frame_lock:
                self.current_frame = cv_image.copy()
            self.latest_image = cv_image
        except Exception as e:
            rospy.logerr(f"Image callback error: {e}")
    
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


@app.route('/')
def index():
    return '''
    <html>
    <head>
        <title>JetBot Data Collection</title>
        <style>
            body { margin:0; background:#000; font-family: monospace; color: #0f0; }
            .container { text-align: center; padding: 20px; }
            img { max-width: 90%; height: auto; border: 2px solid #0f0; }
            .info { margin: 20px; font-size: 18px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>JetBot Data Collection</h1>
            <div class="info">Camera: 640x480 | Saved: 224x224</div>
            <img src="/video">
        </div>
    </body>
    </html>
    '''

@app.route('/video')
def video():
    def generate():
        while True:
            if data_collector is None:
                continue
            
            frame = data_collector.get_current_frame()
            if frame is None:
                continue
            
            frame = cv2.resize(frame, (640, 480))
            
            if data_collector.frame_count > 0:
                cv2.putText(frame, f"Frames: {data_collector.frame_count}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if data_collector.latest_joy and len(data_collector.latest_joy.axes) > JOY_STEERING_AXIS:
                    steering = data_collector.apply_deadzone(data_collector.latest_joy.axes[JOY_STEERING_AXIS])
                    cv2.putText(frame, f"Steering: {steering:+.2f}", 
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
            
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    try:
        node = DataCollectorNode()
        node.run()
    except rospy.ROSInterruptException:
        pass