#!/usr/bin/env python3
"""
CNN Steering Node - ROS Image Version
Publishes to: /steering_cmd (Float32)
"""

import rospy
import torch
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool
import cv2
import os
import sys

import requests
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import (
    CAMERA_TOPIC, STEERING_TOPIC, IMAGE_WIDTH, IMAGE_HEIGHT,
    USE_LIGHTWEIGHT_MODEL,
    PREDICT_ANGULAR_VELOCITY, CNN_INFERENCE_RATE, LAPTOP_IP
)
from core.model import get_model

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'checkpoints', 'best_model.pth')

class CNNSteeringNode:
    """ROS node for CNN-based steering prediction"""
    
    def __init__(self):
        self.laptop_url = f"http://{LAPTOP_IP}:8000/detect?conf=0.35"
        self.last_send = 0.0
        self.send_hz = 5.0 # Send every 0.2 seconds
        self.sending = False

        rospy.init_node('cnn_steering', anonymous=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rospy.loginfo(f"Using device: {self.device}")
        
        self.model = get_model(lightweight=USE_LIGHTWEIGHT_MODEL).to(self.device)
        
        if os.path.exists(MODEL_PATH):
            self.model.load_state_dict(
                torch.load(MODEL_PATH, map_location=self.device)
            )
            rospy.loginfo(f"Model loaded: {MODEL_PATH}")
        else:
            rospy.logerr(f"Model not found: {MODEL_PATH}")
            sys.exit(1)
        
        self.model.eval()
        
        # Add ready publisher
        self.ready_pub = rospy.Publisher('/cnn_ready', Bool, queue_size=1, latch=True)
        
        self.latest_image = None
        self.ready = False
    
        rospy.loginfo("CNN node started, waiting for first image...")
        
        self.image_sub = rospy.Subscriber(
            CAMERA_TOPIC, Image, self.image_callback, queue_size=1
        )
        
        self.steering_pub = rospy.Publisher(
            STEERING_TOPIC, Float32, queue_size=1
        )
        
        self.inference_count = 0
        self.total_time = 0.0
        
        output_type = "angular velocity" if PREDICT_ANGULAR_VELOCITY else "steering angle"
        rospy.loginfo(f"CNN Steering started (predicting {output_type})")
    
    def image_callback(self, msg):
        """Convert ROS Image to numpy array (no cv_bridge needed)"""
        try:
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
                return

            # Forward frames to the laptop YOLO server without blocking the callback
            now = time.time()
            if self.sending or now - self.last_send < 1.0 / self.send_hz:
                return
            self.last_send = now
            self.sending = True
            threading.Thread(
                target=self.send_to_detector, args=(self.latest_image,), daemon=True
            ).start()

        except Exception as e:
            rospy.logerr_throttle(10, f"Image callback error: {e}")
    
    def send_to_detector(self, img):
        """Send one JPEG frame to the laptop YOLO server"""
        try:
            ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if not ok:
                return

            files = {"file": ("frame.jpg", enc.tobytes(), "image/jpeg")}
            requests.post(self.laptop_url, files=files, timeout=0.5)
        except requests.RequestException as e:
            rospy.logwarn_throttle(2.0, f"YOLO request failed: {e}")
        finally:
            self.sending = False
    
    def preprocess_image(self, cv_image):
        """Preprocess image - must match SteeringDataset in scripts/train.py"""
        # SAME ORDER AS TRAINING:
        # 1. BGR to RGB
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # 2. Resize
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        
        # 3. To tensor
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)  # HWC -> CHW
        image = image.unsqueeze(0)  # Add batch dimension
        image = image / 255.0  # Normalize
        
        return image.to(self.device)
    
    def predict_steering(self):
        """Predict steering from latest image"""
        if self.latest_image is None:
            rospy.logwarn_throttle(5, "No image received")
            return None
        
        start_time = rospy.get_time()
        
        with torch.no_grad():
            image_tensor = self.preprocess_image(self.latest_image)
            prediction = self.model(image_tensor)
            steering = prediction.item()
        
        inference_time = rospy.get_time() - start_time
        
        self.inference_count += 1
        self.total_time += inference_time
        
        if self.inference_count % 100 == 0:
            avg_time = self.total_time / self.inference_count
            fps = 1.0 / avg_time if avg_time > 0 else 0
            rospy.loginfo(
                f"Predictions: {self.inference_count} | "
                f"Avg: {avg_time*1000:.1f}ms | "
                f"FPS: {fps:.1f} | "
                f"Latest: {steering:+.3f}"
            )
        
        # First successful prediction = ready
        if not self.ready:
            self.ready = True
            rospy.loginfo("CNN READY")
            self.ready_pub.publish(Bool(data=True))
        
        return steering
    
    def run(self):
        rate = rospy.Rate(CNN_INFERENCE_RATE)
        
        rospy.loginfo(f"Running at {CNN_INFERENCE_RATE} Hz")
        
        try:
            while not rospy.is_shutdown():
                steering = self.predict_steering()
                
                if steering is not None:
                    msg = Float32()
                    msg.data = steering
                    self.steering_pub.publish(msg)
                
                rate.sleep()
        
        except rospy.ROSInterruptException:
            pass


if __name__ == '__main__':
    try:
        node = CNNSteeringNode()
        node.run()
    except rospy.ROSInterruptException:
        pass