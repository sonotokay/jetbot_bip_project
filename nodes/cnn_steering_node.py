#!/usr/bin/env python3
"""
CNN Steering Node with OpenCV Camera
Publishes to: /steering_cmd (Float32)
"""

import rospy
import cv2
import torch
import numpy as np
from std_msgs.msg import Float32
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import (
    STEERING_TOPIC, IMAGE_WIDTH, IMAGE_HEIGHT,
    BEST_MODEL_PATH, USE_LIGHTWEIGHT_MODEL,
    PREDICT_ANGULAR_VELOCITY, CNN_INFERENCE_RATE
)
from core.model import get_model


class CNNSteeringNode:
    """ROS node for CNN-based steering prediction using OpenCV"""
    
    def __init__(self):
        rospy.init_node('cnn_steering', anonymous=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rospy.loginfo(f"Using device: {self.device}")
        
        self.model = get_model(lightweight=USE_LIGHTWEIGHT_MODEL).to(self.device)
        
        if os.path.exists(BEST_MODEL_PATH):
            self.model.load_state_dict(
                torch.load(BEST_MODEL_PATH, map_location=self.device)
            )
            rospy.loginfo(f"Model loaded: {BEST_MODEL_PATH}")
        else:
            rospy.logerr(f"Model not found: {BEST_MODEL_PATH}")
            sys.exit(1)
        
        self.model.eval()
        
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not self.camera.isOpened():
            rospy.logerr("Failed to open camera")
            sys.exit(1)
        
        self.steering_pub = rospy.Publisher(
            STEERING_TOPIC, Float32, queue_size=1
        )
        
        self.inference_count = 0
        self.total_time = 0.0
        
        output_type = "angular velocity" if PREDICT_ANGULAR_VELOCITY else "steering angle"
        rospy.loginfo(f"CNN Steering started (predicting {output_type})")
    
    def preprocess_image(self, cv_image):
        """Preprocess image for CNN"""
        image = cv2.resize(cv_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)
        image = image.unsqueeze(0)
        image = image / 255.0
        return image.to(self.device)
    
    def predict_steering(self):
        """Capture frame and predict steering"""
        ret, frame = self.camera.read()
        if not ret:
            rospy.logwarn_throttle(5, "Failed to read from camera")
            return None
        
        start_time = rospy.get_time()
        
        with torch.no_grad():
            image_tensor = self.preprocess_image(frame)
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
        
        finally:
            self.camera.release()


if __name__ == '__main__':
    try:
        node = CNNSteeringNode()
        node.run()
    except rospy.ROSInterruptException:
        pass