#!/usr/bin/env python3
"""
Grayscale Camera Viewer - To show what CNN sees
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from flask import Flask, Response

app = Flask(__name__)
current_frame = None

def image_callback(msg):
    global current_frame
    
    # Convert ROS Image to numpy
    if msg.encoding == "bgr8":
        img = np.frombuffer(msg.data, dtype=np.uint8)
        img = img.reshape((msg.height, msg.width, 3))
    elif msg.encoding == "rgb8":
        img = np.frombuffer(msg.data, dtype=np.uint8)
        img = img.reshape((msg.height, msg.width, 3))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        return
    
    # Apply SAME preprocessing as CNN
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Back to 3 channels for display
    
    # Resize to 640x480 for viewing
    current_frame = cv2.resize(gray_rgb, (640, 480))

def generate_frames():
    while True:
        if current_frame is None:
            continue
        
        ret, buffer = cv2.imencode('.jpg', current_frame)
        if not ret:
            continue
        
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return '''
    <html>
    <head><title>Grayscale View</title></head>
    <body style="margin:0; background:#000; text-align:center;">
        <h1 style="color:#0f0;">Grayscale View (What CNN Sees)</h1>
        <img src="/video" style="max-width:90%; border:2px solid #0f0;">
    </body>
    </html>
    '''

@app.route('/video')
def video():
    return Response(generate_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    rospy.init_node('grayscale_viewer', anonymous=True)
    
    rospy.Subscriber('/csi_cam_0/image_raw', Image, image_callback, queue_size=1)
    
    print("Grayscale viewer starting...")
    print("View at: http://JETBOT_IP:5001")
    
    app.run(host='0.0.0.0', port=5001, threaded=True)