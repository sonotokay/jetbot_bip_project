#!/usr/bin/env python3
"""
Simple web-based camera viewer
Access at http://JETBOT_IP:5000
"""
import cv2
from flask import Flask, Response

app = Flask(__name__)
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        
        # Resize for faster streaming
        frame = cv2.resize(frame, (640, 480))
        
        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return '''
    <html>
    <head><title>JetBot Camera</title></head>
    <body style="margin:0; background:#000;">
        <img src="/video" style="width:100%; height:auto;">
    </body>
    </html>
    '''

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)