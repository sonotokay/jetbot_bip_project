import os
import requests
from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from ultralytics import YOLO

JETSON_IP = os.environ.get("JETSON_IP", "192.168.137.112")
JETSON_ALARM_URL = os.environ.get("JETSON_ALARM_URL", f"http://{JETSON_IP}:9000/alarm")

app = FastAPI()

model = YOLO("yolov8n.pt")

PERSON_CLASS_ID = 0  # COCO: 0 = person
BOTTLE_CLASS_ID = 39  # COCO: 39 = bottle

@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = 0.35):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"ok": False, "error": "Could not decode image"}

    results = model.predict(img, conf=conf, verbose=False)[0]

    detections = []
    if results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), score, cls in zip(boxes, scores, classes):
            if cls == PERSON_CLASS_ID:
                detections.append({
                    "class": "person",
                    "conf": float(score),
                    "xyxy": [float(x1), float(y1), float(x2), float(y2)]
                })
            if cls == BOTTLE_CLASS_ID:
                detections.append({
                    "class": "bottle",
                    "conf": float(score),
                    "xyxy": [float(x1), float(y1), float(x2), float(y2)]
                })

    if JETSON_ALARM_URL:
        person_found = any(d["class"] == "person" for d in detections)
        if person_found:
            try:
                requests.post(
                    JETSON_ALARM_URL,
                    json={"alarm": True, "reason": "person detected", "source": "laptop-yolo"},
                    timeout=0.2
                )
            except requests.RequestException:
                pass


    return {"ok": True, "num_detections": len(detections), "detections": detections}