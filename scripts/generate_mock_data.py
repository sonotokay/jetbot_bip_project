#!/usr/bin/env python3
"""Generate mock training data for testing pipeline"""

import cv2
import numpy as np
import json
import os
from datetime import datetime
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import DATA_DIR, IMAGE_WIDTH, IMAGE_HEIGHT


def generate_track_image(steering):
    """Generate synthetic track image based on steering"""
    img = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    img[:, :] = (50, 50, 50)
    
    center_x = IMAGE_WIDTH // 2
    offset = int(steering * 50)
    
    left_x = center_x - 60 + offset
    cv2.line(img, (left_x, 0), (left_x, IMAGE_HEIGHT), (255, 255, 255), 3)
    
    right_x = center_x + 60 + offset
    cv2.line(img, (right_x, 0), (right_x, IMAGE_HEIGHT), (255, 0, 255), 3)
    
    noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    
    return img


def generate_mock_dataset(num_samples=500):
    """Generate mock training dataset"""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print(f"Generating {num_samples} mock samples")
    print(f"Saving to: {DATA_DIR}")
    
    metadata = []
    
    for i in range(num_samples):
        steering = np.random.uniform(-0.8, 0.8)
        img = generate_track_image(steering)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_filename = f"img_mock_{i:04d}_{timestamp}.jpg"
        image_path = os.path.join(DATA_DIR, image_filename)
        cv2.imwrite(image_path, img)
        
        metadata.append({
            'image': image_filename,
            'steering': float(steering),
            'throttle': 0.5,
            'omega': float(steering * 1.5),
            'timestamp': timestamp
        })
        
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_samples}")
    
    metadata_path = os.path.join(DATA_DIR, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nGenerated {num_samples} samples")
    print(f"Images: {DATA_DIR}/*.jpg")
    print(f"Metadata: {metadata_path}")
    print("\nReady for training: python3 scripts/train.py")


if __name__ == '__main__':
    num_samples = 500
    if len(sys.argv) > 1:
        num_samples = int(sys.argv[1])
    
    generate_mock_dataset(num_samples)