#!/usr/bin/env python3
"""
Convert training images to video for easy inspection
Overlays frame number and steering value
"""

import json
import cv2
import os
from pathlib import Path

def create_video(data_dir, output_video='training_video.mp4', fps=10, max_frames=None):
    """
    Create video from training images
    
    Args:
        data_dir: Path to dataset directory
        output_video: Output video filename
        fps: Frames per second (10 = same as collection rate)
        max_frames: Limit video length (None = all frames)
    """
    
    # Load metadata
    metadata_path = Path(data_dir) / 'metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    if max_frames:
        metadata = metadata[:max_frames]
    
    print(f"Creating video from {len(metadata)} frames...")
    print(f"Output: {output_video}")
    print(f"FPS: {fps}")
    
    # Get first image to determine size
    first_img_path = Path(data_dir) / metadata[0]['image']
    first_img = cv2.imread(str(first_img_path))
    height, width = first_img.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Process each frame
    for i, sample in enumerate(metadata):
        img_path = Path(data_dir) / sample['image']
        
        if not img_path.exists():
            print(f"Warning: Missing {img_path}")
            continue
        
        # Load image
        frame = cv2.imread(str(img_path))
        
        if frame is None:
            print(f"Warning: Could not read {img_path}")
            continue
        
        # Overlay info
        steering = sample['steering']
        
        # Frame number (top-left, large)
        cv2.putText(frame, f"Frame: {i}", 
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.2, (0, 255, 0), 3)
        
        # Steering value (top-right, large)
        steer_text = f"Steer: {steering:+.2f}"
        color = (0, 255, 0) if abs(steering) < 0.8 else (0, 0, 255)  # Red if extreme
        cv2.putText(frame, steer_text,
                   (width - 250, 40), cv2.FONT_HERSHEY_SIMPLEX,
                   1.2, color, 3)
        
        # Timestamp (bottom)
        timestamp = sample['timestamp'][9:]  # Just time part
        cv2.putText(frame, timestamp,
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
        
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(metadata)} frames...")
    
    out.release()
    print(f"\n✓ Video created: {output_video}")
    print(f"  Total frames: {len(metadata)}")
    print(f"  Duration: {len(metadata)/fps:.1f} seconds")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python images_to_video.py <dataset_dir> [output.mp4] [fps]")
        print("\nExample:")
        print("  python images_to_video.py ./data/challenge-merged")
        print("  python images_to_video.py ./data/challenge-merged training.mp4 15")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else 'training_video.mp4'
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    create_video(data_dir, output, fps)