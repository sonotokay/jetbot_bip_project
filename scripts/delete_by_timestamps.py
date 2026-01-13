#!/usr/bin/env python3
"""
Delete frames by timestamp ranges
Automatically finds frame indices and removes bad data
"""

import json
import os
from pathlib import Path

def find_frame_index(metadata, timestamp_substr):
    """Find frame index matching timestamp substring"""
    for i, sample in enumerate(metadata):
        # Check if timestamp contains the substring (HHMMSS format)
        ts = sample['timestamp'][9:15]  # Extract HHMMSS
        if timestamp_substr in ts:
            return i
    return None

def delete_by_timestamps(data_dir, timestamp_ranges, dry_run=False):
    """
    Delete frames by timestamp ranges
    
    Args:
        data_dir: Dataset directory
        timestamp_ranges: List of (start_ts, end_ts) tuples (HHMMSS format)
        dry_run: If True, only show what would be deleted
    """
    metadata_path = Path(data_dir) / 'metadata.json'
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    total_before = len(metadata)
    
    print(f"{'='*70}")
    print(f"Dataset: {data_dir}")
    print(f"Total frames: {total_before}")
    print(f"{'='*70}\n")
    
    # Convert timestamp ranges to frame indices
    frame_ranges = []
    to_delete = set()
    
    for start_ts, end_ts in timestamp_ranges:
        # Find start and end indices
        start_idx = None
        end_idx = None
        
        for i, sample in enumerate(metadata):
            ts = sample['timestamp'][9:15]  # HHMMSS
            
            if start_ts in ts and start_idx is None:
                start_idx = i
            
            if end_ts in ts:
                end_idx = i
        
        if start_idx is not None and end_idx is not None:
            frame_ranges.append((start_idx, end_idx))
            
            # Add all frames in range to delete set
            for i in range(start_idx, end_idx + 1):
                to_delete.add(i)
            
            print(f"Timestamp {start_ts}-{end_ts} → Frames {start_idx}-{end_idx} "
                f"({end_idx - start_idx + 1} frames)")
        else:
            print(f"⚠️  Warning: Could not find frames for {start_ts}-{end_ts}")
    
    print(f"\nTotal frames to delete: {len(to_delete)}\n")
    
    if dry_run:
        print("DRY RUN - No files will be deleted")
        print(f"Would delete {len(to_delete)} frames")
        return
    
    # Actually delete
    print("Deleting frames...")
    kept_metadata = []
    deleted_count = 0
    
    for i, sample in enumerate(metadata):
        if i in to_delete:
            # Delete image file
            img_path = Path(data_dir) / sample['image']
            if img_path.exists():
                os.remove(img_path)
            deleted_count += 1
            
            if deleted_count % 10 == 0 or deleted_count == len(to_delete):
                print(f"  Deleted {deleted_count}/{len(to_delete)} frames...", end='\r')
        else:
            kept_metadata.append(sample)
    
    print()  # New line after progress
    
    # Save updated metadata
    with open(metadata_path, 'w') as f:
        json.dump(kept_metadata, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✓ Deletion complete!")
    print(f"  Before: {total_before} frames")
    print(f"  Deleted: {deleted_count} frames")
    print(f"  After: {len(kept_metadata)} frames")
    print(f"  Percentage removed: {deleted_count/total_before*100:.1f}%")
    print(f"{'='*70}")

if __name__ == '__main__':
    import sys
    
    # Dataset directory
    data_dir = './data/challenge-merged'
    
    # Timestamp ranges to delete (HHMMSS format)
    # Based on your notes - deleting hand corrections only
    timestamp_ranges = [
        ('155527', '155532'),  # Crash + adjustment
        ('155634', '155641'),  # Similar crash
        ('160430', '160437'),  # Hand correction
        ('160536', '160545'),  # Hand correction  
        ('160640', '160650'),  # End frames with hand correction
    ]
    
    # NOT deleting: 155744-155748 (natural bump recovery - good data!)
    
    print("Frames to DELETE (hand corrections):")
    for start, end in timestamp_ranges:
        print(f"  {start} - {end}")
    
    # Ask for confirmation
    print(f"\n{'='*70}")
    response = input("Proceed with deletion? (yes/no): ")
    
    if response.lower() == 'yes':
        delete_by_timestamps(data_dir, timestamp_ranges, dry_run=False)
    else:
        print("Cancelled - running dry run instead...")
        delete_by_timestamps(data_dir, timestamp_ranges, dry_run=True)