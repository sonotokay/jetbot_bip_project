#!/usr/bin/env python3
"""
Merge multiple training datasets into one
Usage: python merge_datasets.py run1 run2 run3 ...
"""

import json
import os
import shutil
import sys
from pathlib import Path

def merge_datasets(source_dirs, output_dir):
    """
    Merge multiple dataset directories into one
    
    Args:
        source_dirs: List of source dataset paths
        output_dir: Output merged dataset path
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_metadata = []
    total_images = 0
    
    print(f"\n{'='*60}")
    print(f"Merging {len(source_dirs)} datasets into: {output_dir}")
    print(f"{'='*60}\n")
    
    for i, source_dir in enumerate(source_dirs, 1):
        source_path = Path(source_dir)
        
        if not source_path.exists():
            print(f"⚠️  Warning: {source_dir} does not exist, skipping...")
            continue
        
        # Load metadata
        metadata_file = source_path / 'metadata.json'
        if not metadata_file.exists():
            print(f"⚠️  Warning: No metadata.json in {source_dir}, skipping...")
            continue
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"[{i}/{len(source_dirs)}] Processing: {source_dir}")
        print(f"    Samples: {len(metadata)}")
        
        # Copy images
        copied = 0
        for sample in metadata:
            src_img = source_path / sample['image']
            dst_img = output_path / sample['image']
            
            if src_img.exists():
                # Handle duplicate filenames by prefixing with dataset number
                if dst_img.exists():
                    new_name = f"dataset{i}_{sample['image']}"
                    dst_img = output_path / new_name
                    sample['image'] = new_name  # Update metadata
                
                shutil.copy2(src_img, dst_img)
                copied += 1
        
        print(f"    Copied: {copied} images")
        
        all_metadata.extend(metadata)
        total_images += copied
    
    # Save merged metadata
    output_metadata = output_path / 'metadata.json'
    with open(output_metadata, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ Merge complete!")
    print(f"  Total samples: {len(all_metadata)}")
    print(f"  Total images: {total_images}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python merge_datasets.py <output_dir> <source_dir1> <source_dir2> ...")
        print("\nExample:")
        print("  python merge_datasets.py ./data/merged ./data/run1 ./data/run2")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    source_dirs = sys.argv[2:]
    
    merge_datasets(source_dirs, output_dir)