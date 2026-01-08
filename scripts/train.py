#!/usr/bin/env python3
"""
Training Script for CNN Model
(Not a ROS node - runs offline)
"""

import os
import json
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import get_model, count_parameters
from core.config import (
    DATA_DIR, MODEL_SAVE_PATH, BATCH_SIZE, LEARNING_RATE,
    NUM_EPOCHS, VAL_SPLIT, IMAGE_WIDTH, IMAGE_HEIGHT,
    PREDICT_ANGULAR_VELOCITY, USE_LIGHTWEIGHT_MODEL
)


class SteeringDataset(Dataset):
    """Dataset for steering/angular velocity prediction"""
    
    def __init__(self, data_dir, augment=False, use_omega=False):
        self.data_dir = data_dir
        self.augment = augment
        self.use_omega = use_omega  # Use angular velocity instead of steering
        
        # Load metadata
        metadata_path = os.path.join(data_dir, "metadata.json")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"Loaded {len(self.metadata)} samples")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        sample = self.metadata[idx]
        
        # Load image
        image_path = os.path.join(self.data_dir, sample['image'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        
        # Get label (steering or omega)
        if self.use_omega:
            label = sample.get('omega', 0.0)  # Angular velocity
        else:
            label = sample['steering']  # Steering angle
        
        # Augmentation
        if self.augment:
            image, label = self.augment_data(image, label)
        
        # To tensor
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        label = torch.tensor([label], dtype=torch.float32)
        
        return image, label
    
    def augment_data(self, image, label):
        """Apply data augmentation"""
        # Horizontal flip
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
            label = -label  # Flip direction
        
        # Brightness adjustment
        if np.random.rand() > 0.5:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            brightness = 0.5 + np.random.rand() * 0.5
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness, 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Random shadow
        if np.random.rand() > 0.7:
            h, w = image.shape[:2]
            x1, y1 = np.random.randint(0, w), 0
            x2, y2 = np.random.randint(0, w), h
            shadow_mask = np.zeros_like(image)
            cv2.fillPoly(shadow_mask, 
                        [np.array([[x1,y1], [x2,y2], [x2,h], [x1,h]])],
                        (1,1,1))
            image = image * (1 - 0.5 * shadow_mask)
        
        return image, label


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train one epoch"""
    model.train()
    total_loss = 0
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def plot_history(train_losses, val_losses):
    """Plot training curves"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(MODEL_SAVE_PATH, 'training_history.png'))
    print(f"Plot saved: {MODEL_SAVE_PATH}/training_history.png")


def main():
    """Main training function"""
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load data
    use_omega = PREDICT_ANGULAR_VELOCITY
    label_type = "angular velocity" if use_omega else "steering angle"
    print(f"Training to predict: {label_type}\n")
    
    full_dataset = SteeringDataset(DATA_DIR, augment=True, use_omega=use_omega)
    val_size = int(VAL_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"Training: {len(train_dataset)} samples")
    print(f"Validation: {len(val_dataset)} samples\n")
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=2)
    
    # Model
    model = get_model(lightweight=USE_LIGHTWEIGHT_MODEL).to(device)
    print(f"Model parameters: {count_parameters(model):,}\n")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("="*60)
    print("TRAINING STARTED")
    print("="*60 + "\n")
    
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        print(f"Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                      os.path.join(MODEL_SAVE_PATH, 'best_model.pth'))
            print(f"✓ Best model saved")
        
        # Checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(),
                      os.path.join(MODEL_SAVE_PATH, f'checkpoint_{epoch+1}.pth'))
        
        print()
    
    print("="*60)
    print(f"TRAINING COMPLETE")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("="*60 + "\n")
    
    # Save final
    torch.save(model.state_dict(),
              os.path.join(MODEL_SAVE_PATH, 'final_model.pth'))
    
    # Plot
    plot_history(train_losses, val_losses)


if __name__ == "__main__":
    main()