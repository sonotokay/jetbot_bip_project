import torch
import torch.nn as nn

class SteeringCNN(nn.Module):
    """
    CNN for predicting steering angles from images
    Input: RGB image (3, 224, 224)
    Output: Steering angle (scalar, range -1 to 1)
    """
    def __init__(self):
        super(SteeringCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(36),
            
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.flatten_size = 64 * 21 * 21  # Fixed
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(50, 10),
            nn.ReLU(),
            
            nn.Linear(10, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class LightweightSteeringCNN(nn.Module):
    """Lightweight version for faster inference on Jetson Nano"""
    
    def __init__(self):
        super(LightweightSteeringCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            
            nn.Conv2d(32, 48, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        
        self.flatten_size = 48 * 26 * 26
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size, 50),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(50, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


def get_model(lightweight=False):
    if lightweight:
        return LightweightSteeringCNN()
    else:
        return SteeringCNN()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SteeringCNN()
    print(f"Model created: {count_parameters(model):,} parameters")
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output value: {output.item():.4f}")
    
    model_lite = LightweightSteeringCNN()
    print(f"Lightweight model: {count_parameters(model_lite):,} parameters")
    output_lite = model_lite(dummy_input)
    print(f"Output: {output_lite.item():.4f}")