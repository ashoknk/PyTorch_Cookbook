"""
07. Convolutional Neural Network (CNN) with PyTorch

This script demonstrates how to construct, train, and evaluate a Convolutional 
Neural Network (CNN) for image classification. CNNs use localized kernel 
filters and pooling operations to capture spatial hierarchies. We use the pre-downloaded 
FashionMNIST dataset to train a custom CNN featuring Conv2D, BatchNorm2d, MaxPool2d, 
and fully connected layers.

Learning Objectives:
1. Define a 2D convolutional neural network with downsampling layers.
2. Apply Batch Normalization to stabilize and accelerate training.
3. Understand how dimensions change when passing spatial grids through convolutions and pooling layers.
"""

# We import the core torch library.
import torch

# nn houses standard structural layers: Conv2d, BatchNorm2d, MaxPool2d, Linear.
# Documentation: https://pytorch.org/docs/stable/nn.html
import torch.nn as nn

# optim handles parameter optimization.
import torch.optim as optim

# torchvision contains standard computer vision datasets.
import torchvision
import torchvision.transforms as transforms

# DataLoader manages memory-safe streaming batches.
from torch.utils.data import DataLoader

# ==========================================
# 1. DEFINE CNN MODEL ARCHITECTURE
# ==========================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Convolution Block 1:
        # Input channel = 1 (grayscale FashionMNIST images), Output channels = 16.
        # Kernel size = 3x3, Stride = 1, Padding = 1 (retains spatial size 28x28)
        # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        
        # Batch Normalization normalizes activations per mini-batch.
        # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
        self.bn1 = nn.BatchNorm2d(16)
        
        # Activation function
        self.relu1 = nn.ReLU()
        
        # MaxPool2d divides height and width by stride factor. Kernel size 2x2 with stride 2
        # maps 28x28 spatial features to 14x14.
        # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolution Block 2: Output channels = 32. Spatial size maps from 14x14 to 7x7 (after pooling)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        
        # Classification Head:
        # After two pooling operations, a 28x28 input becomes 7x7.
        # Total flattened features = 32 channels * 7 height * 7 width = 1568 features.
        self.fc = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        # Pass through Block 1 + pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)
        
        # Pass through Block 2 + pooling
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool(x)
        
        # Flatten the spatial volume for the classification layer
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

def main():
    # Set computing accelerators
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Running CNN training on: {device} ---")

    # ==========================================
    # 2. PREPARE PRE-DOWNLOADED DATASET
    # ==========================================
    print("--- Preparing FashionMNIST Dataset ---")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=False)
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform, download=False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    # ==========================================
    # 3. INITIALIZE TRAINING CRITERIA
    # ==========================================
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ==========================================
    # 4. CNN TRAINING MODULE (1 epoch for swift verification)
    # ==========================================
    epochs = 1
    print(f"--- Training model for {epochs} epoch ---")
    model.train()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (batch_idx + 1) % 300 == 0:
            print(f"  Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    # ==========================================
    # 5. TESTING AND ACCURACY ASSESSMENT
    # ==========================================
    print("\n--- Evaluating Test Dataset ---")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Final Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    main()
