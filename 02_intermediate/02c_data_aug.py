"""
<<<<<<< Updated upstream
08 - 2c : Modern Data Augmentation with torchvision.transforms.v2
Visual regularization, dynamic data pipelines, and the recommended modern torchvision v2 transforms API 
(e.g., RandomHorizontalFlip, RandomRotation, ColorJitter, RandomErasing).

Once we understand how a basic CNN is trained, our immediate next problem is overfitting 
  (the model performs well on training data but poorly on test data).
This script teaches them how to solve overfitting by dynamically distorting images during training 
so the network never sees the exact same image twice.

Highly practical. Learners can train a small CNN with and without augmentation and
 compare the accuracy curves, proving how data augmentation forces the model to generalize better.
=======
Concept 3: Modern Data Augmentation with torchvision.transforms.v2
Visual regularization, dynamic data pipelines, and the recommended modern torchvision v2 transforms API
 (e.g., RandomHorizontalFlip, RandomRotation, ColorJitter, RandomErasing).

Once you understand how a basic CNN is trained, their immediate next problem is overfitting
(the model performs well on training data but poorly on test data).
This script teaches how to solve overfitting by dynamically distorting images during training 
so the network never sees the exact same image twice.

Highly practical. Learners can train a small CNN with and without augmentation and compare the accuracy curves, 
proving how data augmentation forces the model to generalize better.
>>>>>>> Stashed changes

Learning Objectives:
1. Understand how data augmentation serves as a powerful regularizer to combat overfitting.
2. Adopt the modern torchvision.transforms.v2 API for high-performance visual augmentations.
3. Compare non-augmented vs. augmented data pipelines.
4. Implement pixel-level erasing (v2.RandomErasing) to make the model robust to occlusions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
# We import the modern v2 transform API from torchvision.
# Documentation: https://pytorch.org/vision/stable/transforms.html
from torchvision.transforms import v2
from torch.utils.data import DataLoader

# ==========================================
# 1. DEFINE CNN MODEL
# ==========================================
class RegularizedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(RegularizedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            # Dropout randomly sets active weights to 0, preventing co-adaptation
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Running Regularization & Data Augmentation on: {device} ---")

    # ==========================================
    # 2. DEFINE AUGMENTED PIPELINE (TORCHVISION V2)
    # ==========================================
    print("--- Defining Modern torchvision.transforms.v2 Pipelines ---")
    
    # 2.1 Training Pipeline: Rich and dynamic transformations are applied here.
    # Every time a batch is requested, these transforms occur on-the-fly inside CPU threads.
    train_transform = v2.Compose([
        # Randomly flip the image horizontally with a 50% chance
        v2.RandomHorizontalFlip(p=0.5),
        
        # Randomly rotate the image between -15 and +15 degrees
        v2.RandomRotation(degrees=15),
        
        # Adjust brightness and contrast randomly to simulate different lighting conditions
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        
        # Convert image to float tensor (scales pixels to [0, 1]) so PyTorch tensors can process it.
        v2.ToImage(), #Converts raw input images into a dedicated torchvision.tv_tensors.Image object (a specialized PyTorch tensor).
        v2.ToDtype(torch.float32, scale=True), #Converts the data type of pixel values to 32-bit floating point numbers.
        #scale=True flag automatically scales integer pixel values from the standard range [0, 255] down to [0.0, 1.0].
        
        # Standard Normalization
        v2.Normalize(mean=(0.5,), std=(0.5,)),
        
        # Randomly erase (replcae with random values) a rectangular patch of the image (simulates objects being partially blocked/occluded).
        # Note: RandomErasing expects a tensor, so it must follow ToImage/ToDtype.
        v2.RandomErasing(p=0.2, scale=(0.02, 0.1), value='random')
        # If erasing occurred before normalization, the erased patch values would undergo normalization again, altering the intended noise level
    ])

    # 2.2 Testing Pipeline: Only deterministic transforms (no randomness!).
    # We must NEVER randomly crop, flip, or erase validation/testing images.
    test_transform = v2.Compose([
        v2.ToImage(), #Converts raw input images into a dedicated torchvision.tv_tensors.Image object (a specialized PyTorch tensor).
        v2.ToDtype(torch.float32, scale=True), #Converts the data type of pixel values to 32-bit floating point numbers.
        v2.Normalize(mean=(0.5,), std=(0.5,)) #Normalizes the pixel values.
    ])

    # Load FashionMNIST using the two separate pipelines
    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=train_transform, download=True)
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=test_transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    print("Dynamic training pipeline with Random Flipping, Rotation, Jitter, and Erasing initialized.")

    # ==========================================
    # 3. INITIALIZE MODEL & OPTIMIZER
    # ==========================================
    model = RegularizedCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ==========================================
    # 4. TRAINING LOOP
    # ==========================================
    epochs = 1
    print(f"--- Training Model with Augmentation for {epochs} epoch ---")
    model.train()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward() # Calculates the gradients for every weight and bias based on mistake/loss
        optimizer.step() # Updates the weights & the biases for the Next Iteration

        if (batch_idx + 1) % 300 == 0:
            print(f"  Step [{batch_idx+1}/{len(train_loader)}], Augmented Training Loss: {loss.item():.4f}")

    # ==========================================
    # 5. TESTING AND GENERALIZATION CHECK
    # ==========================================
    print("\n--- Evaluating Test Dataset (Generalization Check) ---")
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

    print(f"Final Test Accuracy under Regularized Pipeline: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    main()
