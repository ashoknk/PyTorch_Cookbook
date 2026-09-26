"""
04. Feedforward Neural Network (MLP) with PyTorch

This script demonstrates how to construct, train, and evaluate a multi-class 
classifier on real visual datasets. We use PyTorch's built-in FashionMNIST 
dataset to train a Feedforward Neural Network (Multi-Layer Perceptron) with 
nonlinear ReLU activations to classify clothing categories (0-9).

ReLU (Rectified Linear Unit) is a non-linear activation function defined as f(x) = max(0, x).
It introduces non-linearity, allowing networks to learn complex shapes and patterns beyond straight lines.
Without ReLU, stacking linear layers (fc1 and fc2) collapses into a single linear equation (y = Wx + b).
It mitigates vanishing gradients (positive inputs have a gradient of 1.0) and is computationally fast.
Note: ReLU outputs 0 for negative inputs and x for positive ones; it does NOT bound output to [0, 1].

Learning Objectives:
1. Load and normalize built-in computer vision datasets using torchvision.
2. Build a Multi-Layer Perceptron (MLP) network with nonlinear activation layers.
3. Understand Multi-Class Cross Entropy loss and output logits.
4. Perform correct dataset evaluation using evaluation mode and disabled tracking.
"""

# We import the core torch library.
import torch

# nn contains the neural network modules (Linear, ReLU, CrossEntropyLoss).
# Documentation: https://pytorch.org/docs/stable/nn.html
import torch.nn as nn

# optim contains optimizer engines.
import torch.optim as optim

# torchvision contains specialized models, transformations, and datasets for computer vision.
# Documentation: https://pytorch.org/vision/stable/index.html
import torchvision
import torchvision.transforms as transforms

# DataLoader manages background batching, shuffling, and multi-threaded loading.
# Documentation: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
from torch.utils.data import DataLoader

# ==========================================
# 1. DEFINE MODEL ARCHITECTURE
# ==========================================
class FeedforwardNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(FeedforwardNet, self).__init__()
        # First layer maps 28x28 flattened image pixels (784 features) to hidden dimensions.
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Rectified Linear Unit (ReLU) introduces non-linearity allowing complex mappings.
        # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        self.relu = nn.ReLU()
        # Output layer maps hidden representations to 10 classes.
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Flatten input tensor from (batch, 1, 28, 28) to (batch, 784)
        x = x.view(x.size(0), -1)
        # Feed through the network
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def main():
    # Set device context
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Running classification on: {device} ---")

    # ==========================================
    # 2. DATA PREPARATION (FashionMNIST)
    # ==========================================
    print("--- Preparing FashionMNIST Dataset ---")
    
    # Transforms define preprocessing steps. We convert image bytes to tensors and normalise values.
    # Documentation: https://pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize around mean=0.5, std=0.5
    ])

    # Download training and testing partitions
    # Documentation: https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html
    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform_pipeline, download=True)
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform_pipeline, download=True)

    # Initialize loaders to yield batches of 64 images
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    # ==========================================
    # 3. INITIALIZE MODEL, LOSS & OPTIMIZER
    # ==========================================
    model = FeedforwardNet().to(device)

    # CrossEntropyLoss expects raw scores (logits) as outputs. Log-Softmax is computed internally.
    # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    # ==========================================
    # 4. TRAINING MODULE
    # ==========================================
    # Common educational value, not a typical production default.
    epochs = 3
    print(f"--- Training model for {epochs} epochs ---")
    
    for epoch in range(epochs):
        model.train()  # Explicitly set training mode
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Move images and labels to target accelerator (GPU/MPS/CPU)
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(f"Inside Epoch [{epoch+1}], Step [{batch_idx+1}]")
            
            # loss.item() converts the tensor loss into a plain Python float
            # The loss values are going down over time:
            # This means the network is getting better at minimizing its prediction error as training continues.
            running_loss += loss.item()
            if (batch_idx + 1) % 300 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {running_loss / 300:.4f}")
                running_loss = 0.0

    # ==========================================
    # 5. TESTING AND ACCURACY ASSESSMENT
    # ==========================================
    print("\n--- Evaluating on Test Dataset ---")
    # Put the model in evaluation mode. This disables dropout or batchnorm operations if present.
    # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval
    model.eval()
    
    correct = 0
    total = 0
    
    # We use torch.no_grad() context to prevent PyTorch from building the autograd graph during testing,
    # which reduces memory usage and speeds up testing.
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Get index of maximum logit score (our predicted label)
            # Using dim=1 means: look along the columns. find the highest score for each row/image
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            
            correct += (predicted == labels).sum().item()
    print(f"Correct : {correct} Total: {total}")
    print(f"Final Test Accuracy of the model on the 10,000 test images: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    main()
