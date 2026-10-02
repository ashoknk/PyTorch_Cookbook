"""
07. Convolutional Neural Network (CNN) with PyTorch

CNN is primarily used for computer vision and image processing tasks. 
This script demonstrates how to construct, train, and evaluate a Convolutional 
Neural Network (CNN) for image classification. CNNs use localized kernel 
filters and pooling operations to capture spatial hierarchies (exactly like a hand exploring an object by touch).
We use the pre-downloaded FashionMNIST dataset to train a custom CNN featuring Conv2D, BatchNorm2d, MaxPool2d, 
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
"""
The 4 Main Tools in Your Neural Network:
Each layer type acts like a specific specialist on a digital factory assembly line:

    1. Conv2d (The Feature Detector):
        What it does: Scans the image with tiny sliding filters (like 3x3 pixel 
        magnifying glasses) to detect edges, curves, and textures.
    
         It converts raw pixels into visual shapes. 
        The 1st Conv2d layer finds simple edges (e.g., straight lines of a trouser). 
        The 2nd Conv2d layer combines those simple edges into complex shapes (e.g., shoe soles or shirt collars).

    2. BatchNorm2d (The Cleaner / Stabilizer):
        What it does: Scales and centers the output numbers from the Conv layer so they 
        don't grow too large or drift too small.
    
         It cleans up the signal so the network doesn't get overwhelmed. 
        It keeps the math stable so the model learns much faster.

    3. ReLU (The Smart Filter / Decision Maker):
        What it does: Replaces all negative numbers with zero (f(x) = max(0, x)).
    
         It throws away useless details and turns on "interest points." 
        If a feature detector finds an edge, ReLU keeps it; if it finds nothing, ReLU mutes it. 
        This allows the model to learn complex, non-linear real-world shapes.

    4. MaxPool2d (The Summarizer / Shrinker):
        What it does: Looks at a 2x2 grid of pixels, picks only the largest value, 
        and discards the rest.
    
         It shrinks the image size in half (28x28 to 14x14 to 7x7) 
        while preserving the most prominent features. This reduces memory usage and 
        ensures the model recognizes a shoe whether it is slightly shifted to the left or right.

    MaxPool2d Parameter Breakdown:
        a. kernel_size=2 -> The 2x2 Pixel Grid Window:
            Sets the dimensions of the sliding window box. It inspects a 2x2 area 
            (4 pixels total) and selects only the maximum value inside that region.
        b. stride=2 -> The Step Size (Shrinks image size by half):
            Sets how many pixels the sliding window shifts over after each step. 
            Jumping 2 pixels at a time halves both the image height and width 
            (for example, turning a 28x28 feature map into 14x14).

    5.  Linear (Classification Head)
        Flatten (32  times 7  times 7 = 1,568 features): 
          Unrolls the 32 small 7  times 7 feature maps into a single flat line of 1,568 numbers.
        Linear(1568 -> 10): 
          Acts as the final judge. It weighs all 1,568 detected high-level features and outputs 10 scores 
          (one for each category: T-shirt, Trouser, Ankle boot, etc.).
        
"""

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # Convolution Block 1:
        # Input channel = 1 (grayscale FashionMNIST images), Output channels = 16.
        # Create 16 separate sliding filters (kernels) for that layer.
        # Instead of searching for just one pattern, the network will look for 16 distinct visual features across the input image simultaneously 
        # e.g. Filter 1 detects horizontal edges, Filter 2 detects vertical edges, Filter 3 detects diagonal lines, etc.
        # 3 X 3 kernel is the universal standard across modern vision models
        # Kernel size = 3x3, Stride = 1, Padding = 1 (retains spatial size 28x28)
        # stride=1: Moves the 3 X 3 filter 1 pixel at a time so no spatial information is skipped during feature extraction
        # padding=1: Adds a 1-pixel border of zeros around the edge of the input tensor.
        # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # Conv2d keeps the spatial size the same when padding=1, kernel_size=3, stride=1.
        # First conv output: 16 × 28 × 28
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        
        # Batch Normalization normalizes activations per mini-batch.
        # Requires num_features=16 to match the number of channels coming out of the preceding layer.
        # Calculates a separate mean and variance for each individual channel across the batch
        # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
        self.bn1 = nn.BatchNorm2d(num_features=16)
        
        # Activation function
        self.relu1 = nn.ReLU()
        
        # MaxPool2d downsamples spatial size by the stride factor. With kernel_size=2 and stride=2,
        # a 28x28 feature map becomes 14x14.
        # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
        # After first pool: 16 × 14 × 14
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolution Block 2: Output channels = 32. The spatial size stays at 14x14 after conv2
        # because kernel_size=3, stride=1, padding=1 keeps the same height/width.
        # The second pooling operation then reduces 14x14 to 7x7.
        # Early layers only need a few filters (16) to capture simple low-level features like straight lines or edges.
        # Deeper layers need more filters (32) because they combine those basic lines into many different complex high-level shapes (e.g., shoe soles, sleeve hems, buttonholes, pocket curves).
        # After second conv2: output is 32 × 14 × 14
        # After second pool: output is 32 × 7 × 7
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.relu2 = nn.ReLU()
        
        # Classification Head:
        # After the two pooling operations, the spatial size is 7x7.
        # Total flattened features = 32 channels * 7 height * 7 width = 1568 features.
        self.fc = nn.Linear(32 * 7 * 7, num_classes)


    def forward(self, x):
        # Input image: 1 × 28 × 28. 
        # Pass through Block 1 + pooling
        x = self.conv1(x)
        # Conv2d keeps the spatial size the same when padding=1, kernel_size=3, stride=1.
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)
        # After first pool: 16 × 14 × 14
        
        # Pass through Block 2 + pooling
        x = self.conv2(x)
        # After second conv2: output is 32 × 14 × 14
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool(x)
        # After second pool: 32 × 7 × 7
        
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
    # By default, when you create a model in PyTorch (model = SimpleCNN()), it is already in training mode. 
    # That is why code without an explicit model.train() still works.
    model.train()
    
    # model(inputs) --> criterion(outputs, labels) -> optimizer.zero_grad() -> loss.backward() -> optimizer.step()
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
    
    """values, predicted = torch.max(outputs.data, 1):
        - values would hold the maximum confidence scores.
        - predicted holds the actual predicted class indices (0 through 9)."""

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # Get index of maximum logit score (our predicted label)
            # Using dim=1 means: look along the columns. Find the highest score for each row/image
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            # test_loader yields images in mini-batches (a batch size of 64).
            correct += (predicted == labels).sum().item()

    print(f"Final Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    main()
