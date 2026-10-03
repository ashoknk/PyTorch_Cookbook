"""
08 - 2b Multi-Channel (RGB) CNN Classification with CIFAR-10 Dataset
Processing 3-channel RGB inputs, how convolution filters handle depth, and 
proper multi-channel normalization

02_intermediate/01_convolutional_neural_network.py uses FashionMNIST which has 1-channel (grayscale) images.
02_intermediate/06_transfer_learning.py uses pre-trained ResNet-18, which expects 3-channel (RGB) images.
Transitioning straight from 1-channel grayscale to pre-trained RGB models to understand how to build a CNN that accepts color images.

CIFAR-10 features color photos of real-world objects (airplanes, automobiles, cats, dogs) 
32X 32 resolution that learners can easily identify and visualize.
The network uses a hierarchical architecture to extract fine details step by step:
Raw RGB (3x32x32) ──> [Block 1] ──> [Block 2] ──> [Block 3] ──> [Classifier Head] ──> 10 Class Scores
                      (32x16x16)    (64x8x8)     (128x4x4)      (Linear -> Dropout -> Linear)

Why Use Three Blocks with Increasing Channels (32 -> 64 -> 128)?
1. Block 1 (3 -> 32 channels):
   Scans the raw 32x32 RGB image to detect basic visual features like 
   simple color gradients, vertical edges, and horizontal lines.
2. Block 2 (32 -> 64 channels):
   Combines basic lines into intermediate shapes (e.g., wheels, circles, corners) 
   on a smaller 16x16 grid.
3. Block 3 (64 -> 128 channels):
   Combines shapes into full object components (e.g., dog faces, car windshields, 
   bird wings) on a compact 8x8 grid.

Learning Objectives:
1. Master processing of 3-channel color (RGB) images in PyTorch.
2. Understand how convolution filters slide across depth dimensions.
3. Apply channel-wise normalization with independent mean and standard deviation values.
4. Scale up the network complexity to handle more complex, multi-color classes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ==========================================
# 1. DEFINE COLOR CNN ARCHITECTURE
# ==========================================

"""
For Conv2d's used as 1st Layer Channel Input Rules:
    1. Grayscale Images (e.g., FashionMNIST, MNIST):
        Have 1 color channel (intensity values from black to white) -> in_channels = 1.
    2. Standard Colored Images (e.g., CIFAR-10, ImageNet, JPEGs/PNGs):
        Have 3 color channels (Red, Green, Blue) -> in_channels = 3.

Convolution Over 3 Channels (Red, Green, Blue):
    When processing a grayscale image, a single filter of shape 
        (3x3) slides over a single channel.
    When processing a color image (RGB), a single filter actually has three dimensions:
        (3x3x3) -> (height x width x depth).
        kernel_size=3 sets the Height and Width (3 X 3 pixels).
        in_channels=3 sets the Depth (matching the 3 RGB color channels).    
    
How it works:
    - The filter slides over the image. At each location, it computes the dot product of 
        its Red kernel with the Red channel, its Green kernel / the Green channel, and its Blue kernel / the Blue channel.
    -  out_channels=32 configures 32 filters. Each filter applies weights to all three input channels RGB, 
    sums those results, and adds a bias at each output position.
    - Therefore, the number of input channels (in_channels) of your Conv2d layer must 
        exactly match the depth of your input image (3 for RGB).
"""

class ColorCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ColorCNN, self).__init__()
        
        # Block 1: Input size (3, 32, 32) -> Output size (32, 16, 16) after pooling
        # Each of the 32 filters in Block 1 is an actual 3D block of numbers with shape 3 X 3 X 3
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            # BatchNorm2d keeps each feature map’s values on a more manageable scale. 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # With a 2x2 kernel and stride 2, each pooling window covers a separate 2x2 patch, 
            # then moves two pixels to the next patch. MaxPool2d outputs the largest value in each patch.
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2: Input size (32, 16, 16) -> Output size (64, 8, 8) after pooling
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 3: Input size (64, 8, 8) -> Output size (128, 4, 4) after pooling
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fully Connected Classification Head:
        # Our final spatial resolution is 4x4 with 128 feature channels.
        # Total flattened features = 128 * 4 * 4 = 2048
        self.classifier = nn.Sequential(
            # The 256 is a chosen hidden-layer reasonable size: this Linear maps those 2048 values to 256 neurons
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            # CIFAR-10 has 50,000 training images, while this CNN has about 620,000 parameters - model has enough capacity to memorize some training examples.
            # Sets 30% of the hidden layer’s activations to zero during each training pass. A different set is dropped each time.
            nn.Dropout(p=0.3), # Dropout helps reduce overfitting by random neuron silencing
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Flatten the feature maps to a 1D tensor for classification
        # x.size(0) extracts the current batch size. Shape (batch_size, 2048)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out

def main():
    # Setup device accelerator (CUDA, MPS, or CPU)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Running Color CNN on: {device} ---")

    # ==========================================
    # 2. COLOR DATASET PREPARATION (CIFAR-10)
    # ==========================================
    print("--- Preparing CIFAR-10 Dataset ---")
    
    # Unlike FashionMNIST which needs only 1 mean and standard deviation value,
    # RGB datasets need 3 separate mean/std values—one for each color channel.
    # The figures below are the calculated Channel-wise Means and STDs of the CIFAR-10 dataset:
    # Channel order: [Red, Green, Blue] ==>[0.4914, 0.4822, 0.4465]. Average brightness of each channel
    # These exact numbers were calculated by analyzing all 60,000 images in the CIFAR-10 dataset:
    # NOTE In other code we used ImageNet stats [0.485, 0.456, 0.406] 
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2470, 0.2435, 0.2616)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])

    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    # CIFAR-10 Category Labels
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print(f"Dataset Loaded: 10 distinct classes from '{classes[0]}' to '{classes[-1]}'")

    # ==========================================
    # 3. INITIALIZE MODEL & LOSS FUNCTION
    # ==========================================
    model = ColorCNN(num_classes=10).to(device)
    # CrossEntropyLoss directly trains the model to assign a higher score to the correct class.
    # It applies the needed log-softmax internally, so pass the raw logits as this code does.
    #NOTE : MSELoss is mainly for regression. BCELoss is useful for binary or multi-label tasks
    criterion = nn.CrossEntropyLoss()
    # The optimizer is chosen based on training behavior, accuracy, and tuning needs. 
    # Adam is a convenient default for training the CNN on CIFAR-10
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ==========================================
    # 4. TRAINING MODULE
    # ==========================================
    epochs = 1
    print(f"--- Training Color CNN for {epochs} epoch ---")
    model.train()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward() # Calculates the gradients for every weight and bias based on mistake/loss
        optimizer.step() # Updates the weights & the biases for the Next Iteration

        if (batch_idx + 1) % 100 == 0:
            print(f"  Step [{batch_idx+1}/{len(train_loader)}], Classification Loss: {loss.item():.4f}")

    # ==========================================
    # 5. TESTING AND ACCURACY ASSESSMENT
    # ==========================================
    print("\n--- Evaluating CIFAR-10 Test Dataset ---")
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
