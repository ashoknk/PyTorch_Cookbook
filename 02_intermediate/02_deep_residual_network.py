"""
08. Deep Residual Network (ResNet) from Scratch with PyTorch

This script demonstrates how to construct and train a Deep Residual Network (ResNet) 
from scratch. ResNets introduce skip connections (identity pathways) that bypass 
intermediate layers, enabling gradients to flow unimpeded. This resolves the 
vanishing/exploding gradient problem, allowing for extremely deep architectures.

Learning Objectives:
1. Implement a ResidualBlock featuring shortcut pathways and dimension matching.
2. Build a full ResNet model by stacking custom ResidualBlocks.
3. Use global average pooling to map spatial features to output dimensions.
4. Verify mathematical structures and shape flow using synthetic tensor checks.
"""

# We import the core torch library.
import torch

# nn contains foundational neural network layers (Conv2d, BatchNorm2d, Linear, Sequential).
# Documentation: https://pytorch.org/docs/stable/nn.html
import torch.nn as nn

# ==========================================
# 1. DEFINE INDIVIDUAL RESIDUAL BLOCK
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        Args:
            in_channels (int): Input channel dimension.
            out_channels (int): Output channel dimension.
            stride (int): Sride factor to match downsampling.
            downsample (nn.Module, optional): Pathway mapping matching sizes.
        """
        super(ResidualBlock, self).__init__()
        # Conv Layer 1: preserves or halves resolution depending on stride.
        # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        # Conv Layer 2: standard 3x3 convolution with stride=1
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Downsample represents a 1x1 convolution pathway needed to scale input shapes 
        # to match output shapes if stride > 1 or channel depth changes.
        self.downsample = downsample

    def forward(self, x):
        # Store original input for shortcut connection
        residual = x
        
        # Pass input through layer 1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Pass through layer 2
        out = self.conv2(out)
        out = self.bn2(out)
        
        # If input and output dimensions do not match, we map the residual using downsample layer
        if self.downsample is not None:
            residual = self.downsample(x)
            
        # Add original input directly to computed activations (skip connection)
        out += residual
        out = self.relu(out)
        return out

# ==========================================
# 2. DEFINE SYSTEM-WIDE RESNET MODEL
# ==========================================
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        """
        Args:
            block (class): ResidualBlock class blueprint.
            layers (list): Number of blocks per layer (e.g., [2, 2, 2]).
            num_classes (int): Category output counts.
        """
        super(ResNet, self).__init__()
        self.in_channels = 16
        
        # Stem Layer: projects 3-channel RGB images to 16 features
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        
        # ResNet stages: stack pairs of ResidualBlocks. Each stage doubles channel depths
        # and halves spatial resolutions.
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        
        # Global Average Pooling averages all activations across spatial grids, mapping (batch, 64, 8, 8) -> (batch, 64, 1, 1).
        # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Linear layer mapping features to category counts
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """Helper to chain sequential ResidualBlocks together and generate matching downsample pathways."""
        downsample = None
        # If strides are not 1, or input features don't match output features, we build a downsample pathway
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        layers = []
        # First block uses downsample and specified stride
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        
        # Subsequent blocks preserve size (stride=1, no downsample)
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        # Entry Stem
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        
        # Stacked blocks
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        # Pooling and projection
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def main():
    print("--- Instantiating Custom ResNet (ResNet-9 Variant) ---")
    # Setup custom model with 2 blocks per stage: Total blocks = 2 + 2 + 2 = 6 blocks (12 conv layers)
    model = ResNet(ResidualBlock, [2, 2, 2])
    print(model)

    # ==========================================
    # 3. VERIFY WITH SYNTHETIC FLOW CHECK
    # ==========================================
    print("\n--- Running Dimensional Check with Synthetic Tensor ---")
    # Feed an artificial batch of 4 RGB images of size 32x32
    synthetic_batch = torch.randn(4, 3, 32, 32)
    print(f"Input Shape: {synthetic_batch.shape}")
    
    # Get forward pass prediction
    output = model(synthetic_batch)
    print(f"Output Shape: {output.shape} (Expected: [4, 10])")
    
    assert output.shape == (4, 10), "Dimensional mismatch in ResNet forward pass."
    print("ResNet structural compilation verified successfully!")

if __name__ == "__main__":
    main()
