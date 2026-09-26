"""
12. Transfer Learning & Fine-Tuning with PyTorch

This script demonstrates how to leverage large-scale pre-trained models for custom 
downstream tasks. It introduces feature extraction (freezing pre-trained convolutional 
backbone layers and training only a newly replaced classification head) and 
fine-tuning (updating entire networks with differential learning rates).

Learning Objectives:
1. Load state-of-the-art vision models with pre-trained ImageNet weights.
2. Freeze parameters by setting requires_grad = False on selected layers.
3. Replace classification heads (e.g., model.fc) to match custom class sizes.
4. Pass selected layer parameters to optimizers for efficient weight updates.
"""

# We import the core torch library.
import torch

# nn contains container and linear mapping classes.
import torch.nn as nn

# optim contains optimization engines.
import torch.optim as optim

# torchvision.models contains state-of-the-art vision models.
# Documentation: https://torchvision.org/stable/models.html
import torchvision.models as models

# Import Weights indices to download pre-trained checkpoints
from torchvision.models import ResNet18_Weights

def main():
    print("--- 1. Loading Pre-trained ResNet-18 Backbone ---")
    
    # We load a pre-trained ResNet-18 model using modern weight APIs.
    # DEFAULT maps to the best available model weights trained on ImageNet.
    # Documentation: https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    
    # Let's inspect the original fully connected classification layer (fc)
    # ResNet-18 maps its final 512 average-pooled spatial features to 1000 ImageNet classes.
    print(f"Original classifier: {model.fc}")

    # ==========================================
    # 2. FEATURE EXTRACTION: FREEZING PARAMETERS
    # ==========================================
    print("\n--- 2. Freezing Feature Extraction Layers ---")
    
    # We iterate over all model parameters and set requires_grad = False.
    # This prevents PyTorch Autograd from tracking gradients and performing updates 
    # on these layers during backpropagation, saving memory and training speed.
    for param in model.parameters():
        param.requires_grad = False

    # ==========================================
    # 3. REPLACING THE CLASSIFICATION HEAD
    # ==========================================
    print("--- 3. Replacing Classifier Head for Custom Task ---")
    
    # We retrieve the number of input features entering the final layer
    in_features = model.fc.in_features
    
    # Suppose we want to classify only 2 custom categories (e.g., "cats" vs. "dogs").
    # We swap out the old fc layer with a new nn.Linear layer.
    # Newly created layers have requires_grad = True by default!
    model.fc = nn.Linear(in_features, 2)
    print(f"Replaced classifier: {model.fc}")

    # ==========================================
    # 4. OPTIMIZER SPECIFICATION
    # ==========================================
    print("\n--- 4. Configuring Parameter-Specific Optimizer ---")
    
    # For feature extraction, we must only pass parameters of the new fc layer 
    # to the optimizer. Passing frozen parameters will cause errors or waste compute.
    # Documentation: https://pytorch.org/docs/stable/optim.html#per-parameter-options
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
    
    # Let's check and audit trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable Parameters: {trainable_params:,} (Should be: 512 * 2 + 2 = 1,026)")
    print(f"Frozen Parameters: {frozen_params:,}")

    # ==========================================
    # 5. SHAPE FLOW VERIFICATION
    # ==========================================
    print("\n--- 5. Shape Flow Check with Synthetic Image Batch ---")
    # ResNet-18 expects input shape: [Batch, 3, 224, 224]
    dummy_images = torch.randn(2, 3, 224, 224)
    
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_images)
        print(f"Output predictions shape: {outputs.shape} (Expected: [2, 2])")
        
    assert outputs.shape == (2, 2), "Transfer learning output shape mismatch."
    print("Transfer learning setup verified successfully!")

if __name__ == "__main__":
    main()
