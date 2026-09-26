"""
Purpose:
    This script sets up a PyTorch image preprocessing and data augmentation pipeline 
    using torchvision.transforms. It prepares raw images for model training by ensuring 
    uniform dimensions, enhancing dataset diversity, and standardizing pixel values.

Data Processing Flow:
    1. Resize: Rescales raw images to a fixed dimension (256x256).
    2. RandomCrop: Crops a sub-region (224x224) to make the model robust to scale/position.
    3. RandomHorizontalFlip: Randomly mirrors images (p=0.5) to augment training data.
    4. ToTensor: Converts PIL images (0 to 255) to PyTorch FloatTensors scaled to [0.0, 1.0].
    5. Normalize: Standardizes tensor channels using ImageNet mean and standard deviation.


    Why Normalize After ToTensor()?

    1. ToTensor() converts raw image numbers (0 to 255) into decimals (0.0 to 1.0).
    While this turns pixels into floats, all numbers are still positive, with an 
    average around 0.5.

    2. Normalize() shifts those decimals so they are centered around 0.0 with a standard spread.
    It uses the formula: (pixel - mean) / std, moving values into a roughly -2.0 to +2.0 range.

    Why is this important for beginners to know?
    - Faster Learning: Neural networks learn best when input numbers are centered around zero. 
    It prevents the model's training path from taking slow "zig-zag" steps.
    - Pretrained Models: Models trained on ImageNet expect inputs in this exact normalized format. 
    Skipping this step causes the model to get confused by unexpected input scales.
    - Stable Gradients: Keeps numbers at a balanced scale so the network doesn't run into training 
    errors like vanishing or exploding gradients.
    
"""

import torch
from torchvision import transforms
from PIL import Image
import os


# 1. Define the transform pipeline pattern
# transforms.Compose chains multiple data preprocessing steps together into one sequence.
# Common sequence: Resize ──> Augment ──> Convert to Tensor ──> Normalize
# transforms.Compose(...) builds a single callable object.It behaves like a function when called, but it is not a function definition
# It chains multiple image transforms in order.It is a composed transform pipeline object

train_transforms = transforms.Compose([
    # Resize the image so every image in the batch is identical in size
    transforms.Resize((256, 256)),
    
    # Crop randomly to teach the model to recognize objects from different angles/crops
    transforms.RandomCrop(224),
    
    # Flips the image horizontally with a 50% chance (Data Augmentation)
    transforms.RandomHorizontalFlip(p=0.5),
    
    # CRITICAL: Converts PIL Image (0-255) to a PyTorch FloatTensor (0.0-1.0)
    transforms.ToTensor(),
    
    # Normalizes pixel values using standard ImageNet mean and standard deviations
    # Documentation: https://docs.pytorch.org/vision/stable/transforms.html
    # We use transforms.Normalize(mean=(0.5,), std=(0.5,)) for gray scale images like MNIST
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

"""
How Were These Numbers Calculated?

    Researchers calculated these exact values by analyzing all 14 million images 
    in the massive ImageNet dataset across three color channels (Red, Green, Blue):

    1. Pixel Average (Mean):
    They averaged the brightness of every pixel in the dataset for each color channel:
    - Red channel mean: 0.485 (about 48.5% brightness)
    - Green channel mean: 0.456 (about 45.6% brightness)
    - Blue channel mean: 0.406 (about 40.6% brightness)

    2. Spread/Variation (Standard Deviation):
    They measured how much the pixel values vary around those average brightness levels:
    - Red channel std: 0.229
    - Green channel std: 0.224
    - Blue channel std: 0.225

    Why Do We Use Them?
    Because popular pretrained models (like ResNet) were originally trained on ImageNet. 
    Using these exact numbers ensures your image data matches what the model expects to see!
"""

def main():
    # Create a dummy RGB image (256x256 pixels) for demonstration purposes
    dummy_image = Image.new("RGB", (256, 256), color=(255, 0, 0))
    
    # Ensure the 'data' directory exists
    os.makedirs("data", exist_ok=True)

    # Create and save the image
    dummy_image = Image.new("RGB", (256, 256), color=(255, 0, 0))
    dummy_image.save("data/dummy_image.png")

    # Apply the transform pipeline
    processed_tensor = train_transforms(dummy_image)
    
    # Verify the transformed tensor properties
    print(f"Processed Tensor Shape: {processed_tensor.shape}")  # Expected: torch.Size([3, 224, 224])
    print(f"Processed Tensor Data Type: {processed_tensor.dtype}")

if __name__ == "__main__":
    main()