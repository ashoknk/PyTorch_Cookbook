"""
Purpose:
    This script downloads and parses the FashionMNIST validation dataset to find 
    and display exactly one example image for each of the 10 fashion categories.

Data Processing Flow & Visualization:
    1. Define Class Labels: Maps class indices (0 to 9) to readable names (e.g., T-shirt, Ankle boot).
    2. Load Dataset: Fetches the test set and converts raw images to PyTorch tensors via transforms.
    3. Batch Extraction: Pulls a large batch (1,000 images) using DataLoader to ensure all 10 classes exist.
    4. Unique Filtering: Iterates through the batch, keeping the first occurrence of each category label.
    5. Grid Plotting: Displays the 10 unique grayscale images in a 2x5 Matplotlib subplot layout.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

# 1. Define the 10 official category text labels in order (indices 0-9)
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

def main():
    # 2. Download and load the FashionMNIST validation dataset
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False)

    # 3. Find exactly one image for each unique category index
    images, labels = next(iter(test_loader))
    unique_images = {}

    for img, label in zip(images, labels):
        label_idx = label.item()
        if label_idx not in unique_images:
            unique_images[label_idx] = img
        if len(unique_images) == 10:  # Stop once we have all 10 classes
            break

    # 4. Plot the 10 categories in a neat 2x5 grid
    # axes.ravel() flattens a multi-dimensional array into a 1D array.This lets you iterate using a single index [axes[0], axes[1]..
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()

    for idx in range(10):
        # Convert PyTorch tensor to numpy array and remove channel dim for grayscale (28x28)
        img_np = unique_images[idx].squeeze().numpy()

        axes[idx].imshow(img_np, cmap="gray")
        axes[idx].set_title(f"[{idx}] {classes[idx]}", fontsize=10, fontweight="bold")
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()