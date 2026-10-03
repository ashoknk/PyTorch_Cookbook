"""
08 -  2e CIFAR-10 Class Image Preview
Goal:
    Show one example image from each of the 10 CIFAR-10 classes with its label.

Code flow:
    1. Load the CIFAR-10 test dataset as RGB image tensors.
    2. Find the first image for each class in a batch of examples.
    3. Display and save the 10 labeled images in a 2x5 grid.
"""

import os

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PNG_FILE = os.path.join(DATA_DIR, "cifar10_examples.png")


def main():
    # ToTensor converts each RGB image to a tensor with shape (channels, height, width).
    transform = transforms.ToTensor()
    test_set = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1000, shuffle=False
    )

    # Keep the first image found for each label in the batch.
    images, labels = next(iter(test_loader))
    class_images = {}
    for image, label in zip(images, labels):
        label_idx = label.item()
        if label_idx not in class_images:
            class_images[label_idx] = image
        if len(class_images) == len(test_set.classes):
            break

    # Plot RGB images in a 2x5 grid. Matplotlib expects channels in the last dimension.
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
     # axes.ravel() flattens a multi-dimensional array into a 1D array.This lets you iterate using a single index [axes[0], axes[1]..
    for class_idx, axis in enumerate(axes.ravel()):
        image = class_images[class_idx].permute(1, 2, 0).numpy()
        axis.imshow(image)
        axis.set_title(test_set.classes[class_idx], fontsize=10, fontweight="bold")
        axis.axis("off")

    plt.tight_layout()
    fig.savefig(PNG_FILE, dpi=300, bbox_inches="tight")
    print(f"Saved CIFAR-10 class examples to: {PNG_FILE}")
    fig.savefig(PNG_FILE, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()