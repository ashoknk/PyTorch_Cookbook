"""
05. Custom Datasets and DataLoaders in PyTorch

This script demonstrates how to construct memory-safe, lazy-loading data pipelines 
for custom file structures. We create a mock disk-based dataset (a CSV index 
and a directory of images), implement a subclass of PyTorch's Dataset, chain 
visual transforms, and load the custom dataset in shuffled batches.

Learning Objectives:
1. Inherit from and implement the three standard methods of torch.utils.data.Dataset.
2. Build custom image loading logic using PIL (Pillow).
3. Chain data augmentations (cropping, flipping, normalizing) using torchvision transforms.
4. Batch and shuffle custom data using torch.utils.data.DataLoader.
"""

import os
import pandas as pd
from PIL import Image

# We import the core torch library.
import torch

# Dataset and DataLoader are the foundational classes for writing data pipelines.
# Documentation: https://pytorch.org/docs/stable/data.html
from torch.utils.data import Dataset, DataLoader

# torchvision.transforms provides image augmentation recipes.
# Documentation: https://pytorch.org/vision/stable/transforms.html
import torchvision.transforms as transforms

# ==========================================
# 1. DEFINE CUSTOM DATASET CLASS
# ==========================================
# All custom datasets in PyTorch must inherit from the Dataset base class.
class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super(CustomDataset, self).__init__()
        # Load the CSV index file containing image filenames and class labels
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        # Must return the total number of items in our dataset
        return len(self.annotations)

    def __getitem__(self, idx):
        # 1.1 Extract image path from annotation dataframe
        img_name = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        
        # 1.2 Read image from disk (using PIL)
        image = Image.open(img_name).convert("RGB")
        
        # 1.3 Extract label index
        label = torch.tensor(int(self.annotations.iloc[idx, 1]), dtype=torch.long)
        
        # 1.4 Apply transformations if specified
        if self.transform:
            image = self.transform(image)
            
        return image, label

def setup_dummy_files():
    """Helper to create dummy CSV index and image files on disk for demonstration."""
    os.makedirs("./dummy_images", exist_ok=True)
    
    # Write 4 random RGB JPEG images to disk
    for i in range(4):
        # We construct a 64x64 dummy image using PyTorch tensors and convert it to PIL
        rand_tensor = torch.randint(0, 255, (64, 64, 3), dtype=torch.uint8)
        img = Image.fromarray(rand_tensor.numpy(), "RGB")
        img.save(f"./dummy_images/img_{i}.jpg")
        
    # Write an annotations CSV indexing these images with dummy class labels
    df = pd.DataFrame({
        "image_file": [f"img_{i}.jpg" for i in range(4)],
        "label": [0, 1, 0, 1]
    })
    df.to_csv("dummy_annotations.csv", index=False)
    print("--- Created dummy image files and dummy_annotations.csv index ---")

def main():
    # Setup dummy directory and csv annotations
    setup_dummy_files()

    # ==========================================
    # 2. DEFINE IMAGE AUGMENTATION PIPELINE
    # ==========================================
    # We use torchvision.transforms to chain multiple operations:
    # 1. Resize images to 32x32.
    # 2. Randomly flip horizontally (data augmentation).
    # 3. Convert PIL to Tensor (maps pixels [0, 255] to [0.0, 1.0]).
    # 4. Standard normalize mean/std.
    # Documentation: https://pytorch.org/vision/stable/generated/torchvision.transforms.Compose.html
    transform_pipeline = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ==========================================
    # 3. INSTANTIATE DATASET & DATALOADER
    # ==========================================
    print("--- Instantiating Custom Dataset and DataLoader ---")
    custom_dataset = CustomDataset(
        csv_file="dummy_annotations.csv",
        img_dir="./dummy_images",
        transform=transform_pipeline
    )

    # DataLoader handles shuffling, batching, and parallel worker processes.
    # We set num_workers=0 here for absolute compatibility across environments (avoiding macOS fork locks).
    # Documentation: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    dataloader = DataLoader(
        dataset=custom_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )

    # ==========================================
    # 4. ITERATE AND VALIDATE BATCHES
    # ==========================================
    print("\n--- Iterating through custom DataLoader ---")
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        # Expected shape: [Batch_Size, Channels, Height, Width]
        print(f"  Images batch shape: {images.shape} (Expected: [2, 3, 32, 32])")
        print(f"  Labels batch shape: {labels.shape} (Labels: {labels.tolist()})")

    # Clean up dummy assets from workspace
    print("\n--- Cleaning up dummy files ---")
    for i in range(4):
        os.remove(f"./dummy_images/img_{i}.jpg")
    os.rmdir("./dummy_images")
    os.remove("dummy_annotations.csv")
    print("Dummy files removed.")

if __name__ == "__main__":
    main()
