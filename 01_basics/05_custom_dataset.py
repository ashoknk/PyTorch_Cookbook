"""
05. Custom Datasets and DataLoaders in PyTorch

This script demonstrates how to construct data pipelines 
for custom file structures. We create a mock disk-based dataset (a CSV index 
and a directory of images), implement a subclass of PyTorch's Dataset, chain 
visual transforms, and load the custom dataset in shuffled batches.

Learning Objectives:
1. Inherit from and implement the three standard methods of torch.utils.data.Dataset.
2. Build custom image loading logic using PIL (Pillow).
3. Compose a pipeline for data augmentations (cropping, flipping, normalizing) using torchvision transforms.
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DUMMY_IMAGE_DIR = os.path.join(DATA_DIR, "dummy_images")
DUMMY_ANNOTATIONS_CSV = os.path.join(DATA_DIR, "dummy_annotations.csv")
NUMBER_OF_IMAGES = 16 
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

    #  Tells PyTorch how many samples are in the dataset. 
    # With shuffle=True, the DataLoader uses that count to determine which samples to shuffle.
    def __len__(self):
        # Must return the total number of items in our dataset
        return len(self.annotations)

    # Defines how to retrieve one sample by index. The DataLoader calls it for each sample it needs, 
    # then groups the returned images and labels into batches.
    def __getitem__(self, idx):
        # 1.1 Extract image path from annotation dataframe
        # iloc[row_index, column_index] stands for integer-location based indexing.
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
    os.makedirs(DUMMY_IMAGE_DIR, exist_ok=True)
    
    # Write 16 random RGB JPEG images to disk
    
    for i in range(NUMBER_OF_IMAGES):
        # We construct a 64x64 dummy image using PyTorch tensors and convert it to PIL
        # numbers 0 and 255 define the minimum and maximum range for generating random integer pixel values
        # 64 (1st dimension): Image Height (64 pixels tall).
        # 64 (2nd dimension): Image Width (64 pixels wide).
        # 3 (3rd dimension): 3 Color Channels (Red, Green, Blue).
        # Standard image formats (PNG, JPEG) store pixel intensities as uint8 values per color channel
        rand_tensor = torch.randint(0, 255, (64, 64, 3), dtype=torch.uint8)
        img = Image.fromarray(rand_tensor.numpy(), "RGB")
        img.save(os.path.join(DUMMY_IMAGE_DIR, f"img_{i}.jpg"))
        
    
    # Generates 16 random binary integers (0 or 1) as a 1D tensor
    binary_tensor = torch.randint(low=0, high=2, size=(NUMBER_OF_IMAGES,))

    print(binary_tensor)
    # Example output: tensor([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1])
  
    # Write an annotations CSV indexing these images with dummy class labels
    df = pd.DataFrame({
        "image_file": [f"img_{i}.jpg" for i in range(NUMBER_OF_IMAGES)],
        "label": binary_tensor
    })
    df.to_csv(DUMMY_ANNOTATIONS_CSV, index=False)
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
    # We use mean=(0.5,), std=(0.5,) for gray scale images.
    # But for colored images we use mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] values. 
    # Researchers calculated these exact values by analyzing all millions of images 
    # in the massive ImageNet dataset across three color channels (Red, Green, Blue)
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
        csv_file=DUMMY_ANNOTATIONS_CSV,
        img_dir=DUMMY_IMAGE_DIR,
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
        # ex. 2 images in this batch, 3 color channels per image , 32 × 32 pixels per image
        print(f"  Images batch shape: {images.shape} (Expected: [2, 3, 32, 32])")
        print(f"  Labels batch shape: {labels.shape} (Labels: {labels.tolist()})")

    # Clean up dummy assets from workspace
    print("\n--- Cleaning up dummy files ---")
    for i in range(NUMBER_OF_IMAGES):
        os.remove(os.path.join(DUMMY_IMAGE_DIR, f"img_{i}.jpg"))
    os.rmdir(DUMMY_IMAGE_DIR)
    os.remove(DUMMY_ANNOTATIONS_CSV)
    print("Dummy files removed.")

if __name__ == "__main__":
    main()


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