"""

04b. Feedforward Neural Network (MLP) with PyTorch - MNIST Dataset

Purpose:
    This script demonstrates how to load, preprocess, and batch an image dataset (MNIST) 
    using PyTorch and Torchvision. It prepares raw grayscale digit images for a neural 
    network by scaling pixel values to a zero-centered range [-1.0, 1.0].

Datasets & DataLoaders:
    Code for processing data samples can get messy and hard to maintain; we ideally want our 
    dataset code to be decoupled from our model training code for better readability and modularity. 
    PyTorch provides two data primitives: 
        torch.utils.data.DataLoader and torch.utils.data.Dataset 
        that allow you to use pre-loaded datasets as well as your own data. 
    Dataset stores the samples and their corresponding labels, 
    and DataLoader wraps an iterable around the Dataset to enable easy access to the samples.
    https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html

Data Processing Pipeline & Flow:
    1. Define Transformation: Converts PIL raw images into PyTorch Tensors and normalizes values.
    2. Instantiate Dataset: Downloads/loads MNIST digits and attaches the pipeline for on-the-fly execution.
    3. Initialize DataLoader: Wraps the dataset into mini-batches, handling batching and shuffling.
    4. Execution (main): Iterates through the DataLoader to extract a single batch and verifies tensor specs.
"""

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# STEP 1: DEFINE THE TRANSFORM PIPELINE
# ---------------------------------------------------------------------------
# transforms.Compose chains multiple data preprocessing steps together into one sequence.
transform_pipeline = transforms.Compose([
    # transforms.ToTensor():
    # 1. Takes a PIL Image with integer pixels in the range [0, 255].
    # 2. Converts it to a PyTorch FloatTensor with values scaled between [0.0, 1.0].
    # 3. Rearranges dimensions from (Height x Width x Channels) to (Channels x Height x Width).
    transforms.ToTensor(),
    
    # transforms.Normalize(mean, std):
    # Applies the formula: output = (input - mean) / std
    # For single-channel (grayscale) images like MNIST, we pass 1-element tuples: (0.5,), (0.5,)
    # Calculation: (0.0 - 0.5) / 0.5 = -1.0  AND  (1.0 - 0.5) / 0.5 = 1.0
    # This shifts pixel values from [0.0, 1.0] to a zero-centered range [-1.0, 1.0], 
    # which helps neural networks learn faster and more stably.
    transforms.Normalize((0.5,), (0.5,))
])

# ---------------------------------------------------------------------------
# STEP 2: LOAD THE DATASET
# ---------------------------------------------------------------------------
# torchvision.datasets provides standard benchmark datasets.
# Documentation: https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html
train_dataset = torchvision.datasets.MNIST(
    root='./data',         # Directory path where dataset files will be stored
    train=True,            # Download/load the training split (60,000 images)
    download=True,         # Automatically download files from internet if not present locally
    transform=transform_pipeline    # Pass the transform pipeline so it runs on each image when loaded
)

# ---------------------------------------------------------------------------
# STEP 3: CREATE THE DATALOADER
# ---------------------------------------------------------------------------
# DataLoader handles batching, shuffling, and multi-threaded loading for training.
# Documentation: https://docs.pytorch.org/docs/2.14/data.html
train_loader = DataLoader(
    train_dataset,         # The PyTorch Dataset object defined above
    batch_size=64,         # Group images into mini-batches of 64 images each
    shuffle=True           # Shuffle data at the start of every epoch to prevent memory bias
)

# ---------------------------------------------------------------------------
# STEP 4: VERIFY AND EXECUTE
# ---------------------------------------------------------------------------
def main():
    # DataLoader is an iterable object (like a generator or stream), not a indexable list
    # iter(train_loader) creates an iterator over the DataLoader.
    # next(...) fetches the very first batch of data from that iterator
    images, labels = next(iter(train_loader))
    
    # Check the dimensions of the batch tensor
    # Expected shape: [64, 1, 28, 28] -> [batch_size, channels, height, width]
    print(f"Batch images tensor shape : {images.shape}")
    print(f"Batch labels tensor shape : {labels.shape}")
    
    # Check min/max values to confirm normalization shifted range to [-1.0, 1.0]
    print(f"Minimum pixel value in batch: {images.min().item():.2f}")  # Outputs -1.00
    print(f"Maximum pixel value in batch: {images.max().item():.2f}")  # Outputs 1.00

if __name__ == "__main__":
    main()

"""
1. `images.shape` -> `torch.Size([64, 1, 28, 28])`

Think of this tensor as a box containing 64 individual image cards stacked together:

`64` (Batch Size): 
    The number of separate images bundled into this single batch. 
    Your `DataLoader` grabbed 64 images at once to pass to the model.
`1` (Channels): 
    The color depth of each image. 
    Because MNIST consists of black-and-white (grayscale) handwritten digits, each image has 1 channel (shades of gray). 
    Standard color photos (RGB) would have 3 channels (Red, Green, Blue).
`28` (Height): 
    The vertical resolution of the image. The image is 28 pixels tall from top to bottom.
`28` (Width): 
    The horizontal resolution of the image. The image is 28 pixels wide from left to right.


2. `labels.shape` -> `torch.Size([64])`

`64` (Labels): 
    This is a 1D array containing 64 single numbers.
    Each number is the answer key (ground truth) for the corresponding image in the batch.
    If image `#1` shows a handwritten digit "7", the first label in this tensor will be the integer `7`.
"""    