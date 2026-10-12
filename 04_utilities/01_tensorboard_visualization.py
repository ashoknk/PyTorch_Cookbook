"""
21. TensorBoard Integration and Model Monitoring in PyTorch

This script demonstrates how to integrate PyTorch's native TensorBoard 
SummaryWriter to monitor and visualize training runs in real-time. It covers 
logging scalar metrics (loss, accuracy), visualizing model graphs, tracking parameter 
weight distributions (histograms), and uploading sample visual image grids.

Learning Objectives:
1. Initialize SummaryWriter pointing to local logging directories.
2. Serialize model computational graph structures using writer.add_graph().
3. Stream training scalar metrics over time using writer.add_scalar().
4. Monitor parameter weights and gradient distributions using writer.add_histogram().
5. Upload visual sample grids using torchvision utilities and writer.add_image().
"""

import shutil

import torch
import torch.nn as nn

# SummaryWriter is the primary class for streaming logs to TensorBoard.
# Documentation: https://pytorch.org/docs/stable/tensorboard.html
from torch.utils.tensorboard import SummaryWriter

# torchvision contains pre-downloaded FashionMNIST dataset and visual helpers.
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Simple toy classifier for logging demonstration
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(8 * 14 * 14, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))
        return self.fc(x.view(x.size(0), -1))

def main():
    print("--- 1. Initializing TensorBoard SummaryWriter ---")
    # Initialize the SummaryWriter. By default, logs are written to './runs/' directory.
    # Documentation: https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter
    writer = SummaryWriter(log_dir="./runs/fashion_mnist_experiment")

    # ==========================================
    # 2. LOG IMAGE GRIDS AND MODEL GRAPHS
    # ==========================================
    print("--- 2. Visualizing Model Graphs & Sample Grids ---")
    
    # Load pre-downloaded FashionMNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    # Fetch a single batch of images and labels
    images, labels = next(iter(dataloader))

    # Create a grid of images using torchvision and upload to TensorBoard
    # Documentation: https://pytorch.org/vision/stable/generated/torchvision.utils.make_grid.html
    img_grid = torchvision.utils.make_grid(images)
    writer.add_image("sample_fashion_mnist_images", img_grid)
    print("  Uploaded sample image grid to TensorBoard.")

    # Initialize model and log its computational graph structure
    model = SimpleCNN()
    dummy_input = torch.randn(1, 1, 28, 28)
    writer.add_graph(model, dummy_input)
    print("  Serialized model computational graph to TensorBoard.")

    # ==========================================
    # 3. TRAINING LOOP WITH SCALAR AND HISTOGRAM LOGGING
    # ==========================================
    print("\n--- 3. Running Dummy Training Loop with Real-time Logging ---")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    epochs = 3
    global_step = 0
    
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (imgs, lbls) in enumerate(dataloader):
            # For rapid testing, we limit to 30 batches per epoch
            if batch_idx >= 30:
                break
                
            global_step += 1
            
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Log individual step loss to TensorBoard
            # The forward slash (/) tells TensorBoard to group both charts under a section called Loss
            writer.add_scalar("Loss/train_step", loss.item(), global_step)
            
        # Log aggregated epoch loss
        epoch_loss = running_loss / 30
        writer.add_scalar("Loss/train_epoch", epoch_loss, epoch)
        print(f"  Epoch [{epoch+1}/{epochs}] complete. Loss logged: {epoch_loss:.4f}")

        # Log parameter weights and biases distributions as Histograms
        # This helps monitor for exploding or vanishing weights over time.
        # Distributions of layer weights, biases, and backprop gradients over time can be visualized in TensorBoard's Histograms tab.
        for name, param in model.named_parameters():
            writer.add_histogram(f"Parameters/{name}", param, epoch)
            if param.grad is not None:
                writer.add_histogram(f"Gradients/{name}", param.grad, epoch)

    # Always close the writer when finished to ensure all buffers are written to disk
    writer.close()
    print("SummaryWriter flushed and closed.")

    # To launch the interactive dashboard and view your graphs, open a separate terminal
    # inside your project directory and run:
    #     tensorboard --logdir=runs
    # Then, open your web browser and navigate to:
    #     http://localhost:6006
    # Inside the TensorBoard web UI, you will see interactive tabs for Scalars,
    # Images, Graphs, and Histograms.
    #
    # ==========================================
    # PART 1: THE RECOMMENDED VISUALIZATION FLOW
    # ==========================================
    #
    # 1. Images tab: sample image grids confirm data loading and normalization.
    # 2. Graphs tab: add_graph() reveals the model architecture and tensor flow.
    # 3. Scalars tab: loss curves show whether training is improving.
    # 4. Histograms tab: weight and gradient distributions expose learning issues.

    # Clean up runs directory
    # shutil.rmtree("./runs")
    # print("\n--- Cleaned up './runs' logging directory ---")


if __name__ == "__main__":
    main()


"""===

1. IMAGES TAB
   - Core Function: writer.add_image("sample_images", img_grid)
   - What It Shows: Visual grid preview of input dataset batches (e.g., FashionMNIST items).
   - Why We Need It: Data Pipeline Verification. Confirms that images are loading, 
     normalizing, and reshaping correctly before feeding them to the neural network.

2. GRAPHS TAB
   - What It Shows: Interactive architectural flowchart of the model's layers (Conv2d, ReLU, Linear).
   - Core Function: writer.add_graph(model, dummy_input)Serializes PyTorch's computational execution graph.
   - Why We Need It: Structural Inspection. Verifies tensor shape transformations between 
     layers and confirms the network topology matches your intended design.
     Debugs dimension mismatches, layer connectivity issues, and tensor 
     broadcasting errors during model design.

3. SCALARS TAB
   - Core Function: writer.add_scalar("Loss/train", loss.item(), global_step)
   Tracks floating-point numbers over step intervals.
   - What It Shows: 2D line plots tracking scalar metrics (like training/validation loss over steps/epochs).
   - Why We Need It: Performance Tracking. Directly answers whether the model is learning 
     by displaying downward-trending loss curves over time.
    Detects underfitting, overfitting, or improper learning rate settings 
     by monitoring optimization convergence.


4. HISTOGRAMS TAB (Parameter Weights & Gradients)
   - Core Function: writer.add_histogram("Parameters/...", param, epoch)
   - What It Shows: 3D density curves of layer weights, biases, and backprop gradients across epochs.
   - Why We Need It: Health Diagnostic. Detects exploding/vanishing gradients and verifies 
     that layer parameters are actively adjusting rather than remaining frozen or dead.
    
"""