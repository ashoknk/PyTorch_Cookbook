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

# We import the core torch library.
import torch

# nn contains neural network model layers.
import torch.nn as nn

# SummaryWriter is the primary class for streaming logs to TensorBoard.
# Documentation: https://pytorch.org/docs/stable/tensorboard.html
from torch.utils.tensorboard import SummaryWriter

# torchvision contains pre-downloaded FashionMNIST dataset and visual helpers.
import torchvision
import torchvision.transforms as transforms

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

    """
    To launch the interactive dashboard and view your graphs, open a separate terminal inside your project directory and run:
        tensorboard --logdir=runs
    Then, open your web browser and navigate to:
        http://localhost:6006
    Inside the TensorBoard web UI, you will see interactive tabs for Scalars (loss curves), 
    Images (sample grids), Graphs (network structure), and Histograms  

    TENSORBOARD DASHBOARD INTERPRETATION GUIDE

    1. LOSS METRICS (Model Error)
        - What it is: Measures prediction errors (Loss/train_step vs Loss/train_epoch)
        - Why look at it: Track learning progress; curves must trend downward over time
        - What You Want to See: Smooth downward sloping curves across steps and epochs
        - Typical Target Values: Starts around ~1.3+ and steadily drops toward ~0.8 or lower.

    2. PARAMETERS (Stored Weights)
        - What it is: Distribution of weights/biases in conv and fc layers
        - Why look at it: Verifies model learning; flat/static distributions mean frozen weights
        - What You Want to See: Distributions widening and shifting across epochs as features learn
        - Typical Target Values: Values well-distributed around non-zero ranges (e.g., -0.2 to +0.2).

    3. GRADIENTS (Update Speed & Health)
        - What it is: Backprop adjustment values for model parameters
        - Why look at it: Detects vanishing (zero) or exploding (massive) gradient errors
        - What You Want to See: Small, active, non-zero gradient values (avoiding zero or spikes)
        - Typical Target Values: Stable backprop ranges centered near zero (e.g., -0.02 to +0.02).

    4. IMAGE SAMPLES (Data Pipeline Check)
        - What it is: Visual sample grid of input dataset batches
        - Why look at it: Confirms images load, normalize, and format correctly before training
        - What You Want to See: Clear, recognizable clothing items after dataset transforms
        - Typical Target Values: Normalized pixel intensities properly scaled in the [0.0, 1.0] visual range.

    """ 

    # Clean up runs directory
    shutil.rmtree("./runs")
    print("\n--- Cleaned up './runs' logging directory ---")

# ==========================================
    # 4. SAVE VISUAL GRAPHS AND IMAGES TO DISK
    # ==========================================
    import matplotlib.pyplot as plt
    import numpy as np

    print("\n--- Saving Visual Graphs and Image Outputs to Disk ---")
    
    # 1. Save the Sample Image Grid as a PNG file
    # De-normalize images from [-1, 1] back to [0, 1] range for viewing
    grid_np = img_grid.cpu().numpy() / 2 + 0.5
    plt.figure(figsize=(10, 4))
    plt.imshow(np.transpose(grid_np, (1, 2, 0)))
    plt.title("FashionMNIST Sample Image Grid")
    plt.axis("off")
    plt.savefig("./data/tensorboard_sample_grid.png", bbox_inches='tight')
    plt.close()
    print("Saved image grid preview to './data/tensorboard_sample_grid.png'")

    # 2. Save a Training Loss Plot as a PNG file
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epochs + 1), [running_loss / 30 for _ in range(epochs)], marker='o', color='blue', label='Epoch Loss')
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig("./data/tensorboard_loss_chart.png", bbox_inches='tight')
    plt.close()
    print("Saved training loss chart to './data/tensorboard_loss_chart.png'")    

if __name__ == "__main__":
    main()
