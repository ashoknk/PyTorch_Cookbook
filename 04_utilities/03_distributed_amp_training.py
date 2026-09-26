"""
23. Distributed Data Parallel (DDP) and Mixed Precision (AMP) in PyTorch

This script demonstrates high-performance training scaling in PyTorch. It 
implements Automatic Mixed Precision (AMP) using GradScaler and autocast (which 
speeds up training by dynamically running compatible layers in FP16/BF16) and 
configures Distributed Data Parallel (DDP) for multi-GPU training clusters.

Learning Objectives:
1. Initialize distributed process groups with torch.distributed.
2. Partition dataset loads across ranks using DistributedSampler.
3. Wrap neural modules inside DistributedDataParallel (DDP) containers.
4. Accelerate training and reduce memory using Automatic Mixed Precision (AMP).
"""

import os

# We import the core torch library.
import torch

# nn contains container classes.
import torch.nn as nn

# optim contains parameter optimization algorithms (SGD).
import torch.optim as optim

# autocast and GradScaler are the core engines of Automatic Mixed Precision.
# Documentation: https://pytorch.org/docs/stable/amp.html
from torch.amp import autocast, GradScaler

# DDP and Distributed Data Samplers scale training across multiple GPUs.
# Documentation: https://pytorch.org/docs/stable/distributed.html
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, TensorDataset

def setup_distributed(rank, world_size):
    """Initializes the distributed communications process group."""
    # We specify host and port configurations for inter-process communications
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the default NCCL (Nvidia) or Gloo (CPU/general) communications backend.
    # We use 'gloo' here as a fallback because it is supported on all devices, including CPU.
    # Documentation: https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

def cleanup_distributed():
    """Destroys active distributed communication groups."""
    dist.destroy_process_group()

def train_worker(rank, world_size):
    print(f"--- Worker Rank {rank}/{world_size} starting up ---")
    setup_distributed(rank, world_size)
    
    # Establish device context for current process rank
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    # Create simple linear network and wrap in DDP.
    # DistributedDataParallel replicates the model and synchronizes gradients across workers.
    # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
    model = nn.Linear(10, 2).to(device)
    if torch.cuda.is_available():
        model = nn.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    else:
        # Fallback DDP wrapper for CPU processing
        model = nn.parallel.DistributedDataParallel(model)

    # Initialize standard optimizer and MSE loss
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # ==========================================
    # 1. SETUP DISTRIBUTED SAMPLER & DATALOADER
    # ==========================================
    # Generate synthetic training tensors
    X = torch.randn(100, 10)
    y = torch.randn(100, 2)
    dataset = TensorDataset(X, y)
    
    # The DistributedSampler partitions the dataset so each rank receives non-overlapping slices.
    # Documentation: https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=10, sampler=sampler)

    # ==========================================
    # 2. INITIALIZE AMP GRADSCALER
    # ==========================================
    # GradScaler prevents floating-point underflow (where small gradients are rounded to zero in FP16).
    # Documentation: https://pytorch.org/docs/stable/amp.html#gradient-scaling
    # Note: On CPU, GradScaler is not required for AMP (bfloat16 doesn't suffer from underflow),
    # but we initialize and demonstrate it here for general multi-GPU usage.
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu', enabled=torch.cuda.is_available())

    # ==========================================
    # 3. HIGH-PERFORMANCE TRAINING LOOP
    # ==========================================
    epochs = 3
    for epoch in range(epochs):
        # We must call set_epoch on the sampler at the beginning of each epoch 
        # to ensure correct data shuffling across workers.
        sampler.set_epoch(epoch)
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass under the 'autocast' context manager.
            # PyTorch automatically selects FP16/BF16 precision for compatible operations.
            # Documentation: https://pytorch.org/docs/stable/amp.html#autocasting
            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            with autocast(device_type=device_type):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
            optimizer.zero_grad()
            
            # Backpropagate using scaled gradients to prevent underflow
            scaler.scale(loss).backward()
            
            # Step the optimizer through the scaler
            scaler.step(optimizer)
            
            # Update scaler expansion factors
            scaler.update()

        # Only print progress on Rank 0 to prevent console spam
        if rank == 0:
            print(f"  Epoch [{epoch+1}/{epochs}] | Worker 0 Loss: {loss.item():.4f}")

    cleanup_distributed()
    print(f"--- Worker Rank {rank} completed training ---")

def main():
    # To demonstrate DDP locally, we simulate a 1-process rank cluster on CPU.
    # Setting world_size=1 prevents process groups from waiting and deadlocking.
    world_size = 1
    print(f"--- Initializing 1-process DDP Simulation ---")
    
    # In production, DDP is spawned using torchrun CLI or mp.spawn.
    # Here we invoke worker 0 directly to demonstrate and verify compile loops.
    train_worker(rank=0, world_size=world_size)

if __name__ == "__main__":
    main()
