"""
21 -3a. Distributed Data Parallel (DDP) and Mixed Precision (AMP) in PyTorch

To demonstrate two production-grade training optimization techniques:
     1. Automatic Mixed Precision (AMP): Speeds up matrix math and cuts VRAM usage by 
        dynamically running compatible operations in 16-bit floating point (FP16/BF16)[cite: 3].
     2. Distributed Data Parallel (DDP): Scales training across multiple GPUs/servers[cite: 3].

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

import torch
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

# def train_worker(rank, world_size):
#     print(f"--- Worker Rank {rank}/{world_size} starting up ---")
#     setup_distributed(rank, world_size)
    
#     # Establish device context for current process rank
#     device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
#     print(f"--- Worker Rank {rank} using device: {device} ---")
    
#     # Create simple linear network and wrap in DDP.
#     # DistributedDataParallel replicates the model and synchronizes gradients across workers.
#     # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
#     model = nn.Linear(10, 2).to(device)
#     if torch.cuda.is_available():
#         model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
#     else:
#         # Fallback DDP wrapper for CPU processing
#         model = nn.parallel.DistributedDataParallel(model)

#     # Initialize standard optimizer and MSE loss
#     criterion = nn.MSELoss()
#     optimizer = optim.SGD(model.parameters(), lr=0.01)

#     # ==========================================
#     # 1. SETUP DISTRIBUTED SAMPLER & DATALOADER
#     # ==========================================
#     # Generate synthetic training tensors
#     X = torch.randn(100, 10)
#     y = torch.randn(100, 2)
#     dataset = TensorDataset(X, y)
    
#     # The DistributedSampler partitions the dataset so each rank receives non-overlapping slices.
#     # Documentation: https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
#     sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
#     dataloader = DataLoader(dataset, batch_size=10, sampler=sampler)

#     # ==========================================
#     # 2. INITIALIZE AMP GRADSCALER
#     # ==========================================
#     # GradScaler prevents floating-point underflow (where small gradients are rounded to zero in FP16).
#     # Documentation: https://pytorch.org/docs/stable/amp.html#gradient-scaling
#     # Note: On CPU, GradScaler is not required for AMP (bfloat16 doesn't suffer from underflow),
#     # but we initialize and demonstrate it here for general multi-GPU usage.
#     scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu', enabled=torch.cuda.is_available())
#     print(f"--- GradScaler using device: {scaler} ---")

#     # ==========================================
#     # 3. HIGH-PERFORMANCE TRAINING LOOP
#     # ==========================================
#     epochs = 3
#     for epoch in range(epochs):
#         # We must call set_epoch on the sampler at the beginning of each epoch 
#         # to ensure correct data shuffling across workers.
#         sampler.set_epoch(epoch)
        
#         for inputs, targets in dataloader:
#             inputs, targets = inputs.to(device), targets.to(device)
            
#             # Forward pass under the 'autocast' context manager.
#             # PyTorch automatically selects FP16/BF16 precision for compatible operations.
#             # Documentation: https://pytorch.org/docs/stable/amp.html#autocasting
#             device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
#             with autocast(device_type=device_type):
#                 outputs = model(inputs)
#                 loss = criterion(outputs, targets)
                
#             optimizer.zero_grad()
            
#             # Backpropagate using scaled gradients to prevent underflow
#             scaler.scale(loss).backward()
            
#             # Step the optimizer through the scaler
#             scaler.step(optimizer)
            
#             # Update scaler expansion factors
#             scaler.update()

#         # Only print progress on Rank 0 to prevent console spam
#         if rank == 0:
#             print(f"  Epoch [{epoch+1}/{epochs}] | Worker 0 Loss: {loss.item():.4f}")

#     cleanup_distributed()
#     print(f"--- Worker Rank {rank} completed training ---")

import time  # Add import at top of file

def train_worker(rank, world_size):
    print(f"--- Worker Rank {rank}/{world_size} starting up ---")
    setup_distributed(rank, world_size)
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print(f"--- Worker Rank {rank} using device: {device} ---")
    
    # -------------------------------------------------------------
    # CHANGE 1: Increase Model Size to visualize GPU/AMP prowess
    # -------------------------------------------------------------
    base_model = nn.Sequential(
        nn.Linear(512, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 128)
    ).to(device)

    if torch.cuda.is_available():
        model = nn.parallel.DistributedDataParallel(base_model, device_ids=[rank])
    else:
        model = nn.parallel.DistributedDataParallel(base_model)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # -------------------------------------------------------------
    # CHANGE 1 (cont.): Increase Dataset Size
    # -------------------------------------------------------------
    X = torch.randn(10000, 512)
    y = torch.randn(10000, 128)
    dataset = TensorDataset(X, y)
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu', enabled=torch.cuda.is_available())

    # -------------------------------------------------------------
    # CHANGE 3: Reset Peak VRAM Stats & Start Timer
    # -------------------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    start_time = time.time()

    epochs = 3
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            with autocast(device_type=device_type):
                outputs = model(inputs)
                
                # -------------------------------------------------------------
                # CHANGE 2: Precision Inspection (Check Dtype)
                # -------------------------------------------------------------
                if epoch == 0 and batch_idx == 0 and rank == 0:
                    print(f"  [AMP PROOF] Output Tensor Dtype under autocast: {outputs.dtype}")
                    
                loss = criterion(outputs, targets)
                
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    # -------------------------------------------------------------
    # CHANGE 4: Compute Final Execution Time & Peak VRAM
    # -------------------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_vram_mb = 0.0

    elapsed_time = time.time() - start_time

    if rank == 0:
        print("\n" + "="*50)
        print("          AMP & GPU EXECUTION REPORT          ")
        print("="*50)
        print(f"  Total Execution Time : {elapsed_time:.4f} seconds")
        print(f"  Peak VRAM Allocation : {peak_vram_mb:.2f} MB")
        print("="*50 + "\n")

    cleanup_distributed()

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
