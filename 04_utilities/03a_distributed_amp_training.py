"""
21 -3a. Distributed Data Parallel (DDP) and Mixed Precision (AMP) in PyTorch

To demonstrate two production-grade training optimization techniques:
     1. Automatic Mixed Precision (AMP): Speeds up matrix math and cuts VRAM usage by 
        dynamically running compatible operations in 16-bit floating point (FP16/BF16).
     2. Distributed Data Parallel (DDP): Scales training across multiple GPUs/servers.

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
import time  # Add import at top of file

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
    """Initializes the distributed communications process group.
    When training a model across multiple GPUs , PyTorch spawns separate Python worker processes
    These workers require a network connection to locate one another, establish 
    a private communication channel, and synchronize gradients during training."""
    # We specify host and port configurations for inter-process communications
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the default NCCL (Nvidia) or Gloo (CPU/general) communications backend.
    # Gloo is the communication library used to send data—like model gradients and tensors—between worker processes.
    # We use 'gloo' here as a fallback because it is supported on all devices, including CPU.
    # Documentation: https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

def cleanup_distributed():
    """Destroys active distributed communication groups."""
    # Documentation: 
    dist.destroy_process_group()

def train_worker(rank, world_size):
    print(f"--- Worker Rank {rank}/{world_size} starting up ---")
    setup_distributed(rank, world_size)
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")
    
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
        torch.cuda.reset_peak_memory_stats(device) #Clears PyTorch's internal VRAM tracker so that only the peak memory used
        torch.cuda.synchronize() #synchronize() forces Python to wait until all queued GPU tasks finish before hitting
    start_time = time.time()

    epochs = 3
    for epoch in range(epochs):
        sampler.set_epoch(epoch) #the dataset is shuffled differently and deterministically across GPUs at every training epoch
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            # Automatically converts compatible mathematical operations (like matrix multiply ) to lower precision formats
            # This speeds up execution and cuts GPU memory usage
            with autocast(device_type=device_type):
                outputs = model(inputs)
                
                # -------------------------------------------------------------
                # CHANGE 2: Precision Inspection (Check Dtype)
                # -------------------------------------------------------------
                # Proof that Automatic Mixed Precision (AMP) is actually working under autocast
                # Displays torch.float16 or torch.bfloat16
                if epoch == 0 and batch_idx == 0 and rank == 0:
                    print(f"  [AMP PROOF] Output Tensor Dtype under autocast: {outputs.dtype}")
                    
                loss = criterion(outputs, targets)
                
            optimizer.zero_grad()
            # scaler - prevents a severe numerical issue in 16-bit floating-point math called gradient underflow
            # Multiply by a large factor, so they don't round down to zero
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            # Dynamically adjusts (increase) the scaling factor for the next iteration 
            scaler.update()

    # -------------------------------------------------------------
    # CHANGE 4: Compute Final Execution Time & Peak VRAM
    # -------------------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.synchronize() # Wait until the GPU finishes all queued operations before checking memory statistics.
        # Measure the max amount of VRAM your model actually used (consumed) during the training loop.
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
    # Setting world_size=1 (or local testing and debugging) prevents process groups from waiting and deadlocking.
    world_size = 1
    print(f"--- Initializing 1-process DDP Simulation ---")
    
    # In production, DDP is spawned using torchrun CLI or mp.spawn.
    # Here we invoke worker 0 directly to demonstrate and verify compile loops.
    # rank is the unique ID assigned to each individual process (or GPU)
    train_worker(rank=0, world_size=world_size)

if __name__ == "__main__":
    main()
