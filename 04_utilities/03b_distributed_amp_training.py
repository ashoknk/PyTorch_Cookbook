"""
21 -3b. Demonstrating Matrix Math Acceleration on Apple Silicon (MPS)
"""

import time
import torch
import torch.nn as nn

def benchmark_mac_acceleration():
    # 1. Check for Apple Silicon GPU (MPS)
    if not torch.backends.mps.is_available():
        print("MPS device not found. Running on CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device("mps")
        print(f"Using Apple Silicon Accelerator: {device}")

    # 2. Setup a large matrix multiplication test
    size = 4096
    print(f"\nBenchmark: Multiplying two {size}x{size} matrices...")
    
    # --- CPU Benchmark ---
    x_cpu = torch.randn(size, size, device="cpu")
    y_cpu = torch.randn(size, size, device="cpu")
    
    start_time = time.time()
    _ = torch.matmul(x_cpu, y_cpu)
    cpu_duration = time.time() - start_time
    print(f"  CPU Time: {cpu_duration:.4f} seconds")

    # --- MPS (Apple GPU) Benchmark ---
    if device.type == "mps":
        x_mps = torch.randn(size, size, device=device)
        y_mps = torch.randn(size, size, device=device)
        
        # Warm-up pass
        _ = torch.matmul(x_mps, y_mps)
        torch.mps.synchronize()  # Ensure GPU operations complete for timing
        
        start_time = time.time()
        _ = torch.matmul(x_mps, y_mps)
        torch.mps.synchronize()
        mps_duration = time.time() - start_time
        
        print(f"  MPS Time: {mps_duration:.4f} seconds")
        print(f"  Speedup:  {cpu_duration / mps_duration:.2f}x faster on Apple GPU!")

if __name__ == "__main__":
    benchmark_mac_acceleration()