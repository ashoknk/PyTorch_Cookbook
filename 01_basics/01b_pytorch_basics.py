"""
01b. PyTorch Tensors & Autograd Basics

This script serves as an entry point for PyTorch learners. It covers the 
fundamental data structure in PyTorch—the Tensor—and demonstrates how to perform 
basic math operations, manage computing hardware (CPUs and GPUs), and use 
Autograd (automatic differentiation) to calculate gradients for training.

Learning Objectives:
1. Understand how to create and manipulate PyTorch Tensors.
2. Learn how to route computations to GPU or Apple Silicon MPS accelerators.
3. Discover how PyTorch automatically tracks math operations to calculate derivatives.
4. Learn how to detach operations from tracking to optimize memory.
"""

# We import the core torch library. This provides the multidimensional tensor 
# data structures and all mathematical operations associated with them.
# Documentation: https://pytorch.org/docs/stable/index.html
import torch

def main():
    # ==========================================
    # 1. TENSOR CREATION AND BASIC CONVERSIONS
    # ==========================================
    print("--- 1. Tensor Creation ---")
    
    # We can create tensors directly from standard Python lists.
    # Documentation: https://pytorch.org/docs/stable/generated/torch.tensor.html
    list_data = [1.0, 2.0, 3.0]
    tensor_from_list = torch.tensor(list_data)
    print(f"Tensor from list: {tensor_from_list}, Type: {tensor_from_list.dtype}")

    # We can create tensors representing random distributions. 
    # torch.randn generates values from a standard normal distribution (mean=0, variance=1).
    # Documentation: https://pytorch.org/docs/stable/generated/torch.randn.html
    random_tensor = torch.randn(2, 3)
    print(f"Random 2x3 normal distribution tensor:\n{random_tensor}\n")

    # ==========================================
    # 2. DEVICE MANAGEMENT & ACCELERATORS
    # ==========================================
    print("--- 2. Device Management ---")
    
    # PyTorch allows selecting where to store a tensor. By default, tensors are created on CPU.
    # We check if active hardware accelerators are available, like Nvidia CUDA or Apple Silicon MPS.
    # Documentation: https://pytorch.org/docs/stable/tensor_attributes.html#torch.device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"  # Nvidia GPUs
    elif torch.backends.mps.is_available():
        device = "mps"   # Apple Silicon GPUs (Metal Performance Shaders)
    
    print(f"Selected computing device: {device}")
    
    # Move our random tensor to the target device using the .to() method.
    device_tensor = random_tensor.to(device)
    print(f"Tensor is now on device: {device_tensor.device}\n")
    

    # ==========================================
    # 3. TENSOR MATHEMATICS AND RESHAPING
    # ==========================================
    print("--- 3. Tensor Math & Reshaping ---")
    
    t = torch.rand(2, 6) # 12 elements total

    # Reshape into a 2D matrix (3 rows, 4 columns)
    t_3x4 = t.view(3, 4)
    print(f"torch.Size([3, 4]): {t_3x4.shape}")   # torch.Size([3, 4])

    # Reshape into a 3D tensor (2 x 2 x 3)
    t_3d = t.view(2, 2, 3)
    print(f"torch.Size([2, 2, 3]): {t_3d.shape}")    # torch.Size([2, 2, 3])

    # We can reshape tensors. The .view() method creates a new view of the tensor data 
    # without copying underlying memory. The size -1 tells PyTorch to calculate that dimension automatically.
    # Documentation: https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
    flat_tensor = random_tensor.view(-1)
    print(f"Original shape: {random_tensor.shape} -> Flattened shape: {flat_tensor.shape}")

    # Matrix multiplication using the '@' operator or 'torch.matmul'.
    # We create a 3x2 tensor to multiply with our 2x3 tensor, resulting in a 2x2 tensor.
    # Documentation: https://pytorch.org/docs/stable/generated/torch.matmul.html
    multiplier = torch.randn(3, 2)
    multiplied_tensor = torch.matmul(random_tensor, multiplier)
    print(f"Matrix multiplication using torch.matmul (Shape {multiplied_tensor.shape}):\n{multiplied_tensor}\n")

    multiplier = torch.randn(3, 2)
    multiplied_tensor = random_tensor @ multiplier
    print(f"Matrix multiplication using @ operator (Shape {multiplied_tensor.shape}):\n{multiplied_tensor}\n")

    # ==========================================
    # 4. AUTOGRAD: AUTOMATIC DIFFERENTIATION
    # ==========================================
    print("--- 4. Autograd Basics ---")
    
    # To calculate gradients, we must set requires_grad=True when creating a tensor.
    # This tells PyTorch to track all operations involving this tensor in a dynamic graph.
    # Documentation: https://pytorch.org/docs/stable/autograd.html
    x = torch.tensor([2.0], requires_grad=True)
    
    # We define an equation: y = 3 * x^2 + 2 * x
    # The derivative of this equation is: dy/dx = 6 * x + 2
    y = 3 * (x ** 2) + 2 * x
    print(f"Inputs: x = {x.item()}")
    print(f"Forward equation value: y = {y.item()}")

    # We trigger the backpropagation step by calling .backward() on our output.
    # This calculates the gradients of y with respect to x and stores them in x.grad.
    # Documentation: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
    y.backward()
    print(f"Calculated gradient dy/dx at x=2 is: {x.grad.item()} (Expected: 6*2 + 2 = 14)\n")

    # ==========================================
    # 5. GRADIENT DETACHMENT AND CONTEXTS
    # ==========================================
    print("--- 5. Gradient Detachment ---")
    
    # When testing or running inference, we do not want PyTorch to track operations.
    # We use 'with torch.no_grad():' block to turn off tracking and save memory and speed up computation.
    # Documentation: https://pytorch.org/docs/stable/generated/torch.no_grad.html
    with torch.no_grad():
        z = x * 10
        print(f"z inside no_grad block requires gradient? {z.requires_grad}")

    # Alternatively, we can use the .detach() method to get a new tensor that shares 
    # the same storage but has gradient tracking removed.
    detached_x = x.detach()
    print(f"Detached tensor requires gradient? {detached_x.requires_grad}")

if __name__ == "__main__":
    main()
