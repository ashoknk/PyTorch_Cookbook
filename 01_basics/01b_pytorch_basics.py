"""
01b. PyTorch Tensors & Autograd Basics

This script serves as an entry point for PyTorch learners. It covers the 
fundamental data structure in PyTorch—the Tensor—and demonstrates how to perform 
basic math operations, manage computing hardware (CPUs and GPUs), and use 
Autograd (automatic differentiation) to calculate gradients for training.

Learning Objectives:
1. Understand how to create and manipulate PyTorch Tensors.
2. Learn how to route computations to GPU or Apple Silicon MPS accelerators.
3. Learn how to perform tensor math, reshape tensors, and multiply matrices.
4. Learn how to calculate gradients using backpropagation.
5. Learn how to detach operations from tracking to optimize memory.
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
    # `torch.randn` specifically samples from the standard normal distribution (mean = 0, standard deviation = 1), 
    # while `torch.normal` allows you to specify custom means and standard deviations for your distribution
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
    # ex. move data from CPU RAM into the GPU
    # Hardware accelerators (CUDA for NVIDIA GPUs, MPS for Apple Silicon) have their own separate, high-speed VRAM memory pools.
    device_tensor = random_tensor.to(device)
    print(f"Tensor is now on device: {device_tensor.device}\n")
    
    
    # ==========================================
    # 3. TENSOR MATHEMATICS AND RESHAPING
    # ==========================================
    print("--- 3. Tensor Math & Reshaping ---")
    
    t = torch.rand(2, 6) # 12 elements total
    print(f"t: {t}")
    print(f"t.Size([3, 4]): {t.shape}") 

    # Reshape into a 2D matrix (3 rows, 4 columns)
    t_3x4 = t.view(3, 4)
    print(f"t_3x4: {t_3x4}")
    print(f"torch.Size([3, 4]): {t_3x4.shape}")   # torch.Size([3, 4])

    # Reshape into a 3D tensor (2 x 2 x 3)
    t_3d = t.view(2, 2, 3)
    print(f"t_3d: {t_3d}")
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

    print(f"\nRandom tensor:\n{random_tensor}\n")
    print(f"Multiplier tensor:\n{multiplier}\n")
    multiplied_tensor = torch.matmul(random_tensor, multiplier)
    print(f"\nMatrix multiplication using torch.matmul (Shape {multiplied_tensor.shape}):\n{multiplied_tensor}\n")

    # multiplier = torch.randn(3, 2)
    multiplied_tensor = random_tensor @ multiplier
    print(f"Matrix multiplication using @ operator (Shape {multiplied_tensor.shape}):\n{multiplied_tensor}\n")
    

    # ==========================================
    # 4. AUTOGRAD: AUTOMATIC DIFFERENTIATION
    # ==========================================
    print("--- 4. Autograd Basics ---")
    
    # To calculate gradients, we must set requires_grad=True when creating a tensor.
    # This tells PyTorch to track all operations involving this tensor in a dynamic graph.
    # Documentation: https://pytorch.org/docs/stable/autograd.html
    # weight_x represents a Model Weight (or Parameter). Track this variable because we are going to tweak it during training using gradient descent.
    weight_x = torch.tensor([2.0], requires_grad=True)
    
    # We define an simple polynomial function: loss_y = 3 * weight_x^2 + 2 * weight_x. An educational example 
    # Mean Squared Error (MSE Loss) or Binary Cross-Entropy (BCE Loss) will be used in real-world scenarios. The goal is to minimize the loss function by adjusting the model weights.
    # The derivative of this equation is: dy/dx = 6 * weight_x + 2
    # loss_y represents the Loss Function:the value measuring error or cost. You want loss_y to be as small as possible
    loss_y = 3 * (weight_x ** 2) + 2 * weight_x
    print(f"Inputs: weight_x = {weight_x.item()}")
    print(f"Forward equation value: loss_y = {loss_y.item()}")

    # We trigger the backpropagation step by calling .backward() on our output.
    # This calculates the gradients of loss_y with respect to weight_x and stores them in weight_x.grad.
    # Documentation: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
    loss_y.backward()
    print(f"Calculated gradient dy/dx at x=2 is: {weight_x.grad.item()} (Expected: 6*2 + 2 = 14)\n")

    # The gradient tells you the slope and direction of the function. 
    # A gradient of 14.0 means that if you increase [weight_x] by a tiny amount, [loss_y] will increase "grad" times faster.
    # New weight_x= weight_x - (Learning_Rate * Gradient)) ==> 2.0 - (0.01 * 14.0)
    
    print(f"Before tensor requires gradient? {weight_x.requires_grad}")
    z = weight_x * 10
    print(f"Before z requires gradient? {z.requires_grad}")
    
    
    # ==========================================
    # 5. GRADIENT DETACHMENT AND CONTEXTS
    # ==========================================
    print("--- 5. Gradient Detachment ---")
    
    # When testing or running inference, we do not want PyTorch to track operations.
    # We use 'with torch.no_grad():' block to turn off tracking and save memory and speed up computation.
    # Documentation: https://pytorch.org/docs/stable/generated/torch.no_grad.html
    with torch.no_grad():
        z = weight_x * 10
        print(f"z inside no_grad block requires gradient? {z.requires_grad}")

    # Alternatively, we can use the .detach() method to get a new tensor that shares 
    # the same storage but has gradient tracking removed.
    detached_x = weight_x.detach()
    print(f"Detached tensor requires gradient? {detached_x.requires_grad}")

if __name__ == "__main__":
    main()
