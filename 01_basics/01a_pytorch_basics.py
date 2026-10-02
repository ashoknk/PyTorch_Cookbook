"""
01a. PyTorch Tensor Creation Methods

This script provides a practical overview of various methods for initializing PyTorch
tensors. It demonstrates how to create tensors from existing data structures like
NumPy arrays without copying memory, instantiate constant and uninitialized tensors,
generate sequences and ranges, and create tensors using alternative random distributions.

Learning Objectives:
1. Learn how to create tensors from existing data (like NumPy arrays) while sharing memory space.
2. Understand how to instantiate static constant tensors (zeros, ones, full) and uninitialized memory.
3. Discover how to generate range sequences, linearly spaced values, and identity matrices.
4. Master generating tensors from uniform, integer, normal, and permuted random distributions.
"""

import numpy as np
import torch

# =====================================================================
# 1. From Existing Data (Without Copying Memory)
# If you already have data in a NumPy array or Python sequence, you can
# create tensors that share the same memory space to optimize performance.
# =====================================================================
print("=" * 32)
# Setup initial data
arr = np.array([10, 20, 30])
python_list = [12, 22, 32]

# torch.from_numpy(ndarray): 
#   Creates a tensor directly from a NumPy array. Modifying the tensor changes the array and vice-versa. 
#   Does not accept Python Lists / Tuples
# Document: https://docs.pytorch.org/docs/2.14/generated/torch.from_numpy.html
t_from_numpy = torch.from_numpy(arr)
print("Creates a tensor from a NumPy array. \nModifying the tensor changes the array and vice-versa:\n", t_from_numpy)
print(f"Type: {t_from_numpy.dtype}")

# torch.as_tensor(data): 
#   Accepts lists, tuples, or NumPy arrays. It avoids copying memory whenever possible.
# Document: https://docs.pytorch.org/docs/2.14/generated/torch.as_tensor.html
t_as_tensor = torch.as_tensor(arr)
# t_as_tensor = torch.as_tensor(python_list) # NOTE testing
print("Creates a tensor from existing data without copying memory whenever possible:\n", t_as_tensor)

print("\nChanged values in the original NumPy array to demonstrate shared memory:\n")
arr[:] = [11, 21, 31]
print("Creates a tensor from a NumPy array. \nModifying the tensor changes the array and vice-versa:\n", t_from_numpy)
print("Creates a tensor from existing data without copying memory whenever possible:\n", t_as_tensor)


print("=" * 32)

# =====================================================================
# 2. Constant and Uninitialized Tensors
# When you need to initialize a specific shape with static values:
# =====================================================================

# torch.zeros(*size): 
#   Returns a tensor filled with the scalar value 0.
# Document: https://docs.pytorch.org/docs/2.14/generated/torch.zeros.html
t_zeros = torch.zeros(2, 3)
print("Creates a tensor filled with zeros:\n", t_zeros)

# torch.ones(*size): 
#   Returns a tensor filled with the scalar value 1.
# Document: https://docs.pytorch.org/docs/2.14/generated/torch.ones.html
t_ones = torch.ones(2, 3)
print("Creates a tensor filled with ones:\n", t_ones)

# torch.full(size, fill_value): 
#   Creates a tensor of the given shape filled entirely with fill_value.
# Document: https://docs.pytorch.org/docs/2.14/generated/torch.full.html
t_full = torch.full((2, 3), fill_value=7)
print("Creates a tensor filled with the value 7:\n", t_full)

# torch.empty(*size): 
#   Allocates memory for a tensor of the given shape but leaves it uninitialized. This is slightly faster if you plan to overwrite the values immediately.
# While torch.zeros guarantees all values will be 0, torch.empty does not fill the tensor with zeros on purpose—it only appeared to contain zeros in your test by coincidence.
# Document: https://docs.pytorch.org/docs/2.14/generated/torch.empty.html
t_empty = torch.empty(2, 3)
print("Creates an uninitialized tensor:\n", t_empty)

print("=" * 32)

# =====================================================================
# 3. Sequences, Ranges, and Linear Spacing
# Great for generating indices, grids, or linearly distributed values:
# =====================================================================

# torch.arange(start, end, step): 
#   Returns a 1D tensor with a sequence of values from start to end (exclusive), incremented by step size (default 1).
# Document: https://docs.pytorch.org/docs/2.14/generated/torch.arange.html
# 1. torch.arange creates a 1D tensor of 12 elements
t_arange1 = torch.arange(start=0, end=12)
print("1D Tensor:\n", t_arange1)
print(f"Original shape: {t_arange1.shape}")

t_arange2 = torch.arange(start=0, end=10, step=2)
print("Creates a tensor containing a sequence with a specified step:\n", t_arange2)
print(f"Original shape: {t_arange2.shape}")

# By design, torch.arange(start, end, step) generates a flat sequence of numbers along a single dimension.
# However, you can easily turn that 1D sequence into a 2D, 3D, or higher-dimensional tensor by chaining .reshape() or .view() right after it:
# Reshape into 2D (3 rows, 4 columns)
t_2d = torch.arange(start=0, end=12).reshape(3, 4)
print("\n2D Tensor:\n", t_2d)
print(f"2D shape: {t_2d.shape}")

# torch.linspace(start, end, steps): 
#   Creates a 1D tensor of a specified number of steps equally spaced between start and end (inclusive).
# Document: https://docs.pytorch.org/docs/2.14/generated/torch.linspace.html
# You specify the number of steps (total count of numbers and not the step size).
t_linspace = torch.linspace(start=0, end=1, steps=5)
print("\nCreates a tensor with evenly spaced values:\n", t_linspace)
print(f"Original shape: {t_linspace.shape}")

# torch.eye(n): 
#   Creates a 2D square identity matrix of size n × n (ones on the diagonal, zeros elsewhere).
# Document: https://docs.pytorch.org/docs/2.14/generated/torch.eye.html
t_eye = torch.eye(3)
print("\nCreates a 3x3 identity matrix:\n", t_eye)

# Rectangular Matrix: n x (n+1) (3 x 4)
t_rect_wide = torch.eye(3, 4)
print("\nCreates 3 x 4 Rectangular Matrix:\n", t_rect_wide)

print("=" * 32)

# =====================================================================
# 4. Alternative Random Distributions
# If you need random variations other than a standard normal distribution:
# =====================================================================

# torch.rand(*size): 
#   Generates random numbers from a uniform distribution between [0, 1).
# Document: https://docs.pytorch.org/docs/2.14/generated/torch.rand.html
t_rand = torch.rand(size=(2, 3))
print("\nCreates a tensor with uniformly distributed random values:\n", t_rand)

# torch.randint(low, high, size): 
#   Generates random integers uniformly distributed between low (inclusive) and high (exclusive).
# Document: https://docs.pytorch.org/docs/2.14/generated/torch.randint.html
t_randint = torch.randint(low=1, high=10, size=(2, 3))
print("\nCreates a tensor with uniformly distributed random integers:\n", t_randint)

# torch.normal(mean, std): 
#   Generates random numbers from a normal distribution where you can explicitly configure a custom mean and standard deviation (std).
# Document: https://docs.pytorch.org/docs/2.14/generated/torch.normal.html
# mean - The "average" or target value you want your numbers to clump around.
# std - How much individual numbers are allowed to stray away from the mean.
t_normal = torch.normal(mean=0.0, std=1.0, size=(2, 3))
print("\nCreates a tensor with normally distributed random values:\n", t_normal)

# torch.randperm(n): 
#   Returns a 1D tensor containing a random permutation (shuffled sequence) of integers from 0 to n-1. Perfect for shuffling dataset indices.
# Document: https://docs.pytorch.org/docs/2.14/generated/torch.randperm.html
t_randperm = torch.randperm(5)
print("\nCreates a tensor containing a random permutation:\n", t_randperm)
t_randperm = torch.randperm(10)
print("\nCreates a tensor containing a random permutation:\n", t_randperm)
