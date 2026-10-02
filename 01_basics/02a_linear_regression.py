"""
02a. Linear Regression with PyTorch
This script demonstrates how to instantiate and 
run data through a single-layer linear regression model using PyTorch's nn.Linear.

Learning Objectives:
- Understand how to configure in_features and out_features for a 1D problem.
- Witness how raw input features transform into a raw scalar prediction.
- Inspect the linear layer's weight and bias alongside its prediction for the input.
"""

import torch
import torch.nn as nn

# 1. Set seed for reproducibility
torch.manual_seed(42)

# 2. Define the linear layer
# in_features=1 (1 input variable), out_features=1 (1 target prediction)
# In standard practice, nn.Linear is almost always used with dimensions larger than (1, 1)
# Documentation: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
linear_layer = nn.Linear(in_features=1, out_features=1)

# 3. Create simulated input data for 1 house
# PyTorch expects data in a 2D shape: [batch_size, num_features] -> [1 row, 1 col]
house_size_input = torch.tensor([[2400.0]]) 

# 4. Pass the data through the layer to get a prediction
with torch.no_grad():
    prediction_output = linear_layer(house_size_input)

# 5. Print the underlying math
print("--- Inside the Layer ---")
print("Weight (w):", linear_layer.weight.item())
print("Bias (b):  ", linear_layer.bias.item())
print("\n--- Output ---")
print("Predicted Price:", prediction_output.item())
