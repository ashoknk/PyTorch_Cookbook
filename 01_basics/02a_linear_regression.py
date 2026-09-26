"""
This script demonstrates how to instantiate and run data through a single-layer linear regression model using PyTorch's nn.Linear.

Learning Objectives:
- Understand how to configure in_features and out_features for a 1D problem.
- Witness how raw input features transform into a raw scalar prediction.
- Verify how PyTorch relies strictly on unitless floating-point calculations before training.
"""

import torch
import torch.nn as nn

# 1. Set seed for reproducibility
torch.manual_seed(42)

# 2. Define the linear layer
# in_features=1 (1 input variable), out_features=1 (1 target prediction)
linear_layer = nn.Linear(in_features=1, out_features=1)

# 3. Create simulated input data for 1 house
# PyTorch expects data in a 2D shape: [batch_size, num_features] -> [1 row, 1 col]
house_size = torch.tensor([[2400.0]]) 

# 4. Pass the data through the layer to get a prediction
with torch.no_grad():
    prediction = linear_layer(house_size)

# 5. Print the underlying math
print("--- Inside the Layer ---")
print("Weight (w):", linear_layer.weight.item())
print("Bias (b):  ", linear_layer.bias.item())
print("\n--- Output ---")
print("Predicted Price:", prediction.item())
