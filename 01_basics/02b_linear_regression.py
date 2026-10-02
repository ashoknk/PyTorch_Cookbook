"""
02b. Linear Regression with PyTorch

This script demonstrates how to train a basic parametric model using PyTorch. 
We generate a noisy linear dataset and train a single-layer neural network 
to fit a line of the form y = w * x + b. This example introduces key PyTorch 
concepts: Modules, Loss Functions, and Optimizers.

SGD (Stochastic Gradient Descent) optimizer is used because it is the engine that 
actually allows the model to learn.In PyTorch, the Model class only defines 
the structure of your model (the math equations and the weights). 
The optimizer (like SGD) is the separate tool that modifies those weights 
to make the model's predictions accurate.

Learning Objectives:
1. Define a custom model by inheriting from torch.nn.Module.
2. Setup and use a standard loss function (MSE) and optimizer (SGD).
3. Implement a complete training loop with forward, backward, and parameter update steps.
4. Visualize data fitting using matplotlib.
"""

# Import standard matplotlib for visualization. We use the 'Agg' backend 
# to ensure it can run on any environment, including headless servers.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# We import the core torch library.
# Documentation: https://pytorch.org/docs/stable/index.html
import torch

# nn (Neural Network) provides modules, layers, and loss functions.
# Documentation: https://pytorch.org/docs/stable/nn.html
import torch.nn as nn

# optim (Optimizers) handles algorithm-driven updates to our parameters.
# Documentation: https://pytorch.org/docs/stable/optim.html
import torch.optim as optim

# ==========================================
# 1. DEFINE MODEL ARCHITECTURE
# ==========================================
# All neural network modules in PyTorch must inherit from nn.Module.
# Documentation: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__() #Python 2 syntax
        # super().__init__() #Python 3 syntax
        # nn.Linear represents a linear transformation: y = xW^T + b.
        # It has 1 input feature and 1 output feature.
        # In standard practice, nn.Linear is almost always used with dimensions larger than (1, 1)
        # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.linear = nn.Linear(1, 1)
        #  nn.Linear(in_features=1, out_features=1)

    # The forward method defines how input data passes through the layers.
    def forward(self, x):
        return self.linear(x)

def main():
    # ==========================================
    # 2. GENERATE SYNTHETIC DATA
    # ==========================================
    print("--- Generating Synthetic Toy Data ---")
    # Set seed for reproducibility. Every time you run your code, you get the exact same "random" numbers
    # Equivalent in sklearn is random_state=42
    torch.manual_seed(42)
    
    # Synthetic data generation 
    # Generate 100 features from normal distribution
    X = torch.randn(100, 1)
    # Target function: y = 2 * x + 1 + noise
    # y = 2 * X + 1 + noise
    #     |   |   |   |
    #     w   x   b   epsilon (random noise)
    # 2: The True Weight (Slope)
    # X: The Input Features
    # 1: The True Bias (Y-intercept)
    # noise: The variance/error that the model has to learn to ignore.

    noise = 0.5 * torch.randn(100, 1)
    y = 2 * X + 1 + noise

    # ==========================================
    # 3. INITIALIZE MODEL, LOSS & OPTIMIZER
    # ==========================================
    print("--- Initializing Linear Regression ---")
    model = LinearRegressionModel()
    
    print(" Before training:")
    [weight, bias] = model.parameters()
    print(f" Weight: {weight.item():.4f}, Bias: {bias.item():.4f}")

    # We use Mean Squared Error loss (MSE) to measure how far predictions are from ground truth.
    # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
    criterion = nn.MSELoss()

    # Stochastic Gradient Descent (SGD) will update the model parameters (weights, biases)
    # based on calculated gradients with a learning rate of 0.05.
    # Documentation: https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
    optimizer = optim.SGD(model.parameters(), lr=0.05)

    # ==========================================
    # 4. IMPLEMENT THE TRAINING LOOP
    # ==========================================
    epochs = 100
    print(f"--- Training model for {epochs} epochs ---")
    
    """optimizer.zero_grad(): 
         You clear your head and forget about your previous shot.
       loss.backward(): 
         The coach analyzes your missed shot and tells you exactly what went wrong: 
         "You aimed 3 inches too high and 2 inches to the left." (This is the gradient calculation).
       optimizer.step(): 
         You actually adjust your arms and stance based on what the coach just told you before taking the next shot."""
    for epoch in range(epochs):
        # 4.1 Forward Pass: Feed data to model to make predictions
        # forward() is called implicitly by model(X)
        predictions = model(X)
        
        # 4.2 Compute loss
        # predictions = model output, i.e. y^ 
        # y = true target values
        # PyTorch internally computes the squared difference between them and averages it
        loss = criterion(predictions, y)
        
        # 4.3 Backward Pass: Reset gradients, compute new ones, and update weights
        # We must call zero_grad() because PyTorch accumulates gradients by default.
        # Gradients measure how much the weights and biases need to change to reduce the model's error.
        optimizer.zero_grad()
        # loss.backward() calculates how much your model's weights contributed to the error.
        # "If I change this specific weight by a tiny amount, will the total loss go up or down, and by how much?"
        # loss.backward() calculates these gradients and attaches them directly to each weight tensor (storing them in weight.grad).
        loss.backward()
        
        # Adjust weight and bias parameters using SGD step
        optimizer.step()
        
        # Print update logs every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            [weight, bias] = model.parameters()
            print(" During training:")
            print(f" Weight: {weight.item():.4f}, Bias: {bias.item():.4f}")


    # ==========================================
    # 5. EVALUATION AND PLOTTING
    # ==========================================
    print("\n--- Model Evaluation ---")
    # Get trained parameters
    [weight, bias] = model.parameters()
    print(f"True Weight: 2.0, Learned Weight: {weight.item():.4f}")
    print(f"True Bias: 1.0, Learned Bias: {bias.item():.4f}")

    # Generate prediction plot (using no_grad to disable gradient tracking during plotting)
    # PyTorch has a strict rule: You cannot convert a tensor to a NumPy array 
    # if it is actively being tracked for gradients.
    # If you generate predictions on a large dataset , PyTorch will keep saving those hidden tracking maps in your computer's RAM or GPU memory
    with torch.no_grad():
        predicted_y = model(X).numpy()
    
    plt.scatter(X.numpy(), y.numpy(), color='blue', label='Ground Truth Data')
    plt.plot(X.numpy(), predicted_y, color='red', linewidth=2, label='Fitted Line')
    plt.title('PyTorch Linear Regression')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    
    # Save the output figure to verify visual output in headless mode
    plt.savefig('01_basics/linear_regression_fit.png')
    print("Regression plot saved as '01_basics/linear_regression_fit.png'.")

if __name__ == "__main__":
    main()
