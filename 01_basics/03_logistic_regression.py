"""
03. Logistic Regression with PyTorch

This script demonstrates how to build and train a single-layer binary 
classification model in PyTorch. We generate synthetic two-dimensional cluster 
data and train a logistic regression network to separate the two classes, 
introducing binary classification loss functions, decision thresholds, and 
decision boundary plotting.

Learning Objectives:
1. Construct a logistic regression network using linear and sigmoid layers.
2. Understand Binary Cross Entropy (BCE) loss and its application.
3. Apply thresholding to map sigmoid probability outputs to binary class labels (0 or 1).
4. Visualize the learned decision boundary in 2D space.
"""

# Import standard libraries for plotting and data generation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# We import the scikit-learn helper to quickly generate clean cluster data.
# Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
from sklearn.datasets import make_blobs

# We import the core torch library.
import torch

# nn handles network architectures and loss criteria.
import torch.nn as nn

# optim contains optimizer updates.
import torch.optim as optim

# ==========================================
# 1. DEFINE MODEL ARCHITECTURE
# ==========================================
# A Logistic Regression model consists of a linear transformation followed by
# a sigmoid activation function to squash output values to [0, 1] probability range.
# Documentation: https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html
# Mathematically, logistic regression is just a linear regression model whose output is squashed 
# through a Sigmoid activation function to turn the prediction into a probability between 0 and 1.
# Because of this, you use the exact same nn.Linear layer to handle the weights and bias.

class LogisticRegressionModel(nn.Module):
    def __init__(self, num_features):
        super(LogisticRegressionModel, self).__init__()
        # 1 linear layer mapping input features to 1 output logit
        self.linear = nn.Linear(num_features, 1)
        # Sigmoid function for probability mapping
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

def main():
    # ==========================================
    # 2. GENERATE AND PREPARE CLUSTERING DATA
    # ==========================================
    print("--- Generating Synthetic Classification Clusters ---")
    
    # Create 150 samples in 2D space grouped into 2 distinct clusters
    # https://sklearn.org/stable/datasets/sample_generators.html
    # https://sklearn.org/stable/modules/generated/sklearn.datasets.make_blobs.html#sklearn.datasets.make_blobs
    raw_X, raw_y = make_blobs(n_samples=150, centers=2, n_features=2, random_state=42, cluster_std=1.2)
    
    # Convert numpy arrays to float tensors. PyTorch models expect float32 inputs by default.
    X = torch.tensor(raw_X, dtype=torch.float32)
    # We reshape the label tensor to have an explicit second dimension of size 1 (shape [150, 1])
    y = torch.tensor(raw_y, dtype=torch.float32).view(-1, 1)
    print("raw_y.shape:", raw_y.shape, "y.shape:", y.shape)
    
    # ==========================================
    # 3. INITIALIZE MODEL, LOSS & OPTIMIZER
    # ==========================================
    print("--- Initializing Logistic Regression Model ---")
    # 2 inputs mapping to 1 class output probability
    model = LogisticRegressionModel(num_features=2)

    # We use Binary Cross-Entropy (BCE) loss because we are doing binary classification.
    # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
    # BCE uses logarithms to heavily penalizes the model if it is confident and wrong. 
    # For example, if a true label is 1 (Spam) and the model predicts a probability 
    # of 0.01 (99% confident it's not spam), BCELoss will output a massive error penalty.
    criterion = nn.BCELoss()

    # We use Adam optimizer, a popular adaptive learning rate algorithm.
    # Configure the optimizer to update your model's weights and biases using calculated gradients, 
    # scaled by a learning rate of 0.1.
    # Documentation: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    # ==========================================
    # 4. IMPLEMENT THE TRAINING LOOP
    # ==========================================
    epochs = 100
    print(f"--- Training model for {epochs} epochs ---")
    
    for epoch in range(epochs):
        # 4.1 Forward Pass: predict probability scores
        # forward() is called implicitly by model(X)
        probabilities = model(X)
        
        # 4.2 Compute loss
        # probabilities = model output, i.e. y^ 
        # y = true target values
        # PyTorch internally computes the squared difference between them and averages it
        loss = criterion(probabilities, y)
        
        # 4.3 Backward Pass: Compute gradients and update network parameters
        # We must call zero_grad() because PyTorch accumulates gradients by default.
        # Gradients measure how much the weights and biases need to change to reduce the model's error.
        optimizer.zero_grad()
        # loss.backward() calculates how much your model's weights contributed to the error.
        # "If I change this specific weight by a tiny amount, will the total loss go up or down, & by how much?"
        # loss.backward() calculates these gradients & attaches them directly to each weight tensor (storing them in weight.grad).
        loss.backward()
        optimizer.step()
        
        # Print update logs every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # ==========================================
    # 5. INFERENCE & ACCURACY ASSESSMENT
    # ==========================================
    with torch.no_grad():
        # Get final output probabilities(using no_grad to disable gradient tracking during plotting)
        # PyTorch has a strict rule: You cannot convert a tensor to a NumPy array 
        # if it is actively being tracked for gradients.
        predictions = model(X)
        print("\n--- Final Predictions (Probabilities) ---")
        print(predictions[:5])  # Show first 5 predictions for brevity
        # Apply binary decision threshold of 0.5: scores >= 0.5 are labeled as 1, otherwise 0
        predicted_classes = (predictions >= 0.5).float()
        print(f"Predicted Classes:\n{predicted_classes[:5]}")  # Show first 5 predicted classes for brevity
        
        # Calculate accuracy by comparing predictions directly with ground truth targets
        # `(predicted_classes == y)` gives one boolean per sample,But for accuracy, we want a single number like: 75% correct
        accuracy = (predicted_classes == y).float().mean()
        print(f"\nFinal Classification Accuracy: {accuracy.item() * 100:.2f}%")
    
    # ==========================================
    # 6. DECISION BOUNDARY VISUALIZATION
    # ==========================================
    print("--- Saving Decision Boundary Visualization ---")
    # Extract weight and bias parameters
    # The decision boundary is the line where: w1*x1 + w2*x2 + b = 0
    # Solving for x2 gives: x2 = -(w1 * x1 + b) / w2
    # By default, PyTorch model parameters (weight and bias) have a property called requires_grad=True. i.e. actively tracking every mathematical operation done
    # .detach() creates a new view of the tensor that hides it from the history tracker, effectively saying, "I just want the current raw values; stop tracking gradients for this copy."
    # .numpy() explicitly converts that PyTorch tensor into a standard NumPy ndarray.
    weights = model.linear.weight.detach().numpy()[0]
    bias = model.linear.bias.detach().numpy()[0]
    
    # 1. Find the lowest and highest X values, then add 1 for breathing room
    x1_min, x1_max = raw_X[:, 0].min() - 1 , raw_X[:, 0].max() + 1
    
    # 2. Create 100 evenly spaced points between the minimum and maximum X values- np.linspace(start, stop, num)
    x1_line = np.linspace(x1_min, x1_max, 100)
    
    # 3. Calculate the matching x2 from x1 values to draw the straight decision line
    x2_line = -(weights[0] * x1_line + bias) / weights[1]

    # 4. Plot the data points (Blue for Class 0, Orange for Class 1)
    # Plot sample data points colored by class
    # raw_y == 0 → keep only rows whose label is 0
    # raw_X[raw_y == 0, 0] → take the first column of those selected rows
    # so this gives all the x-coordinates for class 0
    # alpha=0.7 sets the transparency/see-through of your data points.
    plt.scatter(raw_X[raw_y == 0, 0], raw_X[raw_y == 0, 1], label='Class 0', color='blue', alpha=0.7)
    plt.scatter(raw_X[raw_y == 1, 0], raw_X[raw_y == 1, 1], label='Class 1', color='orange', alpha=0.7)
    
    # 5. Draw the separating line (The Decision Boundary) in dashed red
    plt.plot(x1_line, x2_line, label='Decision Boundary', color='red', linestyle='--', linewidth=2)
    
    # 6. Lock the left and right edges of the graph so the line matches the margins perfectly
    # Plot separating linear boundary line
    # `plt.xlim()` sets the left and right limits of the X-axis on your plot.
    # It controls exactly how much of the horizontal axis is visible in your graph window.
    plt.xlim(x1_min, x1_max)

    plt.title('PyTorch Logistic Regression Classification')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    plt.savefig('01_basics/logistic_regression_boundary.png')
    print("Visualization saved as '01_basics/logistic_regression_boundary.png'.")

if __name__ == "__main__":
    main()
