"""
02d Multi-Task Learning (Classification + Bounding Box Regression)
 Multi-head neural networks, custom target formats, and Multi-Task Loss optimization (e.g., combining nn.CrossEntropyLoss for labels and nn.SmoothL1Loss for box coordinates).

Standard classifiers only predict what is in an image. Real-world applications often need to know where it is.
This teaches how a single shared feature extractor can feed into two separate fully connected heads: 
    one outputting a category, and another outputting 4 spatial coordinates [xmin, ymin, xmax, ymax].
You can programmatically generate a synthetic dataset of simple shapes 
(e.g., drawing circles or squares at random coordinates on a dark canvas), keeping the code 100% self-contained, lightweight, and incredibly fun to train and visualize.

Learning Objectives:
1. Construct a multi-head neural network sharing a single CNN backbone.
2. Formulate and train on multiple distinct labels (categorical label + spatial coordinates).
3. Combine multiple loss functions (Classification Loss and Regression Loss) using weighted loss scaling.
4. Generate custom synthetic shape datasets dynamically inside PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. GENERATE DYNAMIC SYNTHETIC DATASET
# ==========================================
class SyntheticShapeDataset(Dataset):
    """
    Dynamically generates grayscale 28x28 images containing a single random geometric shape 
    (either a Circle or a Square) rendered at random spatial coordinates.
    
    Returns:
        image: Tensor of shape (1, 28, 28)
        class_label: Integer (0 = Circle, 1 = Square)
        bbox: Tensor of shape (4,) containing normalized [xmin, ymin, xmax, ymax] coordinates (0.0 to 1.0)
    """
    def __init__(self, size=1000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Initialize a blank dark canvas of size 28x28
        image = torch.zeros((28, 28), dtype=torch.float32)
        
        # randint(low, high, grid size) Decide shape type: 0 = Circle, 1 = Square
        # (1,) one-item tuple to create a tensor with one element. .item() extracts that single number out of the tensor
        shape_type = torch.randint(0, 2, (1,)).item()
        
        # Determine a random center coordinate and size (radius/half-width)
        # Keep the generated shape completely inside the 28 x 28 image canvas without overflowing
        cx = torch.randint(6, 22, (1,)).item() # Min 6 - 5 =1 or Max 21 + 5 = 26
        cy = torch.randint(6, 22, (1,)).item()
        r = torch.randint(3, 6, (1,)).item() # Radius 3 to 5, Diameter 6 and 10 pixels

        # Render the shape onto the canvas. Create a grid of x, y coordinates. 
        # 1D sequence of numbers [0, 1, 2, ..., 27]. indexing="ij"  enforces matrix notation ordering
        y_indices, x_indices = torch.meshgrid(torch.arange(28), torch.arange(28), indexing="ij")
        
        if shape_type == 0:
            # Circle equation: (x - cx)^2 + (y - cy)^2 <= r^2. 
            # Euclidean Distance - squared distance from the center point
            mask = (x_indices - cx)**2 + (y_indices - cy)**2 <= r**2
            image[mask] = 1.0
        else:
            # Square boundaries: |x - cx| <= r and |y - cy| <= r
            # Chebyshev distance (box distance) r steps to the left or right of the center 
            mask = (torch.abs(x_indices - cx) <= r) & (torch.abs(y_indices - cy) <= r)
            image[mask] = 1.0 # Every pixel where the mask is True, overwrites the value to 1.0 (turning those pixels bright white).
            
        # Add a tiny bit of random background noise to make it realistic
        # Forces the CNN to learn robust visual features rather than relying on unrealistically perfect binary images.
        image += torch.randn_like(image) * 0.05
        image = torch.clamp(image, 0.0, 1.0).unsqueeze(0) # 2D --> 3D Shape: (1, 28, 28) . Remains strictly within valid grayscale boundaries 0.0, 1.0

        # Compute normalized bounding box coordinates: [xmin, ymin, xmax, ymax] to predict where the object is located in the image
        # Dividing by the total image width and height (28 pixels) normalizes the coordinates into the range [0.0, 1.0]
        xmin = (cx - r) / 28.0
        ymin = (cy - r) / 28.0
        xmax = (cx + r) / 28.0
        ymax = (cy + r) / 28.0
        bbox = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)
        
        return image, shape_type, bbox

# ==========================================
# 2. DEFINE MULTI-TASK NETWORK
# ==========================================
"""
Multi-Task Backbone and Heads:

    Instead of having two completely separate models (one to classify, one to detect), 
    we use a single SHARED feature extractor (the "backbone").
    
    This backbone processes the image and extracts high-level semantic features. 
    These features are then passed into two distinct, specialized linear layers:
    
        - Classification Head: Predicts shape class scores (Circle vs. Square).
        - Regression Head: Predicts continuous coordinates for the bounding box.
        
    Benefits: Greatly saves memory, trains faster, and the backbone learns richer features!
"""

class MultiTaskShapeDetector(nn.Module):
    def __init__(self):
        super(MultiTaskShapeDetector, self).__init__()
        
        # Shared CNN Backbone (Feature Extractor)
        # Looks at the raw input image pixels and compresses them into a set of high-level visual features
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 28x28 -> 14x14
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 14x14 -> 7x7
        )
        
        # Head 1: Classification Head - what the item is 
        # Input size: 32 channels * 7 height * 7 width = 1568
        self.class_head = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # 2 Output classes (0 = Circle, 1 = Square)
        )
        
        # Head 2: Bounding Box Regression Head - where the item is located
        self.bbox_head = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4), # 4 Outputs coordinates: [xmin, ymin, xmax, ymax]
            # We use Sigmoid because bounding box coordinates are normalized in the range [0.0, 1.0].
            nn.Sigmoid() 
        )

    def forward(self, x):
        # Pass input through the shared convolutional layers
        features = self.backbone(x)
        features = features.view(features.size(0), -1) # Flatten features
        
        # Branch out into individual heads
        class_logits = self.class_head(features) # The Type of Item ("What is it?")
        bbox_predictions = self.bbox_head(features) # The Location of the Item ("Where is it?")
        
        return class_logits, bbox_predictions

def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Running Multi-Task Learning on: {device} ---")

    # ==========================================
    # 3. PREPARE SYNTHETIC SHAPE DATASETS
    # ==========================================
    train_dataset = SyntheticShapeDataset(size=2000)
    test_dataset = SyntheticShapeDataset(size=500)

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

    print(f"Generated {len(train_dataset)} training shapes & {len(test_dataset)} testing shapes dynamically.")

    # ==========================================
    # 4. INITIALIZE MULTI-TASK LOSS AND WEIGHTS
    # ==========================================
    model = MultiTaskShapeDetector().to(device)
    
    # Classification Criterion (CrossEntropy for categorical prediction)
    class_criterion = nn.CrossEntropyLoss()
    
    # Bounding Box Criterion (SmoothL1 / Huber loss is highly stable for coordinate regression)
    # SmoothL1Loss is a loss function used in machine learning regression tasks that combines the benefits of both L1 (Mean Absolute Error) and L2 (Mean Squared Error) loss
    # Documentation: https://docs.pytorch.org/docs/2.14/generated/torch.nn.SmoothL1Loss.html
    bbox_criterion = nn.SmoothL1Loss()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ==========================================
    # 5. MULTI-TASK TRAINING LOOP
    # ==========================================
    epochs = 1
    print(f"--- Training Multi-Task Network for {epochs} epoch ---")
    model.train()

    for batch_idx, (images, class_labels, bbox_targets) in enumerate(train_loader):
        images = images.to(device)
        class_labels = class_labels.to(device) # The Type of Item ("What is it?")
        bbox_targets = bbox_targets.to(device) # The Location of the Item ("Where is it?")

        # Forward pass returning two separate outputs
        pred_logits, pred_bboxes = model(images)

        # Calculate individual losses
        loss_class = class_criterion(pred_logits, class_labels)
        loss_bbox = bbox_criterion(pred_bboxes, bbox_targets)

        # Combine losses:
        # Multi-task networks combine losses into a single scalar value so we can backward() once.
        # Since bounding boxes are limited between 0 and 1, the bounding box regression loss 
        # can be small compared to classification loss. We use a scaling weight (lambda = 2.0) 
        # to balance the importance of both learning objectives.
        total_loss = loss_class + 2.0 * loss_bbox

        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward() # Calculates the gradients for every weight and bias based on mistake/loss
        optimizer.step() # Updates the weights & the biases for the Next Iteration

        if (batch_idx + 1) % 20 == 0:
            print(f"  Step [{batch_idx+1}/{len(train_loader)}], Total Loss: {total_loss.item():.4f} "
                  f"(Class Loss: {loss_class.item():.4f}, BBox Loss: {loss_bbox.item():.4f})")

    # ==========================================
    # 6. TESTING AND ACCURACY ASSESSMENT
    # ==========================================
    print("\n--- Evaluating Multi-Task Network ---")
    model.eval()
    
    correct_classes = 0
    total_samples = 0
    accumulated_bbox_distance = 0.0

    with torch.no_grad():
        for images, class_labels, bbox_targets in test_loader:
            images = images.to(device)
            class_labels = class_labels.to(device)
            bbox_targets = bbox_targets.to(device)

            pred_logits, pred_bboxes = model(images)
            
            # 1. Assess classification accuracy
            _, predicted_classes = torch.max(pred_logits.data, 1)
            total_samples += class_labels.size(0)
            correct_classes += (predicted_classes == class_labels).sum().item()

            # 2. Assess regression distance error
            # Calculates the standard L1 distance (average difference per coordinate)
            # Predicted coordinates for a whole 1 batch of images (e.g., 32 images, each having 4 coordinates = 128 total numbers).Averages those across all coordinates
            distance_error = torch.mean(torch.abs(pred_bboxes - bbox_targets))
            # Find the overall average error across the entire test set, all batches. size(0) -> Batch Size
            accumulated_bbox_distance += distance_error.item() * class_labels.size(0) 

    avg_bbox_error = accumulated_bbox_distance / total_samples
    print(f"Classification Accuracy:           {100 * correct_classes / total_samples:.2f}%")
    print(f"Average BBox Coordinate Deviation: {avg_bbox_error:.4f} (On a 0.0 to 1.0 normalized grid)")

if __name__ == "__main__":
    main()
