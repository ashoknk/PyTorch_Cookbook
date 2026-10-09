"""
15. Neural Style Transfer (NST) with PyTorch

What is Neural Style Transfer (NST)?
NST is an AI-powered art filter.
Unlike a simple color tint or Instagram filter, NST uses a deep neural network (VGG-19) to look at two separate things:
    a.What is in the picture? (Objects, shapes, edges—extracted from higher layers of VGG).
    b.How does the painting look? (Colors, textures, brushstroke patterns—captured via Gram Matrices across multiple layers).
It then optimizes a blank canvas image pixel-by-pixel until the content matches image #1 and the texture matches image #2.

Situations and Data Types to Use NST:
    Digital Art & Design: Turning real photographs into oil paintings, sketches, or pop-art graphics.
    Gaming & Animation: Transferring texture styles to 3D game models or environment concept art.

This script demonstrates how to perform Neural Style Transfer (NST). NST extracts 
intermediate features from a pre-trained CNN (VGG-19) to blend the semantic 
layout of a "Content" image with the artistic style of a "Style" image. We 
implement content and style loss metrics, construct Gram matrices, and optimize 
an output image using the L-BFGS optimizer.

Learning Objectives:
1. Extract intermediate activations using a pre-trained VGG-19 model.
2. Build Gram Matrices to capture spatial style textures and distributions.
3. Construct composite loss functions combining content matching and style transfer.
4. Perform backpropagation directly onto pixel values of a target image using L-BFGS.
"""

# We import the core torch library.
import torch

# Import ssl to bypass certificate issues on macOS when downloading weights
import ssl
import os
ssl._create_default_https_context = ssl._create_unverified_context
from PIL import Image
import torchvision.transforms as transforms

# nn contains the activations and loss criteria.
import torch.nn as nn

# optim contains L-BFGS, ideal for image-pixel optimization.
import torch.optim as optim

# torchvision contains pre-trained VGG-19 models and visual weights.
import torchvision.models as models
import torchvision.utils as vutils

# ==========================================
# 1. DEFINE VGG FEATURE EXTRACTOR
# ==========================================
class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        # Load a pre-trained VGG-19 model
        # Documentation: https://pytorch.org/vision/stable/models/generated/torchvision.models.vgg19.html
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        
        # We slice VGG-19 layers. We select 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1' 
        # for style representation, and 'conv4_2' for content representation.
        self.style_layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '28': 'conv5_1'}
        self.content_layers = {'21': 'conv4_2'}
        
        # Build self-contained feature sub-networks up to conv5_1 (index 29)
        self.features = vgg[:30]
        # Freeze VGG parameters since we only use it as a static feature extractor
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        """Extracts content and style activations across key intermediate layers."""
        style_features = {}
        content_features = {}
        
        # Feed inputs sequentially layer-by-layer
        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in self.style_layers:
                style_features[self.style_layers[name]] = x
            if name in self.content_layers:
                content_features[self.content_layers[name]] = x
                
        return content_features, style_features

# ==========================================
# 2. DEFINE GRAM MATRIX UTILITY
# ==========================================
def get_gram_matrix(tensor):
    """Computes the Gram Matrix of a layer's activations.
    The Gram Matrix measures the spatial correlations between channel feature maps, 
    capturing artistic textures independent of visual geometry."""
    # tensor shape: [1, Channels, Height, Width]
    b, c, h, w = tensor.size()
    # Flatten spatial height and width
    features = tensor.view(b * c, h * w)
    # Compute outer product: Gram = F * F^T
    # Documentation: https://pytorch.org/docs/stable/generated/torch.mm.html
    gram = torch.mm(features, features.t())
    # Normalize by dividing by total activation volume size to prevent huge loss scores
    return gram.div(b * c * h * w)

def load_image(image_path, image_size=(224, 224)):
    """Opens a real image file and transforms it into a 4D PyTorch tensor."""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    if os.path.exists(image_path):
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0)
    else:
        print(f"Warning: '{image_path}' not found! Generating fallback synthetic image file...")
        # Auto-create the directory if missing
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        # Create a dummy image tensor [1, 3, 224, 224]
        dummy_tensor = torch.rand(3, 224, 224)
        # Convert tensor to PIL image and save to disk
        pil_img = transforms.ToPILImage()(dummy_tensor)
        pil_img.save(image_path)
        
        return dummy_tensor.unsqueeze(0)

def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Running Neural Style Transfer on: {device} ---")

    # ==========================================
    # 3. INITIALIZE CONTENT, STYLE, AND CANVAS
    # ==========================================
    # For self-contained, offline education, we generate synthetic RGB images of size 224x224.
    # content_img = torch.randn(1, 3, 224, 224).to(device)
    # style_img = torch.randn(1, 3, 224, 224).to(device)
    # Pass real image file paths here
    content_img = load_image("data/cats/cat1.jpg").to(device)
    style_img = load_image("data/art_style4.jpg").to(device)
    
    # We initialize our target output canvas directly as a clone of the content image.
    # We set requires_grad = True on the target canvas, telling PyTorch that we want 
    # to update the visual pixel values themselves during optimization!
    target_canvas = content_img.clone().requires_grad_(True).to(device)

    # Initialize VGG extractor and extract source features
    extractor = VGGFeatureExtractor().to(device)
    content_targets, _ = extractor(content_img)
    _, style_targets = extractor(style_img)
    
    # Compute Gram Matrices for style targets
    style_grams = {layer: get_gram_matrix(feat) for layer, feat in style_targets.items()}

    # ==========================================
    # 4. OPTIMIZATION LOOP WITH L-BFGS
    # ==========================================
    # We use L-BFGS because it is a second-order optimization method that yields 
    # much cleaner and faster visual style convergence compared to Adam or SGD.
    # Documentation: https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html
    optimizer = optim.LBFGS([target_canvas])
    
    # Define weighting factors
    content_weight = 1e3
    style_weight = 1e6
    
    epochs = 5
    print(f"--- Optimizing canvas for {epochs} L-BFGS iterations ---")
    
    for epoch in range(epochs):
        # Variable to store the current loss calculated inside the closure
        current_step_loss = [0.0]

        # L-BFGS requires a "closure" function that re-evaluates the model 
        # and computes the loss several times per optimizer step.
        def closure():
            optimizer.zero_grad()
            
            # Constrain target canvas pixels to valid visual bounds [-3.0, 3.0]
            target_canvas.data.clamp_(-3.0, 3.0)
            
            # Extract current target features
            curr_content, curr_style = extractor(target_canvas)
            
            # Compute Content Loss
            c_loss = 0.0
            for layer in content_targets.keys():
                c_loss += nn.functional.mse_loss(curr_content[layer], content_targets[layer])
                
            # Compute Style Loss
            s_loss = 0.0
            for layer, target_gram in style_grams.items():
                curr_gram = get_gram_matrix(curr_style[layer])
                s_loss += nn.functional.mse_loss(curr_gram, target_gram)
                
            # Combine losses
            total_loss = content_weight * c_loss + style_weight * s_loss
            total_loss.backward()

            # Record loss value for progress tracking
            current_step_loss[0] = total_loss.item()
            return total_loss
            
        optimizer.step(closure)
        
        # Print progress with the total loss value
        print(f"  Step [{epoch+1}/{epochs}] | Total Loss: {current_step_loss[0]:,.2f}")

    # Clamping final output
    final_img = target_canvas.detach().cpu().clamp_(0, 1)
    print(f"Visual optimization complete. Output shape: {final_img.shape}")


    # ==========================================
    # 5. SAVE INDIVIDUAL IMAGES (ORIGINAL & STYLED)
    # ==========================================
    print("\n--- Saving Output Images ---")
    
    # Ensure destination directory exists
    output_dir = "./data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare and clamp pixel values to valid [0.0, 1.0] visual range
    content_display = content_img.detach().cpu().clamp(0, 1)
    final_img = target_canvas.detach().cpu().clamp(0, 1)
    
    # Define distinct file paths
    orig_path = os.path.join(output_dir, "ns_original.png")
    result_path = os.path.join(output_dir, "ns_style_transfer_result.png")
    
    # Save as separate files
    vutils.save_image(content_display, orig_path)
    vutils.save_image(final_img, result_path)
    
    print(f"Saved original image to: '{orig_path}'")
    print(f"Saved style transfer result to: '{result_path}'")

if __name__ == "__main__":
    main()
