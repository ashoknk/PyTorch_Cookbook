"""
08 - 02a Simple Autoencoder & Image Denoising (Unsupervised Vision)

  Encoder-Decoder Architectures, Unsupervised Learning, Reconstruction Loss (nn.MSELoss), & Spatial Upsampling (nn.ConvTranspose2d).
  You can use this exact combination of techniques to build an unsupervised anomaly detection system 
  that automatically finds physical defects on manufactured products, 
  such as scratches on metal parts or cracks in microchips.

  02_intermediate/01_convolutional_neural_network.py teaches how to compress a 2D image down to a single class label.
  An Autoencoder teaches the natural next step: 
    How to compress an image into a low-dimensional bottleneck (the Encoder) and 
    then reconstruct the original image back from that bottleneck (the Decoder).
 Why the bottleneck is important:
    The autoencoder is forced to represent the image in a compressed form:(batch, 32, 7, 7)
    This is much smaller than the original: (batch, 1, 28, 28) = (batch, 784)
    The representation uses many channels but fewer spatial dimensions, 
    which captures important structure while reducing detail.
    
- You can add synthetic random noise to the FashionMNIST dataset and 
    train the network to "clean" (denoise) the clothing images. 
    The model is being asked:“If I give you a noisy clothes image, can you reconstruct the clean version?”

Learning Objectives:
1. Understand the bottleneck concept of autoencoders (unsupervised representation learning).
2. Learn how to reconstruct spatial grids using Transposed Convolutions (nn.ConvTranspose2d).
3. Use Mean Squared Error Loss (nn.MSELoss) to measure similarity between raw inputs and reconstructions.
4. Implement a denoising autoencoder that learns to remove added gaussian noise.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ==========================================
# 1. DEFINE DENOISING AUTOENCODER MODEL
# ==========================================
"""
An Autoencoder consists of two matched components:

    1. Encoder (Downsampling):
       Uses Conv2d layers to extract high-level feature maps while shrinking spatial sizes. 
       The final layer of the encoder forms the "bottleneck" or latent representation, 
       which represents the core, compressed information of the image.

    2. Decoder (Upsampling / Transposed Convolutions):
       Takes the compressed bottleneck vector and uses ConvTranspose2d layers to project 
       it back to the original image dimensions.
       
       How ConvTranspose2d (Transposed Convolution) works:
       Often colloquially called "deconvolution," it performs the reverse operation of Conv2d. 
       Instead of sliding a filter to reduce sizes, it associates each input value with a larger 
       kernel window output, scaling up spatial resolutions (e.g., from 7x7 back to 14x14 and then 28x28).
"""
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        
        # --- ENCODER ---

        # Documentation: https://docs.pytorch.org/docs/2.14/generated/torch.nn.Sequential.html
        self.encoder = nn.Sequential(
            # Input: (batch, 1, 28, 28)
            # Layer 1: 1 -> 16 channels, reduces size to 14x14 (stride=2)
            # After first conv with stride 2: 16 × 14 × 14
            # Documentation: https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html
            # Conv2D) is a two-dimensional convolutional layer to extract spatial features like edges, corners, and shapes from images
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # Layer 2: 16 -> 32 channels, reduces size to 7x7 (stride=2)
            # After second conv with stride 2: 32 × 7 × 7
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # --- DECODER ---
        self.decoder = nn.Sequential(
            # Input: (batch, 32, 7, 7)
            # Layer 1: 32 -> 16 channels, increases size back to 14x14
            # output_padding=1 ensures that the rounding is handled correctly to get exactly 14x14
            # ConvTranspose2d is a 2D transposed convolution operator that increases the spatial dimensions (height and width) of a feature map
            # Documentation: https://docs.pytorch.org/docs/2.14/generated/torch.nn.ConvTranspose2d.html
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # Layer 2: 16 -> 1 channel, increases size back to 28x28
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),
            # We use Tanh because our input data is normalized to the range [-1, 1].
            # Tanh limits outputs strictly between -1 and 1, matching our clean image targets.
            nn.Tanh()
        )

    def forward(self, x):
        # 1. Compress raw image into latent space representation
        latent = self.encoder(x)
        # 2. Reconstruct original image shape from the latent space
        # Then the model reconstructs from that latent representation
        reconstructed = self.decoder(latent)
        return reconstructed

def main():
    # Setup device accelerator (CUDA, MPS, or CPU)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Running Denoising Autoencoder on: {device} ---")

    # ==========================================
    # 2. PREPARE DATASET (FashionMNIST)
    # ==========================================
    print("--- Preparing FashionMNIST Dataset ---")
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Normalize to [-1, 1] range to match the Tanh output activation
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load pre-downloaded dataset if available, otherwise download
    # Documentation: https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html
    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    # ==========================================
    # 3. INITIALIZE AUTOENCODER MODEL & LOSS
    # ==========================================
    model = DenoisingAutoencoder().to(device)
    
    # We use MSELoss (Mean Squared Error) because we are performing a regression task: 
    # predicting individual pixel intensity values (0 to 1 or -1 to 1) to match the target.
    # Documentation: https://docs.pytorch.org/docs/2.14/generated/torch.nn.MSELoss.html
    # NOTE: Binary Cross-Entropy (BCE) loss for binary classification
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ==========================================
    # 4. TRAINING WITH ARTIFICIAL NOISE
    # ==========================================
    epochs = 1
    print(f"--- Training Autoencoder for {epochs} epoch ---")
    model.train()

    for batch_idx, (images, _) in enumerate(train_loader):
        # For autoencoders, we do not need the category labels! We are learning unsupervised representations.
        images = images.to(device)
        
        # Add random Gaussian noise to the clean images
        # torch.randn_like creates a tensor of random values (mean=0, std=1) with the same size as input images.
        # multiplying by 0.25 makes the noise smaller and more controlled
        # This creates training examples with corruption so the model learns denoising.
        noise = torch.randn_like(images) * 0.25
        noisy_images = images + noise
        # Clip values to ensure the noisy image remains within the normalized range [-1.0, 1.0]
        noisy_images = torch.clamp(noisy_images, -1.0, 1.0)

        # Forward pass: Feed NOISY images into autoencoder, compare output against CLEAN images
        reconstructed = model(noisy_images)
        # NOTE: Training target is the original clean image, not the label
        loss = criterion(reconstructed, images)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward() # Calculates the gradients for every weight and bias based on mistake/loss
        optimizer.step() # Updates the weights & the biases for the Next Iteration

        if (batch_idx + 1) % 300 == 0:
            print(f"  Step [{batch_idx+1}/{len(train_loader)}], Reconstruction Loss: {loss.item():.4f}")

    # ==========================================
    # 5. TESTING AND DIMENSIONAL VERIFICATION
    # ==========================================
    """This is unsupervised because:
        - there is no class label used
        - the network learns from the image itself
        - it tries to recreate the original image from a compressed version
    So the “representation learning” is:
        - encoder learns a compressed representation of the image
        - decoder learns to reconstruct it
        - both are trained together through the loss"""

    print("\n--- Verifying Autoencoder Output on Test Sample ---")
    model.eval()
    
    with torch.no_grad():
        # Grab a single batch from the test loader
        images, _ = next(iter(test_loader))
        images = images.to(device)
        
        # Add noise
        # This is just to check whether the trained model can reconstruct noisy test examples.
        noise = torch.randn_like(images) * 0.25
        noisy_images = torch.clamp(images + noise, -1.0, 1.0)
        
        # Forward pass
        reconstructed_images = model(noisy_images)
        
        print(f"Original Input Shape:      {images.shape}")
        print(f"Noisy Corrupted Shape:     {noisy_images.shape}")
        print(f"Reconstructed Shape:       {reconstructed_images.shape} (Expected matching input shape!)")
        
        assert reconstructed_images.shape == images.shape, "Reconstructed dimensions do not match raw inputs."
        print("Autoencoder dimensional and reconstruction flow verified successfully!")

if __name__ == "__main__":
    main()
