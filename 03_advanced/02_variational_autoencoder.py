"""
14. Variational Autoencoder (VAE) with PyTorch

A Variational Autoencoder (VAE) in PyTorch is a generative deep learning architecture that compresses 
input data into a continuous, probabilistic latent space and reconstructs it back into new data samples.

ex. a. Reconstructs an input image of a "7". 
    b. Compresses a photo of a person to save space. Generates non-existent human faces, or allows you to add glasses to an existing face by tweaking a single latent vector

VAE DATA FLOW ARCHITECTURE

    Input Image Tensor (images / x)
    │ [Shape: Batch, 1, 28, 28] -> Flattened to [Batch, 784]
    ▼
    1. The Encoder (self.encode)
    │
    ├─► self.fc_mu(h1) ──────► mu ─────┐
    │                                  ├─► (Feeds into kl_divergence calculation)
    └─► self.fc_logvar(h1) ──► logvar ─┘
    │
    ▼
    2. The Reparameterization Trick (self.reparameterize)
    │
    ├─► std = torch.exp(0.5 * logvar)       Standard Deviation (sigma)
    ├─► eps = torch.randn_like(std)         Epsilon (epsilon)
    └─► z = mu + eps * std  <── [Stochastic Sampling]       
    │   z = mean + (wiggle room * random noise)
    │   z is called Latent vector/Compressed fingerprint
    ▼
    Latent Coordinates Vector (z) [Shape: Batch, 20]
    │
    ▼
    3. The Decoder (self.decode)
    │
    └─► reconstructed / recon_x = self.sigmoid(self.fc4(h3))
    │
    ▼
    4. Dual-Loss Evaluation (loss_function)
    │
    ├─► recon_loss: Binary Cross Entropy (BCE) comparing recon_x ◄─► x_flat
    └─► kl_divergence: Statistical regularization on mu & logvar
        (Enforces latent distribution toward standard normal prior N(0, I))

This script demonstrates how to construct, train, and sample from a Variational 
Autoencoder (VAE) on the FashionMNIST dataset. VAEs model data distributions 
by mapping inputs to continuous, Gaussian-regularized latent spaces. This script 
implements the reparameterization trick, combines reconstruction (BCE) and Kullback-Leibler (KL) 
divergence losses, and generates new outputs by sampling from the learned prior.

Learning Objectives:
1. Build an Encoder network estimating latent Gaussian mean and variance.
2. Implement the Reparameterization Trick to permit backpropagation through stochastic variables.
3. Construct a Decoder network mapping latent vectors back to pixel grids.
4. Integrate a dual-loss objective combining reconstruction quality and statistical regularisation.
"""

import os
# We import the core torch library.
import torch

# nn contains the standard neural network layers (Linear, ReLU).
import torch.nn as nn

# optim contains optimizer engines.
import torch.optim as optim

# torchvision contains pre-downloaded FashionMNIST utilities.
import torchvision
import torchvision.transforms as transforms

# ==========================================
# 1. DEFINE VARIATIONAL AUTOENCODER MODULE
# ==========================================
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        
        # 1.1 Encoder Layer definitions
        # self.fc1 (784 -> 400): 
        # Compresses 784 raw image pixels into 400 hidden visual features (like edges, shapes, and contours).
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Latent Mu (mean) represents the central coordinate of the data sample in latent space
        # self.fc_mu (400 -> 20): 
        # Maps the 400 hidden features into 20 exact mean coordinates representing core visual traits (ex. shoe length, sleeve width, or color shade).
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        # Latent Logvar (log variance) represents the size of the uncertainty sphere around Mu
        # self.fc_logvar (400 -> 20): 
        # Maps the 400 hidden features into 20 uncertainty values that define how much variation or fuzziness is allowed around each visual trait.
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # 1.2 Decoder Layer definitions
        # Expansion & Feature Extraction: Maps 20 -> 400 (hidden_dim). It takes the compressed fingerprint coordinates 
        # and rebuilds mid-level visual features (like shapes, curves, and edges). 
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        # Pixel Reconstruction: Maps 400 -> 784 (input_dim). It takes those mid-level features 
        # and projects them onto the final 784 individual pixel values of the image
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        
        # ReLU sets negative values to zero (0), allowing the network to learn non-linear patterns 
        # like smooth clothing contours, folds, and outlines rather than just simple linear combinations.
        self.relu = nn.ReLU()
        # Squashes raw pixel logits from the final linear layer so every generated pixel matches the normalized intensity range of the input images.
        # # Sigmoid activation maps reconstruction pixels to [0.0, 1.0] matching normalized inputs
        # Input: 784 unconstrained real numbers -> Output: 784 probability values strictly between 0.0 and 1.0)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        """The Reparameterization Trick: samples from N(mu, var) by computing mu + eps * std.
        This shifts the stochastic node out of the backpropagation pathway."""
        # Calculate standard deviation: std = e^(0.5 * logvar)
        std = torch.exp(0.5 * logvar)
        print(f"Reparameterization: mu shape {mu.shape}, std shape {std.shape}")
        # Draw epsilon noise from standard normal distribution N(0, I)
        # This draws random noise epsilon to help sample points around an existing input image's mean (mu) during training
        eps = torch.randn_like(std)
        # Return scaled continuous coordinate
        return mu + eps * std

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        # Flatten image input [Batch, 1, 28, 28] to [Batch, 784]
        x_flat = x.view(x.size(0), -1)
        # Encode to latent parameters
        mu, logvar = self.encode(x_flat)
        # Sample coordinates
        z = self.reparameterize(mu, logvar)
        # Reconstruct pixel intensities
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

# ==========================================
# 2. DEFINE THE DUAL-LOSS FUNCTION
# ==========================================
def loss_function(recon_x, x, mu, logvar):
    """Calculates combined VAE loss: Reconstruction Loss + Kullback-Leibler (KL) Divergence."""
    # Reconstruction loss measures pixel-wise binary classification differences (BCE).
    # We flatten the input pixels first.
    x_flat = x.view(x.size(0), -1)
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x_flat, reduction='sum')
    
    # KL Divergence measures how close the learned latent distribution is to standard normal prior N(0, I).
    # Enforcing this prior prevents coordinates from drifting apart, ensuring smooth interpolations.
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_divergence

def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Running VAE training on: {device} ---")

    # ==========================================
    # 3. PREPARE CACHED DATASET
    # ==========================================
    # We only apply ToTensor (scaling to [0, 1]) without normalization to match the Sigmoid decoder
    transform = transforms.ToTensor()
    dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ==========================================
    # 4. RUN VAE TRAINING MODULE
    # ==========================================
    epochs = 1
    print(f"--- Training VAE for {epochs} epoch ---")
    model.train()
    
    # In standard supervised terms:
    # y_predicted = recon_batch (the reconstructed pixel values generated by the decoder)
    # y_target    = images      (the original ground-truth input image pixels)
    # The Reconstruction Loss computes Binary Cross Entropy (BCE) between y_predicted and y_target.
    for batch_idx, (images, _) in enumerate(dataloader):
        # We only train for 30 batches for rapid testing
        if batch_idx >= 30:
            break
            
        images = images.to(device)
        
        # Forward pass
        recon_batch, mu, logvar = model(images)
        # The loss_function evaluates VAE performance using a dual-comparison approach across its 4 arguments:
        # 1. Reconstruction Loss compares recon_batch against original images pixel-by-pixel to ensure visual quality.
        # 2. KL Divergence Loss compares mu and logvar against a standard normal distribution N(0, I) to keep the latent space smooth, continuous, and sampleable.
        loss = loss_function(recon_batch, images, mu, logvar)
        
        # Backward optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Step [{batch_idx+1}/30], Loss: {loss.item() / images.size(0):.4f}")
        
    # ==========================================
    # 5. SAMPLE GENERATION FROM LATENT PRIOR
    # ==========================================
    print("\n--- Generating New Image Sample by Sampling Prior N(0, I) ---")
    model.eval()
    with torch.no_grad():
        # Sample directly from the standard normal prior
        sample_z = torch.randn(1, 20).to(device)
        generated_pixel_probabilities = model.decode(sample_z)
        # Reshape flat outputs back to a 28x28 grayscale image tensor
        generated_image = generated_pixel_probabilities.view(1, 28, 28)
        print(f"Successfully generated visual tensor of shape: {generated_image.shape}")

    # ==========================================
    # 6. VISUALIZE ORIGINAL VS RECONSTRUCTED IMAGES
    # ==========================================
    print("\n--- Comparing Original vs. Reconstructed Images ---")
    
    # Ensure the 'data' directory exists
    output_dir = "./data"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Grab a small batch of real test images (e.g., 8 images)
    real_images, _ = next(iter(dataloader))
    real_images = real_images[:8].to(device)
    
    # 2. Pass them through the VAE to get reconstructed outputs
    model.eval()
    with torch.no_grad():
        reconstructed_images, _, _ = model(real_images) #ignore mu and logvar for reconstruction
        # Reshape flat reconstructions [8, 784] back to image grid [8, 1, 28, 28]
        reconstructed_images = reconstructed_images.view(-1, 1, 28, 28)
    
    # 3. Stack original (top row) and reconstructed (bottom row) together
    comparison = torch.cat([real_images, reconstructed_images])
    
    # 4. Save to the ./data folder
    save_path = os.path.join(output_dir, "vae_reconstruction_comparison.png")
    torchvision.utils.save_image(comparison, save_path, nrow=8)
    print(f"Saved side-by-side comparison image to '{save_path}'!")

if __name__ == "__main__":
    main()
