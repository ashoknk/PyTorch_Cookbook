"""
14. Variational Autoencoder (VAE) with PyTorch

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
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Latent Mu (mean) represents the central coordinate of the data sample in latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        # Latent Logvar (log variance) represents the size of the uncertainty sphere around Mu
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # 1.2 Decoder Layer definitions
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        
        self.relu = nn.ReLU()
        # Sigmoid activation maps reconstruction pixels to [0.0, 1.0] matching normalized inputs
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        """The Reparameterization Trick: samples from N(mu, var) by computing mu + eps * std.
        This shifts the stochastic node out of the backpropagation pathway."""
        # Calculate standard deviation: std = e^(0.5 * logvar)
        std = torch.exp(0.5 * logvar)
        # Draw epsilon noise from standard normal distribution N(0, I)
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
    
    for batch_idx, (images, _) in enumerate(dataloader):
        # We only train for 30 batches for rapid testing
        if batch_idx >= 30:
            break
            
        images = images.to(device)
        
        # Forward pass
        recon_batch, mu, logvar = model(images)
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

if __name__ == "__main__":
    main()
