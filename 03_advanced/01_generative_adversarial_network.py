"""
13. Deep Convolutional Generative Adversarial Network (DCGAN) with PyTorch

This script demonstrates how to construct and optimize a Deep Convolutional 
Generative Adversarial Network (DCGAN). GANs use a zero-sum minimax game between 
two competing modules: a Generator that synthesizes images from random noise 
vectors, and a Discriminator that classifies inputs as authentic (real) or 
fabricated (fake). We train on pre-downloaded FashionMNIST images.

Learning Objectives:
1. Construct a Generator utilizing spatial transpose convolutions (ConvTranspose2d).
2. Build a Discriminator classifier utilizing strided convolution filters.
3. Coordinate stable adversarial minimax training loops using Binary Cross Entropy (BCE) Loss.
"""

# We import the core torch library.
import torch

# nn houses structural modules: Conv2d, ConvTranspose2d, BatchNorm2d, LeakyReLU, Sigmoid, Tanh.
# Documentation: https://pytorch.org/docs/stable/nn.html
import torch.nn as nn

# optim contains optimization algorithms.
import torch.optim as optim

# torchvision transforms and data utilities.
import torchvision.transforms as transforms
import torchvision

# ==========================================
# 1. DEFINE GENERATOR ARCHITECTURE
# ==========================================
class Generator(nn.Module):
    def __init__(self, latent_dim=100, out_channels=1):
        super(Generator, self).__init__()
        # ConvTranspose2d layers (deconvolutions) project low-resolution inputs 
        # to higher resolutions. We map a 1x1 noise vector of size latent_dim to a 28x28 image.
        # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1 -> 128 x 7 x 7
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 128 x 7 x 7 -> 64 x 14 x 14
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 64 x 14 x 14 -> out_channels x 28 x 28
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            # Tanh activation squashes output pixel channels to [-1.0, 1.0] range
            nn.Tanh()
        )

    def forward(self, z):
        # Ensure input noise has spatial shape: [Batch, Latent_Dim, 1, 1]
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.main(z)

# ==========================================
# 2. DEFINE DISCRIMINATOR ARCHITECTURE
# ==========================================
class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()
        # Strided Conv2d layers map a 28x28 image to a single probability confidence score.
        # We do not use max pooling here; downsampling is handled by stride=2.
        # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        self.main = nn.Sequential(
            # Input: in_channels x 28 x 28 -> 32 x 14 x 14
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1, bias=False),
            # LeakyReLU prevents zero gradient bottlenecks on inactive units
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32 x 14 x 14 -> 64 x 7 x 7
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64 x 7 x 7 -> 1 x 1 x 1
            nn.Conv2d(64, 1, kernel_size=7, stride=1, padding=0, bias=False),
            # Sigmoid outputs a probability score between 0 (fake) and 1 (real)
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.main(img)
        # Flatten outputs to single scores [Batch, 1]
        return out.view(out.size(0), -1)

def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Running DCGAN on: {device} ---")

    # ==========================================
    # 3. PREPARE CACHED DATASET
    # ==========================================
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Scale to [-1.0, 1.0] matching Generator Tanh
    ])
    # Load pre-downloaded dataset
    dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    # Initialize weights
    latent_dim = 100
    netG = Generator(latent_dim=latent_dim).to(device)
    netD = Discriminator().to(device)

    # Setup loss and optimizers
    # We use standard Binary Cross Entropy (BCE) Loss for minimax classification
    # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # ==========================================
    # 4. ADVERSARIAL MINIMAX OPTIMIZATION LOOP
    # ==========================================
    # We run 30 mini-batches for quick structural validation
    print("--- Starting Minimax Optimization Steps ---")
    netG.train()
    netD.train()
    
    for batch_idx, (real_images, _) in enumerate(dataloader):
        if batch_idx >= 30:
            break
            
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        
        # 4.1 Update Discriminator (netD): maximize log(D(x)) + log(1 - D(G(z)))
        # Create ground-truth labels for real images (1) and fake images (0)
        label_real = torch.ones(batch_size, 1).to(device)
        label_fake = torch.zeros(batch_size, 1).to(device)
        
        # Train on real images
        output_real = netD(real_images)
        lossD_real = criterion(output_real, label_real)
        
        # Train on fake images generated from random noise vectors
        noise = torch.randn(batch_size, latent_dim).to(device)
        fake_images = netG(noise)
        output_fake = netD(fake_images.detach())
        lossD_fake = criterion(output_fake, label_fake)
        
        lossD = lossD_real + lossD_fake
        optimizerD.zero_grad()
        lossD.backward()
        optimizerD.step()
        
        # 4.2 Update Generator (netG): maximize log(D(G(z)))
        # We pass fake images to netD again, but this time with label_real (1) targets
        # to calculate how well the generator "fooled" the discriminator.
        output_generator = netD(fake_images)
        lossG = criterion(output_generator, label_real)
        
        optimizerG.zero_grad()
        lossG.backward()
        optimizerG.step()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Step [{batch_idx+1}/30] | Loss_D: {lossD.item():.4f} | Loss_G: {lossG.item():.4f}")

    print("DCGAN validation complete. Networks compiled and executed without errors!")

if __name__ == "__main__":
    main()
