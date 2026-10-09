"""
13. Deep Convolutional Generative Adversarial Network (DCGAN) with PyTorch

A Generative Adversarial Network (GAN) is a machine learning framework that uses two neural networks 
competing against each other . 
a.The Generator network creates new data instances. 
  The goal is to produce data that is fake but indistinguishable from real data. 
b.The Discriminator network evaluates them for authenticity.
   It aims to correctly identify whether the data is real or generated.

A Deep Convolutional Generative Adversarial Network (DCGAN) is an extension of the standard 
Generative Adversarial Network (GAN) that uses Convolutional Neural Networks (CNNs) in 
both the generator and discriminator to synthesize realistic images from random noise. 
No more flattening: It keeps images in their natural 2D shapes.

This script demonstrates how to construct and optimize a DCGAN. We train on pre-downloaded FashionMNIST images.
Some other data types we can use are Photos / Faces, Fashion & Products, Medical Imaging.

This code takes completely random numerical noise and transforms it into 
realistic, brand-new grayscale images of clothing items (from FashionMNIST) 
that look like real product photos, even though they were generated entirely by AI. 

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
        # self.main is an instance attribute containing a callable object. The input tensor z and automatically passes it through every layer
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1 -> 128 x 7 x 7
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels=128, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 128 x 7 x 7 -> 64 x 14 x 14
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 64 x 14 x 14 -> out_channels x 28 x 28
            nn.ConvTranspose2d(in_channels=64, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            # Tanh activation function squashes output pixel channels to [-1.0, 1.0] range
            #Documentation: https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html
            nn.Tanh()
        )

    def forward(self, z):
        # Ensure input noise has spatial shape: [Batch, Latent_Dim, 1, 1]
        # It takes a 2D batch of latent noise vectors z (usually shaped [Batch Size, Latent Dimension]) 
        # and reshapes it into a 4D tensor of shape [Batch Size, Latent Dimension, 1, 1].
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
        # self.main is an instance attribute containing a callable object. The input tensor z and automatically passes it through every layer
        self.main = nn.Sequential(
            # Input: in_channels x 28 x 28 -> 32 x 14 x 14
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            # LeakyReLU prevents zero gradient bottlenecks on inactive units
            # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32 x 14 x 14 -> 64 x 7 x 7
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64 x 7 x 7 -> 1 x 1 x 1
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, stride=1, padding=0, bias=False),
            # Sigmoid outputs a probability score between 0 (fake) and 1 (real)
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.main(img)
        # Flatten outputs to single scores [Batch, 1] -> a 2D tensor of shape [Batch Size, Total Features]
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

    # Initialize weights. An industry-standard heuristic that provides a balanced balance between capacity and trainability 
    latent_dim = 100
    # The Generator (netG) — "The Forger": 
    # Tries to paint fake images of clothing items (like shoes, shirts, or dresses) to trick the discriminator
    netG = Generator(latent_dim=latent_dim).to(device)
    # The Discriminator (netD) — "The Art Inspector": 
    # Looks at real images from the dataset and fake images created by the Generator. Outputs a score, 0.0 (Fake) to 1.0 (Real).
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
        # When you show Discriminator a real image, you compare its guess to label_real (1). 
        # If it guesses 0.9, it gets an 'A'. If it guesses 0.2, it fails.
       
        lossD_real = criterion(output_real, label_real)
        
        # Train on fake images generated from random noise vectors
        noise = torch.randn(batch_size, latent_dim).to(device)
        fake_images = netG(noise)
        output_fake = netD(fake_images.detach())
        # When you show it an AI-generated image, you compare its guess to label_fake (0).
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

        # ==========================================
        # CALCULATE DISCRIMINATOR ACCURACY & METRICS
        # ==========================================
        # Average confidence scores (0.0 to 1.0)
        D_x = output_real.mean().item()           # How real the Discriminator thinks real images are
        D_G_z1 = output_fake.mean().item()         # How real the Discriminator thinks fake images are
        
        # Calculate percentage accuracy
        acc_real = (output_real > 0.5).float().mean().item() * 100
        acc_fake = (output_fake < 0.5).float().mean().item() * 100
        d_acc = (acc_real + acc_fake) / 2.0

        # Calculate and print Real Accuracy and Fake Accuracy

        if (batch_idx + 1) % 10 == 0:
            print(f"\nStep [{batch_idx+1}/30] | Loss_D: {lossD.item():.4f} | Loss_G: {lossG.item():.4f} "
                  f"Real Acc: {acc_real:.1f}% | Fake Acc: {acc_fake:.1f}% | "
                  f"Avg D(x): {D_x:.2f} | Avg D(G(z)): {D_G_z1:.2f}")
            # print(f"  Step [{batch_idx+1}/30] | Loss_D: {lossD.item():.4f} | Loss_G: {lossG.item():.4f}")
    
    print("DCGAN validation complete. Networks compiled and executed without errors!")

if __name__ == "__main__":
    main()

"""
    === Example ===
    Discriminator Performance Metrics:
        - Real Acc (85.2%): The Discriminator correctly classified 85.2% of real images 
        as real (score > 0.5).
        - Fake Acc (82.0%): The Discriminator correctly identified 82.0% of generated fake 
        images as fake (score < 0.5).
        - Avg D(x) (0.81): On average, the Discriminator gives real images an 81% 
        probability score of being real.
        - Avg D(G(z)) (0.18): On average, the Discriminator gives generated images only an 
        18% chance of being real (meaning it successfully catches fakes).

    === Training Dynamics ===
        If Fake Acc drops close to 0% early on, the Discriminator is failing. 
        If Fake Acc stays at 100% forever without dropping as epochs progress, 
        the Generator is failing to improve.

    === SUMMARY & INTERPRETATION ===
    - Both networks are learning cleanly! Loss_D is hovering around ~0.8 to ~1.3, 
    which indicates a balanced game where neither network is completely crushing the other.
    - Early Steps (10-40): The Discriminator easily caught fakes (Fake Acc ~98%).
    - Middle/Late Steps (50-100): The Generator improved, pushing Avg D(G(z)) up to ~0.50 
    and dropping Discriminator accuracy to ~66%. This shows the Generator is learning 
    to produce much more convincing clothing images.
        
"""