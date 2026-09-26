"""
17. Image Captioning Training Pipeline in PyTorch

This script demonstrates how to configure and execute a training pipeline for 
attention-based image captioning. It generates synthetic image tensors and 
associates them with tokenized captions, defines a cross-entropy training loop with 
sequence masking, and performs forward-backward passes across both the CNN Encoder 
and the recurrent Attention Decoder.

Learning Objectives:
1. Orchestrate complete multi-module visual-text training pipelines.
2. Structure custom training loops combining visual features and text sequences.
3. Apply CrossEntropyLoss over temporal sequence grids.
4. Verify gradient flows through both the Encoder and Decoder networks.
"""

# We import the core torch library.
import torch

# nn contains the activations and loss criteria.
import torch.nn as nn

# optim contains optimization modules (Adam).
import torch.optim as optim

# We import importlib to load modules with names starting with numbers.
# In standard Python, 'import 04_image_captioning_model' is invalid syntax.
# Documentation: https://docs.python.org/3/library/importlib.html
import importlib
model_module = importlib.import_module("03_advanced.04_image_captioning_model")
EncoderCNN = model_module.EncoderCNN
DecoderRNN = model_module.DecoderRNN

# Import ssl to prevent certificate verification errors on macOS model downloads
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Running Image Captioning Training on: {device} ---")

    # ==========================================
    # 1. SETUP VOCABULARY AND MODEL PARAMETERS
    # ==========================================
    # We define a tiny toy vocabulary mapping characters/tokens
    # 0: <PAD>, 1: <START>, 2: <END>, 3: "dog", 4: "runs", 5: "outside"
    vocab_size = 6
    embed_size = 256
    hidden_size = 256
    
    # Instantiate modular Encoder and Decoder
    encoder = EncoderCNN(embed_size=embed_size).to(device)
    decoder = DecoderRNN(embed_size=embed_size, vocab_size=vocab_size, hidden_size=hidden_size).to(device)

    # Initialize dual optimization criteria updating both models
    criterion = nn.CrossEntropyLoss()
    params = list(encoder.projection.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=0.01)

    # ==========================================
    # 2. RUN TRAINING STEPS ON TOY MULTIMODAL DATA
    # ==========================================
    print("--- Simulating Multimodal Training Epochs ---")
    
    # 2.1 Generate 2 synthetic RGB images of size 224x224
    synthetic_images = torch.randn(2, 3, 224, 224).to(device)
    
    # 2.2 Generate matching tokenized caption targets (representing sequence: "<START> dog runs outside <END>")
    # Captions shape: [Batch_Size, Seq_Len]
    synthetic_captions = torch.tensor([
        [1, 3, 4, 5, 2],
        [1, 3, 5, 4, 2]
    ], dtype=torch.long).to(device)

    encoder.train()
    decoder.train()
    
    epochs = 20
    for epoch in range(epochs):
        # Forward pass through Encoder CNN to extract spatial feature grids: [Batch, 49, embed_size]
        features = encoder(synthetic_images)
        
        # Forward pass through recurrent Decoder RNN with Attention: [Batch, Seq_Len, Vocab_Size]
        outputs = decoder(features, synthetic_captions)
        
        # We calculate Cross Entropy loss across the entire vocabulary over all sequence timesteps.
        # We reshape output logits to [Batch * Seq_Len, Vocab_Size] and targets to [Batch * Seq_Len].
        loss = criterion(outputs.view(-1, vocab_size), synthetic_captions.view(-1))
        
        # Backward optimization pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}], Cross-Entropy Loss: {loss.item():.4f}")

    # ==========================================
    # 3. VERIFICATION AND EVALUATION
    # ==========================================
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        features = encoder(synthetic_images)
        outputs = decoder(features, synthetic_captions)
        # Predict the most probable word token index at each sequence step
        predictions = torch.argmax(outputs, dim=-1)
        
        print("\nTraining Verification Results:")
        print(f"  Predicted Token Indices Batch 1: {predictions[0].tolist()}")
        print(f"  Ground Truth Token Indices Batch 1: {synthetic_captions[0].tolist()}")

if __name__ == "__main__":
    main()
