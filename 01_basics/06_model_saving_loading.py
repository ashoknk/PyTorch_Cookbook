"""
06. Model Saving, Loading, and Checkpointing in PyTorch

This script demonstrates best-practice routines for serializing and resuming 
model training. It covers saving and loading raw weights (state_dict), saving 
and loading comprehensive training checkpoints (including optimizer states, 
epochs, and historic losses) to resume training after interruptions, and cross-device 
loading (e.g., loading GPU models onto CPUs).

Learning Objectives:
1. Save and load raw model parameter weights using PyTorch state_dict dictionary.
2. Build comprehensive checkpoints representing complete paused training sessions.
3. Reload weights across different computing devices (e.g., CUDA to CPU).
"""

import os

# We import the core torch library.
import torch

# nn holds model layers and container classes.
import torch.nn as nn

# optim holds optimizer definitions and their internal states.
import torch.optim as optim

# ==========================================
# 1. DEFINE A SIMPLE TOY MODEL
# ==========================================
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        # A simple linear model for demonstration
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

def main():
    print("--- 1. Initializing Model & Optimizer ---")
    model = SimpleMLP()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Let's inspect the weights before any changes to verify later
    original_bias = model.fc.bias.clone().detach()
    print(f"Original bias weights: {original_bias}")

    # ==========================================
    # 2. SAVING & LOADING WEIGHTS ONLY (STATE_DICT)
    # ==========================================
    print("\n--- 2. Saving State Dict (Weights Only) ---")
    os.makedirs("./models", exist_ok=True)
    weights_path = "./models/model_weights.pth"
    
    # A state_dict is a Python dictionary mapping each layer name to its parameter tensor.
    # We save only this dictionary for production deployment as it is highly efficient.
    # Documentation: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    torch.save(model.state_dict(), weights_path)
    print(f"Saved state_dict to {weights_path}")

    # To restore: create an instance of the exact same network structure first
    new_model = SimpleMLP()
    # Then load the saved weights mapping into the structure
    new_model.load_state_dict(torch.load(weights_path))
    print("Successfully reloaded state_dict into new model instance.")

    # ==========================================
    # 3. SAVING & LOADING FULL CHECKPOINTS
    # ==========================================
    print("\n--- 3. Saving Complete Training Checkpoint ---")
    checkpoint_path = "./models/checkpoint.tar"
    
    # We modify model weights to simulate a dummy training step
    with torch.no_grad():
        model.fc.bias.fill_(5.0)
    print(f"Simulated updated bias weights: {model.fc.bias}")

    # A complete checkpoint should contain model state, optimizer state, current epoch, and loss
    # so we can fully reconstruct the training state.
    checkpoint = {
        'epoch': 42,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': 0.1234
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved full checkpoint to {checkpoint_path}")

    # To restore and resume training:
    resume_model = SimpleMLP()
    resume_optimizer = optim.SGD(resume_model.parameters(), lr=0.01)
    
    # Read the checkpoint dictionary from disk
    loaded_checkpoint = torch.load(checkpoint_path)
    
    # Map the stored state dictionaries back to their active containers
    resume_model.load_state_dict(loaded_checkpoint['model_state_dict'])
    resume_optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
    epoch = loaded_checkpoint['epoch']
    loss = loaded_checkpoint['loss']
    
    print("Checkpoint restored successfully:")
    print(f"  Resumed Epoch: {epoch}")
    print(f"  Resumed Bias: {resume_model.fc.bias}")
    print(f"  Resumed Saved Loss: {loss}")

    # ==========================================
    # 4. SAVING AND LOADING ACROSS DEVICES
    # ==========================================
    print("\n--- 4. Cross-Device Loading (GPU to CPU) ---")
    
    # Suppose a model was trained on a GPU and saved. If we load it on a machine
    # with only a CPU, we must specify map_location=torch.device('cpu') to map
    # storage pointers to the CPU memory layout.
    # Documentation: https://pytorch.org/docs/stable/generated/torch.load.html
    cpu_device = torch.device('cpu')
    loaded_state_dict = torch.load(weights_path, map_location=cpu_device)
    cpu_model = SimpleMLP()
    cpu_model.load_state_dict(loaded_state_dict)
    print("Loaded model saved on accelerator device directly onto CPU.")

    # Clean up serialized models
    os.remove(weights_path)
    os.remove(checkpoint_path)
    os.rmdir("./models")
    print("\n--- Cleaned up saved model files ---")

if __name__ == "__main__":
    main()
