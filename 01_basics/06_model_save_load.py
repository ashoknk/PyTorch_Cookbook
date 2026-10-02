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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(DATA_DIR, "models")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "model_weights.pth")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "checkpoint.tar") 


def main():
    print("--- 1. Initializing Model & Optimizer ---")
    model = SimpleMLP()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Let's inspect the weights before any changes to verify later
    original_bias = model.fc.bias.clone().detach()
    print(f"Original bias weights: {original_bias} Type:{type(original_bias)}")
    # A detached copy of the bias is a regular torch.Tensor, not a trainable model parameter.

    # ==========================================
    # 2. SAVING & LOADING WEIGHTS ONLY (STATE_DICT)
    # ==========================================
    print("\n--- 2. Saving State Dict (Weights Only) ---")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # A state_dict is a Python dictionary mapping each layer name to its parameter tensor.
    # We save only this dictionary for production deployment as it is highly efficient.
    # Documentation: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"Saved state_dict to {WEIGHTS_PATH}")

    # To restore: create an instance of the exact same network structure first
    new_model = SimpleMLP()
    # Then load the saved weights mapping into the structure
    new_model.load_state_dict(torch.load(WEIGHTS_PATH))
    print("Successfully reloaded state_dict into new model instance.")
    
    # ==========================================
    # 3. SAVING & LOADING FULL CHECKPOINTS
    # ==========================================
    print("\n--- 3. Saving Complete Training Checkpoint ---")
    
    # We modify model weights to simulate a dummy training step
    with torch.no_grad():
        model.fc.bias.fill_(5.0)
    print(f"Simulated updated bias weights: {model.fc.bias} Type:{type(model.fc.bias)}")
    # Parameter is a special kind of tensor, tracks for training; displays includes requires_grad=True.

    # A complete checkpoint should contain model state, optimizer state, current epoch, and loss
    # so we can fully reconstruct the training state.
    checkpoint = {
        'epoch': 42,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': 0.1234
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"Saved full checkpoint to {CHECKPOINT_PATH}")

    # To restore and resume training:
    resume_model = SimpleMLP()
    resume_optimizer = optim.SGD(resume_model.parameters(), lr=0.01)
    
    # Read the checkpoint dictionary from disk
    loaded_checkpoint = torch.load(CHECKPOINT_PATH)
    
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
    loaded_state_dict = torch.load(WEIGHTS_PATH, map_location=cpu_device)
    cpu_model = SimpleMLP()
    cpu_model.load_state_dict(loaded_state_dict)
    print("Loaded model saved on accelerator device directly onto CPU.")

    # Clean up serialized models
    os.remove(WEIGHTS_PATH)
    os.remove(CHECKPOINT_PATH)
    os.rmdir(MODEL_DIR)
    print("\n--- Cleaned up saved model files ---")

if __name__ == "__main__":
    main()
