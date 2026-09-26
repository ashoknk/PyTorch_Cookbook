"""
26. Parameter-Efficient Fine-Tuning (LoRA from Scratch) in PyTorch

This script demonstrates Low-Rank Adaptation (LoRA), a popular Parameter-Efficient 
Fine-Tuning (PEFT) methodology. LoRA freezes pre-trained weight matrices and injects 
pairs of trainable low-rank decomposition matrices (A and B) into linear layers. 
This dramatically reduces trainable parameter counts, permitting large-model 
fine-tuning on consumer hardware.

Learning Objectives:
1. Implement a custom LoraLinear layer wrapper from scratch.
2. Inject LoRA adapters into an existing neural network architecture.
3. Freeze base network weights while leaving only adapter parameters trainable.
4. Audit and compare trainable versus frozen parameter counts.
"""

import math

# We import the core torch library.
import torch

# nn contains the neural network layers (Linear) and parameters.
import torch.nn as nn

# ==========================================
# 1. IMPLEMENT CUSTOM LORALINEAR LAYER
# ==========================================
class LoraLinear(nn.Module):
    def __init__(self, base_layer, r=8, alpha=16):
        """
        Args:
            base_layer (nn.Linear): The pre-trained linear layer to wrap and freeze.
            r (int): The rank of the low-rank decomposition (r << in_features).
            alpha (int): The scaling hyperparameter.
        """
        super(LoraLinear, self).__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        # Scaling factor: alpha / r stabilizes adapter weight updates
        self.scaling = alpha / r
        
        # Freeze the pre-trained base linear layer's weights
        for param in self.base_layer.parameters():
            param.requires_grad = False
            
        # Initialize Low-Rank matrices: A and B
        # Matrix A maps: in_features -> r (initialized with Gaussian values)
        # Matrix B maps: r -> out_features (initialized with zeros, so adapter starts as identity 0)
        self.lora_A = nn.Parameter(torch.zeros(base_layer.in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, base_layer.out_features))
        
        # Standard Gaussian/Kaiming initialization of matrix A
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Matrix B is zero-initialized so the adapter initially adds nothing to the base layer's outputs
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # 1.1 Compute the standard pre-trained base layer's forward pass
        base_output = self.base_layer(x)
        
        # 1.2 Compute the adapter's low-rank pathway: (x * A) * B * scaling
        # We do (x @ A) @ B to project the dimensions efficiently
        adapter_output = (x @ self.lora_A) @ self.lora_B * self.scaling
        
        # 1.3 Combine both outputs (additive decomposition)
        return base_output + adapter_output

# ==========================================
# 2. INJECT ADAPTERS AND FREEZE BASE WEIGHTS
# ==========================================
class PretrainedClassifier(nn.Module):
    def __init__(self):
        super(PretrainedClassifier, self).__init__()
        # Simulate a pre-trained network with multiple linear layers
        self.fc1 = nn.Linear(10, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def main():
    print("--- 1. Initializing Pre-trained Base Classifier ---")
    model = PretrainedClassifier()
    print(model)

    # Calculate pre-trained parameter statistics
    total_base_params = sum(p.numel() for p in model.parameters())
    print(f"Total Base Parameters: {total_base_params:,}")

    # ==========================================
    # 3. APPLY LORA ADAPTER INJECTION
    # ==========================================
    print("\n--- 2. Injecting LoRA Adapters into fc1 & fc2 layers ---")
    # Wrap self.fc1 and self.fc2 in our LoraLinear adapters
    model.fc1 = LoraLinear(model.fc1, r=4, alpha=8)
    model.fc2 = LoraLinear(model.fc2, r=4, alpha=8)
    print(model)

    # ==========================================
    # 4. AUDIT TRAINABLE PARAMETERS
    # ==========================================
    print("\n--- 3. Auditing Trainable vs. Frozen Parameters ---")
    
    # Count trainable and frozen parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    print(f"Trainable Parameters (adapters only): {trainable_params:,}")
    print(f"Frozen Parameters (base network only): {frozen_params:,}")
    print(f"Adapter parameter density: {100 * trainable_params / (trainable_params + frozen_params):.2f}%")

    # Verify that only the adapters (lora_A, lora_B) have requires_grad=True
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert "lora_A" in name or "lora_B" in name, f"Unexpected trainable parameter: {name}"

    # ==========================================
    # 5. RUN TRAINING VERIFICATION EPOCH
    # ==========================================
    print("\n--- 4. Running Verification Optimization Epoch ---")
    # Generate synthetic training batch
    X = torch.randn(5, 10)
    y = torch.randint(0, 2, (5,))
    
    criterion = nn.CrossEntropyLoss()
    # Ensure our optimizer ONLY receives the trainable parameters of the adapters!
    trainable_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_list, lr=0.01)

    # Capture initial weights of frozen fc1 to confirm they do NOT change
    initial_fc1_weight = model.fc1.base_layer.weight.clone().detach()

    # Perform a single optimization step
    outputs = model(X)
    loss = criterion(outputs, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"  Single step Cross Entropy Loss: {loss.item():.4f}")

    # Check that frozen weights are unchanged
    post_fc1_weight = model.fc1.base_layer.weight.clone().detach()
    assert torch.equal(initial_fc1_weight, post_fc1_weight), "Frozen base weights were altered!"
    print("  Assertion Verified: Pre-trained base weights remained completely frozen.")
    print("PEFT LoRA script compiled and verified successfully!")

if __name__ == "__main__":
    main()
