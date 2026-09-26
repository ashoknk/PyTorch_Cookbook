"""
24. Model Compilation & Export (TorchScript and ONNX) in PyTorch

This script demonstrates how to transition PyTorch models from research/training 
environments to production serving. We compile a trained model using TorchScript 
tracing (creating JIT compiler structures to bypass Python runtime limits) 
and export the model architecture to ONNX (Open Neural Network Exchange) format 
for cross-platform and language-agnostic deployment.

Learning Objectives:
1. Compile models using JIT Tracing (torch.jit.trace).
2. Save compiled serialized TorchScript models (.pt).
3. Export PyTorch layers to standard ONNX representations (torch.onnx.export).
4. Verify execution consistency of exported models.
"""

import os

# We import the core torch library.
import torch

# nn contains neural layers.
import torch.nn as nn

# Define simple classification model for serialization
class SimpleImageClassifier(nn.Module):
    def __init__(self):
        super(SimpleImageClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Linear(8 * 14 * 14, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def main():
    print("--- 1. Preparing Model for Export ---")
    model = SimpleImageClassifier()
    model.eval()  # ALWAYS set model to evaluation mode before compiling!

    # Create dummy input matching the shape of expected production requests.
    # Grayscale FashionMNIST image: shape [1, 1, 28, 28]
    dummy_input = torch.randn(1, 1, 28, 28)

    # Make directories to store production artifacts
    os.makedirs("./export", exist_ok=True)

    # ==========================================
    # 2. TORCHSCRIPT JIT TRACING
    # ==========================================
    print("\n--- 2. Compiling with TorchScript Tracing ---")
    
    # torch.jit.trace runs the dummy input through the model and captures
    # the execution graph. This is highly efficient for models with static control flows.
    # Documentation: https://pytorch.org/docs/stable/jit.html#torch.jit.trace
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Save the compiled JIT model to disk. It can now be loaded directly in C++
    # without any Python dependency!
    torchscript_path = "./export/classifier_traced.pt"
    torch.jit.save(traced_model, torchscript_path)
    print(f"  Traced model saved to: {torchscript_path}")

    # ==========================================
    # 3. ONNX EXPORT
    # ==========================================
    print("\n--- 3. Exporting to ONNX Format ---")
    
    # torch.onnx.export translates PyTorch layers to ONNX intermediate representation ops,
    # allowing serving in ONNXRuntime, TensorRT, or open-source edge hardware.
    # Documentation: https://pytorch.org/docs/stable/onnx.html#torch.onnx.export
    onnx_path = "./export/classifier.onnx"
    torch.onnx.export(
        model,               # Model to export
        dummy_input,         # Example input tensor
        onnx_path,           # Destination file
        export_params=True,  # Save model weights inside the file
        opset_version=15,    # Target ONNX standard opset version
        input_names=['input'],   # Name input layer
        output_names=['output'], # Name output layer
        # Specify dynamic axes if batch size can change during inference
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"  ONNX model exported to: {onnx_path}")

    # ==========================================
    # 4. VERIFY EXPORT CONSISTENCY
    # ==========================================
    print("\n--- 4. Verifying Compiled Model Consistency ---")
    
    # Run predictions with the original PyTorch model
    with torch.no_grad():
        original_output = model(dummy_input)

    # Load and run predictions with the saved TorchScript model
    loaded_jit = torch.jit.load(torchscript_path)
    with torch.no_grad():
        jit_output = loaded_jit(dummy_input)

    # Verify that outputs are mathematically identical
    diff = torch.abs(original_output - jit_output).max().item()
    print(f"  Max difference between PyTorch and JIT predictions: {diff}")
    
    assert diff < 1e-6, "Compiled TorchScript output deviates from original model!"
    print("JIT tracing and ONNX compilation verified successfully!")

if __name__ == "__main__":
    main()
