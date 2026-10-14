"""
22 - 04. Modern Model Compilation & Export (torch.export and ONNX) in PyTorch

"I trained a model in PyTorch on Python File/Jupyter/Colab... 
how do I actually put this into a C++ app, a web server, a mobile app, or TensorRT?"

The main purpose of this code is to show how to take a trained PyTorch model 
out of Python so it can be deployed to real-world production environments—such as 
C++ apps, web servers, mobile devices, or hardware accelerators (like NVIDIA TensorRT) 

Use .pt2 when your entire deployment and execution stack remains inside the PyTorch ecosystem or relies on PyTorch-native compilation pipelines.
• PyTorch-Native Inference: 
You are running inference with PyTorch 2.x, torch.compile, or AOTInductor C++ wrappers

Use .onnx when you need to escape the Python/PyTorch runtime dependency and run your model across different frameworks or high-speed hardware backends.
• Cross-Platform / Cross-Framework Deployment: 
You need to run a model trained in PyTorch inside another framework (like C++, C#, Java, or TensorRT/DirectML environments).

Learning Objectives:
1. Capture PyTorch 2.x computational graphs into standalone artifacts using torch.export.export().
2. Save and reload compiled PyTorch 2.x exported models (.pt2) to disk.
3. Define dynamic shapes (dynamic_shapes) so exported models can handle flexible batch sizes in production.
4. Export neural network layers to cross-platform ONNX representations (torch.onnx.export).
5. Verify that predictions from exported models remain mathematically identical to the original PyTorch model.
"""

import os
import torch

# nn contains neural layers.
import torch.nn as nn
from torch.export import Dim
import onnxruntime as ort

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

FOLDER_EXPORT= "./data/export/"

def main():
    print("--- 1. Preparing Model for Export ---")
    model = SimpleImageClassifier()
    model.eval()  # ALWAYS set model to evaluation mode before compiling!

    # Create dummy input matching the shape of expected production requests.
    # Grayscale FashionMNIST image: shape [1, 1, 28, 28]
    dummy_input = torch.randn(1, 1, 28, 28)

    # Make directories to store production artifacts
    os.makedirs(FOLDER_EXPORT, exist_ok=True)

    # ==========================================
    # 2. TORCHSCRIPT TRACING. PT2 EXPORT
    # ==========================================
    print("\n--- 2. Compiling with TorchScript Tracing ---")
    
    # torch.export.export captures the graph structure of the model for PyTorch 2.x+
    exported_program = torch.export.export(model, (dummy_input,))
    
    # Save the exported model (.pt2) to disk
    export_path_pt2 = os.path.join(FOLDER_EXPORT, "classifier_exported.pt2")
    torch.export.save(exported_program, export_path_pt2)
    print(f"  Exported model saved to: {export_path_pt2}")

    # ==========================================
    # 3. ONNX EXPORT
    # ==========================================
    print("\n--- 3. Exporting to ONNX Format ---")
    
    # torch.onnx.export translates PyTorch layers to ONNX intermediate representation ops,
    # allowing serving in ONNXRuntime, TensorRT, or open-source edge hardware.
    # Documentation: https://pytorch.org/docs/stable/onnx.html#torch.onnx.export
    onnx_path = os.path.join(FOLDER_EXPORT, "classifier.onnx")

    # Define dynamic dimension symbol using PyTorch's export module
    # Passing dynamic_shapes tells the exporter explicitly not to hardcode those dimensions
    batch_dim = torch.export.Dim("batch_size")
    dynamic_shapes = {"x": {0: batch_dim}}

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        input_names=['input'],
        output_names=['output'],
        dynamic_shapes=dynamic_shapes
    )

    print(f"  ONNX model exported to: {onnx_path}")

    # ==========================================
    # 4. VERIFY EXPORT CONSISTENCY
    # ==========================================
    print("\n--- 4. Verifying Compiled Model Consistency ---")
    
    # 4.1 Run predictions with original PyTorch model
    with torch.no_grad():
        original_output = model(dummy_input)

    # ========= 4.2 Verify PyTorch 2.x Exported Model (.pt2). Load and run predictions with the saved torch.export model
    loaded_program = torch.export.load(export_path_pt2)
    loaded_module = loaded_program.module()
    with torch.no_grad():
        exported_output = loaded_module(dummy_input)

    # Verify that outputs are mathematically identical
    diff = torch.abs(original_output - exported_output).max().item()
    print(f"  Max difference between PyTorch and Exported predictions: {diff}")
    
    assert diff < 1e-6, "Exported model output deviates from original model!"
    print("Modern torch.export and ONNX compilation verified successfully!")

    # ========= 4.3 Verify ONNX Model (.onnx) using ONNX Runtime
    try:
        
        # Create ONNX Runtime session. 
        # ort.InferenceSession is the primary class used to load, configure, and execute compiled ONNX ML models
        ort_session = ort.InferenceSession(onnx_path)
        
        # ONNX Runtime requires inputs as NumPy arrays mapped to input layer names
        onnx_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        onnx_output = ort_session.run(None, onnx_inputs)[0]

        # Convert ONNX numpy array output to PyTorch tensor for comparison
        onnx_output_tensor = torch.from_numpy(onnx_output)
        
        onnx_diff = torch.abs(original_output - onnx_output_tensor).max().item()
        print(f"  Max difference (PyTorch vs. .onnx Export): {onnx_diff:.6f}")
        assert onnx_diff < 1e-5, "Exported ONNX model output deviates from original PyTorch model!"

        print("\nBoth PyTorch 2.x (.pt2) and ONNX (.onnx) exports verified successfully!")

    except ImportError:
        print("  [INFO] 'onnxruntime' is not installed. Skipping ONNX runtime verification.")
        print("  Install it via 'pip install onnxruntime' to enable ONNX validation.")    

if __name__ == "__main__":
    main()
