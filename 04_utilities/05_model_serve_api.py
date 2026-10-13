"""
23 - 05 . Web API Model Serving (FastAPI and JIT) in PyTorch

This script demonstrates how to deploy a compiled PyTorch model as a high-performance, 
production-ready web service endpoint using the FastAPI framework. We load our 
pre-compiled TorchScript model, write an endpoint that accepts image uploads, 
apply real-time image preprocessing (PIL and torchvision transforms), and return 
inference classifications as JSON.

Learning Objectives:
1. Initialize FastAPI application structures with request routing.
2. Load serialized TorchScript models into memory during application startup.
3. Preprocess incoming raw byte images in real-time.
4. Execute JIT inference and serialize output distributions to JSON.
"""

import io
import torch

# torchvision contains standard image transforms.
import torchvision.transforms as transforms

# PIL handles reading incoming image byte streams.
from PIL import Image
from contextlib import asynccontextmanager

# FastAPI and Uvicorn provide high-speed, asynchronous ASGI web services.
# Documentation: https://fastapi.tiangolo.com/
from fastapi import FastAPI, UploadFile, File
import uvicorn

# Global variable to store our loaded TorchScript model
model = None

# We define standard vision transformations matching our training data (FashionMNIST).
# Documentation: https://pytorch.org/vision/stable/transforms.html
transform_pipeline = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# FastAPI life-cycle events. We load the model once during application startup.
# @app.on_event("startup")
# def load_production_model():
#     global model
#     model_path = "./export/classifier_traced.pt"
#     # Load JIT-compiled TorchScript model
#     # Documentation: https://pytorch.org/docs/stable/generated/torch.jit.load.html
#     model = torch.jit.load(model_path)
#     model.eval()  # Set to evaluation mode to ensure static layers (dropout, batchnorm) behave correctly
#     print(f"--- Production TorchScript model loaded from {model_path} ---")

# Define modern FastAPI lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ACTIONS ---
    global model
    # The loader below is torch.export, so this must be a .pt2 archive.
    # A TorchScript .pt file requires torch.jit.load instead.
    model_path = "./data/export/classifier_exported.pt2"
    
    # Load model into memory when server starts
    # Load PyTorch 2.x exported program
    exported_program = torch.export.load(model_path)
    # Extract the callable PyTorch module.
    # No .eval() call here: an exported program has its train/eval mode frozen at
    # export time, so .module() replaces eval()/train() with methods that raise.
    model = exported_program.module()

    print(f"--- Production model loaded successfully from {model_path} ---")
    
    yield  # Server runs and handles incoming API requests here
    
    # --- SHUTDOWN ACTIONS ---
    # Clean up resources when the server stops
    model = None
    print("--- Server shutting down. Model unloaded from memory. ---")


# The lifespan handler must already be defined before it is passed in here.
app = FastAPI(title="PyTorch Production Model Server", lifespan=lifespan)


@app.get("/")
def health_check():
    """Health check endpoint that verifies model presence and runs a test inference."""
    global model
    if model is None:
        return {"status": "unhealthy", "error": "Model not loaded"}

    # Run a dummy input tensor [1, 1, 28, 28] to prove the model can perform inference
    dummy_input = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        test_output = model(dummy_input)

    return {
        "status": "healthy",
        "model_architecture": "SimpleImageClassifier",
        "export_format": "PyTorch 2.x (torch.export .pt2)",
        "inference_test": "PASSED",
        "output_shape": list(test_output.shape)
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """POST endpoint that receives an image file and returns prediction probabilities."""
    global model
    if model is None:
        return {"error": "Model has not been initialized."}
        
    try:
        # 1.1 Read incoming image bytes
        image_bytes = await file.read()
        
        # 1.2 Open byte stream using PIL
        image = Image.open(io.BytesIO(image_bytes))
        
        # 1.3 Apply transformations to map to size [1, 1, 28, 28]
        # unsqueeze(0) inserts the batch dimension
        tensor_image = transform_pipeline(image).unsqueeze(0)
        
        # 1.4 Execute fast JIT inference (using no_grad to disable Autograd tracking)
        with torch.no_grad():
            logits = model(tensor_image)
            # Calculate Softmax probability scores
            probabilities = torch.softmax(logits, dim=-1).flatten().tolist()
            
        # Return class predictions. Index matches FashionMNIST categories (0-9).
        predictions = {f"class_{i}": round(prob, 4) for i, prob in enumerate(probabilities)}
        top_class = int(torch.argmax(logits).item())
        
        return {
            "prediction": predictions,
            "top_predicted_class": top_class,
            "confidence": round(probabilities[top_class], 4)
        }
    except Exception as e:
        return {"error": f"Failed to execute inference: {str(e)}"}

def run_server():
    print("--- Starting FastAPI Web Server locally on port 8000 ---")
    # In production, this server would be started from terminal using: uvicorn 05_model_serve_api:app --host 0.0.0.0 --port 8000
    # Here we run a short diagnostic import check to verify syntax and routing completeness.
    uvicorn.run("05_model_serve_api:app", host="0.0.0.0", port=8000, reload=True)
    print("FastAPI serving structures successfully compiled!")

if __name__ == "__main__":
    run_server()


"""
The Output Breakdownprediction 
(class_0 through class_9): 
    These are the Softmax probability scores for each of the 10 possible categories (e.g., FashionMNIST classes like T-shirt, etc.). 
    Because softmax normalizes outputs, all 10 values sum to exactly 1.0 (100% total probability).   
top_predicted_class: 2: 
    The class index that received the highest predicted probability score.   
confidence: 0.1314: 
    The probability assigned to class 2 (13.14% confidence).   

ex. { "prediction": {
    "class_0": 0.1124,"class_1": 0.0857,
    "class_2": 0.1314,"class_3": 0.1005,
    "class_4": 0.1042,"class_5": 0.1088,
    "class_6": 0.085,"class_7": 0.0765,
    "class_8": 0.0861,"class_9": 0.1093
    },
  "top_predicted_class": 2,
  "confidence": 0.1314
}    
"""
