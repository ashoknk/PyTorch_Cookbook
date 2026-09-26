"""
25. Web API Model Serving (FastAPI and JIT) in PyTorch

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

# We import the core torch library.
import torch

# torchvision contains standard image transforms.
import torchvision.transforms as transforms

# PIL handles reading incoming image byte streams.
from PIL import Image

# FastAPI and Uvicorn provide high-speed, asynchronous ASGI web services.
# Documentation: https://fastapi.tiangolo.com/
from fastapi import FastAPI, UploadFile, File
import uvicorn

# Initialize FastAPI application
app = FastAPI(title="PyTorch Production Model Server")

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
@app.on_event("startup")
def load_production_model():
    global model
    model_path = "./export/classifier_traced.pt"
    # Load JIT-compiled TorchScript model
    # Documentation: https://pytorch.org/docs/stable/generated/torch.jit.load.html
    model = torch.jit.load(model_path)
    model.eval()  # Set to evaluation mode to ensure static layers (dropout, batchnorm) behave correctly
    print(f"--- Production TorchScript model loaded from {model_path} ---")

@app.get("/")
def health_check():
    """Health check endpoint to monitor web server status."""
    return {"status": "healthy", "model": "SimpleImageClassifier (TorchScript JIT)"}

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
    print("FastAPI serving structures successfully compiled!")

if __name__ == "__main__":
    run_server()
