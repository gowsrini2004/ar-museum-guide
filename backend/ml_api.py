"""
ML-based API server that uses the trained ResNet50 model for artifact recognition
"""
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import json
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models

app = FastAPI(title="AR Museum Guide - ML API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load artifacts and model
DATA_DIR = Path(__file__).parent.parent / "data"
ARTIFACTS_FILE = DATA_DIR / "artifacts.json"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "artifact_model.pth"
CLASS_MAPPING_PATH = MODEL_DIR / "class_mapping.json"

def load_artifacts():
    """Load trained artifacts"""
    if ARTIFACTS_FILE.exists():
        print(f"Loading artifacts from: {ARTIFACTS_FILE.absolute()}")
        try:
            with open(ARTIFACTS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"Loaded {len(data)} artifacts")
                return data
        except Exception as e:
            print(f"Error loading artifacts: {e}")
            return []
    print(f"Artifacts file not found at: {ARTIFACTS_FILE.absolute()}")
    return []

def load_model():
    """Load the trained model"""
    if not MODEL_PATH.exists():
        return None, None
    
    # Load class mapping
    with open(CLASS_MAPPING_PATH, 'r') as f:
        class_mapping = json.load(f)
    
    # Create model architecture
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(class_mapping))
    
    # Load trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    
    return model, class_mapping

# Load on startup
artifacts = load_artifacts()
model, class_mapping = load_model()

if model is not None:
    print(f"✅ Loaded trained model with {len(class_mapping)} classes")
else:
    print("⚠️  No trained model found. Please train the model first.")

print(f"✅ Loaded {len(artifacts)} artifacts")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


@app.get("/")
async def root():
    return {
        "message": "AR Museum Guide - ML API",
        "version": "1.0",
        "artifacts": len(artifacts),
        "model_loaded": model is not None,
        "endpoints": {
            "/predict": "POST - Upload image for recognition",
            "/health": "GET - Health check"
        }
    }


@app.post("/predict")
async def predict_artifact(file: UploadFile = File(...)):
    """
    Recognize artifact from uploaded image using trained ML model
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Try reloading if empty
        global artifacts
        if not artifacts or len(artifacts) == 0:
            artifacts = load_artifacts()

        if not artifacts:
            return JSONResponse({
                "success": False,
                "message": f"No artifacts trained yet (Path: {ARTIFACTS_FILE.absolute()}). Please add artifacts via admin panel."
            })
        
        if model is None:
            return JSONResponse({
                "success": False,
                "message": "Model not trained yet. Please train the model first."
            })
        
        # Preprocess image
        img_tensor = transform(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_idx = predicted_idx.item()
            confidence = confidence.item()
        
        # Get predicted class name
        predicted_class = class_mapping[str(predicted_idx)]
        
        # Find matching artifact
        matching_artifact = None
        for artifact in artifacts:
            if artifact["id"] == predicted_class:
                matching_artifact = artifact
                break
        
        if not matching_artifact:
            return JSONResponse({
                "success": False,
                "message": f"Predicted class '{predicted_class}' not found in artifacts"
            })
        
        # Get top 3 predictions
        top_probs, top_indices = torch.topk(probabilities[0], min(3, len(class_mapping)))
        top_predictions = []
        for prob, idx in zip(top_probs, top_indices):
            class_name = class_mapping[str(idx.item())]
            for artifact in artifacts:
                if artifact["id"] == class_name:
                    top_predictions.append({
                        "name": artifact["name"],
                        "confidence": float(prob.item())
                    })
                    break
        
        return JSONResponse({
            "success": True,
            "prediction": {
                "artifact_id": matching_artifact["id"],
                "artifact_name": matching_artifact["name"],
                "category": matching_artifact["category"],
                "period": matching_artifact["period"],
                "origin": matching_artifact["origin"],
                "description": matching_artifact["description"],
                "confidence": confidence,
                "model": "ResNet50 (Trained)"
            },
            "top_3_predictions": top_predictions,
            "explanation": matching_artifact["description"],
            "sources": matching_artifact.get("sources", [matching_artifact.get("curator", "Museum Curator")]),
            "grounded": True,
            "inference_time_ms": 0  # Can add timing if needed
        })
    
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error: {str(e)}"
        }, status_code=500)


@app.get("/health")
async def health_check():
    # Reload artifacts to check current state
    current_artifacts = load_artifacts()
    return {
        "status": "healthy",
        "artifacts_loaded": len(current_artifacts),
        "artifacts_file_path": str(ARTIFACTS_FILE.absolute()),
        "file_exists": ARTIFACTS_FILE.exists(),
        "model_loaded": model is not None,
        "mode": "ml_trained"
    }


if __name__ == "__main__":
    import uvicorn
    print(f"""
╔══════════════════════════════════════════════════════════╗
║     AR Museum Guide - ML API Server                     ║
╚══════════════════════════════════════════════════════════╝

🌐 API: http://localhost:8000
📚 Artifacts loaded: {len(artifacts)}
🤖 Model: {'Trained ✅' if model else 'Not trained ❌'}

Endpoints:
  - POST /predict - Upload image for recognition
  - GET /health - Check status

🎯 Using trained ResNet50 model for real artifact recognition!

Press Ctrl+C to stop
""")
    uvicorn.run(app, host="0.0.0.0", port=8000)
