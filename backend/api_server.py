"""
FastAPI backend with real ML model integration
Handles image upload and returns artifact predictions
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import time
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from real_recognition_model import ArtifactRecognitionModel
from knowledge_grounder import KnowledgeGrounder

app = FastAPI(title="AR Museum Guide API - Real ML")

# CORS for mobile access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
print("Loading ML models...")
recognition_model = ArtifactRecognitionModel()
grounder = KnowledgeGrounder()
print("✅ Models loaded successfully!")


@app.get("/")
async def root():
    return {
        "message": "AR Museum Guide API with Real ML",
        "version": "2.0",
        "model": "ResNet50 Transfer Learning",
        "endpoints": {
            "/predict": "POST - Upload image for artifact recognition",
            "/health": "GET - Health check"
        }
    }


@app.post("/predict")
async def predict_artifact(file: UploadFile = File(...)):
    """
    Predict artifact from uploaded image using real ML model
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON with artifact predictions and grounded explanation
    """
    start_time = time.time()
    
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Get predictions from ML model
        predictions = recognition_model.predict(image, top_k=3)
        
        if not predictions:
            return JSONResponse({
                "success": False,
                "message": "Could not recognize artifact in image"
            })
        
        # Get top prediction
        top_prediction = predictions[0]
        artifact = top_prediction["artifact"]
        confidence = top_prediction["confidence"]
        
        # Generate grounded explanation
        explanation_result = grounder.generate_explanation(
            artifact=artifact,
            user_interest="general overview"
        )
        
        inference_time = (time.time() - start_time) * 1000
        
        return JSONResponse({
            "success": True,
            "prediction": {
                "artifact_id": artifact["id"],
                "artifact_name": artifact["name"],
                "category": artifact["category"],
                "period": artifact["period"],
                "origin": artifact["origin"],
                "description": artifact["description"],
                "confidence": confidence,
                "model": "ResNet50"
            },
            "top_3_predictions": [
                {
                    "name": p["artifact"]["name"],
                    "confidence": p["confidence"]
                }
                for p in predictions
            ],
            "explanation": explanation_result["explanation"],
            "sources": explanation_result["sources"],
            "grounded": explanation_result["grounded"],
            "inference_time_ms": inference_time
        })
    
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse({
            "success": False,
            "message": f"Error processing image: {str(e)}"
        }, status_code=500)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": recognition_model is not None,
        "device": str(recognition_model.device),
        "num_classes": recognition_model.num_classes
    }


if __name__ == "__main__":
    import uvicorn
    print("""
╔══════════════════════════════════════════════════════════╗
║     AR Museum Guide - Real ML API Server                ║
╚══════════════════════════════════════════════════════════╝

🤖 Model: ResNet50 Transfer Learning
🌐 API: http://localhost:8000
📱 Mobile: http://YOUR_IP:8000

Endpoints:
  - POST /predict - Upload image for recognition
  - GET /health - Check API status

Press Ctrl+C to stop
""")
    uvicorn.run(app, host="0.0.0.0", port=8000)
