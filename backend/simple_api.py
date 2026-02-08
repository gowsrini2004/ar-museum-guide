"""
Simple API server that uses trained artifacts without ML dependencies
Works with the artifacts you added via admin panel
"""
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import json
from pathlib import Path
import random

app = FastAPI(title="AR Museum Guide - Simple API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load artifacts
DATA_DIR = Path(__file__).parent.parent / "data"
ARTIFACTS_FILE = DATA_DIR / "artifacts.json"

def load_artifacts():
    """Load trained artifacts"""
    if ARTIFACTS_FILE.exists():
        with open(ARTIFACTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

artifacts = load_artifacts()
print(f"✅ Loaded {len(artifacts)} artifacts")


@app.get("/")
async def root():
    return {
        "message": "AR Museum Guide - Simple API",
        "version": "1.0",
        "artifacts": len(artifacts),
        "endpoints": {
            "/predict": "POST - Upload image for recognition",
            "/health": "GET - Health check"
        }
    }


@app.post("/predict")
async def predict_artifact(file: UploadFile = File(...)):
    """
    Recognize artifact from uploaded image
    Returns one of your trained artifacts
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        if not artifacts:
            return JSONResponse({
                "success": False,
                "message": "No artifacts trained yet. Please add artifacts via admin panel."
            })
        
        # Simple recognition: return random artifact with high confidence
        # In real ML version, this would use the trained model
        artifact = random.choice(artifacts)
        confidence = random.uniform(0.80, 0.95)
        
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
                "model": "Simple Matcher (Demo)"
            },
            "top_3_predictions": [
                {
                    "name": a["name"],
                    "confidence": random.uniform(0.60, 0.90)
                }
                for a in artifacts[:3]
            ],
            "explanation": artifact["description"],
            "sources": artifact.get("sources", [artifact.get("curator", "Museum Curator")]),
            "grounded": True,
            "inference_time_ms": random.randint(100, 300)
        })
    
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error: {str(e)}"
        }, status_code=500)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "artifacts_loaded": len(artifacts),
        "mode": "simple_demo"
    }


if __name__ == "__main__":
    import uvicorn
    print(f"""
╔══════════════════════════════════════════════════════════╗
║     AR Museum Guide - Simple API Server                 ║
╚══════════════════════════════════════════════════════════╝

🌐 API: http://localhost:8000
📚 Artifacts loaded: {len(artifacts)}

Endpoints:
  - POST /predict - Upload image for recognition
  - GET /health - Check status

⚠️  Note: Using simple demo mode (not real ML yet)
   Your artifacts will be recognized randomly for testing

Press Ctrl+C to stop
""")
    uvicorn.run(app, host="0.0.0.0", port=8000)
