"""
Training API endpoints for managing artifacts and training the model
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
import json
import shutil
from pathlib import Path
from PIL import Image
import io

app = FastAPI(title="AR Museum Guide - Training API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data directories
DATA_DIR = Path(__file__).parent.parent / "data"
TRAINING_DIR = DATA_DIR / "training"
ARTIFACTS_FILE = DATA_DIR / "artifacts.json"

# Create directories
TRAINING_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_artifacts():
    """Load artifacts from JSON file"""
    if ARTIFACTS_FILE.exists():
        with open(ARTIFACTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_artifacts(artifacts):
    """Save artifacts to JSON file"""
    with open(ARTIFACTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(artifacts, f, indent=2, ensure_ascii=False)


@app.post("/api/artifacts/add")
async def add_artifact(
    name: str = Form(...),
    category: str = Form(...),
    period: str = Form(...),
    origin: str = Form(...),
    description: str = Form(...),
    curator: str = Form("Museum Curator"),
    images: List[UploadFile] = File(...)
):
    """
    Add a new artifact with training images
    """
    try:
        # Generate artifact ID
        artifacts = load_artifacts()
        artifact_id = f"artifact_{len(artifacts) + 1}"
        
        # Create directory for this artifact's images
        artifact_dir = TRAINING_DIR / artifact_id
        artifact_dir.mkdir(exist_ok=True)
        
        # Save images
        saved_images = []
        for idx, image_file in enumerate(images):
            # Read and validate image
            contents = await image_file.read()
            img = Image.open(io.BytesIO(contents))
            
            # Save image
            filename = f"{artifact_id}_{idx+1}.jpg"
            filepath = artifact_dir / filename
            img.convert('RGB').save(filepath, 'JPEG', quality=95)
            saved_images.append(str(filepath.relative_to(DATA_DIR)))
        
        # Create artifact entry
        artifact = {
            "id": artifact_id,
            "name": name,
            "category": category,
            "period": period,
            "origin": origin,
            "description": description,
            "curator": curator,
            "sources": [
                f"{curator}, Chief Curator",
                "Museum Historical Archive, 2024"
            ],
            "images": saved_images,
            "num_images": len(saved_images)
        }
        
        # Save to artifacts list
        artifacts.append(artifact)
        save_artifacts(artifacts)
        
        return JSONResponse({
            "success": True,
            "message": f"Artifact '{name}' added successfully",
            "artifact": artifact
        })
    
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error adding artifact: {str(e)}"
        }, status_code=500)


@app.get("/api/artifacts/list")
async def list_artifacts():
    """Get all artifacts"""
    artifacts = load_artifacts()
    return JSONResponse({
        "success": True,
        "artifacts": artifacts,
        "total": len(artifacts)
    })


@app.post("/api/model/train")
async def train_model():
    """
    Train the model on all artifacts
    """
    try:
        artifacts = load_artifacts()
        
        if len(artifacts) < 2:
            return JSONResponse({
                "success": False,
                "message": "Need at least 2 artifacts to train the model"
            }, status_code=400)
        
        # Import training code
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        
        from train_model import train_artifact_model
        
        # Train the model
        results = train_artifact_model(
            data_dir=str(TRAINING_DIR),
            num_epochs=10,
            batch_size=8
        )
        
        return JSONResponse({
            "success": True,
            "message": "Model trained successfully!",
            "results": results
        })
    
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Training failed: {str(e)}"
        }, status_code=500)


@app.get("/api/stats")
async def get_stats():
    """Get training statistics"""
    artifacts = load_artifacts()
    total_images = sum(a.get('num_images', 0) for a in artifacts)
    
    model_path = Path(__file__).parent.parent / "models" / "artifact_model.pth"
    model_trained = model_path.exists()
    
    return JSONResponse({
        "total_artifacts": len(artifacts),
        "total_images": total_images,
        "model_trained": model_trained,
        "artifacts": artifacts
    })


if __name__ == "__main__":
    import uvicorn
    print("""
╔══════════════════════════════════════════════════════════╗
║     AR Museum Guide - Training API Server               ║
╚══════════════════════════════════════════════════════════╝

🌐 API: http://localhost:8001
📁 Data: {DATA_DIR}

Endpoints:
  - POST /api/artifacts/add - Add new artifact with images
  - GET /api/artifacts/list - List all artifacts
  - POST /api/model/train - Train the model
  - GET /api/stats - Get statistics

Press Ctrl+C to stop
""")
    uvicorn.run(app, host="0.0.0.0", port=8001)
