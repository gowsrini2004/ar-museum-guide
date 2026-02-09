"""
Training API endpoints for managing artifacts and training the model
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import json
import shutil
from pathlib import Path
from PIL import Image
import io
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import RAG service
try:
    from rag_service import get_rag_service
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    rag_service = get_rag_service(api_key=GEMINI_API_KEY)
    RAG_ENABLED = True
except Exception as e:
    print(f"RAG service not available: {e}")
    RAG_ENABLED = False

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
DOCUMENTS_DIR = DATA_DIR / "documents"
ARTIFACTS_FILE = DATA_DIR / "artifacts.json"

# Create directories
TRAINING_DIR.mkdir(parents=True, exist_ok=True)
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
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
    images: List[UploadFile] = File(...),
    documents: Optional[List[UploadFile]] = File(None)
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
        
        # Process PDF documents if provided
        saved_documents = []
        if documents and RAG_ENABLED:
            doc_dir = DOCUMENTS_DIR / artifact_id
            doc_dir.mkdir(exist_ok=True)
            
            for idx, doc_file in enumerate(documents):
                if doc_file.filename.lower().endswith('.pdf'):
                    doc_id = f"doc_{idx+1}"
                    filename = f"{artifact_id}_{doc_id}.pdf"
                    filepath = doc_dir / filename
                    
                    # Save PDF
                    contents = await doc_file.read()
                    with open(filepath, 'wb') as f:
                        f.write(contents)
                    
                    # Process with RAG service
                    success = rag_service.create_embeddings(
                        artifact_id=artifact_id,
                        pdf_path=str(filepath),
                        document_id=doc_id
                    )
                    
                    saved_documents.append({
                        "id": doc_id,
                        "filename": doc_file.filename,
                        "path": str(filepath.relative_to(DATA_DIR)),
                        "uploaded_at": datetime.now().isoformat(),
                        "processed": success
                    })
        
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
            "num_images": len(saved_images),
            "documents": saved_documents,
            "num_documents": len(saved_documents)
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


@app.delete("/api/artifacts/{artifact_id}")
async def delete_artifact(artifact_id: str):
    """
    Delete an artifact and its training images
    """
    try:
        artifacts = load_artifacts()
        
        # Find the artifact
        artifact = None
        artifact_index = None
        for idx, a in enumerate(artifacts):
            if a['id'] == artifact_id:
                artifact = a
                artifact_index = idx
                break
        
        if artifact is None:
            return JSONResponse({
                "success": False,
                "message": f"Artifact '{artifact_id}' not found"
            }, status_code=404)
        
        # Delete the training images directory
        artifact_dir = TRAINING_DIR / artifact_id
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)
        
        # Delete documents and embeddings
        doc_dir = DOCUMENTS_DIR / artifact_id
        if doc_dir.exists():
            shutil.rmtree(doc_dir)
        
        if RAG_ENABLED:
            rag_service.delete_artifact_embeddings(artifact_id)
        
        # Remove from artifacts list
        artifacts.pop(artifact_index)
        save_artifacts(artifacts)
        
        return JSONResponse({
            "success": True,
            "message": f"Artifact '{artifact['name']}' deleted successfully",
            "remaining_artifacts": len(artifacts)
        })
    
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error deleting artifact: {str(e)}"
        }, status_code=500)


@app.get("/api/stats")
async def get_stats():
    """Get training statistics"""
    artifacts = load_artifacts()
    total_images = sum(a.get('num_images', 0) for a in artifacts)
    
    model_path = Path(__file__).parent.parent / "models" / "artifact_model.pth"
    model_trained = model_path.exists()
    
    # Get model training timestamp if available
    model_timestamp = None
    if model_trained:
        import os
        model_timestamp = os.path.getmtime(model_path)
    
    # Add training status to each artifact
    artifacts_with_status = []
    for artifact in artifacts:
        artifact_copy = artifact.copy()
        artifact_dir = TRAINING_DIR / artifact['id']
        artifact_copy['has_images'] = artifact_dir.exists() and len(list(artifact_dir.glob('*.jpg'))) > 0
        artifacts_with_status.append(artifact_copy)
    
    return JSONResponse({
        "total_artifacts": len(artifacts),
        "total_images": total_images,
        "model_trained": model_trained,
        "model_timestamp": model_timestamp,
        "artifacts": artifacts_with_status
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
  - POST /api/artifacts/add - Add new artifact with images and PDFs
  - GET /api/artifacts/list - List all artifacts
  - DELETE /api/artifacts/{id} - Delete an artifact
  - POST /api/model/train - Train the model
  - GET /api/stats - Get statistics

RAG Q&A: {'Enabled' if RAG_ENABLED else 'Disabled (install dependencies)'}

Press Ctrl+C to stop
""")
    uvicorn.run(app, host="0.0.0.0", port=8001)
