"""
Training API endpoints for managing artifacts and training the model
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
import json
from datetime import datetime
import shutil
from pathlib import Path
from PIL import Image
import io
import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import RAG service
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
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

# Disable caching for static files to prevent stale images
@app.middleware("http")
async def add_no_cache_header(request, call_next):
    response = await call_next(request)
    # Apply to all API and static requests to prevent stale training progress or images
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Data and Model directories
DATA_DIR = Path(__file__).parent.parent / "data"
TRAINING_DIR = DATA_DIR / "training"
DOCUMENTS_DIR = DATA_DIR / "documents"
ARTIFACTS_FILE = DATA_DIR / "artifacts.json"
STATS_FILE = DATA_DIR / "training_stats.json"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "artifact_model.pth"
CLASS_MAPPING_PATH = MODEL_DIR / "class_mapping.json"

# Create directories
TRAINING_DIR.mkdir(parents=True, exist_ok=True)
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODELS_3D_DIR = DATA_DIR / "models_3d"
MODELS_3D_DIR.mkdir(parents=True, exist_ok=True)
MEDIA_DIR = DATA_DIR / "media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/static/training", StaticFiles(directory=str(TRAINING_DIR)), name="training_images")
app.mount("/static/documents", StaticFiles(directory=str(DOCUMENTS_DIR)), name="documents_files")
app.mount("/static/models", StaticFiles(directory=str(MODELS_3D_DIR)), name="artifact_models")
app.mount("/static/media", StaticFiles(directory=str(MEDIA_DIR)), name="media_files")

# ─── AR 3D Model Upload ────────────────────────────────────────────────────────

@app.post("/api/artifacts/{artifact_id}/upload-model")
async def upload_3d_model(artifact_id: str, model_file: UploadFile = File(...)):
    """Upload a GLB/GLTF 3D model for an artifact."""
    artifacts = load_artifacts()
    artifact = next((a for a in artifacts if a["id"] == artifact_id), None)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")

    if not model_file.filename.lower().endswith(('.glb', '.gltf')):
        raise HTTPException(status_code=400, detail="Only .glb or .gltf files are accepted")

    model_dir = MODELS_3D_DIR / artifact_id
    model_dir.mkdir(parents=True, exist_ok=True)
    dest = model_dir / "model.glb"

    with open(dest, "wb") as f:
        content = await model_file.read()
        f.write(content)

    # Update artifact record
    artifact["model_3d_path"] = f"models_3d/{artifact_id}/model.glb"
    save_artifacts(artifacts)

    return JSONResponse({"success": True, "message": f"3D model uploaded for {artifact['name']}"})


@app.delete("/api/artifacts/{artifact_id}/upload-model")
async def delete_3d_model(artifact_id: str):
    """Remove the 3D model for an artifact."""
    artifacts = load_artifacts()
    artifact = next((a for a in artifacts if a["id"] == artifact_id), None)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")

    model_path = MODELS_3D_DIR / artifact_id / "model.glb"
    if model_path.exists():
        model_path.unlink()

    artifact.pop("model_3d_path", None)
    save_artifacts(artifacts)
    return JSONResponse({"success": True, "message": "3D model removed"})


# ─── AR Audio/Video Upload ────────────────────────────────────────────────────

@app.post("/api/artifacts/{artifact_id}/upload-audio")
async def upload_audio(artifact_id: str, audio_file: UploadFile = File(...)):
    """Upload an MP3 audio file for an artifact."""
    artifacts = load_artifacts()
    artifact = next((a for a in artifacts if a["id"] == artifact_id), None)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")

    if not audio_file.filename.lower().endswith('.mp3'):
        raise HTTPException(status_code=400, detail="Only .mp3 files are accepted")

    media_dir = MEDIA_DIR / artifact_id
    media_dir.mkdir(parents=True, exist_ok=True)
    dest = media_dir / "audio.mp3"

    with open(dest, "wb") as f:
        content = await audio_file.read()
        f.write(content)

    artifact["audio_path"] = f"media/{artifact_id}/audio.mp3"
    save_artifacts(artifacts)
    return JSONResponse({"success": True, "message": f"Audio uploaded for {artifact['name']}"})


@app.delete("/api/artifacts/{artifact_id}/upload-audio")
async def delete_audio(artifact_id: str):
    artifacts = load_artifacts()
    artifact = next((a for a in artifacts if a["id"] == artifact_id), None)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")

    audio_path = MEDIA_DIR / artifact_id / "audio.mp3"
    if audio_path.exists():
        audio_path.unlink()

    artifact.pop("audio_path", None)
    save_artifacts(artifacts)
    return JSONResponse({"success": True, "message": "Audio removed"})


@app.post("/api/artifacts/{artifact_id}/upload-video")
async def upload_video(artifact_id: str, video_file: UploadFile = File(...)):
    """Upload a video file for an artifact."""
    artifacts = load_artifacts()
    artifact = next((a for a in artifacts if a["id"] == artifact_id), None)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")

    if not video_file.filename.lower().endswith(('.mp4', '.webm')):
        raise HTTPException(status_code=400, detail="Only .mp4 or .webm files are accepted")

    media_dir = MEDIA_DIR / artifact_id
    media_dir.mkdir(parents=True, exist_ok=True)
    
    ext = video_file.filename.split('.')[-1].lower()
    filename = f"video.{ext}"
    dest = media_dir / filename

    with open(dest, "wb") as f:
        content = await video_file.read()
        f.write(content)

    artifact["video_path"] = f"media/{artifact_id}/{filename}"
    save_artifacts(artifacts)
    return JSONResponse({"success": True, "message": f"Video uploaded for {artifact['name']}"})


@app.delete("/api/artifacts/{artifact_id}/upload-video")
async def delete_video(artifact_id: str):
    artifacts = load_artifacts()
    artifact = next((a for a in artifacts if a["id"] == artifact_id), None)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")

    media_dir = MEDIA_DIR / artifact_id
    if media_dir.exists():
        for file in media_dir.glob("video.*"):
            file.unlink()

    artifact.pop("video_path", None)
    save_artifacts(artifacts)
    return JSONResponse({"success": True, "message": "Video removed"})


# ─── AR Info Cards ────────────────────────────────────────────────────────────

@app.get("/api/artifacts/{artifact_id}/ar-cards")
async def get_ar_cards(artifact_id: str):
    """Get the AR info cards for an artifact."""
    artifacts = load_artifacts()
    artifact = next((a for a in artifacts if a["id"] == artifact_id), None)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return JSONResponse({"cards": artifact.get("ar_cards", [])})


@app.put("/api/artifacts/{artifact_id}/ar-cards")
async def save_ar_cards(artifact_id: str, request: dict):
    """Save AR info cards for an artifact. Body: { cards: [{icon, title, body, color}] }"""
    artifacts = load_artifacts()
    artifact = next((a for a in artifacts if a["id"] == artifact_id), None)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")

    cards = request.get("cards", [])
    # Validate structure
    validated = []
    for c in cards:
        validated.append({
            "icon": str(c.get("icon", "📌"))[:8],
            "title": str(c.get("title", "Info"))[:100],
            "body": str(c.get("body", ""))[:2000],
            "color": str(c.get("color", "#667eea"))[:20],
        })

    artifact["ar_cards"] = validated
    save_artifacts(artifacts)
    return JSONResponse({"success": True, "count": len(validated)})


async def run_training_task():
    try:
        print("Starting background training task...")
        # Import inside function to avoid circular imports or path issues
        try:
            from train_model import train_artifact_model
        except ImportError:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from train_model import train_artifact_model
            
        artifacts = load_artifacts()
        if len(artifacts) < 2:
            print("Not enough artifacts to train")
            return

        print(f"Training on {len(artifacts)} artifacts...")
        # Add a small delay to ensure the reset file is readable by frontend
        await asyncio.sleep(1) 
        
        results = train_artifact_model(
            data_dir=str(TRAINING_DIR),
            num_epochs=10,
            batch_size=8
        )
        
        # Save stats
        stats = {
            "last_training_accuracy": results['best_accuracy'],
            "avg_training_accuracy": results['avg_accuracy'],
            "training_epochs": results['num_epochs'],
            "last_trained": datetime.now().isoformat(),
            "total_classes": results['num_classes']
        }
        
        with open(STATS_FILE, 'w') as f:
            json.dump(stats, f)
            
        # Notify ML API to reload the new model
        try:
            import urllib.request
            # We use localhost:8000 for the ML API
            req = urllib.request.Request("http://localhost:8000/reload", method="POST")
            with urllib.request.urlopen(req, timeout=5) as response:
                print(f"[OK] ML API notified to reload. Status: {response.getcode()}")
        except Exception as reload_err:
            print(f"[WARN] Could not auto-reload ML API: {reload_err}")

        print("Background training completed successfully")
    except Exception as e:
        print(f"Background training failed: {e}")
        import traceback
        traceback.print_exc()
        # Update progress file with error status
        try:
            progress_file = DATA_DIR / "training_progress.json"
            with open(progress_file, 'w') as f:
                json.dump({"status": "error", "message": str(e), "percent": 0}, f)
        except:
            pass

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


def run_embedding_task(artifact_id: str, pdf_path: str, doc_id: str, filename: str):
    """Background task wrapper for embedding generation"""
    print(f"[*] Starting background embedding generation for {filename}...")
    try:
        success = rag_service.create_embeddings(
            artifact_id=artifact_id,
            pdf_path=pdf_path,
            document_id=doc_id,
            filename=filename
        )
        if success:
            print(f"[OK] Embeddings created for {filename}")
        else:
            print(f"[WARN] Failed to create embeddings for {filename}")
    except Exception as e:
        print(f"[ERR] Error in background embedding task: {e}")


@app.post("/api/artifacts/add")
async def add_artifact(
    background_tasks: BackgroundTasks,
    name: str = Form(...),

    category: str = Form(...),
    period: str = Form(...),
    origin: str = Form(...),
    description: str = Form(...),
    curator: str = Form(...),
    images: List[UploadFile] = File(...),
    documents: List[UploadFile] = File(default=[])
):
    """Add a new artifact with images and documents"""
    print(f"[*] Received upload request for artifact: {name}")
    print(f"   - Images: {len(images)}")
    print(f"   - Documents: {len(documents)}")

    try:
        # Generate artifact ID
        loop = asyncio.get_running_loop()
        
        # Load artifacts in threadpool to avoid blocking
        artifacts = await loop.run_in_executor(None, load_artifacts)

        # Generate unique artifact ID using timestamp
        import time
        artifact_id = f"artifact_{int(time.time())}"
        
        # Create directory for this artifact's images
        artifact_dir = TRAINING_DIR / artifact_id
        artifact_dir.mkdir(exist_ok=True)
        
        # Save images
        saved_images = []
        for idx, image_file in enumerate(images):
            # Read image content (async)
            contents = await image_file.read()
            
            # Process image in threadpool
            def process_and_save_image(content, path):
                img = Image.open(io.BytesIO(content))
                img.convert('RGB').save(path, 'JPEG', quality=95)
                return str(path.relative_to(DATA_DIR))

            # Save image
            filename = f"{artifact_id}_{idx+1}.jpg"
            filepath = artifact_dir / filename
            
            relative_path = await loop.run_in_executor(None, process_and_save_image, contents, filepath)
            saved_images.append(relative_path)
            print(f"   [OK] Saved image: {filename}")
        
        if len(saved_images) < 5:
            # Cleanup if not enough valid images
            if artifact_dir.exists():
                shutil.rmtree(artifact_dir)
            return JSONResponse({
                "success": False,
                "message": f"Need at least 5 valid images. Only {len(saved_images)} were valid."
            }, status_code=400)
            
        # Handle documents
        saved_documents = []
        if documents:
            doc_dir = DOCUMENTS_DIR / artifact_id
            doc_dir.mkdir(exist_ok=True)
            
            for idx, doc_file in enumerate(documents):
                if doc_file.filename.lower().endswith('.pdf'):
                    doc_id = f"doc_{idx+1}"
                    # Save PDF with original filename (sanitized)
                    original_filename = doc_file.filename
                    safe_filename = "".join([c for c in original_filename if c.isalpha() or c.isdigit() or c in (' ', '.', '_', '-')]).strip()
                    if not safe_filename.lower().endswith('.pdf'):
                        safe_filename += ".pdf"
                        
                    filename = safe_filename
                    filepath = doc_dir / filename
                    
                    contents = await doc_file.read()
                    
                    # Save PDF to disk (blocking IO in threadpool)
                    await loop.run_in_executor(None, lambda: filepath.write_bytes(contents))
                    print(f"   [OK] Saved document: {filename}")
                    
                    # Create embeddings (BACKGROUND TASK)
                    if RAG_ENABLED:
                        print(f"   [...] Scheduled background embedding generation for {filename}")
                        background_tasks.add_task(
                            run_embedding_task,
                            artifact_id,
                            str(filepath),
                            doc_id,
                            original_filename
                        )
                    
                    saved_documents.append({
                        "id": doc_id,
                        "filename": original_filename,
                        "path": str(filepath.relative_to(DATA_DIR))
                    })
        
        # Create artifact object
        artifact = {
            "id": artifact_id,
            "name": name,
            "category": category,
            "period": period,
            "origin": origin,
            "description": description,
            "curator": curator,
            "images": saved_images,
            "num_images": len(saved_images),
            "documents": saved_documents,
            "num_documents": len(saved_documents),
            "created_at": datetime.now().isoformat()
        }
        
        artifacts.append(artifact)
        await loop.run_in_executor(None, save_artifacts, artifacts)
        print(f"[OK] Artifact '{name}' added successfully with ID {artifact_id}")
        
        # Trigger training in background
        if len(artifacts) >= 2:
            background_tasks.add_task(run_training_task)
            print(f"Triggered background training after adding {artifact_id}")
        
        return JSONResponse({
            "success": True,
            "message": f"Artifact '{name}' added successfully. Training started in background.",
            "artifact": artifact
        })
    
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error adding artifact: {str(e)}"
        }, status_code=500)



@app.get("/api/artifacts/list")
async def list_artifacts():
    """Get all artifacts with minimal data for fast loading"""
    artifacts = load_artifacts()
    
    # Return only summary data for performance
    summary_artifacts = []
    for artifact in artifacts:
        summary_artifacts.append({
            "id": artifact["id"],
            "name": artifact["name"],
            "category": artifact["category"],
            "period": artifact["period"],
            "origin": artifact["origin"],
            "description": artifact["description"],
            "images": artifact.get("images", []),
            "num_images": artifact.get("num_images", 0),
            "num_documents": artifact.get("num_documents", 0),
            "model_3d_path": artifact.get("model_3d_path", None),
            "audio_path": artifact.get("audio_path", None),
            "video_path": artifact.get("video_path", None),
            "ar_cards": artifact.get("ar_cards", [])
        })
    
    return JSONResponse({
        "success": True,
        "artifacts": summary_artifacts,
        "total": len(summary_artifacts)
    })


@app.post("/api/model/train")
async def train_model(background_tasks: BackgroundTasks):
    """
    Train the model on all artifacts in the background
    """
    try:
        artifacts = load_artifacts()
        
        if len(artifacts) < 2:
            return JSONResponse({
                "success": False,
                "message": "Need at least 2 artifacts to train the model"
            }, status_code=400)
        
        # Start training in background
        # FIRST: Explicitly reset the progress file to "loading_data" 0% 
        # This ensures the UI updates IMMEDIATELY even before the background thread kicks in fully
        progress_file = DATA_DIR / "training_progress.json"
        with open(progress_file, 'w') as f:
            json.dump({
                "status": "loading_data", 
                "percent": 0, 
                "epoch": 0, 
                "total_epochs": 10,
                "message": "Initializing training engine...",
                "timestamp": datetime.now().isoformat()
            }, f)
        
        background_tasks.add_task(run_training_task)
        
        return JSONResponse({
            "success": True,
            "message": "Training started in background. Follow progress in the dashboard."
        })
    
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Could not start training: {str(e)}"
        }, status_code=500)


@app.get("/api/model/progress")
async def get_training_progress():
    """Get the current training progress from the progress file"""
    progress_file = DATA_DIR / "training_progress.json"
    if not progress_file.exists():
        return JSONResponse({
            "status": "idle",
            "percent": 0,
            "message": "No training in progress"
        })
    
    try:
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        return JSONResponse(progress)
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)


@app.delete("/api/model")
async def delete_model():
    """Delete the trained model and its statistics"""
    try:
        deleted_files = []
        for file_path in [MODEL_PATH, CLASS_MAPPING_PATH, STATS_FILE]:
            if file_path.exists():
                file_path.unlink()
                deleted_files.append(file_path.name)
        
        return JSONResponse({
            "success": True,
            "message": f"Successfully deleted model files: {', '.join(deleted_files) if deleted_files else 'None'}"
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error deleting model: {str(e)}"
        }, status_code=500)


@app.delete("/api/artifacts/{artifact_id}")
async def delete_artifact(
    artifact_id: str,
    background_tasks: BackgroundTasks
):
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
            shutil.rmtree(artifact_dir, ignore_errors=True)
        
        # Delete documents and embeddings
        doc_dir = DOCUMENTS_DIR / artifact_id
        if doc_dir.exists():
            shutil.rmtree(doc_dir, ignore_errors=True)
        
        if RAG_ENABLED:
            rag_service.delete_artifact_embeddings(artifact_id)
        
        # Remove from artifacts list
        artifacts.pop(artifact_index)
        save_artifacts(artifacts)
        
        # Trigger training in background if we still have enough artifacts
        if len(artifacts) >= 2:
            background_tasks.add_task(run_training_task)
            print(f"Triggered background training after deleting {artifact_id}")
        
        return JSONResponse({
            "success": True,
            "message": f"Artifact '{artifact['name']}' deleted successfully. Model will be retrained in background.",
            "remaining_artifacts": len(artifacts)
        })
    
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error deleting artifact: {str(e)}"
        }, status_code=500)
    
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error deleting artifact: {str(e)}"
        }, status_code=500)




@app.get("/api/artifacts/{artifact_id}")
async def get_artifact(artifact_id: str):
    """Get detailed information about a specific artifact"""
    try:
        artifacts = load_artifacts()
        artifact = None
        for a in artifacts:
            if a['id'] == artifact_id:
                artifact = a.copy()
                break
        
        if not artifact:
            return JSONResponse({
                "success": False,
                "message": f"Artifact '{artifact_id}' not found"
            }, status_code=404)
        
        # Get image files
        artifact_dir = TRAINING_DIR / artifact_id
        image_files = []
        if artifact_dir.exists():
            for img_path in sorted(artifact_dir.glob('*.jpg')):
                image_files.append({
                    "filename": img_path.name,
                    "path": str(img_path.relative_to(DATA_DIR))
                })
        
        artifact['image_files'] = image_files
        
        # Get document stats if RAG enabled
        if RAG_ENABLED:
            stats = rag_service.get_artifact_stats(artifact_id)
            artifact['document_stats'] = stats
            artifact['documents_list'] = rag_service.list_documents(artifact_id)
        
        return JSONResponse({
            "success": True,
            "artifact": artifact
        })
    
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error getting artifact: {str(e)}"
        }, status_code=500)


@app.post("/api/artifacts/{artifact_id}/images")
async def add_images_to_artifact(
    artifact_id: str,
    images: List[UploadFile] = File(...)
):
    """Add more training images to an existing artifact"""
    try:
        artifacts = load_artifacts()
        artifact = None
        artifact_index = None
        
        for idx, a in enumerate(artifacts):
            if a['id'] == artifact_id:
                artifact = a
                artifact_index = idx
                break
        
        if not artifact:
            return JSONResponse({
                "success": False,
                "message": f"Artifact '{artifact_id}' not found"
            }, status_code=404)
        
        # Get artifact directory
        artifact_dir = TRAINING_DIR / artifact_id
        artifact_dir.mkdir(exist_ok=True)
        
        # Find next image number
        existing_images = list(artifact_dir.glob('*.jpg'))
        next_num = len(existing_images) + 1
        
        # Save new images
        saved_images = []
        for image_file in images:
            contents = await image_file.read()
            img = Image.open(io.BytesIO(contents))
            
            filename = f"{artifact_id}_{next_num}.jpg"
            filepath = artifact_dir / filename
            img.convert('RGB').save(filepath, 'JPEG', quality=95)
            saved_images.append(str(filepath.relative_to(DATA_DIR)))
            next_num += 1
        
        # Update artifact
        artifact['images'].extend(saved_images)
        artifact['num_images'] = len(artifact['images'])
        artifacts[artifact_index] = artifact
        save_artifacts(artifacts)
        
        return JSONResponse({
            "success": True,
            "message": f"Added {len(saved_images)} images to '{artifact['name']}'",
            "new_images": saved_images,
            "total_images": artifact['num_images']
        })
    
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error adding images: {str(e)}"
        }, status_code=500)


@app.delete("/api/artifacts/{artifact_id}/images/{image_filename}")
async def delete_image_from_artifact(artifact_id: str, image_filename: str):
    """Delete a specific image from an artifact"""
    try:
        artifacts = load_artifacts()
        artifact = None
        artifact_index = None
        
        for idx, a in enumerate(artifacts):
            if a['id'] == artifact_id:
                artifact = a
                artifact_index = idx
                break
        
        if not artifact:
            return JSONResponse({
                "success": False,
                "message": f"Artifact '{artifact_id}' not found"
            }, status_code=404)
        
        # Delete the image file
        artifact_dir = TRAINING_DIR / artifact_id
        image_path = artifact_dir / image_filename
        
        if not image_path.exists():
            return JSONResponse({
                "success": False,
                "message": f"Image '{image_filename}' not found"
            }, status_code=404)
        
        image_path.unlink()
        
        # Update artifact images list
        relative_path = str(image_path.relative_to(DATA_DIR))
        if relative_path in artifact['images']:
            artifact['images'].remove(relative_path)
        
        artifact['num_images'] = len(artifact['images'])
        artifacts[artifact_index] = artifact
        save_artifacts(artifacts)
        
        return JSONResponse({
            "success": True,
            "message": f"Image '{image_filename}' deleted successfully",
            "remaining_images": artifact['num_images']
        })
    
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error deleting image: {str(e)}"
        }, status_code=500)


@app.post("/api/artifacts/{artifact_id}/documents")
async def upload_document_to_artifact(
    artifact_id: str,
    background_tasks: BackgroundTasks,
    documents: List[UploadFile] = File(...)
):
    """Upload PDF documents to an artifact"""
    try:
        if not RAG_ENABLED:
            return JSONResponse({
                "success": False,
                "message": "RAG service is not available"
            }, status_code=503)
        
        artifacts = load_artifacts()
        artifact = None
        artifact_index = None
        
        for idx, a in enumerate(artifacts):
            if a['id'] == artifact_id:
                artifact = a
                artifact_index = idx
                break
        
        if not artifact:
            return JSONResponse({
                "success": False,
                "message": f"Artifact '{artifact_id}' not found"
            }, status_code=404)
        
        # Create documents directory
        doc_dir = DOCUMENTS_DIR / artifact_id
        doc_dir.mkdir(exist_ok=True)
        
        # Get existing documents
        if 'documents' not in artifact:
            artifact['documents'] = []
        
        next_doc_num = len(artifact['documents']) + 1
        saved_documents = []
        
        for doc_file in documents:
            if doc_file.filename.lower().endswith('.pdf'):
                doc_id = f"doc_{next_doc_num}"
                doc_id = f"doc_{next_doc_num}"
                
                # Save PDF with original filename (sanitized)
                original_filename = doc_file.filename
                safe_filename = "".join([c for c in original_filename if c.isalpha() or c.isdigit() or c in (' ', '.', '_', '-')]).strip()
                if not safe_filename.lower().endswith('.pdf'):
                    safe_filename += ".pdf"
                    
                filename = safe_filename
                filepath = doc_dir / filename
                
                # Save PDF
                contents = await doc_file.read()
                with open(filepath, 'wb') as f:
                    f.write(contents)
                
                # Process with RAG service (BACKGROUND TASK)
                print(f"   [...] Scheduled background embedding generation for {original_filename}")
                background_tasks.add_task(
                    run_embedding_task,
                    artifact_id,
                    str(filepath),
                    doc_id,
                    original_filename
                )
                
                saved_documents.append({
                    "id": doc_id,
                    "filename": original_filename,
                    "path": str(filepath.relative_to(DATA_DIR)),
                    "uploaded_at": datetime.now().isoformat(),
                    "processed": False # Will be true after background task
                })
                next_doc_num += 1
        
        # Update artifact
        artifact['documents'].extend(saved_documents)
        artifact['num_documents'] = len(artifact['documents'])
        artifacts[artifact_index] = artifact
        save_artifacts(artifacts)
        
        return JSONResponse({
            "success": True,
            "message": f"Uploaded {len(saved_documents)} documents to '{artifact['name']}'",
            "documents": saved_documents
        })
    
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error uploading documents: {str(e)}"
        }, status_code=500)


@app.delete("/api/artifacts/{artifact_id}/documents/{document_id}")
async def delete_document_from_artifact(artifact_id: str, document_id: str):
    """Delete a document and its embeddings from an artifact"""
    try:
        if not RAG_ENABLED:
            return JSONResponse({
                "success": False,
                "message": "RAG service is not available"
            }, status_code=503)
        
        artifacts = load_artifacts()
        artifact = None
        artifact_index = None
        
        for idx, a in enumerate(artifacts):
            if a['id'] == artifact_id:
                artifact = a
                artifact_index = idx
                break
        
        if not artifact:
            return JSONResponse({
                "success": False,
                "message": f"Artifact '{artifact_id}' not found"
            }, status_code=404)
        
        # Find and delete the document
        document = None
        doc_index = None
        for idx, doc in enumerate(artifact.get('documents', [])):
            if doc['id'] == document_id:
                document = doc
                doc_index = idx
                break
        
        if not document:
            return JSONResponse({
                "success": False,
                "message": f"Document '{document_id}' not found"
            }, status_code=404)
        
        # Delete the PDF file
        doc_path = DATA_DIR / document['path']
        if doc_path.exists():
            doc_path.unlink()
        
        # Delete embeddings
        rag_service.delete_document_embeddings(artifact_id, document_id)
        
        # Update artifact
        artifact['documents'].pop(doc_index)
        artifact['num_documents'] = len(artifact['documents'])
        artifacts[artifact_index] = artifact
        save_artifacts(artifacts)
        
        return JSONResponse({
            "success": True,
            "message": f"Document '{document['filename']}' and its embeddings deleted successfully",
            "remaining_documents": artifact['num_documents']
        })
    
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error deleting document: {str(e)}"
        }, status_code=500)


@app.get("/api/stats")
async def get_stats():
    """Get training statistics"""
    artifacts = load_artifacts()
    total_images = sum(a.get('num_images', 0) for a in artifacts)
    
    model_trained = MODEL_PATH.exists()
    
    # Get model training timestamp if available
    model_timestamp = None
    if model_trained:
        model_timestamp = os.path.getmtime(MODEL_PATH)
    
    # Load detailed stats
    training_stats = {}
    if STATS_FILE.exists():
        try:
            with open(STATS_FILE, 'r') as f:
                training_stats = json.load(f)
        except Exception:
            pass
    
    # Check if retraining is needed (simple heuristic: if artifact count changed)
    needs_retraining = False
    if training_stats:
        if len(artifacts) != training_stats.get('total_classes', 0):
            needs_retraining = True
    elif model_trained:
        # If we have a model but no stats, assume we need retraining if we can't verify
        pass
        
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
        "needs_retraining": needs_retraining,
        "last_training_accuracy": training_stats.get("last_training_accuracy"),
        "training_epochs": training_stats.get("training_epochs"),
        "artifacts": artifacts_with_status
    })


if __name__ == "__main__":
    import uvicorn
    print("""
============================================================
      AR Museum Guide - Training API Server               
============================================================

Data: {DATA_DIR}

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
