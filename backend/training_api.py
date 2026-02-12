"""
Training API endpoints for managing artifacts and training the model
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
import json
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
    if request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response

# Data directories
DATA_DIR = Path(__file__).parent.parent / "data"
TRAINING_DIR = DATA_DIR / "training"
DOCUMENTS_DIR = DATA_DIR / "documents"
ARTIFACTS_FILE = DATA_DIR / "artifacts.json"
STATS_FILE = DATA_DIR / "training_stats.json"

# Create directories
TRAINING_DIR.mkdir(parents=True, exist_ok=True)
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files to serve images and documents
app.mount("/static/training", StaticFiles(directory=str(TRAINING_DIR)), name="training_images")
app.mount("/static/documents", StaticFiles(directory=str(DOCUMENTS_DIR)), name="documents_files")

def run_training_task():
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
        results = train_artifact_model(
            data_dir=str(TRAINING_DIR),
            num_epochs=10,
            batch_size=8
        )
        
        # Save stats
        stats = {
            "last_training_accuracy": results['best_accuracy'],
            "training_epochs": results['num_epochs'],
            "last_trained": datetime.now().isoformat(),
            "total_classes": results['num_classes']
        }
        
        with open(STATS_FILE, 'w') as f:
            json.dump(stats, f)
            
        print("Background training completed successfully")
    except Exception as e:
        print(f"Background training failed: {e}")
        import traceback
        traceback.print_exc()

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
    print(f"🔄 Starting background embedding generation for {filename}...")
    try:
        success = rag_service.create_embeddings(
            artifact_id=artifact_id,
            pdf_path=pdf_path,
            document_id=doc_id,
            filename=filename
        )
        if success:
            print(f"✨ Embeddings created for {filename}")
        else:
            print(f"⚠️ Failed to create embeddings for {filename}")
    except Exception as e:
        print(f"❌ Error in background embedding task: {e}")


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
    print(f"📥 Received upload request for artifact: {name}")
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
            print(f"   ✅ Saved image: {filename}")
        
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
                    print(f"   ✅ Saved document: {filename}")
                    
                    # Create embeddings (BACKGROUND TASK)
                    if RAG_ENABLED:
                        print(f"   ⏳ Scheduled background embedding generation for {filename}")
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
        print(f"✅ Artifact '{name}' added successfully with ID {artifact_id}")
        
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
            "num_documents": artifact.get("num_documents", 0)
        })
    
    return JSONResponse({
        "success": True,
        "artifacts": summary_artifacts,
        "total": len(summary_artifacts)
    })


@app.post("/api/model/train")
async def train_model(background_tasks: BackgroundTasks):
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
        try:
            from train_model import train_artifact_model
        except ImportError:
            # Fallback for when running from different directory
            import sys
            sys.path.append(str(Path(__file__).parent))
            from train_model import train_artifact_model
        
        # Train the model synchronously for now to report success/failure immediately
        # (User asked for it to work, so we ensure it runs)
        results = train_artifact_model(
            data_dir=str(TRAINING_DIR),
            num_epochs=10,
            batch_size=8
        )
        
        # Save training stats
        stats_file = DATA_DIR / "training_stats.json"
        
        # Calculate stats
        stats = {
            "last_training_accuracy": results['best_accuracy'],
            "training_epochs": results['num_epochs'],
            "last_trained": datetime.now().isoformat(),
            "total_classes": results['num_classes']
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f)
        
        return JSONResponse({
            "success": True,
            "message": "Model trained successfully!",
            "results": results
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "message": f"Training failed: {str(e)}"
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
                print(f"   ⏳ Scheduled background embedding generation for {original_filename}")
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
    
    model_path = Path(__file__).parent.parent / "models" / "artifact_model.pth"
    model_trained = model_path.exists()
    
    # Get model training timestamp if available
    model_timestamp = None
    if model_trained:
        model_timestamp = os.path.getmtime(model_path)
    
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
