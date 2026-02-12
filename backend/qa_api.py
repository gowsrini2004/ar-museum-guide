"""
Q&A API Server for Artifact Question Answering
Handles user questions about artifacts using RAG
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import RAG service
from rag_service import get_rag_service

app = FastAPI(title="AR Museum Guide - Q&A API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG service
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATA_DIR = Path(__file__).parent.parent / "data"
rag_service = get_rag_service(str(DATA_DIR), GEMINI_API_KEY)


class QuestionRequest(BaseModel):
    artifact_id: str
    artifact_name: str
    question: str


@app.post("/api/ask")
async def ask_question(request: QuestionRequest):
    """
    Ask a question about an artifact
    """
    try:
        result = rag_service.query_artifact(
            artifact_id=request.artifact_id,
            question=request.question,
            artifact_name=request.artifact_name
        )
        
        return JSONResponse(result)
    
    except Exception as e:
        return JSONResponse({
            "success": False,
            "answer": f"An error occurred: {str(e)}",
            "sources": []
        }, status_code=500)


@app.get("/api/health")
async def health_check():
    """Check if the service is ready"""
    has_api_key = GEMINI_API_KEY is not None and GEMINI_API_KEY != ""
    
    return JSONResponse({
        "status": "healthy" if has_api_key else "missing_api_key",
        "gemini_configured": has_api_key,
        "message": "Q&A service is ready" if has_api_key else "Please configure GEMINI_API_KEY in .env file"
    })


@app.get("/api/artifact/{artifact_id}/stats")
async def get_artifact_stats(artifact_id: str):
    """Get statistics about an artifact's documents"""
    try:
        chunk_count = rag_service.get_document_count(artifact_id)
        
        return JSONResponse({
            "artifact_id": artifact_id,
            "total_chunks": chunk_count,
            "has_documents": chunk_count > 0
        })
    
    except Exception as e:
        return JSONResponse({
            "error": str(e)
        }, status_code=500)


@app.get("/api/artifact/{artifact_id}/documents")
async def list_artifact_documents(artifact_id: str):
    """List all documents for an artifact"""
    try:
        documents = rag_service.list_documents(artifact_id)
        
        return JSONResponse({
            "success": True,
            "artifact_id": artifact_id,
            "documents": documents,
            "total": len(documents)
        })
    
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "documents": []
        }, status_code=500)


if __name__ == "__main__":
    import uvicorn
    print("""
╔══════════════════════════════════════════════════════════╗
║     AR Museum Guide - Q&A API Server                    ║
╚══════════════════════════════════════════════════════════╝

🌐 API: http://localhost:8002
🤖 Powered by: Google Gemini + ChromaDB

Endpoints:
  - POST /api/ask - Ask a question about an artifact
  - GET /api/health - Check service health
  - GET /api/artifact/{id}/stats - Get document statistics

Press Ctrl+C to stop
""")
    uvicorn.run(app, host="0.0.0.0", port=8002)
