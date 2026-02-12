"""
RAG Service for Artifact Q&A System
Handles PDF processing, embedding generation, and question answering using Google Gemini
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Optional
import hashlib

# PDF Processing
import PyPDF2
import pdfplumber

# Embeddings
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Google Gemini
import google.genai as genai
from chromadb.utils import embedding_functions


class RAGService:
    """Service for handling PDF processing, embeddings, and Q&A"""
    
    def __init__(self, data_dir: str, api_key: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.documents_dir = self.data_dir / "documents"
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.data_dir / "chroma_db"),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize embedding model (lazy loaded)
        self._embedding_model = None
        
        # Initialize Gemini
        if api_key:
            self.client = genai.Client(api_key=api_key)
            # Use models/ prefix as per 2026 SDK discovery
            self.model_id = 'models/gemini-flash-lite-latest'
        else:
            self.client = None
            self.model_id = None
            
    @property
    def embedding_model(self):
        """Lazy load the embedding model and return as a ChromaDB compliant function"""
        if self._embedding_model is None:
            # Use specific embedding function wrapper for ChromaDB
            self._embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name='all-MiniLM-L6-v2'
            )
        return self._embedding_model

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Extract text from PDF with page numbers
        Returns list of dicts with page_number and text
        """
        chunks = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if text and text.strip():
                        chunks.append({
                            'page_number': page_num,
                            'text': text.strip()
                        })
        except Exception as e:
            print(f"Error extracting text with pdfplumber: {e}")
            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num, page in enumerate(pdf_reader.pages, start=1):
                        text = page.extract_text()
                        if text and text.strip():
                            chunks.append({
                                'page_number': page_num,
                                'text': text.strip()
                            })
            except Exception as e2:
                print(f"Error extracting text with PyPDF2: {e2}")
                return []
        
        return chunks
    
    def split_into_chunks(self, text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap
        
        return chunks
    
    def create_embeddings(self, artifact_id: str, pdf_path: str, document_id: str, filename: str = None) -> bool:
        """
        Process PDF and create embeddings in ChromaDB
        Returns True if successful
        """
        try:
            # Extract filename from path if not provided
            if filename is None:
                filename = Path(pdf_path).name
            
            # Extract text from PDF
            pages = self.extract_text_from_pdf(pdf_path)
            if not pages:
                print(f"No text extracted from {pdf_path}")
                return False
            
            # Get or create collection for this artifact
            collection_name = f"artifact_{artifact_id}"
            collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"artifact_id": artifact_id},
                embedding_function=self.embedding_model
            )
            
            # Process each page
            all_chunks = []
            all_metadatas = []
            all_ids = []
            
            for page_data in pages:
                page_num = page_data['page_number']
                page_text = page_data['text']
                
                # Split page into chunks
                chunks = self.split_into_chunks(page_text)
                
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_id = f"{document_id}_p{page_num}_c{chunk_idx}"
                    all_chunks.append(chunk)
                    all_metadatas.append({
                        'document_id': document_id,
                        'filename': filename,
                        'page_number': page_num,
                        'chunk_index': chunk_idx,
                        'artifact_id': artifact_id
                    })
                    all_ids.append(chunk_id)
            
            # Add to ChromaDB (it will generate embeddings automatically)
            if all_chunks:
                collection.add(
                    documents=all_chunks,
                    metadatas=all_metadatas,
                    ids=all_ids
                )
                print(f"Added {len(all_chunks)} chunks for {filename} to collection {collection_name}")
                return True
            
            return False
            
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            return False
    
    def query_artifact(self, artifact_id: str, question: str, artifact_name: str = "") -> Dict[str, any]:
        """
        Query the artifact's documents and generate an answer using Gemini
        """
        try:
            collection_name = f"artifact_{artifact_id}"
            
            # Check if collection exists
            try:
                collection = self.chroma_client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_model
                )
            except Exception:
                return {
                    'success': False,
                    'answer': f"No documents found for this artifact. Please upload PDF documents first.",
                    'sources': []
                }
            
            # Query the collection
            results = collection.query(
                query_texts=[question],
                n_results=5  # Get top 5 most relevant chunks
            )
            
            if not results['documents'] or not results['documents'][0]:
                return {
                    'success': False,
                    'answer': "I couldn't find relevant information to answer your question.",
                    'sources': []
                }
            
            # Extract relevant chunks and metadata
            relevant_chunks = results['documents'][0]
            metadatas = results['metadatas'][0]
            
            # Build context from retrieved chunks
            context = "\n\n".join([f"[Page {meta['page_number']}]: {chunk}" 
                                   for chunk, meta in zip(relevant_chunks, metadatas)])
            
            # Generate answer using Gemini
            if not self.client:
                return {
                    'success': False,
                    'answer': "Gemini API is not configured. Please set your API key.",
                    'sources': []
                }
            
            prompt = f"""You are a knowledgeable museum guide assistant. Answer the question about the artifact based on the provided context.

Artifact: {artifact_name}

Context from documents:
{context}

Question: {question}

Instructions:
- Answer the question based ONLY on the provided context
- Be informative and engaging
- If the context doesn't contain enough information to answer, say so
- Keep your answer concise but complete
- Cite page numbers when relevant

Answer:"""
            
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt
            )
            answer = response.text
            
            # Format sources
            sources = []
            seen_pages = set()
            for meta in metadatas:
                page_num = meta['page_number']
                if page_num not in seen_pages:
                    sources.append({
                        'page': page_num,
                        'document_id': meta['document_id']
                    })
                    seen_pages.add(page_num)
            
            return {
                'success': True,
                'answer': answer,
                'sources': sources,
                'context_used': len(relevant_chunks)
            }
            
        except Exception as e:
            print(f"Error querying artifact: {e}")
            return {
                'success': False,
                'answer': f"An error occurred while processing your question: {str(e)}",
                'sources': []
            }
    
    def delete_artifact_embeddings(self, artifact_id: str) -> bool:
        """Delete all embeddings for an artifact"""
        try:
            collection_name = f"artifact_{artifact_id}"
            self.chroma_client.delete_collection(name=collection_name)
            print(f"Deleted collection {collection_name}")
            return True
        except Exception as e:
            print(f"Error deleting embeddings: {e}")
            return False
    
    def delete_document_embeddings(self, artifact_id: str, document_id: str) -> bool:
        """Delete embeddings for a specific document"""
        try:
            collection_name = f"artifact_{artifact_id}"
            collection = self.chroma_client.get_collection(name=collection_name)
            
            # Get all IDs for this document
            results = collection.get(
                where={"document_id": document_id}
            )
            
            if results['ids']:
                collection.delete(ids=results['ids'])
                print(f"Deleted {len(results['ids'])} chunks for document {document_id}")
            
            return True
        except Exception as e:
            print(f"Error deleting document embeddings: {e}")
            return False
    
    def get_document_count(self, artifact_id: str) -> int:
        """Get number of documents in an artifact's collection"""
        try:
            collection_name = f"artifact_{artifact_id}"
            collection = self.chroma_client.get_collection(name=collection_name)
            return collection.count()
        except Exception:
            return 0
    
    def list_documents(self, artifact_id: str) -> List[Dict]:
        """List all documents for an artifact with metadata"""
        try:
            collection_name = f"artifact_{artifact_id}"
            try:
                collection = self.chroma_client.get_collection(name=collection_name)
            except Exception:
                # Collection doesn't exist, which is fine (no docs)
                return []
            
            # Get all items in collection
            results = collection.get()
            
            # Debug logging
            print(f"Listing documents for {artifact_id}, found {len(results.get('ids', []))} chunks")
            
            metadatas = results.get('metadatas')
            if not metadatas:
                return []
            
            # Group by document_id
            documents = {}
            for metadata in metadatas:
                if not metadata:
                    continue
                    
                doc_id = metadata.get('document_id')
                filename = metadata.get('filename', 'Unknown')
                
                if doc_id and doc_id not in documents:
                    documents[doc_id] = {
                        'document_id': doc_id,
                        'filename': filename,
                        'chunks': 0
                    }
                
                if doc_id:
                    documents[doc_id]['chunks'] += 1
            
            return list(documents.values())
        except Exception as e:
            print(f"Error listing documents: {e}")
            return []
    
    def get_artifact_stats(self, artifact_id: str) -> Dict:
        """Get detailed stats for an artifact"""
        try:
            collection_name = f"artifact_{artifact_id}"
            collection = self.chroma_client.get_collection(name=collection_name)
            
            results = collection.get()
            total_chunks = len(results.get('ids', []))
            
            # Count unique documents
            document_ids = set()
            for metadata in results.get('metadatas', []):
                doc_id = metadata.get('document_id')
                if doc_id:
                    document_ids.add(doc_id)
            
            return {
                'total_chunks': total_chunks,
                'total_documents': len(document_ids),
                'has_documents': len(document_ids) > 0
            }
        except Exception:
            return {
                'total_chunks': 0,
                'total_documents': 0,
                'has_documents': False
            }


# Singleton instance
_rag_service = None

def get_rag_service(data_dir: str = None, api_key: str = None) -> RAGService:
    """Get or create RAG service instance"""
    global _rag_service
    if _rag_service is None:
        if data_dir is None:
            data_dir = str(Path(__file__).parent.parent / "data")
        _rag_service = RAGService(data_dir, api_key)
    return _rag_service
