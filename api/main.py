import time
import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag.config import settings
from rag.embeddings.nvidia_embedder import NvidiaEmbedder
from rag.vectorstore.qdrant_store import QdrantStore
from rag.retrieval.dense_retriever import DenseRetriever
from rag.retrieval.sparse_retriever import SparseRetriever
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.retrieval.reranker import Reranker
from rag.llm.generator import RAGGenerator
from rag.llm.query_analyzer import QueryAnalyzer
from rag.ingestion.models import Chunk, ChunkMetadata

# -----------------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str

class SourceSnippet(BaseModel):
    id: int
    ticker: str
    company_name: str
    text: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceSnippet]
    execution_time_seconds: float

# -----------------------------------------------------------------------------
# Global State Management
# -----------------------------------------------------------------------------
# We load the heavy ML models exactly ONCE when the server starts.
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("="*80)
    print("INITIALISING ML MODELS (COLD START)...")
    print("="*80)
    
    t0 = time.time()
    # 1. Load Qdrant Store
    print("[1] Connecting to Qdrant Cloud Database...")
    store = QdrantStore()
    
    # 2. Load Embedder
    print("[2] Loading Nvidia Embedder...")
    embedder = NvidiaEmbedder()
    dense_retriever = DenseRetriever(embedder=embedder, store=store)

    # 3. Load Sparse Retriever (BM25)
    print("[3] Building BM25 Sparse Index in RAM (Downloading from Qdrant)...")
    chunks = store.get_all_chunks()
    sparse_retriever = SparseRetriever()
    sparse_retriever.build_index(chunks)

    # 4. Init Hybrid & Reranker
    print("[4] Loading Reranker Weights...")
    app.state.hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever)
    app.state.reranker = Reranker()
    
    # 5. Init Generator & LLM
    print("[5] Initialising Groq Generator & LLMs...")
    app.state.generator = RAGGenerator()
    app.state.query_analyzer = QueryAnalyzer()
    
    print(f"COLD START COMPLETE in {time.time() - t0:.2f} seconds!")
    print("="*80)
    
    yield
    
    # Cleanup on shutdown
    print("Shutting down ML models...")

# -----------------------------------------------------------------------------
# App Initialization
# -----------------------------------------------------------------------------
app = FastAPI(lifespan=lifespan, title="Company SEC RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        t0 = time.time()
        
        # 0. Analyze Query for Metadata Filters
        filters = app.state.query_analyzer.analyze(request.query)
        
        # 1. Retrieve & Rerank
        candidates = app.state.hybrid_retriever.retrieve(request.query, top_k=settings.dense_top_k, filters=filters)
        reranked = app.state.reranker.rerank(request.query, candidates, top_n=settings.rerank_top_n)
        
        # 2. Generate Answer
        answer = app.state.generator.generate_answer(request.query, reranked)
        
        # 3. Format Sources for the UI
        sources = []
        for i, chunk in enumerate(reranked):
            sources.append(SourceSnippet(
                id=i+1,
                ticker=getattr(chunk.metadata, "ticker", "UNKNOWN"),
                company_name=getattr(chunk.metadata, "company_name", "UNKNOWN"),
                text=chunk.text
            ))
            
        execution_time = round(time.time() - t0, 2)
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            execution_time_seconds=execution_time
        )
        
    except Exception as e:
        print(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------------------------------
# Mount Frontend
# -----------------------------------------------------------------------------
frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
# Create the frontend directory if it doesn't exist yet to prevent startup crashes
os.makedirs(frontend_path, exist_ok=True)
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
