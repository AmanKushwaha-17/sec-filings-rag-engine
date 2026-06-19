import time
import os
import threading
from contextlib import asynccontextmanager
from typing import List, Optional, Dict
import json
import glob

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
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
    chunk_id: str
    ticker: str
    company_name: str
    section: str
    filing_date: str
    rerank_score: float
    char_start: int
    char_end: int
    text: str

class QueryAnalyzerInfo(BaseModel):
    detected_ticker: Optional[str] = None
    metadata_filtering_applied: bool = False

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceSnippet]
    analyzer_info: QueryAnalyzerInfo
    execution_time_seconds: float

class APIErrorResponse(BaseModel):
    code: str
    message: str
    retryable: bool

# -----------------------------------------------------------------------------
# Global State Management
# -----------------------------------------------------------------------------
def sync_init(app: FastAPI):
    """Heavy synchronous initialization that runs in a background thread."""
    try:
        t0 = time.time()
        
        # 1. Load Qdrant Store
        print("[1] Connecting to Qdrant Cloud Database...")
        store = QdrantStore()
        
        # 2. Load Embedder
        print("[2] Loading Nvidia Embedder...")
        embedder = NvidiaEmbedder()
        dense_retriever = DenseRetriever(embedder=embedder, store=store)

        # 3. Load Sparse Retriever (BM25) & Extract Companies from Index
        print("[3] Building BM25 Sparse Index in RAM (Downloading from Qdrant)...")
        chunks = store.get_all_chunks()
        sparse_retriever = SparseRetriever()
        sparse_retriever.build_index(chunks)
        
        # Build companies list strictly from indexed chunks
        print("[3b] Extracting active company list from indexed chunks...")
        companies = {}
        for c in chunks:
            t = getattr(c.metadata, "ticker", "")
            n = getattr(c.metadata, "company_name", "")
            if t and n:
                companies[t] = n
                
        app.state.companies = [{"ticker": k, "company_name": v} for k, v in companies.items()]
        app.state.companies.sort(key=lambda x: x["ticker"])

        # 4. Init Hybrid & Reranker
        print("[4] Loading Reranker Weights...")
        app.state.hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever)
        app.state.reranker = Reranker()
        
        # 5. Init Generator & LLM
        print("[5] Initialising Groq Generator & LLMs...")
        app.state.generator = RAGGenerator()
        app.state.query_analyzer = QueryAnalyzer()
        
        app.state.ready = True
        print(f"COLD START COMPLETE in {time.time() - t0:.2f} seconds!")
        print("="*80)
        
    except Exception as e:
        print(f"CRITICAL ERROR during initialization: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    app.state.companies = []
    
    print("="*80)
    print("INITIALISING ML MODELS IN BACKGROUND THREAD...")
    print("="*80)
    
    # Launch heavy initialization in a background thread so FastAPI can start serving immediately
    thread = threading.Thread(target=sync_init, args=(app,))
    thread.daemon = True
    thread.start()
    
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
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=APIErrorResponse(
            code="HTTP_ERROR",
            message=exc.detail,
            retryable=True
        ).model_dump()
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"INTERNAL ERROR: {exc}") # Log raw error for backend debugging
    return JSONResponse(
        status_code=500,
        content=APIErrorResponse(
            code="INTERNAL_ERROR",
            message="An unexpected internal error occurred. Please try again later.",
            retryable=True
        ).model_dump()
    )

@app.get("/api/health")
async def health_endpoint():
    if getattr(app.state, "ready", False):
        return {"status": "API ready"}
    return {"status": "model cold start in progress"}

@app.get("/api/companies")
async def companies_endpoint():
    return getattr(app.state, "companies", [])

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not getattr(app.state, "ready", False):
        raise HTTPException(
            status_code=503, 
            detail="AI models are currently warming up. Please try again in a few seconds."
        )
        
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
                chunk_id=chunk.chunk_id,
                ticker=getattr(chunk.metadata, "ticker", "UNKNOWN"),
                company_name=getattr(chunk.metadata, "company_name", "UNKNOWN"),
                section=getattr(chunk.metadata, "section", "UNKNOWN"),
                filing_date=getattr(chunk.metadata, "filing_date", "UNKNOWN"),
                rerank_score=round(chunk.score, 4) if chunk.score is not None else 0.0,
                char_start=getattr(chunk.metadata, "char_start", 0),
                char_end=getattr(chunk.metadata, "char_end", 0),
                text=chunk.text
            ))
            
        execution_time = round(time.time() - t0, 2)
        
        analyzer_info = QueryAnalyzerInfo(
            detected_ticker=filters.get("ticker") if filters else None,
            metadata_filtering_applied=bool(filters)
        )
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            analyzer_info=analyzer_info,
            execution_time_seconds=execution_time
        )
        
    except Exception as e:
        print(f"Error during generation: {e}")
        # The global_exception_handler will catch this and format it as APIErrorResponse
        raise

# -----------------------------------------------------------------------------
# Mount Frontend
# -----------------------------------------------------------------------------
frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend_2", "dist")
# Create the frontend_2/dist directory if it doesn't exist yet to prevent startup crashes
os.makedirs(frontend_path, exist_ok=True)

# Note: We mount the "dist" directory which contains the built Vite output.
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
