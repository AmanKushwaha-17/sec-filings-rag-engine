"""
scripts/evaluate_retrieval.py
=============================
A diagnostic script to evaluate and compare the performance of:
1. Dense Retrieval (FAISS / Nvidia Embeddings)
2. Sparse Retrieval (BM25 Keyword Matching)
3. Hybrid Retrieval (RRF fusion of Dense + Sparse)
4. Reranked Hybrid (CrossEncoder)

This allows us to quantitatively and qualitatively evaluate the pipeline.
"""

import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.config import settings
from rag.embeddings.nvidia_embedder import NvidiaEmbedder
from rag.vectorstore.faiss_store import FAISSStore
from rag.retrieval.dense_retriever import DenseRetriever
from rag.retrieval.sparse_retriever import SparseRetriever
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.retrieval.reranker import Reranker

def main():
    print("="*80)
    print("  RAG RETRIEVAL PIPELINE EVALUATION")
    print("="*80)

    # 1. Load Stores
    print("\n[1] Loading FAISS Database...")
    store = FAISSStore(index_dir=settings.faiss_index_dir)
    print(f"    Loaded {store.count():,} chunks.")

    print("\n[2] Loading Embedder...")
    embedder = NvidiaEmbedder()
    dense_retriever = DenseRetriever(embedder=embedder, store=store)

    print("\n[3] Building BM25 Sparse Index (from FAISS metadata)...")
    t0 = time.time()
    # Extract all chunks from metadata to build BM25
    from rag.ingestion.models import Chunk, ChunkMetadata
    
    # We need to construct Chunk objects from the FAISSStore metadata dictionary
    chunks = []
    for meta_dict in store._meta:
        # Clone dict, removing text and chunk_id
        meta_fields = {k: v for k, v in meta_dict.items() if k not in ("chunk_id", "text")}
        metadata = ChunkMetadata(**meta_fields)
        chunks.append(Chunk(
            chunk_id=meta_dict["chunk_id"],
            text=meta_dict["text"],
            metadata=metadata
        ))
        
    sparse_retriever = SparseRetriever()
    sparse_retriever.build_index(chunks)
    print(f"    BM25 Index built in {time.time()-t0:.1f} seconds.")

    print("\n[4] Initialising Hybrid & Reranker...")
    hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever)
    reranker = Reranker()

    queries = [
        "What are the major supply chain risks involving TSMC or Taiwan?",
        "How is artificial intelligence driving revenue growth?",
        "What export control restrictions to China impact the business?"
    ]

    for query in queries:
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        print(f"{'='*80}")

        # A) Dense Only
        t0 = time.time()
        dense_res = dense_retriever.retrieve(query, top_k=3)
        print(f"\n--- A) DENSE ONLY ({time.time()-t0:.2f}s) ---")
        for i, res in enumerate(dense_res, 1):
            print(f"[{i}] {res.metadata.ticker} | Score: {res.score:.4f} | Snippet: {res.text[:120].replace('\n', ' ')}...")

        # B) Sparse Only
        t0 = time.time()
        sparse_res = sparse_retriever.retrieve(query, top_k=3)
        print(f"\n--- B) SPARSE ONLY (BM25) ({time.time()-t0:.2f}s) ---")
        for i, res in enumerate(sparse_res, 1):
            print(f"[{i}] {res.metadata.ticker} | Score: {res.score:.4f} | Snippet: {res.text[:120].replace('\n', ' ')}...")

        # C) Hybrid
        t0 = time.time()
        hybrid_res = hybrid_retriever.retrieve(query, top_k=5)
        print(f"\n--- C) HYBRID (RRF Fusion) ({time.time()-t0:.2f}s) ---")
        for i, res in enumerate(hybrid_res[:3], 1):
            print(f"[{i}] {res.metadata.ticker} | RRF Score: {res.score:.4f} | Snippet: {res.text[:120].replace('\n', ' ')}...")

        # D) Reranked
        t0 = time.time()
        reranked_res = reranker.rerank(query, hybrid_res, top_n=3)
        print(f"\n--- D) HYBRID + CROSS-ENCODER RERANKER ({time.time()-t0:.2f}s) ---")
        for i, res in enumerate(reranked_res, 1):
            print(f"[{i}] {res.metadata.ticker} | CeScore: {res.score:.4f} | Snippet: {res.text[:120].replace('\n', ' ')}...")

    print("\nEvaluation complete! Notice how Sparse/Hybrid finds exact keywords, and Reranker orders by semantic relevance.")

if __name__ == "__main__":
    main()
