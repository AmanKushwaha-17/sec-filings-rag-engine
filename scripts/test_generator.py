"""
scripts/test_generator.py
=========================
Tests the Generator layer by running a full retrieval and passing the chunks to Groq.
"""

import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.config import settings
from rag.embeddings.nvidia_embedder import NvidiaEmbedder
from rag.vectorstore.qdrant_store import QdrantStore
from rag.retrieval.dense_retriever import DenseRetriever
from rag.retrieval.sparse_retriever import SparseRetriever
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.retrieval.reranker import Reranker
from rag.llm.generator import RAGGenerator
from rag.llm.query_analyzer import QueryAnalyzer

def main():
    total_t0 = time.time()
    print("="*80)
    print("  TESTING LANGGRAPH AGENT")
    print("="*80)

    # 1. Load Vector Store
    print("\n[1] Loading Qdrant Database...")
    store = QdrantStore()
    
    # 2. Load Embedder
    print("\n[2] Loading Embedder...")
    embedder = NvidiaEmbedder()
    dense_retriever = DenseRetriever(embedder=embedder, store=store)

    # 3. Load Sparse Retriever
    print("\n[3] Building BM25 Sparse Index (loading from Qdrant)...")
    chunks = store.get_all_chunks()
    sparse_retriever = SparseRetriever()
    sparse_retriever.build_index(chunks)

    # 4. Initialise Retrievers
    print("\n[4] Initialising Hybrid & Reranker...")
    hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever)
    reranker = Reranker()
    
    # 5. Init Generator & Analyzer
    print("\n[5] Initialising Groq Generator & Analyzer...")
    generator = RAGGenerator()
    query_analyzer = QueryAnalyzer()

    # The Queries
    queries = [
        "Which of the 33 companies disclosed highest China export restriction exposure?",
        "Summarise top 5 supply chain risks appearing across majority of companies",

    ]

    for query in queries:
        print("\n" + "="*80)
        print(f"QUERY: {query}")
        print("-" * 80)
        
        # Retrieve
        print("Retrieving documents...")
        t0 = time.time()
        
        # Extract filters
        filters = query_analyzer.analyze(query)
        if filters:
            print(f"-> Applied Metadata Filter: {filters}")
            
        candidates = hybrid_retriever.retrieve(query, top_k=settings.dense_top_k, filters=filters)
        reranked = reranker.rerank(query, candidates, top_n=settings.rerank_top_n)
        print(f"Retrieved {len(reranked)} chunks in {time.time()-t0:.2f}s")
        
        # Generate
        print("Generating answer via Groq...")
        t0 = time.time()
        answer = generator.generate_answer(query, reranked)
        print(f"Generated answer in {time.time()-t0:.2f}s\n")
        
        import textwrap
        print("FINAL ANSWER:")
        print(textwrap.fill(answer, width=80))

    print("="*80)
    print(f"\nTOTAL PIPELINE EXECUTION TIME: {time.time() - total_t0:.2f} seconds\n")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
