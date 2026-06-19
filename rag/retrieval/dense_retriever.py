"""
rag/retrieval/dense_retriever.py
=================================
Dense (semantic) retriever using FAISS vector similarity.

Wraps FAISSStore.query() with a clean interface:
  1. Embed the user query with the BGE query prefix
  2. Run ANN search in FAISS
  3. Return ranked QueryResult list

This is Stage 1a of the hybrid pipeline.

Usage:
    from rag.retrieval.dense_retriever import DenseRetriever

    retriever = DenseRetriever(embedder, store)
    results   = retriever.retrieve("NVDA supply chain risks", top_k=20)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from rag.config import settings
from rag.embeddings.base import BaseEmbedder
from rag.vectorstore.qdrant_store import QdrantStore, QueryResult

logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    Semantic similarity retriever backed by Qdrant.

    Args:
        embedder: Initialised NvidiaEmbedder.
        store:    Initialised QdrantStore.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        store: QdrantStore,
    ) -> None:
        self._embedder = embedder
        self._store    = store

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[QueryResult]:
        """
        Retrieve the top-K semantically similar chunks.

        Args:
            query:   User query string (raw, no prefix needed here).
            top_k:   Number of results. Defaults to settings.dense_top_k.
            filters: Metadata filter dict e.g. {"ticker": "NVDA"}.

        Returns:
            List of QueryResult sorted by cosine similarity (descending).
        """
        k = top_k or settings.dense_top_k

        # Embed query with BGE instruction prefix
        q_vector = self._embedder.embed_query(query)

        results = self._store.query(
            vector=q_vector,
            top_k=k,
            filters=filters,
        )

        logger.debug(
            "Dense retrieval | query=%r | top_k=%d | filters=%s | got=%d",
            query[:60],
            k,
            filters,
            len(results),
        )

        return results
