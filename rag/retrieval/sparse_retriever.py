"""
rag/retrieval/sparse_retriever.py
==================================
Sparse (keyword) retriever using BM25 via LangChain.

BM25 scores chunks based on exact keyword matches — critical for
financial text where specific terms like "TSMC", "export controls",
and ticker symbols must be found precisely.

Why BM25 alongside dense retrieval?
  Dense embeddings find semantically similar chunks even with different
  words. BM25 finds chunks with exact keyword matches. Together via RRF
  they cover both cases.

Usage:
    retriever = SparseRetriever()
    retriever.build_index(all_chunks)
    results = retriever.retrieve("NVDA supply chain risks", top_k=20)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from rag.config import settings
from rag.ingestion.models import Chunk, ChunkMetadata
from rag.vectorstore.qdrant_store import QueryResult

logger = logging.getLogger(__name__)


class SparseRetriever:
    """
    BM25 keyword retriever built on LangChain's BM25Retriever.

    Must call build_index(chunks) before calling retrieve().
    """

    def __init__(self) -> None:
        self._retriever: Optional[BM25Retriever] = None
        self._chunks: list[Chunk] = []

    def build_index(self, chunks: list[Chunk]) -> None:
        """
        Build BM25 index from all chunks.
        Call once at startup after loading chunks.

        Args:
            chunks: All Chunk objects in the corpus.
        """
        if not chunks:
            raise ValueError("Cannot build BM25 index from empty chunk list.")

        self._chunks = chunks

        # Convert chunks to LangChain Documents
        docs = [
            Document(
                page_content=c.text,
                metadata={**c.metadata.to_chroma_dict(), "chunk_id": c.chunk_id},
            )
            for c in chunks
        ]

        self._retriever = BM25Retriever.from_documents(
            docs,
            k=settings.sparse_top_k,
        )

        logger.info("BM25 index built | %d chunks", len(chunks))

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[QueryResult]:
        """
        Retrieve top-K chunks matching the query via BM25.

        Args:
            query:   Raw query string.
            top_k:   Number of results. Defaults to settings.sparse_top_k.
            filters: Optional metadata filter e.g. {"ticker": "NVDA"}.

        Returns:
            List of QueryResult sorted by BM25 score descending.

        Raises:
            RuntimeError: If build_index() has not been called yet.
        """
        if self._retriever is None:
            raise RuntimeError(
                "Call build_index(chunks) before retrieve()."
            )

        k = top_k or settings.sparse_top_k
        self._retriever.k = k

        # BM25 retrieval
        docs = self._retriever.invoke(query)

        # Post-filter by metadata if needed
        if filters:
            docs = [
                d for d in docs
                if self._matches_filters(d.metadata, filters)
            ]

        # Convert to QueryResult
        # BM25Retriever doesn't expose raw scores, so we assign
        # descending rank-based scores (1.0 for rank 1, etc.)
        results = []
        for rank, doc in enumerate(docs[:k]):
            # Normalised rank score: top result = 1.0
            score = 1.0 / (1.0 + rank)
            results.append(QueryResult(
                chunk_id=doc.metadata["chunk_id"],
                text=doc.page_content,
                metadata=self._get_chunk_metadata(doc.metadata["chunk_id"]),
                score=score,
            ))

        logger.debug(
            "Sparse | query=%r | got=%d after filter",
            query[:60],
            len(results),
        )

        return results

    def _get_chunk_metadata(self, chunk_id: str):
        """Look up ChunkMetadata from original chunks list by chunk_id."""
        for c in self._chunks:
            if c.chunk_id == chunk_id:
                return c.metadata
        # Fallback: should never happen if index is built correctly
        raise ValueError(f"chunk_id '{chunk_id}' not found in corpus")

    @staticmethod
    def _matches_filters(metadata: dict, filters: dict[str, Any]) -> bool:
        """
        Check if metadata satisfies all filter conditions.
        Supports equality and list (OR) values.
        """
        for key, value in filters.items():
            meta_val = metadata.get(key)
            if isinstance(value, list):
                if meta_val not in value:
                    return False
            else:
                if meta_val != value:
                    return False
        return True
