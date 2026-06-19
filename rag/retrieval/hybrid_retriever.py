"""
rag/retrieval/hybrid_retriever.py
==================================
Hybrid retriever: fuses dense (semantic) + sparse (BM25) results
using Reciprocal Rank Fusion (RRF).

Why RRF?
  Dense and BM25 scores are on completely different scales and cannot
  be directly combined (cosine in [0,1] vs BM25 in [0, ∞]).
  RRF uses only the RANK of each result, not the raw score, making it
  scale-invariant and extremely robust in practice.

RRF Formula:
  For a chunk appearing at rank r in a result list:
      rrf_score = 1 / (K + r)     where K = 60 (standard constant)

  If a chunk appears in multiple lists, its scores are summed.
  A chunk ranked 1st in both lists gets the highest combined score.

Example:
  Dense:  [A@1, B@2, C@3, D@4 ...]
  Sparse: [C@1, A@2, E@3, B@4 ...]

  RRF(A) = 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252
  RRF(C) = 1/(60+3) + 1/(60+1) = 0.01587 + 0.01639 = 0.03226
  RRF(B) = 1/(60+2) + 1/(60+4) = 0.01613 + 0.01563 = 0.03175

  Final order: A > C > B > E > D ...

Usage:
    retriever = HybridRetriever(dense_retriever, sparse_retriever)
    results   = retriever.retrieve("NVDA supply chain risks", top_k=5)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from rag.config import settings
from rag.retrieval.dense_retriever import DenseRetriever
from rag.retrieval.sparse_retriever import SparseRetriever
from rag.vectorstore.qdrant_store import QueryResult

logger = logging.getLogger(__name__)

# Standard RRF smoothing constant — 60 is the widely accepted default
_RRF_K = 60


class HybridRetriever:
    """
    Combines dense and sparse retrieval via Reciprocal Rank Fusion.

    Args:
        dense:  Initialised DenseRetriever.
        sparse: Initialised SparseRetriever (with index already built).
        dense_top_k:  Candidates to fetch from dense stage.
        sparse_top_k: Candidates to fetch from sparse stage.
    """

    def __init__(
        self,
        dense:  DenseRetriever,
        sparse: SparseRetriever,
        dense_top_k:  Optional[int] = None,
        sparse_top_k: Optional[int] = None,
    ) -> None:
        self._dense       = dense
        self._sparse      = sparse
        self._dense_top_k  = dense_top_k  or settings.dense_top_k
        self._sparse_top_k = sparse_top_k or settings.sparse_top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[QueryResult]:
        """
        Retrieve and fuse results from dense + sparse retrievers.

        Steps:
          1. Dense retrieval  → up to dense_top_k candidates
          2. Sparse retrieval → up to sparse_top_k candidates
          3. RRF fusion       → combined, deduplicated ranking
          4. Return top_k from fused results

        Args:
            query:   Raw user query string.
            top_k:   Final number of results to return.
                     Defaults to settings.rerank_top_n * 3
                     (gives reranker enough to work with).
            filters: Metadata filter dict passed to both retrievers.

        Returns:
            Fused list of QueryResult with RRF scores, sorted descending.
        """
        # Default: fetch enough for the reranker to have candidates
        k = top_k or (settings.rerank_top_n * 6)

        # ── Stage 1: retrieve from both sources
        dense_results  = self._dense.retrieve(
            query, top_k=self._dense_top_k, filters=filters
        )
        sparse_results = self._sparse.retrieve(
            query, top_k=self._sparse_top_k, filters=filters
        )

        logger.debug(
            "Hybrid | dense=%d | sparse=%d | query=%r",
            len(dense_results),
            len(sparse_results),
            query[:60],
        )

        # ── Stage 2: RRF fusion
        fused = self._rrf_fuse(dense_results, sparse_results)

        # ── Stage 3: return top_k
        top = fused[:k]

        logger.debug(
            "Hybrid after RRF | unique chunks=%d | returning top %d",
            len(fused),
            len(top),
        )

        return top

    # -----------------------------------------------------------------------
    # Internal — RRF fusion
    # -----------------------------------------------------------------------

    @staticmethod
    def _rrf_fuse(
        *result_lists: list[QueryResult],
    ) -> list[QueryResult]:
        """
        Apply Reciprocal Rank Fusion across multiple result lists.

        For each chunk, sums 1/(K + rank) across all lists it appears in.
        Chunks not appearing in a list simply get no contribution from it.

        Args:
            *result_lists: Variable number of QueryResult lists to fuse.

        Returns:
            Deduplicated list of QueryResult with .score = RRF score,
            sorted by RRF score descending.
        """
        # chunk_id → accumulated RRF score
        rrf_scores: dict[str, float] = {}
        # chunk_id → representative QueryResult (for metadata + text)
        chunk_map: dict[str, QueryResult] = {}

        for result_list in result_lists:
            for rank, result in enumerate(result_list, start=1):
                cid = result.chunk_id
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (_RRF_K + rank))
                # Keep the first-seen result as the representative
                if cid not in chunk_map:
                    chunk_map[cid] = result

        # Build final list with RRF scores
        fused: list[QueryResult] = []
        for cid, rrf_score in sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        ):
            result = chunk_map[cid]
            # Replace raw retrieval score with RRF score
            fused_result = QueryResult(
                chunk_id=result.chunk_id,
                text=result.text,
                metadata=result.metadata,
                score=rrf_score,
            )
            fused.append(fused_result)

        return fused
