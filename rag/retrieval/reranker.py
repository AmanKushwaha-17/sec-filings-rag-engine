"""
rag/retrieval/reranker.py
==========================
Cross-encoder reranker: re-scores the top candidates from hybrid
retrieval and selects the most relevant chunks for the LLM.

Why rerank?
  Bi-encoders (FAISS + BM25) embed query and chunks independently.
  A cross-encoder reads query + chunk TOGETHER, producing a much more
  accurate relevance score — but is too slow to run on all 4,873 chunks.

  Two-stage pipeline:
    Stage 1 (fast): hybrid retrieval → top 30 candidates  (~100ms)
    Stage 2 (precise): cross-encoder → re-score 30, keep 5 (~500ms)

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Size:     22 MB (tiny, CPU-friendly)
  - Training: MS MARCO passage ranking benchmark
  - Output:   Raw logit score (higher = more relevant)
  - Latency:  ~500ms for 30 pairs on CPU

Usage:
    from rag.retrieval.reranker import Reranker

    reranker = Reranker()
    reranked = reranker.rerank(
        query="NVDA supply chain risks",
        results=hybrid_results,   # list of QueryResult (top 30)
        top_n=5,
    )
    # reranked → top 5 most relevant chunks, ready for LLM
"""

from __future__ import annotations

import logging
from typing import Optional

try:
    from sentence_transformers import CrossEncoder
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False

from rag.config import settings
from rag.vectorstore.qdrant_store import QueryResult

logger = logging.getLogger(__name__)

# Cross-encoder model — 22MB, fast on CPU, trained on MS MARCO
_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """
    Cross-encoder reranker for the final retrieval stage.

    Takes the top-N candidates from hybrid retrieval and re-scores
    each (query, chunk) pair with a cross-encoder, which reads both
    the query and the chunk text together for much higher accuracy.

    Args:
        model_name: HuggingFace cross-encoder model ID.
        device:     Torch device ("cpu", "cuda", "mps").
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: Optional[str] = None,
    ) -> None:
        self._model_name = model_name
        self._device     = device or settings.embedding_device

        if not _HAS_SENTENCE_TRANSFORMERS:
            logger.warning("sentence-transformers not installed. Reranker will act as a pass-through.")
            self._model = None
            return

        logger.info(
            "Loading cross-encoder '%s' on device='%s'...",
            self._model_name,
            self._device,
        )

        self._model = CrossEncoder(
            self._model_name,
            device=self._device,
            max_length=512,  # cross-encoders have shorter context than bi-encoders
        )

        logger.info("Reranker ready | model=%s", self._model_name)

    def rerank(
        self,
        query: str,
        results: list[QueryResult],
        top_n: Optional[int] = None,
    ) -> list[QueryResult]:
        """
        Re-score a list of QueryResult objects using the cross-encoder.

        Args:
            query:   Original user query string.
            results: Candidate chunks from hybrid retrieval (typically 20-30).
            top_n:   Number of results to return after reranking.
                     Defaults to settings.rerank_top_n (5).

        Returns:
            Top top_n QueryResult objects sorted by cross-encoder score
            (most relevant first). Scores are replaced with the
            cross-encoder's logit output (higher = more relevant).

        Example:
            >>> reranked = reranker.rerank("NVDA risks", hybrid_results, top_n=5)
            >>> len(reranked)
            5
            >>> reranked[0].score  # cross-encoder logit
            8.742
        """
        n = top_n or settings.rerank_top_n

        if not results:
            logger.warning("rerank() called with empty results — returning []")
            return []

        if self._model is None:
            # Fallback to pure hybrid ranking if no cross-encoder is loaded
            logger.debug("Reranker pass-through mode active.")
            return results[:n]

        # Build (query, chunk_text) pairs for cross-encoder
        pairs = [(query, r.text) for r in results]

        # Batch-score all pairs
        # CrossEncoder.predict() returns a numpy array of raw logit scores
        scores = self._model.predict(
            pairs,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Attach scores to results and sort descending
        scored: list[tuple[float, QueryResult]] = [
            (float(score), result)
            for score, result in zip(scores, results)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        # Rebuild QueryResult with updated scores
        reranked: list[QueryResult] = []
        for score, result in scored[:n]:
            reranked.append(QueryResult(
                chunk_id=result.chunk_id,
                text=result.text,
                metadata=result.metadata,
                score=score,
            ))

        logger.debug(
            "Reranker | input=%d | output=%d | top score=%.3f | bottom score=%.3f",
            len(results),
            len(reranked),
            reranked[0].score if reranked else 0,
            reranked[-1].score if reranked else 0,
        )

        return reranked
