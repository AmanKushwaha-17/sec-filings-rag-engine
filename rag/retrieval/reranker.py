"""
rag/retrieval/reranker.py
==========================
Cross-encoder reranker using NVIDIA NIM API.

This completely replaces the local HuggingFace `sentence-transformers` model
with a lightweight API call to NVIDIA, meaning this runs with almost zero
RAM usage and easily fits within Render's 512MB free tier limit.

Model: nvidia/nv-rerankqa-mistral-4b-v3
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.documents import Document

try:
    from langchain_nvidia_ai_endpoints import NVIDIARerank
    _HAS_NVIDIA = True
except ImportError:
    _HAS_NVIDIA = False

from rag.config import settings
from rag.vectorstore.qdrant_store import QueryResult

logger = logging.getLogger(__name__)

# Cloud-hosted NVIDIA NIM reranking model
_DEFAULT_MODEL = "nvidia/llama-nemotron-rerank-1b-v2"


class Reranker:
    """
    Cross-encoder reranker for the final retrieval stage.

    Takes the top-N candidates from hybrid retrieval and re-scores
    each (query, chunk) pair using NVIDIA's cloud NIM API.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: Optional[str] = None,
    ) -> None:
        self._model_name = model_name

        if not _HAS_NVIDIA or not settings.nvidia_api_key:
            logger.warning("NVIDIA SDK not installed or missing API key. Reranker will act as a pass-through.")
            self._model = None
            return

        logger.info(
            "Loading NVIDIA NIM reranker '%s'...",
            self._model_name
        )

        self._model = NVIDIARerank(
            model=self._model_name,
            nvidia_api_key=settings.nvidia_api_key
        )

        logger.info("NVIDIA Reranker ready | model=%s", self._model_name)

    def rerank(
        self,
        query: str,
        results: list[QueryResult],
        top_n: Optional[int] = None,
    ) -> list[QueryResult]:
        n = top_n or settings.rerank_top_n

        if not results:
            return []

        if self._model is None:
            logger.debug("Reranker pass-through mode active.")
            return results[:n]

        # Convert to LangChain Documents for the NVIDIARerank interface
        docs = [
            Document(
                page_content=r.text,
                metadata={"original_result": r}
            )
            for r in results
        ]

        # Call NVIDIA NIM API
        try:
            reranked_docs = self._model.compress_documents(
                documents=docs,
                query=query
            )
        except Exception as e:
            logger.error(f"NVIDIA Reranker failed: {e}. Falling back to hybrid results.")
            return results[:n]

        # Rebuild QueryResult list from the returned sorted documents
        reranked: list[QueryResult] = []
        for doc in reranked_docs[:n]:
            orig: QueryResult = doc.metadata["original_result"]
            score = doc.metadata.get("relevance_score", 0.0)
            
            reranked.append(QueryResult(
                chunk_id=orig.chunk_id,
                text=orig.text,
                metadata=orig.metadata,
                score=float(score),
            ))

        logger.debug(
            "NVIDIA Reranker | input=%d | output=%d | top score=%.3f",
            len(results),
            len(reranked),
            reranked[0].score if reranked else 0,
        )

        return reranked
