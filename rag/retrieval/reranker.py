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

        keys_str = settings.nvidia_api_keys or settings.nvidia_api_key
        
        if not _HAS_NVIDIA or not keys_str:
            logger.warning("NVIDIA SDK not installed or missing API keys. Reranker will act as a pass-through.")
            self.models = []
            return

        self.api_keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        if not self.api_keys:
            logger.warning("No valid API keys found. Reranker will act as a pass-through.")
            self.models = []
            return

        logger.info(
            "Loading NVIDIA NIM reranker '%s' with %d keys...",
            self._model_name,
            len(self.api_keys)
        )

        self.models = [
            NVIDIARerank(
                model=self._model_name,
                nvidia_api_key=key,
                top_n=settings.rerank_top_n
            ) for key in self.api_keys
        ]
        self.current_key_idx = 0

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

        if not self.models:
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
        for attempt in range(len(self.api_keys)):
            try:
                model = self.models[self.current_key_idx]
                if hasattr(model, "top_n"):
                    model.top_n = n
                    
                reranked_docs = model.compress_documents(
                    documents=docs,
                    query=query
                )
                break
            except Exception as e:
                err_msg = str(e).lower()
                if "429" in err_msg or "rate limit" in err_msg or "rate_limit_exceeded" in err_msg:
                    logger.warning(f"NVIDIA Reranker Key {self.current_key_idx+1} hit rate limit. Trying next key...")
                    self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
                    continue
                else:
                    logger.error(f"NVIDIA Reranker failed: {e}. Falling back to hybrid results.")
                    return results[:n]
        else:
            logger.error("All NVIDIA API keys hit rate limits. Falling back to hybrid results.")
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
