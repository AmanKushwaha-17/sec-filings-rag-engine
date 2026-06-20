"""
rag/embeddings/nvidia_embedder.py
==================================
NVIDIA NIM API-based embedder using nv-embedqa-e5-v5.

Drop-in replacement for SentenceTransformerEmbedder:
  - Same BaseEmbedder interface
  - Same 1024-dim output
  - Asymmetric: "passage" for documents, "query" for queries
  - API-based — zero CPU load, no local model download

Model: nvidia/nv-embedqa-e5-v5
  - 1024-dimensional output (same as BGE-large)
  - Trained for retrieval/QA tasks
  - Supports asymmetric passage/query input types

Usage:
    from rag.embeddings.nvidia_embedder import NvidiaEmbedder

    embedder = NvidiaEmbedder()
    vectors  = embedder.embed_documents(texts, show_progress=True)
    q_vec    = embedder.embed_query("What are NVDA's supply chain risks?")
    chunks   = embedder.embed_chunks(chunks, show_progress=True)
"""

from __future__ import annotations

import logging
import math
import time
from typing import Optional

import requests
from tqdm import tqdm

from rag.config import settings
from rag.embeddings.base import BaseEmbedder
from rag.ingestion.models import Chunk

logger = logging.getLogger(__name__)

_API_URL    = "https://integrate.api.nvidia.com/v1/embeddings"
_MODEL      = "nvidia/nv-embedqa-e5-v5"
_BATCH_SIZE = 96   # NVIDIA NIM handles large batches well


class NvidiaEmbedder(BaseEmbedder):
    """
    NVIDIA NIM API embedder — nv-embedqa-e5-v5.

    Implements BaseEmbedder so it can be swapped in anywhere
    SentenceTransformerEmbedder is used with zero other changes.

    Args:
        api_key:    NVIDIA NIM API key. Reads from NVIDIA_API_KEY in .env if not passed.
        batch_size: Number of texts per API call. Default 96.
    """

    def __init__(
        self,
        api_keys: Optional[list[str]] = None,
        batch_size: int = _BATCH_SIZE,
    ) -> None:
        if api_keys:
            self.api_keys = api_keys
        else:
            keys_str = settings.nvidia_api_keys or settings.nvidia_api_key
            self.api_keys = [k.strip() for k in keys_str.split(",") if k.strip()]
            
        if not self.api_keys:
            raise ValueError("No valid NVIDIA API keys found.")
            
        self.current_key_idx = 0
        self._batch_size = batch_size
        self._dim        = 1024

        # Quick connectivity check
        self._ping()

        logger.info(
            "NvidiaEmbedder ready | model=%s | dim=%d | batch_size=%d",
            _MODEL,
            self._dim,
            self._batch_size,
        )

    def _ping(self) -> None:
        """Verify API key works with a single test embedding."""
        try:
            result = self._call_api(["ping"], input_type="passage")
            assert len(result[0]) == self._dim
            logger.info("NVIDIA API connectivity check passed.")
        except Exception as exc:
            raise RuntimeError(
                f"NVIDIA NIM API connectivity check failed: {exc}\n"
                "Check your NVIDIA_API_KEYS in .env"
            )

    def _call_api(
        self,
        texts: list[str],
        input_type: str,
        retries: int = 3,
    ) -> list[list[float]]:
        """
        Call NVIDIA NIM embeddings API for a batch of texts.

        Args:
            texts:      List of text strings to embed.
            input_type: "passage" for documents, "query" for queries.
            retries:    Number of retry attempts on rate limit / server error.

        Returns:
            List of 1024-dim float vectors.
        """
        payload = {
            "model": _MODEL,
            "input": texts,
            "input_type": input_type,
        }
        
        max_attempts = max(retries, len(self.api_keys) * 2)

        for attempt in range(max_attempts):
            current_key = self.api_keys[self.current_key_idx]
            headers = {
                "Authorization": f"Bearer {current_key}",
                "Content-Type": "application/json",
            }
            
            try:
                response = requests.post(
                    _API_URL,
                    headers=headers,
                    json=payload,
                    timeout=60,
                )

                if response.status_code == 200:
                    data = response.json()
                    # Sort by index to preserve order
                    items = sorted(data["data"], key=lambda x: x["index"])
                    return [item["embedding"] for item in items]

                elif response.status_code == 429:
                    # Rate limited — wait and retry with next key
                    wait = 2 ** attempt
                    logger.warning(
                        "Rate limited (429). Waiting %ds before retry %d/%d. Switching API key...",
                        wait, attempt + 1, max_attempts,
                    )
                    self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
                    time.sleep(wait)

                else:
                    logger.warning(f"NVIDIA API error {response.status_code}: {response.text}")
                    self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
                    time.sleep(1)

            except requests.exceptions.Timeout:
                logger.warning("Request timeout on attempt %d/%d", attempt + 1, max_attempts)
                if attempt == max_attempts - 1:
                    raise

        raise RuntimeError(f"NVIDIA API failed after {max_attempts} retries.")

    # -----------------------------------------------------------------------
    # BaseEmbedder interface
    # -----------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return _MODEL

    def embed_documents(
        self,
        texts: list[str],
        show_progress: bool = False,
    ) -> list[list[float]]:
        """
        Embed a list of document texts for indexing.

        Uses input_type="passage" — required for asymmetric retrieval.

        Args:
            texts:         List of raw text strings.
            show_progress: Show tqdm progress bar.

        Returns:
            List of 1024-dim float vectors.
        """
        if not texts:
            return []

        all_vectors: list[list[float]] = []
        n_batches = math.ceil(len(texts) / self._batch_size)

        batch_iter = range(n_batches)
        if show_progress:
            batch_iter = tqdm(
                batch_iter,
                desc="Embedding chunks (NVIDIA NIM)",
                unit="batch",
                total=n_batches,
            )

        for i in batch_iter:
            start = i * self._batch_size
            end   = start + self._batch_size
            batch = texts[start:end]

            vectors = self._call_api(batch, input_type="passage")
            all_vectors.extend(vectors)

        logger.debug(
            "embed_documents: %d texts → %d vectors (dim=%d)",
            len(texts),
            len(all_vectors),
            self._dim,
        )

        return all_vectors

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single user query for retrieval.

        Uses input_type="query" for asymmetric retrieval quality.

        Args:
            text: Raw user query string.

        Returns:
            A single 1024-dim float vector.
        """
        vectors = self._call_api([text.strip()], input_type="query")
        return vectors[0]

    # -----------------------------------------------------------------------
    # Convenience method
    # -----------------------------------------------------------------------

    def embed_chunks(
        self,
        chunks: list[Chunk],
        show_progress: bool = True,
    ) -> list[Chunk]:
        """
        Embed a list of Chunk objects, filling chunk.embedding in-place.

        Args:
            chunks:        List of Chunk objects from the chunker.
            show_progress: Show tqdm progress bar.

        Returns:
            Same list of Chunk objects with .embedding populated.
        """
        if not chunks:
            logger.warning("embed_chunks called with empty list — nothing to do")
            return chunks

        texts = [c.text for c in chunks]

        logger.info(
            "Embedding %d chunks via NVIDIA NIM (batch_size=%d)...",
            len(chunks),
            self._batch_size,
        )

        vectors = self.embed_documents(texts, show_progress=show_progress)

        for chunk, vector in zip(chunks, vectors):
            chunk.embedding = vector

        logger.info(
            "Embedding complete: %d chunks | dim=%d | all populated: %s",
            len(chunks),
            self._dim,
            all(c.embedding is not None for c in chunks),
        )

        return chunks
