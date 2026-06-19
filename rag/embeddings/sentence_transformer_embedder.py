"""
rag/embeddings/sentence_transformer_embedder.py
===============================================
Concrete embedder using sentence-transformers.

Default model: BAAI/bge-large-en-v1.5
  - 1024-dimensional output vectors
  - State-of-the-art on financial / domain-specific retrieval tasks
  - Asymmetric: uses a query instruction prefix for better search results
  - Fully local — no API key, no internet call after first download

Key design choices:
  1. Batched encoding   — processes chunks in groups of `batch_size`
                          to fit in RAM and show a clean progress bar.
  2. Query prefix       — BAAI/bge models require a special prefix when
                          encoding user queries (not documents). Omitting
                          this prefix measurably hurts retrieval quality.
  3. Normalised vectors — L2 normalisation makes cosine similarity equal
                          to dot product, which ChromaDB's HNSW index
                          uses internally (faster ANN search).
  4. embed_chunks()     — convenience method that fills chunk.embedding
                          in-place, returning the same Chunk list.

Usage:
    from rag.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder()          # loads model once

    # At index-build time (slow, batched, with progress bar):
    vectors = embedder.embed_documents(texts, show_progress=True)

    # At query time (instant, single vector):
    q_vector = embedder.embed_query("What are NVDA's supply chain risks?")

    # Convenience — fill chunks directly:
    chunks = embedder.embed_chunks(chunks, show_progress=True)
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from rag.config import settings
from rag.embeddings.base import BaseEmbedder
from rag.ingestion.models import Chunk

logger = logging.getLogger(__name__)

# BGE models perform best on retrieval tasks when the query (not documents)
# is prefixed with this instruction. Do NOT add this prefix to documents.
_BGE_QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Sentence-transformers embedder backed by BAAI/bge-large-en-v1.5.

    Args:
        model_name:  HuggingFace model ID. Defaults to settings.embedding_model.
        device:      Torch device string ("cpu", "cuda", "mps").
                     Defaults to settings.embedding_device.
        batch_size:  Number of texts per encoding batch.
                     32–64 works well on CPU; 128–256 on GPU.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 64,
    ) -> None:
        self._model_name = model_name or settings.embedding_model
        self._device     = device     or settings.embedding_device
        self._batch_size = batch_size

        logger.info(
            "Loading sentence-transformer model '%s' on device='%s'...",
            self._model_name,
            self._device,
        )
        self._model = SentenceTransformer(
            self._model_name,
            device=self._device,
        )

        # Cache the output dimension by encoding a single dummy text
        self._dim: int = self._model.get_sentence_embedding_dimension()

        logger.info(
            "Embedder ready | model=%s | dim=%d | batch_size=%d | device=%s",
            self._model_name,
            self._dim,
            self._batch_size,
            self._device,
        )

    # -----------------------------------------------------------------------
    # BaseEmbedder interface
    # -----------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Output vector dimension (1024 for bge-large-en-v1.5)."""
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(
        self,
        texts: list[str],
        show_progress: bool = False,
    ) -> list[list[float]]:
        """
        Embed a list of document texts for indexing.

        Processes texts in batches of `self._batch_size` with an optional
        tqdm progress bar. Each vector is L2-normalised.

        Args:
            texts:         List of raw text strings.
            show_progress: Show a tqdm progress bar (recommended for
                           large builds).

        Returns:
            List of 1024-dim float vectors, one per input text.

        Example:
            >>> vectors = embedder.embed_documents(["NVDA supplies GPUs..."])
            >>> len(vectors[0])
            1024
        """
        if not texts:
            return []

        all_vectors: list[list[float]] = []
        n_batches = math.ceil(len(texts) / self._batch_size)

        batch_iter = range(n_batches)
        if show_progress:
            batch_iter = tqdm(
                batch_iter,
                desc="Embedding chunks",
                unit="batch",
                total=n_batches,
            )

        for i in batch_iter:
            start = i * self._batch_size
            end   = start + self._batch_size
            batch = texts[start:end]

            vectors = self._model.encode(
                batch,
                batch_size=self._batch_size,
                normalize_embeddings=True,   # unit-length for cosine sim
                show_progress_bar=False,     # tqdm is handled above
                convert_to_numpy=True,
            )

            # Convert numpy arrays to plain Python lists for JSON/ChromaDB compat
            all_vectors.extend(v.tolist() for v in vectors)

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

        Prepends the BGE query instruction prefix, which significantly
        improves retrieval quality for asymmetric search tasks.

        Args:
            text: Raw user query string.

        Returns:
            A single 1024-dim normalised float vector.

        Example:
            >>> vec = embedder.embed_query("What are NVDA's supply chain risks?")
            >>> len(vec)
            1024
        """
        # Add the instruction prefix for BGE models
        prefixed = _BGE_QUERY_PREFIX + text.strip()

        vector = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return vector.tolist()

    # -----------------------------------------------------------------------
    # Convenience method — embed chunks in-place
    # -----------------------------------------------------------------------

    def embed_chunks(
        self,
        chunks: list[Chunk],
        show_progress: bool = True,
    ) -> list[Chunk]:
        """
        Embed a list of Chunk objects, filling chunk.embedding in-place.

        This is the primary method called by scripts/build_index.py.
        After this call, each chunk.embedding is a 1024-dim float list
        ready to be upserted into ChromaDB.

        Args:
            chunks:        List of Chunk objects from the chunker.
            show_progress: Show a tqdm progress bar.

        Returns:
            The same list of Chunk objects, each with .embedding populated.

        Example:
            >>> chunks = chunker.chunk_all_filings(filings)
            >>> chunks = embedder.embed_chunks(chunks, show_progress=True)
            >>> chunks[0].embedding is not None
            True
            >>> len(chunks[0].embedding)
            1024
        """
        if not chunks:
            logger.warning("embed_chunks called with empty list — nothing to do")
            return chunks

        texts = [c.text for c in chunks]

        logger.info(
            "Embedding %d chunks with model '%s' (batch_size=%d)...",
            len(chunks),
            self._model_name,
            self._batch_size,
        )

        vectors = self.embed_documents(texts, show_progress=show_progress)

        # Assign embeddings back to each chunk
        for chunk, vector in zip(chunks, vectors):
            chunk.embedding = vector

        logger.info(
            "Embedding complete: %d chunks | dim=%d | "
            "all embeddings populated: %s",
            len(chunks),
            self._dim,
            all(c.embedding is not None for c in chunks),
        )

        return chunks
