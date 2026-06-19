"""
rag/embeddings/base.py
======================
Abstract base class for all embedding implementations.

Why have an abstract base?
  Every other module (vector store, retriever, pipeline) types against
  BaseEmbedder — not a specific implementation. This means you can swap
  sentence-transformers for OpenAI or any other provider in the future
  by writing one new class, with zero changes elsewhere.

Usage:
    from rag.embeddings.base import BaseEmbedder

    # Type hint your functions against the base, not the implementation:
    def build_index(embedder: BaseEmbedder, chunks: list[Chunk]) -> None:
        ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """
    Contract that every embedder must satisfy.

    Two distinct methods are intentional:
      embed_documents — used at index-build time (batched, may show progress)
      embed_query     — used at search time (single text, instant)

    The distinction matters for BGE models, which need a special query
    instruction prefix to maximise retrieval quality.
    """

    @abstractmethod
    def embed_documents(
        self,
        texts: list[str],
        show_progress: bool = False,
    ) -> list[list[float]]:
        """
        Embed a list of document texts for indexing.

        Args:
            texts:         List of raw text strings to embed.
            show_progress: If True, display a progress bar (useful for
                           large batches during build_index.py).

        Returns:
            List of embedding vectors, one per input text.
            Each vector is a list of floats with length == embedding_dim.
            All vectors are L2-normalised (unit length).
        """
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single user query for retrieval.

        For asymmetric models (like BGE), this method adds the required
        query instruction prefix before encoding.

        Args:
            text: The user's raw search query.

        Returns:
            A single embedding vector (list of floats).
            L2-normalised (unit length).
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The output vector dimension of this embedder."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable name of the underlying model."""
        ...
