"""
rag/vectorstore/qdrant_store.py
===============================
Qdrant-backed vector store for SEC 10-K chunk embeddings.

Uses Qdrant Cloud. Requires QDRANT_URL and QDRANT_API_KEY.
Metadata filtering is done natively in Qdrant before returning candidates.
"""

from __future__ import annotations

import uuid
import logging
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from rag.config import settings
from rag.ingestion.models import Chunk, ChunkMetadata

logger = logging.getLogger(__name__)

# A consistent namespace UUID to generate deterministic chunk UUIDs
NAMESPACE_QDRANT = uuid.UUID('12345678-1234-5678-1234-567812345678')

class QueryResult:
    """One result returned by QdrantStore.query()."""
    def __init__(
        self,
        chunk_id: str,
        text: str,
        metadata: ChunkMetadata,
        score: float,
    ) -> None:
        self.chunk_id = chunk_id
        self.text     = text
        self.metadata = metadata
        self.score    = score

    def __repr__(self) -> str:
        return (
            f"QueryResult(id={self.chunk_id!r}, "
            f"score={self.score:.4f}, "
            f"ticker={self.metadata.ticker!r}, "
            f"section={self.metadata.section!r})"
        )


class QdrantStore:
    """
    Persistent Qdrant Cloud vector store for SEC 10-K chunk embeddings.
    """

    def __init__(self, dim: Optional[int] = None) -> None:
        self._dim = dim or settings.embedding_dim
        self._collection_name = settings.qdrant_collection_name
        
        if not settings.qdrant_url or not settings.qdrant_api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in .env")
            
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=60
        )
        
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Creates the collection if it doesn't already exist."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self._collection_name for c in collections)
        
        if not exists:
            logger.info("Creating Qdrant collection: %s", self._collection_name)
            self.client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(size=self._dim, distance=Distance.DOT),
            )
        else:
            logger.info("Connected to Qdrant collection: %s", self._collection_name)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(self, chunks: list[Chunk]) -> None:
        """Insert embedded chunks into Qdrant."""
        missing = [c.chunk_id for c in chunks if c.embedding is None]
        if missing:
            raise ValueError(
                f"{len(missing)} chunks have no embedding. "
                f"Run embedder.embed_chunks() first."
            )

        points = []
        for c in chunks:
            payload = c.metadata.to_chroma_dict()
            payload["text"] = c.text
            payload["chunk_id"] = c.chunk_id
            
            # Deterministic UUID generation for Qdrant
            point_id = str(uuid.uuid5(NAMESPACE_QDRANT, c.chunk_id))
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=c.embedding,
                    payload=payload
                )
            )
            
        # Batch upload
        BATCH_SIZE = 100
        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i:i + BATCH_SIZE]
            self.client.upsert(
                collection_name=self._collection_name,
                points=batch
            )

        logger.info(
            "Upsert complete: %d chunks added to Qdrant",
            len(chunks)
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(
        self,
        vector: list[float],
        top_k: int = 20,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[QueryResult]:
        """
        Find the top-K most similar chunks to a query vector.
        Uses native Qdrant metadata pre-filtering.
        """
        qdrant_filter = None
        if filters:
            must_conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
                    # Qdrant OR logic: match any of the values
                    # FieldCondition(key=key, match=MatchAny(any=value)) in modern Qdrant,
                    # but for broader compatibility we can use MatchValue inside a Should block
                    # Actually, qdrant supports MatchAny
                    must_conditions.append(
                        FieldCondition(key=key, match=rest.MatchAny(any=value))
                    )
                else:
                    must_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
            qdrant_filter = Filter(must=must_conditions)

        search_response = self.client.query_points(
            collection_name=self._collection_name,
            query=vector,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True
        )

        results: list[QueryResult] = []

        for hit in search_response.points:
            payload = hit.payload or {}
            
            try:
                meta_fields = {
                    k: v for k, v in payload.items()
                    if k not in ("chunk_id", "text")
                }
                metadata = ChunkMetadata(**meta_fields)
            except Exception as exc:
                logger.warning(
                    "Could not reconstruct ChunkMetadata for point %s: %s", hit.id, exc
                )
                continue

            results.append(QueryResult(
                chunk_id=payload.get("chunk_id", str(hit.id)),
                text=payload.get("text", ""),
                metadata=metadata,
                score=hit.score,
            ))

        return results

    def count(self) -> int:
        """Total number of chunks in the collection."""
        try:
            return self.client.count(collection_name=self._collection_name).count
        except Exception:
            return 0

    def get_stored_tickers(self) -> list[str]:
        """Qdrant doesn't easily expose distinct values directly without a scroll or aggregation plugin.
        Returning a placeholder for now, or we could just skip it."""
        # For a full implementation, you'd scroll and extract, but for this RAG engine it is primarily used for logging.
        return ["Qdrant Cloud (Tickers hidden)"]

    def get_all_chunks(self) -> list[Chunk]:
        """
        Scrolls through the entire Qdrant collection and returns all chunks.
        Needed for building the in-memory BM25 index.
        """
        chunks = []
        offset = None
        while True:
            points, next_page_offset = self.client.scroll(
                collection_name=self._collection_name,
                offset=offset,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )
            for point in points:
                payload = point.payload or {}
                meta_fields = {k: v for k, v in payload.items() if k not in ("chunk_id", "text")}
                try:
                    metadata = ChunkMetadata(**meta_fields)
                    chunks.append(Chunk(
                        chunk_id=payload.get("chunk_id", str(point.id)),
                        text=payload.get("text", ""),
                        metadata=metadata
                    ))
                except Exception as exc:
                    logger.warning("Could not reconstruct Chunk for point %s: %s", point.id, exc)
                    
            if next_page_offset is None:
                break
            offset = next_page_offset
            
        return chunks
