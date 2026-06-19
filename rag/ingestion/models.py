"""
rag/ingestion/models.py
=======================
Core Pydantic data models that flow through the entire RAG pipeline.

Two main models:
    FilingDocument  — raw JSON filing as loaded from disk
    Chunk           — the unit of text that gets embedded & stored
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# FilingDocument
# ---------------------------------------------------------------------------

class FilingDocument(BaseModel):
    """
    Represents one company's 10-K filing as loaded from a JSON file.

    Maps 1-to-1 with the JSON files produced by data.py.
    All section fields are optional because some filings may have
    missing sections — we handle that gracefully instead of crashing.
    """

    ticker: str = Field(..., description="Stock ticker symbol e.g. 'NVDA'")
    company_name: str = Field(..., description="Full legal company name")
    cik: str = Field(..., description="SEC Central Index Key")
    filing_date: str = Field(..., description="Filing date as ISO string YYYY-MM-DD")
    accession_number: str = Field(..., description="SEC accession number")

    # The three 10-K sections we care about
    business: str = Field(default="", description="Item 1 – Business Overview")
    risk_factors: str = Field(default="", description="Item 1A – Risk Factors")
    management_discussion: str = Field(
        default="", description="Item 7 – Management's Discussion & Analysis"
    )

    @field_validator("ticker", mode="before")
    @classmethod
    def ticker_uppercase(cls, v: str) -> str:
        """Normalise ticker to uppercase for consistent downstream lookup."""
        return v.strip().upper()

    @field_validator("cik", mode="before")
    @classmethod
    def cik_as_string(cls, v) -> str:
        """CIK can arrive as int from JSON — normalise to str."""
        return str(v).strip()

    def non_empty_sections(self) -> dict[str, str]:
        """
        Return only the sections that have actual content.

        Returns:
            dict mapping section_name -> text for all non-empty sections.

        Example:
            {"business": "...", "risk_factors": "..."}
        """
        candidates = {
            "business": self.business,
            "risk_factors": self.risk_factors,
            "management_discussion": self.management_discussion,
        }
        return {k: v for k, v in candidates.items() if v.strip()}

    def total_chars(self) -> int:
        """Total character count across all sections."""
        return (
            len(self.business)
            + len(self.risk_factors)
            + len(self.management_discussion)
        )


# ---------------------------------------------------------------------------
# ChunkMetadata  (embedded inside Chunk)
# ---------------------------------------------------------------------------

class ChunkMetadata(BaseModel):
    """
    Rich metadata attached to every chunk.

    Stored alongside the vector in ChromaDB so we can:
      - filter by ticker / section before retrieval
      - display source attribution in the UI
      - reconstruct the original context window

    All fields must be ChromaDB-compatible types (str, int, float, bool).
    ChromaDB does NOT support nested dicts or lists in metadata.
    """

    ticker: str = Field(..., description="Stock ticker e.g. 'NVDA'")
    company_name: str = Field(..., description="Full company name")
    cik: str = Field(..., description="SEC CIK")
    filing_date: str = Field(..., description="Filing date YYYY-MM-DD")
    accession_number: str = Field(..., description="SEC accession number")
    section: str = Field(
        ...,
        description="Which 10-K section: business | risk_factors | management_discussion",
    )
    chunk_index: int = Field(
        ..., description="Zero-based position of this chunk within its section"
    )
    total_chunks_in_section: int = Field(
        ..., description="How many chunks the section was split into (for context)"
    )
    char_start: int = Field(
        ..., description="Character offset of this chunk's start in the original section text"
    )
    char_end: int = Field(
        ..., description="Character offset of this chunk's end in the original section text"
    )

    def to_chroma_dict(self) -> dict:
        """
        Serialise to a flat dict suitable for ChromaDB's metadata field.
        ChromaDB requires all values to be str | int | float | bool.
        """
        return self.model_dump()


# ---------------------------------------------------------------------------
# Chunk  — the main unit of the pipeline
# ---------------------------------------------------------------------------

class Chunk(BaseModel):
    """
    A single text chunk ready to be embedded and stored.

    Lifecycle:
        FilingDocument
            → chunker splits into Chunk(text=..., metadata=..., embedding=None)
            → embedder fills  Chunk.embedding
            → vector store stores text + metadata + embedding
    """

    chunk_id: str = Field(
        ...,
        description=(
            "Unique, deterministic ID: '{ticker}_{section}_{chunk_index}'. "
            "Used as ChromaDB document ID to allow safe re-indexing (upsert)."
        ),
    )
    text: str = Field(..., description="The raw text content of this chunk")
    metadata: ChunkMetadata

    # Filled by the embedding step; None until then
    embedding: Optional[list[float]] = Field(
        default=None,
        description="Dense vector produced by the embedding model",
        exclude=True,   # don't serialise embedding to JSON by default (it's huge)
    )

    @classmethod
    def build_id(cls, ticker: str, section: str, chunk_index: int) -> str:
        """
        Build a deterministic chunk ID from its three key identifiers.

        Using a predictable ID lets us safely re-run the ingestion pipeline
        with ChromaDB's upsert — existing chunks are updated, not duplicated.

        Example:
            Chunk.build_id("NVDA", "risk_factors", 3) -> "NVDA_risk_factors_3"
        """
        return f"{ticker}_{section}_{chunk_index}"

    def word_count(self) -> int:
        """Approximate word count — useful for logging and validation."""
        return len(self.text.split())

    def char_count(self) -> int:
        """Character length of the chunk text."""
        return len(self.text)

    def __repr__(self) -> str:
        return (
            f"Chunk(id={self.chunk_id!r}, "
            f"chars={self.char_count()}, "
            f"section={self.metadata.section!r})"
        )
