"""
rag/ingestion/chunker.py
========================
Section-aware text chunking for SEC 10-K filings.

Supports three strategies (set via settings.chunking_strategy):
  "recursive" → RecursiveCharacterTextSplitter only
                Fast, deterministic, guaranteed max size.
  "semantic"  → Custom SemanticSplitter only
                Embedding-driven boundary detection, variable sizes.
  "hybrid"    → SemanticSplitter first (meaning boundaries),
                then RecursiveCharacterTextSplitter as a size guard.
                Best quality — recommended for production.

NOTE: We implement our own SemanticSplitter using sentence-transformers
directly instead of langchain-experimental.SemanticChunker, which is
deprecated and being sunset.

Key design rule:
  A chunk NEVER crosses a section boundary.
  business / risk_factors / management_discussion are always chunked
  independently and tagged with their section in metadata.

Usage:
    from rag.ingestion.chunker import Chunker

    chunker = Chunker()                             # loads model once

    chunks = chunker.chunk_filing(filing_doc)       # single company
    all_chunks = chunker.chunk_all_filings(filings) # all 33 companies
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import settings
from rag.ingestion.models import Chunk, ChunkMetadata, FilingDocument

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy Enum
# ---------------------------------------------------------------------------

class ChunkingStrategy(str, Enum):
    RECURSIVE = "recursive"   # fast, fixed size, deterministic
    SEMANTIC  = "semantic"    # ML boundary detection, variable size
    HYBRID    = "hybrid"      # semantic → recursive size guard (recommended)


# ---------------------------------------------------------------------------
# _SemanticSplitter  (internal — do not import directly)
# ---------------------------------------------------------------------------

class _SemanticSplitter:
    """
    Custom semantic text splitter using sentence-transformers.

    Algorithm:
      1. Split text into sentences on paragraph / line / period boundaries.
      2. Embed every sentence with the configured model.
      3. Compute cosine distance between adjacent sentence embeddings.
      4. Find breakpoints where the distance spike exceeds the Nth percentile.
      5. Group consecutive sentences between breakpoints into chunks.

    This produces chunks that align with natural semantic shifts in the text —
    ideal for long financial prose like 10-K sections.

    Args:
        model:      A loaded SentenceTransformer model (shared with embedder).
        percentile: Distance percentile used as the breakpoint threshold (0–100).
                    Higher → fewer, larger chunks. Lower → more, smaller chunks.
                    95 works well for SEC filings.
    """

    # Sentence boundary pattern — prefer paragraph breaks, then lines, then periods
    # Uses negative lookbehinds to avoid splitting on common SEC abbreviations.
    _SENTENCE_RE = re.compile(
        r"(?<!\bU\.S\.)"
        r"(?<!\bU\.K\.)"
        r"(?<!\bInc\.)"
        r"(?<!\bCorp\.)"
        r"(?<!\bLtd\.)"
        r"(?<!\bCo\.)"
        r"(?<!\bvs\.)"
        r"(?<!\b[A-Z]\.)"
        r"((?<=[.!?])\s+|(?<=\n\n)|\n{2,})"
    )

    def __init__(self, model: SentenceTransformer, percentile: float = 95.0) -> None:
        self._model = model
        self._percentile = percentile

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split raw text into a list of sentence/paragraph strings, PRESERVING whitespace."""
        parts = self._SENTENCE_RE.split(text)
        sentences = []
        for i in range(0, len(parts), 2):
            sentence = parts[i]
            if i + 1 < len(parts):
                sentence += parts[i + 1]
            if sentence.strip():
                sentences.append(sentence)
        return sentences

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine distance between two unit vectors (1 - cosine_similarity)."""
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 1.0
        return float(1.0 - dot / norm)

    def split_text(self, text: str) -> list[str]:
        """
        Split text into semantically coherent chunks.

        Args:
            text: Raw section text to split.

        Returns:
            List of text strings, each representing one semantic unit.
            May include very large strings if no breakpoints are detected —
            use the hybrid strategy to guard against this.
        """
        sentences = self._split_into_sentences(text)

        if len(sentences) <= 1:
            # Nothing to split
            return [text] if text.strip() else []

        # Embed all sentences in one batched call (efficient)
        embeddings: np.ndarray = self._model.encode(
            sentences,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,  # unit vectors → cosine = dot product
            convert_to_numpy=True,
        )

        # Compute cosine distance between consecutive sentence embeddings
        distances: list[float] = [
            self._cosine_distance(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]

        if not distances:
            return sentences

        # Find the distance threshold at the Nth percentile
        threshold = float(np.percentile(distances, self._percentile))

        # Identify breakpoint indices (where distance exceeds threshold)
        breakpoints: list[int] = [
            i + 1  # +1 because distance[i] is between sentence[i] and sentence[i+1]
            for i, d in enumerate(distances)
            if d >= threshold
        ]

        # Group sentences between breakpoints into chunks
        chunks: list[str] = []
        prev = 0
        for bp in breakpoints:
            group = sentences[prev:bp]
            if group:
                chunks.append("".join(group))
            prev = bp

        # Don't forget the final group after the last breakpoint
        final_group = sentences[prev:]
        if final_group:
            chunks.append("".join(final_group))

        return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Chunker  (public API)
# ---------------------------------------------------------------------------

class Chunker:
    """
    Converts FilingDocument objects into Chunk objects ready for embedding.

    The chunker is initialised once and reused across all filings.
    The heavy part (loading the sentence-transformer model) happens
    only once during __init__ when strategy is "semantic" or "hybrid".

    Args:
        strategy: One of "recursive", "semantic", "hybrid".
                  Defaults to settings.chunking_strategy.
    """

    def __init__(self, strategy: Optional[str] = None) -> None:
        self.strategy = ChunkingStrategy(
            strategy or settings.chunking_strategy
        )

        # ── Recursive splitter — always created (used as size guard in hybrid)
        self._recursive = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            keep_separator=True,
        )

        # ── Semantic splitter — only for "semantic" and "hybrid"
        self._semantic: Optional[_SemanticSplitter] = None
        if self.strategy in (ChunkingStrategy.SEMANTIC, ChunkingStrategy.HYBRID):
            # We use a tiny, lightning-fast model specifically for detecting topic shifts.
            # Using the massive main embedding model here would take hours on CPU.
            fast_chunking_model = "all-MiniLM-L6-v2"
            logger.info(
                "Loading fast sentence-transformer model '%s' for semantic chunking...",
                fast_chunking_model,
            )
            model = SentenceTransformer(
                fast_chunking_model,
                device=settings.embedding_device,
            )
            self._semantic = _SemanticSplitter(model=model, percentile=95.0)
            logger.info("Semantic splitter ready.")

        logger.info("Chunker initialised with strategy='%s'", self.strategy.value)

    # -----------------------------------------------------------------------
    # Internal — split a single section's text into raw strings
    # -----------------------------------------------------------------------

    def _split_text(self, text: str) -> list[str]:
        """
        Split raw section text into a list of text strings.

        Applies the configured strategy:
          recursive → RecursiveCharacterTextSplitter
          semantic  → _SemanticSplitter (may produce large chunks)
          hybrid    → semantic split, then recursive guard on oversized chunks

        Args:
            text: The full text of one 10-K section.

        Returns:
            List of text strings, each within the size limit (for hybrid/recursive).
        """
        text = text.strip()
        if not text:
            return []

        if self.strategy == ChunkingStrategy.RECURSIVE:
            return self._recursive.split_text(text)

        if self.strategy == ChunkingStrategy.SEMANTIC:
            return self._semantic.split_text(text)

        # ── HYBRID: semantic first, recursive guard on oversized chunks
        semantic_chunks = self._semantic.split_text(text)
        final_chunks: list[str] = []

        for chunk in semantic_chunks:
            if len(chunk) <= settings.chunk_size:
                # Within size limit → keep as-is
                final_chunks.append(chunk)
            else:
                # Oversized semantic chunk → recursively split further
                logger.debug(
                    "Oversized semantic chunk (%d chars) → applying recursive guard",
                    len(chunk),
                )
                sub_chunks = self._recursive.split_text(chunk)
                final_chunks.extend(sub_chunks)

        return final_chunks

    # -----------------------------------------------------------------------
    # Public — chunk one section
    # -----------------------------------------------------------------------

    def chunk_section(
        self,
        text: str,
        section_name: str,
        filing: FilingDocument,
    ) -> list[Chunk]:
        """
        Split one 10-K section into a list of Chunk objects.

        Args:
            text:         Raw text of the section (e.g. filing.risk_factors).
            section_name: One of "business", "risk_factors", "management_discussion".
            filing:       The parent FilingDocument (for metadata).

        Returns:
            Ordered list of Chunk objects with full metadata attached.
            Empty list if section text is blank.
        """
        raw_texts = self._split_text(text)

        if not raw_texts:
            logger.debug(
                "[%s] Section '%s' is empty — skipping", filing.ticker, section_name
            )
            return []

        chunks: list[Chunk] = []
        char_cursor = 0  # tracks character position in original text

        for idx, chunk_text in enumerate(raw_texts):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            # Find where this chunk starts in the original text
            char_start = text.find(chunk_text, char_cursor)
            if char_start == -1:
                # Fallback: approximate (can happen after recursive guard)
                char_start = char_cursor
            char_end = char_start + len(chunk_text)
            char_cursor = max(char_cursor, char_end - settings.chunk_overlap)

            metadata = ChunkMetadata(
                ticker=filing.ticker,
                company_name=filing.company_name,
                cik=filing.cik,
                filing_date=filing.filing_date,
                accession_number=filing.accession_number,
                section=section_name,
                chunk_index=idx,
                total_chunks_in_section=len(raw_texts),
                char_start=char_start,
                char_end=char_end,
            )

            chunk = Chunk(
                chunk_id=Chunk.build_id(filing.ticker, section_name, idx),
                text=chunk_text,
                metadata=metadata,
            )
            chunks.append(chunk)

        logger.debug(
            "[%s] '%s' → %d chunks (avg %.0f chars)",
            filing.ticker,
            section_name,
            len(chunks),
            sum(c.char_count() for c in chunks) / max(len(chunks), 1),
        )

        return chunks

    # -----------------------------------------------------------------------
    # Public — chunk one filing
    # -----------------------------------------------------------------------

    def chunk_filing(self, filing: FilingDocument) -> list[Chunk]:
        """
        Chunk all non-empty sections of a single FilingDocument.

        Processes sections in a fixed order:
          business → risk_factors → management_discussion

        Args:
            filing: The FilingDocument to chunk.

        Returns:
            Flat list of all Chunk objects for this filing.

        Example:
            >>> chunks = chunker.chunk_filing(nvda_doc)
            >>> len(chunks)
            161   # varies by company and strategy
        """
        all_chunks: list[Chunk] = []

        sections = filing.non_empty_sections()

        if not sections:
            logger.warning(
                "[%s] All sections are empty — no chunks produced", filing.ticker
            )
            return []

        for section_name, text in sections.items():
            section_chunks = self.chunk_section(text, section_name, filing)
            all_chunks.extend(section_chunks)

        logger.info(
            "[%s] Total chunks: %d across %d section(s)",
            filing.ticker,
            len(all_chunks),
            len(sections),
        )

        return all_chunks

    # -----------------------------------------------------------------------
    # Public — chunk all filings
    # -----------------------------------------------------------------------

    def chunk_all_filings(
        self,
        filings: list[FilingDocument],
    ) -> list[Chunk]:
        """
        Chunk all filings and return a single flat list of chunks.

        This is the main entry point called by scripts/build_index.py.

        Args:
            filings: List of FilingDocument objects (from loader.load_all_filings).

        Returns:
            All chunks from all 33 companies in order:
            [ADI chunks...] [ALAB chunks...] ... [WOLF chunks...]

        Example:
            >>> filings = load_all_filings()
            >>> all_chunks = chunker.chunk_all_filings(filings)
            >>> len(all_chunks)
            ~4500  # approximate for 33 semiconductor companies
        """
        all_chunks: list[Chunk] = []

        for i, filing in enumerate(filings, start=1):
            logger.info(
                "Chunking [%d/%d] %s (%s)...",
                i,
                len(filings),
                filing.ticker,
                filing.company_name,
            )
            chunks = self.chunk_filing(filing)
            all_chunks.extend(chunks)

        total_chunks    = len(all_chunks)
        total_companies = len(filings)

        logger.info(
            "Chunking complete: %d chunks from %d companies (avg %.1f chunks/company)",
            total_chunks,
            total_companies,
            total_chunks / max(total_companies, 1),
        )

        return all_chunks
