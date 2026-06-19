"""
scripts/build_index.py
======================
One-time script to build and persist the FAISS vector index.

Pipeline:
  1. Load all 33 filings from filings_data/
  2. Chunk with hybrid strategy (semantic boundaries + recursive size guard)
  3. Embed all chunks via NVIDIA NIM API (fast, no GPU needed)
  4. Upsert into FAISS and save to faiss_db/

Usage:
    venv\\Scripts\\python scripts\\build_index.py           # build fresh
    venv\\Scripts\\python scripts\\build_index.py --force   # rebuild even if index exists
"""

import io
import sys
import time
import logging
import argparse
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_index")


def sep(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build FAISS index for SEC RAG system")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild index even if it already exists on disk",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from rag.config import settings

    # -----------------------------------------------------------------------
    # Qdrant Cloud Upsert is Idempotent
    # -----------------------------------------------------------------------
    # We don't check for local files anymore since we are pushing to the cloud.
    # Qdrant will overwrite vectors with the same ID, making re-runs safe.

    total_start = time.time()

    # -----------------------------------------------------------------------
    # STEP 1 — Load filings
    # -----------------------------------------------------------------------
    sep("STEP 1: Loading filings")

    from rag.ingestion.loader import load_all_filings

    t0 = time.time()
    filings = load_all_filings()
    print(f"\n  Loaded {len(filings)} filings in {time.time() - t0:.1f}s")

    for f in filings:
        print(
            f"  [{f.ticker:<6}] {f.company_name[:40]:<40} "
            f"| {f.total_chars():>10,} chars"
        )

    # -----------------------------------------------------------------------
    # STEP 2 — Chunk
    # -----------------------------------------------------------------------
    sep(f"STEP 2: Chunking — strategy='{settings.chunking_strategy}'")

    from rag.ingestion.chunker import Chunker
    from collections import defaultdict

    t0 = time.time()
    chunker    = Chunker()
    all_chunks = chunker.chunk_all_filings(filings)
    elapsed    = time.time() - t0

    print(f"\n  Total chunks : {len(all_chunks):,}")
    print(f"  Time taken   : {elapsed:.1f}s")
    print(f"  Avg / company: {len(all_chunks) / len(filings):.0f} chunks")
    print(f"  Avg chars    : {sum(c.char_count() for c in all_chunks) // len(all_chunks):,} chars/chunk")

    by_section: dict = defaultdict(int)
    for c in all_chunks:
        by_section[c.metadata.section] += 1

    print(f"\n  Section breakdown:")
    for section, count in sorted(by_section.items()):
        pct = count / len(all_chunks) * 100
        print(f"    {section:<30} {count:>5,} chunks ({pct:.1f}%)")

    # -----------------------------------------------------------------------
    # STEP 3 — Embed via NVIDIA NIM
    # -----------------------------------------------------------------------
    sep("STEP 3: Embedding chunks via NVIDIA NIM API")
    print(f"  Model  : nvidia/nv-embedqa-e5-v5")
    print(f"  Chunks : {len(all_chunks):,}")
    print(f"  Dim    : 1024")
    print("  (API-based — no GPU needed)\n")

    from rag.embeddings.nvidia_embedder import NvidiaEmbedder

    t0 = time.time()
    embedder   = NvidiaEmbedder()
    all_chunks = embedder.embed_chunks(all_chunks, show_progress=True)
    elapsed    = time.time() - t0

    missing = [c.chunk_id for c in all_chunks if c.embedding is None]
    if missing:
        print(f"\n  ❌ ERROR: {len(missing)} chunks have no embedding!")
        sys.exit(1)

    print(f"\n  Embedding complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Embedding dim : {len(all_chunks[0].embedding)}")
    print(f"  All populated : ✅")

    # -----------------------------------------------------------------------
    # STEP 4 — Upsert into Qdrant Cloud
    # -----------------------------------------------------------------------
    sep("STEP 4: Upserting into Qdrant Cloud")
    print(f"  URL: {settings.qdrant_url}")

    from rag.vectorstore.qdrant_store import QdrantStore

    t0    = time.time()
    store = QdrantStore()
    store.upsert(all_chunks)
    elapsed = time.time() - t0

    print(f"\n  Chunks indexed : {store.count():,}")
    print(f"  Time taken     : {elapsed:.1f}s")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    sep("BUILD COMPLETE")
    total_elapsed = time.time() - total_start
    print(f"  Total time     : {total_elapsed / 60:.1f} minutes")
    print(f"  Filings loaded : {len(filings)}")
    print(f"  Chunks indexed : {store.count():,}")
    print(f"  Index dim      : {settings.embedding_dim}")
    print(f"  Cloud Store    : Qdrant")
    print(f"\n  ✅ Ready to query!\n")


if __name__ == "__main__":
    main()
