# Company SEC RAG — Project Vision & Goal

## What This Project Is

An **advanced Hybrid RAG system** built over SEC 10-K filings for 33
U.S. semiconductor companies. It goes significantly beyond naive
retrieve-once-then-generate RAG by adding an LLM-driven Query Analyzer for metadata filtering, cross-company comparison capabilities, and a
3-stage retrieval pipeline that most tutorial RAG projects skip entirely.

---

## Why This Exists (The Problem)

Standard RAG breaks in three specific, well-known ways that this project
is designed to solve:

1. **Single-document limitation** — naive RAG retrieves from one document at
   a time. A question like *"which semiconductor companies disclosed the most
   supply chain risk in their 2025 10-K?"* is fundamentally unanswerable
   by single-document retrieval. You need cross-corpus synthesis.

2. **Retrieve-once blindness** — if the first retrieval pass returns
   insufficient or irrelevant context, naive RAG generates an answer anyway,
   often confidently wrong. This system explicitly instructs the LLM to admit
   when context is insufficient rather than hallucinating.

3. **Keyword-only or semantic-only retrieval** — BM25 (sparse) is great for
   exact terms (tickers, legal clauses). Dense embeddings are great for
   semantic similarity. Neither alone is optimal. Most tutorials pick one.

---

## Architecture

### Data Layer (Complete)
- **Corpus:** 33 semiconductor companies' most recent 10-K filings (2025–2026)
  pulled directly from SEC EDGAR via `edgartools`
- **Sections indexed:** `business` (Item 1), `risk_factors` (Item 1A),
  `management_discussion` (Item 7)
- **Companies:** ADI, ALAB, ALGM, AMAT, AMBA, AVGO, CDNS, COHR, CRDO, DIOD,
  ENTG, KLAC, LRCX, LSCC, MCHP, MKSI, MPWR, MRVL, MU, MXL, NXPI, ON, ONTO,
  POWI, QCOM, RMBS, SITM, SLAB, SNPS, TER, TXN, WOLF — intentionally excludes
  AMD and SWKS (missing risk_factors sections)

### Ingestion & Chunking (Complete)
- **Three chunking strategies** configurable via `settings.chunking_strategy`:
  - `recursive` — RecursiveCharacterTextSplitter, fast, deterministic
  - `semantic` — custom SemanticSplitter using sentence-transformer embeddings
    and cosine distance percentile thresholds for boundary detection
  - `hybrid` — semantic boundaries first, recursive size guard on oversized
    chunks (recommended, default)
- **Section-aware:** a chunk never crosses a section boundary
- **Rich metadata per chunk:** ticker, company name, CIK, filing date,
  accession number, section name, chunk index, char offsets

### Embedding (Complete)
- **Model:** `BAAI/bge-large-en-v1.5` (1024-dim, state-of-the-art on
  financial retrieval tasks)
- **Asymmetric:** BGE query instruction prefix applied at query time only
  (not at indexing time) — omitting this measurably hurts retrieval quality
- **Batched + normalised:** L2 normalisation makes cosine similarity equal
  to dot product (inner product index)

### Vector Store (Complete)
- **Qdrant Cloud** — Stores vector embeddings alongside full metadata payload
- **Upsert-safe:** deterministic chunk IDs (`uuid5`) allow
  safe re-indexing without duplicates
- **Native metadata filtering** with strict conditions applied before search
- **Extensible:** enables adding new companies later
  without rebuilding the entire index

### 3-Stage Retrieval Pipeline (Complete)
```
User Query
    │
    ▼
Stage 1 — Dense Retrieval (Qdrant Cloud)
    top_k=20 semantic candidates
    │
    ├── Stage 2 — Sparse Retrieval (BM25)
    │       top_k=20 keyword candidates
    │
    ▼
Reciprocal Rank Fusion (RRF, K=60)
    fuses dense + sparse by rank (scale-invariant)
    deduplicates, returns top 30
    │
    ▼
Stage 3 — Cross-Encoder Reranking
    model: cross-encoder/ms-marco-MiniLM-L-6-v2
    re-scores (query, chunk) pairs together
    returns top 5 for LLM
    │
    ▼
LLM Generation (Groq — llama-3.3-70b-versatile)
```

---

## Demo Queries (What Makes This Impressive)

These are the queries naive RAG cannot handle that this system can:

1. **Cross-company comparison**
   > "Which of the 33 semiconductor companies disclosed the highest exposure
   > to China export restrictions in their most recent 10-K?"

2. **Sector-wide pattern detection**
   > "Summarise the top 5 supply chain risks that appear across the majority
   > of semiconductor companies in this corpus."

3. **Single-company deep dive**
   > "What does NVDA say about its dependence on TSMC for manufacturing,
   > and how has that risk been framed?"

4. **Outlier identification**
   > "Which companies mention AI inference demand as a growth driver, and
   > which ones describe it as a risk?"

5. **Negative capability stress test**
   > "What is WOLF's exact revenue guidance for FY2026?"
   > (Tests: does the system admit uncertainty when context is insufficient,
   > rather than hallucinating a number?)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data ingestion | `edgartools`, SEC EDGAR API |
| Chunking | `langchain-text-splitters`, custom SemanticSplitter |
| Embeddings | `sentence-transformers` (BAAI/bge-large-en-v1.5) |
| Vector store | Qdrant Cloud |
| Sparse retrieval | BM25 |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Reranking | `sentence-transformers` CrossEncoder (ms-marco-MiniLM-L-6-v2) |
| LLM | Groq — llama-3.3-70b-versatile |
| Config | Pydantic-settings |
| Validation | Pydantic v2 |
| Language | Python 3.12 |

---

## What Is Left To Build

In priority order:

1. **Ultra Premium Web UI** — Vite + React frontend with a glassmorphism aesthetic using vanilla CSS.

*(Note: Ingestion, Qdrant Vectorstore, Hybrid Retrieval, BM25, FastAPI, Groq LLM Generation, Query Analyzer are all 100% complete).*

---

## Key Engineering Decisions Worth Explaining In Interviews

- **Why RRF over score normalisation?** Dense and BM25 scores are on
  incompatible scales. RRF uses only rank, making it scale-invariant and
  empirically more robust than weighted score combination.

- **Why BGE with an instruction prefix?** BGE models are asymmetrically
  trained — documents and queries are expected to have different input
  formats. Skipping the prefix is a common mistake that measurably hurts
  retrieval quality.

- **Why cross-encoder reranking instead of just taking top-5 from Qdrant?**
  Bi-encoders embed query and chunk independently (fast, approximate).
  Cross-encoders read both together (slow, precise). Two-stage is the
  standard production pattern: fast coarse retrieval then slow precise
  reranking on a small candidate set.

- **Why hybrid chunking?** Purely semantic chunking can produce a 15,000-char
  chunk if no semantic boundary is detected in a long section. The recursive
  guard prevents the embedder and LLM context window from being blown out by
  a single oversized chunk.

- **Why AMD and SWKS were excluded?** Their risk_factors sections returned 0
  chars during ingestion — silently, without errors. Caught by manual log
  inspection, not automatic validation. Exclusion is intentional and documented.

---

*Last updated: June 2026*
*Author: Aman Kushwaha*
