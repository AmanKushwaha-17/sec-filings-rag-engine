# SEC 10-K RAG System — Semiconductor Industry

An **advanced Hybrid Retrieval-Augmented Generation (RAG) system** built over SEC 10-K filings for **33 U.S. semiconductor companies**. The system goes beyond traditional retrieve-once-and-generate RAG by implementing an LLM-driven Query Analyzer for metadata filtering and a 3-stage hybrid retrieval pipeline with cross-encoder reranking.

---

## Problem Statement

Standard RAG systems fail in three specific, well-known ways that this project is designed to solve:

- **Single-document limitation** — Naive RAG retrieves from one document at a time. A question like *"which semiconductor companies disclosed the most supply chain risk in their 2025 10-K?"* requires cross-corpus synthesis across all 33 companies simultaneously.

- **Retrieve-once blindness** — If the first retrieval pass returns insufficient or irrelevant context, naive RAG generates an answer anyway — often confidently wrong. This system is heavily prompted to declare when context is insufficient rather than hallucinate.

- **Keyword-only or semantic-only retrieval** — BM25 (sparse retrieval) is excellent at matching exact terms like ticker symbols and legal clauses. Dense embeddings are excellent at capturing semantic similarity and paraphrases. Using either one alone is provably sub-optimal. This system uses both, fused together.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                         User Query                               │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                   LLM QUERY ANALYZER                             │
│       → Extracts ticker symbols for metadata filtering           │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    3-STAGE RETRIEVAL PIPELINE                     │
│                                                                  │
│  Stage 1a: Dense Retriever (Qdrant Cloud)                        │
│       → Embeds query with BGE instruction prefix                 │
│       → Returns top-20 semantically similar chunks               │
│                                                                  │
│  Stage 1b: Sparse Retriever (BM25)                               │
│       → Scores chunks by exact keyword frequency                 │
│       → Returns top-20 keyword-matched chunks                    │
│                                                                  │
│  Stage 2:  Reciprocal Rank Fusion (RRF)                          │
│       → Fuses dense + sparse results by rank (not score)         │
│       → Produces ~30 unique, deduplicated candidates             │
│                                                                  │
│  Stage 3:  Cross-Encoder Reranker                                │
│       → Re-scores each (query, chunk) pair together              │
│       → Returns the top 5 most relevant chunks                   │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│              LLM Generation (Groq — Llama 3.3 70B)               │
│       → Receives the 5 highest-quality chunks as context         │
│       → Generates a grounded, citation-backed answer             │
└──────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Why This Choice |
|-------|-----------|-----------------|
| **Data Source** | SEC EDGAR via `edgartools` | Direct, official API access to 10-K filings |
| **Configuration** | `pydantic-settings` | Type-safe, validated configuration from `.env` files with a singleton pattern |
| **Data Validation** | Pydantic v2 | Strict schema enforcement on filing documents and chunk metadata |
| **Chunking** | `langchain-text-splitters` + custom `SemanticSplitter` | Hybrid strategy: semantic boundaries with recursive size guardrails |
| **Embeddings** | `sentence-transformers` — `BAAI/bge-large-en-v1.5` | State-of-the-art on financial retrieval benchmarks, 1024-dim output, asymmetric query support |
| **Vector Store** | Qdrant Cloud | Stores payload alongside vectors, native metadata filtering, high-availability |
| **Sparse Retrieval** | `rank-bm25` (BM25Okapi) | Industry-standard keyword ranking, complementary to dense retrieval |
| **Rank Fusion** | Reciprocal Rank Fusion (RRF) | Scale-invariant fusion — uses ranks not scores, handles incompatible score distributions |
| **Reranking** | `sentence-transformers` CrossEncoder — `ms-marco-MiniLM-L-6-v2` | Reads query + chunk jointly for much higher accuracy than bi-encoders, only 22 MB |
| **LLM** | Groq — `llama-3.3-70b-versatile` | Ultra-fast inference with 128K context window |
| **Language** | Python 3.12 | Modern features, type hints, performance improvements |

---

## Corpus

**33 semiconductor companies**, each with their most recent 10-K annual filing (2025–2026 fiscal years), sourced from SEC EDGAR.

Three sections are extracted and indexed per company:

| Section | SEC Item | What It Contains |
|---------|----------|-----------------|
| `business` | Item 1 | Company overview, products, markets, competitive landscape |
| `risk_factors` | Item 1A | All disclosed risks — regulatory, supply chain, geopolitical, financial |
| `management_discussion` | Item 7 (MD&A) | Management's analysis of financial condition and results of operations |

**Companies included:**
ADI, ALAB, ALGM, AMAT, AMBA, AVGO, CDNS, COHR, CRDO, DIOD, ENTG, KLAC, LRCX, LSCC, MCHP, MKSI, MPWR, MRVL, MU, MXL, NVDA, NXPI, ON, ONTO, POWI, QCOM, RMBS, SITM, SLAB, SNPS, TER, TXN, WOLF

**Intentionally excluded:** AMD and SWKS — their `risk_factors` sections returned 0 characters during ingestion. Caught by manual log inspection; exclusion is documented.

### Corpus Statistics

| Metric | Value |
|--------|-------|
| Total companies | 33 |
| Total chunks (recursive strategy) | 4,873 |
| Avg chunks per company | ~148 |
| Avg characters per chunk | ~1,580 |
| Section breakdown | risk_factors 51.8% · business 24.2% · MD&A 24.0% |

---

## Module-by-Module Breakdown

### 1. Configuration — `rag/config.py`

A centralised, singleton configuration system using `pydantic-settings`. Every tuneable parameter in the system — API keys, model names, chunk sizes, retrieval depths — is defined once and read from a `.env` file. Pydantic automatically validates types, provides defaults, and raises clear errors when required values (like `GROQ_API_KEY`) are missing.

**Why Pydantic-settings and not raw `os.environ`?**
Raw environment variable access returns strings with no validation. `os.environ["CHUNK_SIZE"]` returns `"2000"` as a string — if someone sets it to `"abc"`, the error shows up deep inside the chunker, not at startup. Pydantic-settings validates at import time and converts to the correct type.

---

### 2. Ingestion — `rag/ingestion/`

Three files handle the full ingestion pipeline:

#### Data Models (`models.py`)
Defines strict Pydantic schemas for every data structure that flows through the system:
- **`FilingDocument`** — Represents one company's complete 10-K filing with all three sections, company metadata (ticker, CIK, filing date, accession number), and validation rules.
- **`ChunkMetadata`** — The metadata attached to every single chunk: ticker, company name, section, chunk index, character offsets, filing date. This metadata travels with the chunk through embedding, storage, and retrieval — enabling filtered queries like "only search NVDA risk_factors."
- **`Chunk`** — The atomic unit of the system: text content + metadata + an optional embedding vector.

#### Loader (`loader.py`)
Discovers and loads all 33 filing JSON files from the `filings_data/` directory. Validates each file against the `FilingDocument` schema (rejects files with missing or empty required sections). Provides both bulk loading (`load_all_filings`) and targeted single-company loading (`get_filing("NVDA")`).

#### Chunker (`chunker.py`)
Splits each filing section into overlapping text chunks. Three strategies are available, selectable via `.env`:

- **`recursive`** — Uses LangChain's `RecursiveCharacterTextSplitter`. Deterministic, fast (~0.12s for all 33 companies), splits on paragraph → sentence → word boundaries in priority order. Chunk size: 2,000 characters with 400-character overlap.

- **`semantic`** — A custom-built `_SemanticSplitter` that uses `sentence-transformers` to detect semantic boundaries. It embeds each sentence, computes cosine distance between consecutive sentences, and places chunk boundaries where distance spikes above the 95th percentile. This produces semantically coherent chunks that don't cut mid-thought.

- **`hybrid`** (recommended, default) — Runs the semantic splitter first, then applies the recursive splitter as a size guard on any chunk exceeding the maximum size. This prevents the edge case where a semantically uniform passage produces a single 15,000-character chunk that would blow out the embedding model's context window.

**Critical design rule:** A chunk never crosses a section boundary. Risk factors text never mixes with business text in the same chunk.

---

### 3. Embedding — `rag/embeddings/`

Two files implementing an abstract interface pattern:

#### Abstract Base (`base.py`)
Defines the `BaseEmbedder` contract with separate methods for document embedding and query embedding. This separation exists because of how BGE models work (see below). Any future embedder (OpenAI, Cohere, etc.) can be swapped in by implementing this interface — nothing else in the codebase needs to change.

#### Sentence Transformer Embedder (`sentence_transformer_embedder.py`)
Wraps the `BAAI/bge-large-en-v1.5` model from HuggingFace via `sentence-transformers`.

**Why `bge-large-en-v1.5`?**
It consistently ranks among the top models on the MTEB benchmark for retrieval tasks, especially on domain-specific and financial text. Its 1024-dimensional output provides rich representational capacity for nuanced financial language.

**Key design decisions:**

- **Asymmetric query prefix** — BGE models are trained with a specific instruction prefix on query inputs: `"Represent this sentence for searching relevant passages: "`. This prefix is applied only during query embedding (search time), never during document embedding (index time). Omitting this prefix measurably degrades retrieval quality — it is a common mistake in BGE implementations.

- **Batched encoding** — The 4,873 chunks are embedded in batches of 64. Processing all chunks one-by-one would incur per-call overhead 4,873 times. Processing all at once could exhaust RAM. Batching at 64 maximises hardware parallelism while keeping memory usage predictable.

- **L2 normalisation** — All output vectors are normalised to unit length. This makes cosine similarity equivalent to inner product (dot product), which is what FAISS's `IndexFlatIP` computes. One less computation per query.

- **Progress bar** — Embedding all 33 companies takes approximately 2-3 minutes on CPU (one-time cost). A `tqdm` progress bar keeps the user informed during the build process.

---

### 4. Vector Store — `rag/vectorstore/`

#### Qdrant Store (`qdrant_store.py`)
A persistent vector store backed by Qdrant Cloud.

**Why Qdrant and not FAISS?**
We initially built the system on FAISS, which was extremely fast but required a parallel JSON file to store metadata and relied on an "over-fetch and post-filter in Python" hack. Qdrant natively stores payload metadata alongside the vectors and provides highly-optimised pre-filtering using complex nested conditions. It dramatically simplifies the architecture.

**Metadata filtering:**
Instead of post-filtering, the Qdrant store converts incoming python filter dicts into native Qdrant `Filter` and `FieldCondition` objects before executing the search.

**Deterministic IDs:**
Qdrant requires vector IDs to be either integers or UUIDs. We generate deterministic UUIDs (`uuid.uuid5`) from our string-based chunk IDs (e.g. `NVDA_risk_factors_3`) to ensure idempotent upserts without duplicates.

---

### 5. Retrieval Pipeline — `rag/retrieval/`

Four files implementing the full 3-stage retrieval pipeline:

#### Dense Retriever (`dense_retriever.py`)
Wraps the embedding + Qdrant query into a clean interface. Takes a raw query string, embeds it using the BGE query prefix, and runs the search in Qdrant. Returns the top-K most semantically similar chunks.

**What it's good at:** Understanding meaning, synonyms, paraphrases. "semiconductor production bottleneck" will match "chip supply disruption" even though they share zero keywords.

**What it misses:** Exact keyword matches. If a user asks about "COGS" (cost of goods sold), the dense retriever might return chunks about "expenses" or "cost management" instead of chunks containing the exact term "COGS."

#### Sparse Retriever (`sparse_retriever.py`)
A keyword-based retriever using the BM25Okapi algorithm from the `rank-bm25` library. BM25 is a probabilistic ranking function that scores each chunk based on:
- **Term frequency** — How many times the query words appear in the chunk
- **Inverse document frequency** — Rare words (like "TSMC") contribute more signal than common words (like "the")
- **Length normalisation** — Penalises very long chunks to prevent them from dominating just because they contain more words

## Project Structure

```
Company_SEC_Rag/
│
├── .env                         # Environment variables (secrets, model config)
├── .env.example                 # Template for .env (safe to commit)
├── GOAL.md                      # Project vision and architecture goals
├── README.md                    # This file
│
├── filings_data/                # 33 company filing JSONs
│   ├── ADI.json                 # Each file: {business, risk_factors,
│   ├── NVDA.json                #   management_discussion, metadata}
│   └── ... (33 files)
│
├── rag/                         # Core RAG package
│   ├── __init__.py
│   ├── config.py                # Pydantic-settings singleton
│   │
│   ├── ingestion/               # Data loading and chunking
│   │   ├── models.py            # FilingDocument, Chunk, ChunkMetadata schemas
│   │   ├── loader.py            # File discovery and validation
│   │   └── chunker.py           # Recursive, semantic, and hybrid chunking
│   │
│   ├── embeddings/              # Text-to-vector conversion
│   │   ├── base.py              # Abstract BaseEmbedder interface
│   │   └── sentence_transformer_embedder.py  # BGE-large implementation
│   │
│   ├── vectorstore/             # Persistent vector storage
│   │   └── qdrant_store.py      # Qdrant Cloud client
│   │
│   └── retrieval/               # 3-stage retrieval pipeline
│       ├── dense_retriever.py   # Semantic similarity via Qdrant
│       ├── sparse_retriever.py  # BM25 keyword matching
│       ├── hybrid_retriever.py  # RRF fusion of dense + sparse
│       └── reranker.py          # Cross-encoder precision reranking
│
├── scripts/                     # Runnable entry points
│   ├── test_ingestion.py        # Tests config, loader, chunker
│   └── test_retrieval.py        # Tests sparse, dense, hybrid, reranker
│
├── test_results/                # Saved test outputs
│   └── ingestion_test_*.txt
│
└── data.py                      # Original data download script (edgartools)
```

---

## Configuration

All configuration is centralised in a single `.env` file. Copy `.env.example` to `.env` and fill in your values:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *(required)* | API key from [console.groq.com](https://console.groq.com) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq-hosted LLM model |
| `GROQ_TEMPERATURE` | `0.1` | Low temperature for factual, grounded answers |
| `GROQ_MAX_TOKENS` | `2048` | Maximum tokens in LLM response |
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | HuggingFace model ID for embeddings |
| `EMBEDDING_DIM` | `1024` | Embedding vector dimension |
| `EMBEDDING_DEVICE` | `cpu` | Device for sentence-transformers (`cpu`, `cuda`, `mps`) |
| `CHUNK_SIZE` | `2000` | Chunk size in characters (~512 tokens) |
| `CHUNK_OVERLAP` | `400` | Overlap between consecutive chunks (~100 tokens) |
| `DENSE_TOP_K` | `20` | Candidates fetched from dense retrieval |
| `SPARSE_TOP_K` | `20` | Candidates fetched from sparse retrieval |
| `RERANK_TOP_N` | `5` | Final chunks kept after reranking (sent to LLM) |
| `CHUNKING_STRATEGY` | `hybrid` | One of `recursive`, `semantic`, or `hybrid` |
| `FILINGS_DIR` | `filings_data` | Directory containing company filing JSONs |

---
