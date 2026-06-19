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

**Why BM25 and not TF-IDF?**
BM25 includes a term frequency saturation parameter (`k1=1.5`) that prevents a single word appearing 50 times from dominating the score. TF-IDF has no such control. BM25 is the standard in production search systems (Elasticsearch uses it by default).

**Tokenisation:**
Uses a custom financial-aware tokeniser that preserves hyphenated terms. `"10-K filing"` tokenises as `["10-k", "filing"]`, not `["10", "k", "filing"]`. This matters for financial terminology like `"chip-on-board"`, `"risk-adjusted"`, and `"year-over-year"`.

**Index lifecycle:**
The BM25 index is built entirely in memory from all 4,873 chunks at application startup (~0.5 seconds). It is not persisted to disk because rebuilding is so fast.

#### Hybrid Retriever (`hybrid_retriever.py`)
Combines dense and sparse results using **Reciprocal Rank Fusion (RRF)**.

**The core problem RRF solves:**
Dense retrieval returns cosine similarity scores in [0, 1]. BM25 returns scores that can range from 0 to 30+. These scales are incomparable — you cannot simply average them. Naive weighted combination (`0.7 * dense_score + 0.3 * bm25_score`) is unreliable because the score distributions shift depending on the query.

**How RRF works:**
RRF ignores raw scores entirely and uses only the **rank** of each result. For a chunk appearing at rank `r`, its RRF score is `1 / (K + r)` where `K = 60` (the standard smoothing constant). If a chunk appears in both the dense and sparse result lists, its RRF scores from both lists are summed. A chunk ranked 1st in both lists gets the highest combined score.

**Why K = 60?**
This is the value from the original RRF paper (Cormack, Clarke & Butt, 2009). It ensures that a chunk ranked 1st gets only modestly more credit than one ranked 5th, preventing any single result from dominating. The value is standard across production systems.

#### Reranker (`reranker.py`)
The precision stage. Uses a **cross-encoder** model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to re-score the top ~30 hybrid candidates and select the best 5 for the LLM.

**Why a cross-encoder?**
Bi-encoders (used in dense retrieval) embed the query and chunk independently, then compare the resulting vectors. This is fast but loses information — the model never "reads" the query and chunk together. A cross-encoder takes the concatenation `[query] [SEP] [chunk]` as input and produces a single relevance score. It reads both simultaneously, capturing fine-grained interactions that bi-encoders miss.

**Why not use the cross-encoder for everything?**
Speed. The cross-encoder must process each (query, chunk) pair individually. Running it on all 4,873 chunks would take ~80 seconds per query. Running it on 30 candidates takes ~500ms. The two-stage approach (fast retrieval first, precise cross-encoder second) is the standard production pattern.

**Model choice:**
`ms-marco-MiniLM-L-6-v2` is 22 MB, runs well on CPU (~500ms for 30 pairs), and was trained on the MS MARCO passage ranking benchmark — one of the most widely used datasets for training retrieval models.

---

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

## Key Design Decisions

### Why hybrid chunking instead of fixed-size chunks?
Fixed-size chunking (e.g., every 500 tokens) inevitably cuts through the middle of sentences and paragraphs, splitting related ideas across two chunks. Semantic chunking detects natural topic boundaries by measuring embedding distance between consecutive sentences. The hybrid approach adds a recursive size guard to prevent the edge case where a semantically uniform passage produces a single oversized chunk.

### Why RRF over weighted score normalisation?
Dense retrieval scores (cosine similarity, 0 to 1) and BM25 scores (0 to 30+) live on fundamentally different scales. Normalising both to [0, 1] and then averaging is fragile because the score distributions shift depending on the query. RRF uses only rank positions, which are directly comparable regardless of the underlying scoring function.

### Why BGE with an asymmetric query prefix?
BGE models are trained with a specific instruction prefix on queries but not on documents. This asymmetry reflects the reality that a search query ("find risks about supply chain") has different intent than the document text it should match ("The company's operations depend on..."). Skipping the prefix is a common implementation mistake that measurably hurts retrieval quality on financial text.

### Why cross-encoder reranking instead of just taking top-5 from FAISS?
Bi-encoders embed query and chunk independently — fast but approximate. Cross-encoders read both together — slow but precise. For a 5-chunk retrieval, the difference between "good enough" and "the actual best 5 chunks" directly affects the quality of the LLM's answer. The ~500ms latency cost is negligible compared to the LLM generation time.

### Why Qdrant over FAISS?
We initially used FAISS but quickly hit limits with metadata filtering. FAISS required us to over-fetch vectors and post-filter them in Python, which is messy and unscalable. Qdrant solves this by storing payloads natively alongside vectors and executing pre-filtering before the similarity search. Additionally, using Qdrant Cloud removes the need to store the heavy index files locally.

### Why were AMD and SWKS excluded?
Their `risk_factors` sections returned 0 characters during ingestion — silently, without errors. This was caught during manual review of ingestion logs. Rather than including companies with incomplete data (which would produce misleading retrieval results), they were intentionally excluded and the exclusion is documented.

---

## Current Progress

| Module | Status | Details |
|--------|--------|---------|
| Configuration | ✅ Complete | Pydantic-settings singleton |
| Data Ingestion | ✅ Complete | 33 companies loaded and validated |
| Chunking | ✅ Complete | 3 strategies, 4,873 chunks produced |
| Embedding | ✅ Complete | BGE-large, batched, with query prefix |
| Vector Store | ✅ Complete | Qdrant Cloud |
| Dense Retrieval | ✅ Complete | Qdrant semantic search |
| Sparse Retrieval | ✅ Complete | BM25Okapi keyword search |
| Hybrid Retrieval | ✅ Complete | RRF fusion |
| Reranker | ✅ Complete | Cross-encoder ms-marco-MiniLM |
| Build Index Script | ✅ Complete | Wires ingestion → embedding → Qdrant |
| LLM Generation | ✅ Complete | Groq generator with citation prompts |
| API | ✅ Complete | FastAPI endpoint with cold-start |
| Frontend | 🔲 Not started | Ultra-premium React/Vite UI |

---

## Setup and Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd Company_SEC_Rag

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install pydantic-settings langchain-text-splitters langchain-huggingface
pip install sentence-transformers numpy qdrant-client rank-bm25 tqdm

# 4. Configure environment
copy .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Run tests
python scripts/test_ingestion.py    # Tests config, loader, chunker
python scripts/test_retrieval.py    # Tests full retrieval pipeline
```

---

*Last updated: June 2026*
*Author: Aman Kushwaha*
