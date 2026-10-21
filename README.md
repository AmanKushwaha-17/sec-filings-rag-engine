# SEC 10-K RAG Intelligence Engine

An **advanced Hybrid Retrieval-Augmented Generation (RAG) system** built over SEC 10-K filings for **33 U.S. semiconductor companies**. The system features a modern decoupled architecture: a **FastAPI backend** (deployed on Render) and a stunning **React/Vite frontend** (deployed on Vercel).

At its core, this engine implements an LLM-driven Query Analyzer for metadata pre-filtering, and a high-performance **3-stage hybrid retrieval pipeline** featuring Nvidia NIM embeddings, BM25 sparse retrieval, and cross-encoder reranking.

---

## 🚀 Live Demo

- **Frontend UI:** [https://sec-filings-rag-engine.vercel.app/](https://sec-filings-rag-engine.vercel.app/)
- **Backend API:** Hosted on Render (`https://sec-filings-rag-engine.onrender.com/api/health` to check status)

---

## 🏗️ Architecture Overview

```mermaid
flowchart TD
    %% Black & White Theme
    classDef default fill:#ffffff,stroke:#000000,stroke-width:1.5px,color:#000000;

    subgraph ClientLayer ["🖥️ Frontend UI (Vercel)"]
        User(["👤 User Query"]) -->|Submits Prompt| UI["React + Vite UI<br/><i>(Split-Pane Layout & Citation Inspector)</i>"]
    end

    subgraph BackendLayer ["⚡ API Orchestration (FastAPI on Render)"]
        UI -->|POST /api/chat| Router["FastAPI Server"]
        Router --> QA["LLM Query Analyzer<br/><b>Groq (Llama 3.3 70B)</b><br/><i>(Entity & Ticker Extraction)</i>"]
        QA --> Filter{"Metadata Filter<br/>Detected?"}
    end

    subgraph RetrievalLayer ["🔍 3-Stage Hybrid Retrieval Pipeline"]
        Filter -->|Ticker Payload Filters| DenseRet["Stage 1a: Dense Retriever<br/><b>Nvidia NIM Embeddings</b><br/><i>(nemotron-3-embed-1b)</i>"]
        Filter -->|Keyword Query| SparseRet["Stage 1b: Sparse Retriever<br/><b>BM25 Scoring</b><br/><i>(Exact keyword matching)</i>"]

        DenseRet <-->|Filtered ANN Search| Qdrant[("Qdrant Cloud<br/>Vector Store")]
        DenseRet -->|Top-40 Dense Chunks| RRF["Stage 2: Reciprocal Rank Fusion (RRF)<br/><i>Score = Σ 1 / (60 + Rank)</i><br/><i>Deduplicated Candidate Pool</i>"]
        SparseRet -->|Top-40 Sparse Chunks| RRF

        RRF -->|Candidate Chunks| Rerank["Stage 3: Cross-Encoder Reranker<br/><b>Nvidia NIM Reranker</b><br/><i>(llama-nemotron-rerank-1b-v2)</i>"]
    end

    subgraph GenerationLayer ["🧠 LLM Generation & Response"]
        Rerank -->|Top 10 High-Relevance Chunks| GenLLM["LLM Generation<br/><b>Groq (Llama 3.3 70B)</b><br/><i>(Dynamic Rate-Limit API Key Cycling)</i>"]
        GenLLM -->|Grounded Citation-Backed Answer| Response["Response Payload<br/><i>(Answer + Verified Citations + Latency)</i>"]
    end

    Response -->|Render Stream/Answer| UI
```

### Data Ingestion & Indexing Pipeline

```mermaid
flowchart LR
    %% Black & White Theme
    classDef default fill:#ffffff,stroke:#000000,stroke-width:1.5px,color:#000000;

    SEC[("SEC EDGAR<br/>10-K Filings")] --> Loader["Filing Loader<br/><i>33 Semiconductor Companies</i>"]
    Loader --> Extractor["Section Extractor<br/>• Item 1: Business<br/>• Item 1A: Risk Factors<br/>• Item 7: MD&A"]
    Extractor --> Chunker["Smart Text Chunker<br/>• Recursive Character Splitting<br/>• Metadata Attachment"]
  
    Chunker --> BM25Index[("BM25 Sparse Index<br/>(Keyword Index)")]
    Chunker --> Embedder["Nvidia NIM Embedder<br/><i>nemotron-3-embed-1b</i>"]
    Embedder --> QdrantDB[("Qdrant Cloud<br/>Vector Store")]
```

---

## 🛠️ Tech Stack

| Component                  | Technology                                   | Why This Choice                                                                                                     |
| -------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Frontend UI**      | React + Vite                                 | Blazing fast local development, deployed globally on Vercel. Features a split-pane layout for inspecting citations. |
| **Backend API**      | FastAPI                                      | High-performance Python async framework, deployed on Render.                                                        |
| **LLM Inference**    | Groq (`openai/gpt-oss-120b`)               | Ultra-fast LPU inference enabling both Query Analysis and Generation to run in under ~2 seconds.                    |
| **Embeddings**       | Nvidia NIM (`nemotron-3-embed-1b`)         | Cloud-hosted state-of-the-art embedding model preventing local OOM crashes.                                         |
| **Reranker**         | Nvidia NIM (`llama-nemotron-rerank-1b-v2`) | Cloud-hosted cross-encoder for high-accuracy semantic ranking.                                                      |
| **Vector Store**     | Qdrant Cloud                                 | Stores vectors + metadata payload natively. Supports complex nested metadata pre-filtering.                         |
| **Sparse Retrieval** | `rank-bm25`                                | Industry-standard keyword ranking to complement dense retrieval.                                                    |
| **Data Validation**  | Pydantic v2                                  | Strict schema enforcement on config, filing documents, and chunk metadata.                                          |

---

## 🗄️ The Corpus

**33 semiconductor companies**, each with their most recent 10-K annual filing (2025–2026 fiscal years), sourced directly from SEC EDGAR.

Three specific critical sections are extracted and indexed per company:

- `business` (Item 1): Company overview, products, competitive landscape.
- `risk_factors` (Item 1A): All disclosed risks (supply chain, geopolitical, financial).
- `management_discussion` (Item 7): MD&A on financial condition and results.

**Companies included:**
ADI, ALAB, ALGM, AMAT, AMBA, AVGO, CDNS, COHR, CRDO, DIOD, ENTG, KLAC, LRCX, LSCC, MCHP, MKSI, MPWR, MRVL, MU, MXL, NVDA, NXPI, ON, ONTO, POWI, QCOM, RMBS, SITM, SLAB, SNPS, TER, TXN, WOLF.

---

## 💻 Local Development Setup

### 1. Environment Variables

Copy `.env.example` to `.env` in the root directory and fill in your API keys. You will need:

- `GROQ_API_KEY` (comma-separated if using multiple for rate limits)
- `NVIDIA_API_KEY`
- `QDRANT_URL` and `QDRANT_API_KEY`

### 2. Backend (FastAPI)

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install requirements
pip install -r requirements.txt

# Start the API server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Frontend (React/Vite)

```bash
# Open a new terminal
cd frontend

# Install Node dependencies
npm install

# Start the Vite development server
npm run dev
```

### 4. Running the Indexer (First Time Only)

If your Qdrant cloud is empty, you'll need to run the data ingestion and vector embedding script:

```bash
python -m scripts.build_index
```

---

## 🚀 Deployment Guide

### Deploying the Backend (Render)

1. Connect your GitHub repository to Render as a **Web Service**.
2. **Build Command:** `pip install -r requirements.txt`
3. **Start Command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
4. Add your Environment Variables (`GROQ_API_KEY`, `NVIDIA_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`).
   *Note: The `render.yaml` file is included in this repository to automate this setup.*

### Deploying the Frontend (Vercel)

1. Import the repository in Vercel.
2. Set the **Root Directory** to `frontend`.
3. Select **Vite** as the Framework Preset.
4. Set the `VITE_API_URL` environment variable if needed, or rely on the `vercel.json` rewrite rules to automatically proxy `/api` requests to your Render backend.

---

## 🧠 Key Differentiators vs. "Standard" RAG

- **Multi-Document Synthesis:** Standard RAG struggles to compare two documents. This system uses a Query Analyzer to extract specific tickers from the prompt (e.g. *"Compare NVDA and AVGO"*), executing targeted metadata pre-filtering across Qdrant before vector search even begins.
- **RRF + Reranker Precision:** Fusing Sparse (BM25) and Dense (Nvidia Embeddings) retrieval ensures we catch both exact keywords and semantic meaning. Feeding the top 30-40 results into a Cross-Encoder Reranker guarantees only the absolute highest-fidelity context makes it into the prompt.
- **Stateless Cloud Architecture:** Moving embeddings and reranking to the Nvidia NIM cloud APIs eliminated local OOM (Out-of-Memory) crashes on small Render instances, making the API incredibly lightweight.
- **Rate Limit Cycling:** The Groq API is famously fast but has strict free-tier rate limits. This system automatically rotates through an array of API keys dynamically on HTTP 429 errors to guarantee uptime.
