"""
rag/config.py
=============
Centralised settings for the SEC 10-K RAG system.

All values are read from environment variables (or a .env file).
Import the singleton `settings` object everywhere instead of reading
os.environ directly.

Usage:
    from rag.config import settings

    print(settings.groq_model)
    print(settings.chunk_size)
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Single source of truth for every tuneable parameter and secret.

    Pydantic-settings automatically:
      - reads from a .env file in the project root
      - validates types (e.g. int, bool, Path)
      - raises a clear error if a required secret is missing
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,       # GROQ_API_KEY == groq_api_key
        extra="ignore",             # silently ignore unknown env vars
    )

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    filings_dir: Path = Path("filings_data")

    # ------------------------------------------------------------------
    # NVIDIA NIM  (API-based embeddings)
    # ------------------------------------------------------------------
    embedding_model: str = "nvidia/nv-embedqa-e5-v5"
    embedding_dim: int = 1024
    embedding_device: str = "cpu"
    nvidia_api_keys: str = ""                   # REQUIRED — comma separated keys



    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
    chunk_size: int = 800
    chunk_overlap: int = 150

    # ------------------------------------------------------------------
    # Vector store  (Qdrant Cloud)
    # ------------------------------------------------------------------
    qdrant_url: str = ""                        # REQUIRED — must be in .env
    qdrant_api_key: str = ""                    # REQUIRED — must be in .env
    qdrant_collection_name: str = "sec_10k_chunks"

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    dense_top_k: int = 40
    sparse_top_k: int = 40
    rerank_top_n: int = 10

    # ------------------------------------------------------------------
    # LLM  (Groq — ultra-fast inference)
    # ------------------------------------------------------------------
    groq_api_keys: str                           # REQUIRED — must be in .env
    groq_model: str = "llama-3.3-70b-versatile"
    groq_temperature: float = 0.1
    groq_max_tokens: int = 2048

    # ------------------------------------------------------------------
    # Chunking strategy
    # ------------------------------------------------------------------
    chunking_strategy: str = "hybrid"

    # ------------------------------------------------------------------
    # Sections available in each filing JSON
    # ------------------------------------------------------------------
    known_sections: list[str] = [
        "business",
        "risk_factors",
        "management_discussion",
    ]


# ---------------------------------------------------------------------------
# Singleton — import this everywhere
# ---------------------------------------------------------------------------
settings = Settings()
