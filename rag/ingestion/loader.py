"""
rag/ingestion/loader.py
=======================
Loads and validates SEC 10-K filing JSON files from disk.

Responsibilities:
  - Discover all .json files in the configured filings directory
  - Parse and validate each file into a FilingDocument Pydantic model
  - Handle corrupt / missing files gracefully (log warning, skip)
  - Expose helpers to load a single ticker or all filings at once

Usage:
    from rag.ingestion.loader import load_all_filings, get_filing

    filings = load_all_filings()          # all 33 companies
    nvda    = get_filing("NVDA")          # single company
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from rag.config import settings
from rag.ingestion.models import FilingDocument

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_single_file(path: Path) -> Optional[FilingDocument]:
    """
    Read one JSON file and return a validated FilingDocument.

    Returns None (and logs a warning) if:
      - The file cannot be read (PermissionError, etc.)
      - The JSON is malformed
      - Required fields (ticker, company_name, etc.) are missing

    Args:
        path: Absolute or relative path to the .json filing file.

    Returns:
        A validated FilingDocument, or None on failure.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Cannot read file %s: %s", path, exc)
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Malformed JSON in %s: %s", path, exc)
        return None

    try:
        filing = FilingDocument(**data)
    except ValidationError as exc:
        logger.warning("Validation failed for %s:\n%s", path.name, exc)
        return None

    # Log section sizes so we can spot empty sections early
    logger.info(
        "[%s] Loaded | business=%d chars | risk_factors=%d chars | md&a=%d chars",
        filing.ticker,
        len(filing.business),
        len(filing.risk_factors),
        len(filing.management_discussion),
    )

    return filing


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_all_filings(
    filings_dir: Optional[Path] = None,
) -> list[FilingDocument]:
    """
    Load every .json file from the filings directory.

    Files that fail validation are skipped with a warning — the rest
    of the pipeline continues unaffected.

    Args:
        filings_dir: Directory containing the JSON files.
                     Defaults to ``settings.filings_dir``.

    Returns:
        List of validated FilingDocument objects (one per company).

    Raises:
        FileNotFoundError: If the filings directory does not exist.

    Example:
        >>> filings = load_all_filings()
        >>> len(filings)
        33
    """
    directory = Path(filings_dir or settings.filings_dir)

    if not directory.exists():
        raise FileNotFoundError(
            f"Filings directory not found: {directory.resolve()}"
        )

    json_files = sorted(directory.glob("*.json"))

    if not json_files:
        logger.warning("No .json files found in %s", directory)
        return []

    logger.info("Found %d JSON files in %s", len(json_files), directory)

    filings: list[FilingDocument] = []
    failed: list[str] = []

    for path in json_files:
        filing = _load_single_file(path)
        if filing is not None:
            filings.append(filing)
        else:
            failed.append(path.name)

    logger.info(
        "Loaded %d/%d filings successfully. Failed: %s",
        len(filings),
        len(json_files),
        failed or "none",
    )

    return filings


def get_filing(
    ticker: str,
    filings_dir: Optional[Path] = None,
) -> FilingDocument:
    """
    Load a single company's filing by ticker symbol.

    Args:
        ticker: Stock ticker (case-insensitive). E.g. "nvda" or "NVDA".
        filings_dir: Override the default filings directory.

    Returns:
        The validated FilingDocument for that ticker.

    Raises:
        FileNotFoundError: If no JSON file exists for that ticker.
        ValueError: If the file exists but fails validation.

    Example:
        >>> doc = get_filing("NVDA")
        >>> doc.company_name
        'NVIDIA Corporation'
    """
    directory = Path(filings_dir or settings.filings_dir)
    ticker_upper = ticker.strip().upper()
    path = directory / f"{ticker_upper}.json"

    if not path.exists():
        raise FileNotFoundError(
            f"No filing found for ticker '{ticker_upper}' at {path.resolve()}"
        )

    filing = _load_single_file(path)

    if filing is None:
        raise ValueError(
            f"Filing file for '{ticker_upper}' exists but failed to load/validate. "
            "Check logs for details."
        )

    return filing


def list_available_tickers(
    filings_dir: Optional[Path] = None,
) -> list[str]:
    """
    Return sorted list of ticker symbols available on disk.

    Useful for the API's GET /companies endpoint.

    Args:
        filings_dir: Override the default filings directory.

    Returns:
        Sorted list of ticker strings e.g. ['ADI', 'ALAB', ..., 'WOLF']
    """
    directory = Path(filings_dir or settings.filings_dir)
    return sorted(p.stem for p in directory.glob("*.json"))
