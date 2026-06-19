"""
Starter script: pull a company's latest 10-K, split into sections, save as JSON.
Run this LOCALLY (not in a sandbox) since it needs real internet access to SEC EDGAR.

Install:
    pip install edgartools

SEC requires every requester to identify themselves with a name + email.
Replace the identity string below with your own before running.
"""

import json
import os
import sys
from edgar import set_identity, Company

# REQUIRED by SEC -- must be a real name/email, or they will block your requests
set_identity("Aman Kushwaha amankushwaha2323@gmail.com")

OUTPUT_DIR = "filings_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def ingest_company(ticker: str) -> dict:
    """
    Fetch the latest 10-K for a given ticker, pull out the sections that
    actually matter for a risk/strategy RAG system, and save to JSON.
    """
    company = Company(ticker)
    filings = company.get_filings(form="10-K")
    latest_filing = filings.latest()

    if latest_filing is None:
        raise ValueError(f"No 10-K found for {ticker}")

    print(f"[{ticker}] Found 10-K filed {latest_filing.filing_date}", flush=True)

    # .obj() parses the raw filing into a structured TenK object,
    # which already exposes the standard 10-K Items as separate properties
    print(f"[{ticker}] Parsing 10-K document (this may take a moment)...", flush=True)
    tenk = latest_filing.obj()

    sections = {
        "ticker": ticker,
        "company_name": company.name,
        "cik": company.cik,
        "filing_date": str(latest_filing.filing_date),
        "accession_number": latest_filing.accession_no,
        # Item 1: Business overview
        "business": str(tenk.business) if tenk.business else "",
        # Item 1A: Risk Factors -- the highest-value section for this project
        "risk_factors": str(tenk.risk_factors) if tenk.risk_factors else "",
        # Item 7: Management's Discussion & Analysis
        "management_discussion": str(tenk.management_discussion) if tenk.management_discussion else "",
    }

    out_path = os.path.join(OUTPUT_DIR, f"{ticker}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2)

    print(f"[{ticker}] Saved -> {out_path}", flush=True)
    print(f"[{ticker}] business length:             {len(sections['business'])} chars", flush=True)
    print(f"[{ticker}] risk_factors length:         {len(sections['risk_factors'])} chars", flush=True)
    print(f"[{ticker}] management_discussion length:{len(sections['management_discussion'])} chars", flush=True)

    return sections


if __name__ == "__main__":
    # Test with one company first before scaling to all 35
    # ingest_company("NVDA")

    # Once this works, loop over your full sector list:
    
    SEMICONDUCTOR_TICKERS = [
        "NVDA", "AMD", "INTC", "QCOM", "AVGO", "TXN", "MU", "MRVL", "ADI", "NXPI",
        "AMAT", "LRCX", "KLAC", "TER", "ENTG", "MKSI", "COHR", "ONTO",
        "ON", "SWKS", "MCHP", "SNPS", "CDNS", "MPWR", "ALGM", "POWI", "LSCC",
        "RMBS", "DIOD", "SLAB", "MXL", "WOLF", "AMBA", "CRDO", "ALAB", "SITM",
    ]
    
    all_data = {}
    for ticker in SEMICONDUCTOR_TICKERS:
        try:
            all_data[ticker] = ingest_company(ticker)
        except Exception as e:
            print(f"[{ticker}] FAILED: {e}", flush=True)
        # be polite to SEC's servers -- avoid hammering them in a tight loop
        import time
        time.sleep(1)