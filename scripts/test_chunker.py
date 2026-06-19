import sys
import os
import io

# Ensure stdout handles emojis and special characters cleanly on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Add the project root to the python path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rag.ingestion.loader import load_all_filings
from rag.ingestion.chunker import Chunker

def test_chunking_quality():
    print("Loading filings...")
    # Load all filings, but we will only grab the first one (e.g. ADI)
    filings = load_all_filings()
    if not filings:
        print("No filings found!")
        return
        
    test_filing = filings[0]
    print(f"\n--- Testing Chunking on: {test_filing.company_name} ({test_filing.ticker}) ---")
    
    # Initialize the chunker. 
    # It will use the 'hybrid' strategy and 'all-MiniLM-L6-v2' as defined in the .env/code
    print("Initializing chunker (this will load the fast MiniLM model)...")
    chunker = Chunker(strategy="hybrid")
    
    print(f"\nChunking the 'risk_factors' section...")
    risk_factors_text = test_filing.risk_factors
    chunks = chunker.chunk_section(risk_factors_text, "risk_factors", test_filing)
    
    print(f"\nTotal chunks produced for risk_factors: {len(chunks)}\n")
    print("===================================================================")
    print(" EYEBALL TEST: Showing the first 5 chunks to inspect boundaries")
    print("===================================================================")
    
    for i in range(min(5, len(chunks))):
        chunk = chunks[i]
        print(f"\n\n[CHUNK {i+1} | {chunk.char_count()} chars]")
        print("-" * 60)
        print(chunk.text)
        print("-" * 60)

if __name__ == "__main__":
    test_chunking_quality()
