import logging
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

from rag.config import settings

logger = logging.getLogger(__name__)

class QueryFilter(BaseModel):
    tickers: Optional[List[str]] = Field(
        default=None, 
        description="A list of stock ticker symbols of the companies mentioned in the query, if any (e.g., ['NVDA', 'AAPL', 'MSFT']). Must be uppercase. If no specific company is mentioned, leave null."
    )

class QueryAnalyzer:
    """
    Lightweight LLM call to extract metadata filters (e.g. ticker) from a query 
    before hitting the vector database.
    """
    def __init__(self):
        if not settings.groq_api_keys:
            raise ValueError("GROQ_API_KEYS is not set in .env")
        self.api_keys = [k.strip() for k in settings.groq_api_keys.split(",") if k.strip()]
        
    def analyze(self, query: str) -> Optional[dict]:
        """
        Analyzes the query and extracts metadata filters.
        Returns a dict e.g., {"ticker": ["NVDA", "AVGO"]} or None.
        """
        logger.info(f"Analyzing query for metadata filters: '{query}'")
        
        system_prompt = """You are a financial query analyzer. 
        Your job is to extract a list of all stock ticker symbols of the companies mentioned in the user's question.
        If the user asks about a specific company (e.g., 'What does NVDA say...'), return ['NVDA'].
        If the user asks to compare multiple companies (e.g., 'Compare NVDA and AVGO...'), return ['NVDA', 'AVGO'].
        If the user asks a general question without specifying companies (e.g., 'Summarize supply chain risks'), return null.
        Always return the tickers in uppercase."""
        
        for i, api_key in enumerate(self.api_keys):
            try:
                llm = ChatGroq(
                    api_key=api_key,
                    model=settings.groq_model,
                    temperature=0
                )
                analyzer = llm.with_structured_output(QueryFilter)
                
                result = analyzer.invoke([
                    ("system", system_prompt),
                    ("human", query)
                ])
                
                if result and result.tickers:
                    logger.info(f"Extracted filter: tickers={result.tickers}")
                    return {"ticker": result.tickers}
                
                return None
                
            except Exception as e:
                err_msg = str(e).lower()
                if "429" in err_msg or "rate limit" in err_msg or "rate_limit_exceeded" in err_msg:
                    logger.warning(f"QueryAnalyzer: Key {i+1} hit rate limit. Trying next key...")
                    continue
                logger.error(f"Error in QueryAnalyzer: {e}")
                return None
                
        logger.error("QueryAnalyzer exhausted all API keys due to rate limits.")
        return None
