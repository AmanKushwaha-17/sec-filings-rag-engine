"""
rag/llm/generator.py
====================
Wrapper for the Groq LLM Generation step using LangChain.
"""

import logging
from typing import List

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

from rag.config import settings
from rag.vectorstore.qdrant_store import QueryResult
from rag.llm.prompts import get_rag_prompt

logger = logging.getLogger(__name__)

class RAGGenerator:
    """
    Handles generating answers from the Groq API based on retrieved SEC contexts.
    """
    def __init__(self):
        if not settings.groq_api_keys:
            raise ValueError("GROQ_API_KEYS is not set in .env")
            
        self.api_keys = [k.strip() for k in settings.groq_api_keys.split(",") if k.strip()]
        if not self.api_keys:
            raise ValueError("No valid keys found in GROQ_API_KEYS")
            
        self.prompt = get_rag_prompt()
        logger.info(f"RAGGenerator initialized with {len(self.api_keys)} keys for model: {settings.groq_model}")

    def generate_answer(self, query: str, contexts: List[QueryResult]) -> str:
        """
        Takes the user query and a list of retrieved chunks, formats them,
        and generates an LLM response with citations.
        """
        formatted_context = self._format_contexts(contexts)
        
        logger.info(f"Generating answer for query: '{query}'")
        
        last_error = None
        for i, api_key in enumerate(self.api_keys):
            try:
                # Initialize LLM dynamically for each key attempt
                llm = ChatGroq(
                    api_key=api_key,
                    model=settings.groq_model,
                    temperature=settings.groq_temperature,
                    max_tokens=settings.groq_max_tokens,
                )
                chain = self.prompt | llm | StrOutputParser()
                
                response = chain.invoke({
                    "context": formatted_context,
                    "query": query
                })
                return response
                
            except Exception as e:
                last_error = e
                err_msg = str(e).lower()
                
                # Check if it is a rate limit error (HTTP 429)
                if "429" in err_msg or "rate limit" in err_msg or "rate_limit_exceeded" in err_msg:
                    logger.warning(f"Key {i+1}/{len(self.api_keys)} hit rate limit. Trying next key...")
                    continue
                else:
                    # If it's a different error, just fail immediately
                    logger.error(f"Error calling Groq API: {e}")
                    return f"An error occurred while generating the answer: {e}"
                    
        # If we exhausted all keys
        logger.error(f"All {len(self.api_keys)} Groq API keys exhausted due to rate limits.")
        return f"An error occurred while generating the answer: All API keys hit rate limits. Last error: {last_error}"

    @staticmethod
    def _format_contexts(contexts: List[QueryResult]) -> str:
        """
        Formats retrieved QueryResults into an XML-like string for the LLM.
        Example:
        <doc id="1" ticker="NVDA">
        Text snippet...
        </doc>
        """
        formatted = []
        for i, res in enumerate(contexts, start=1):
            doc = (
                f'<doc id="{i}" ticker="{res.metadata.ticker}">\n'
                f'{res.text}\n'
                f'</doc>'
            )
            formatted.append(doc)
            
        return "\n\n".join(formatted)
