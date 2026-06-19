"""
rag/llm/prompts.py
==================
System prompts and templates for the Groq LLM Generator.
"""

from langchain_core.prompts import ChatPromptTemplate

# System prompt that strictly enforces citations and forbids hallucination.
RAG_SYSTEM_PROMPT = """You are a highly analytical AI assistant analyzing SEC 10-K filings.
Your task is to answer the user's question using ONLY the provided context documents.

RULES:
1. You MUST NOT hallucinate or bring in outside knowledge. If the answer is not contained in the context, say "I cannot find the answer in the provided documents."
2. You MUST cite your sources using the chunk index and ticker provided in the context. Format citations like this: [1] or [1, NVDA].
3. Write your answer in well-structured, professional paragraphs. Avoid using bulleted lists unless explicitly asked.

CONTEXT DOCUMENTS:
{context}
"""

def get_rag_prompt() -> ChatPromptTemplate:
    """Returns the LangChain ChatPromptTemplate for standard RAG generation."""
    return ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "{query}")
    ])
