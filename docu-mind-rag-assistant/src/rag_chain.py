"""
Alias for src.rag_engine to maintain backwards compatibility.
"""
from src.rag_engine import (
    get_llm,
    build_rag_chain,
    query_rag,
    FALLBACK_MESSAGE,
    DEFAULT_GROQ_MODEL
)
