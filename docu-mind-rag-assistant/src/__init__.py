"""
DocuMind RAG Assistant - Core Package
"""
from src.document_loader import load_and_chunk_pdf, load_and_chunk_text, load_document_chunks
from src.vector_db import build_vector_store, save_index, load_index, get_embedding_model
from src.rag_engine import build_rag_chain, query_rag, get_llm, sanitize_api_key, FALLBACK_MESSAGE
from src.summarizer import (
    summarize_document,
    summarize_executive,
    summarize_technical_key_points,
    summarize_action_items,
)

__all__ = [
    "load_and_chunk_pdf",
    "load_and_chunk_text",
    "load_document_chunks",
    "build_vector_store",
    "save_index",
    "load_index",
    "get_embedding_model",
    "build_rag_chain",
    "query_rag",
    "get_llm",
    "sanitize_api_key",
    "FALLBACK_MESSAGE",
    "summarize_document",
    "summarize_executive",
    "summarize_technical_key_points",
    "summarize_action_items",
]

__version__ = "0.1.0"
