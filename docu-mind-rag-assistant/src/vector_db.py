"""
FAISS Vector Database module using HuggingFace Embeddings.
"""
import os
from typing import List, Union
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """
    Instantiate HuggingFaceEmbeddings with the specified model name.
    """
    # Handles both 'all-MiniLM-L6-v2' and 'sentence-transformers/all-MiniLM-L6-v2'
    full_model_name = (
        model_name
        if "/" in model_name
        else f"sentence-transformers/{model_name}"
    )
    return HuggingFaceEmbeddings(model_name=full_model_name)


def build_vector_store(
    chunks: List[Document],
    model_name: str = DEFAULT_EMBEDDING_MODEL
) -> FAISS:
    """
    Build a FAISS vector store from document chunks using HuggingFaceEmbeddings.

    Args:
        chunks: List of document chunks to embed and index.
        model_name: HuggingFace sentence transformer model name.

    Returns:
        FAISS vector store instance.
    """
    if not chunks:
        raise ValueError("Cannot build vector store with empty chunks list.")
    
    embeddings = get_embedding_model(model_name)
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def save_index(vector_store: FAISS, path: Union[str, Path]) -> None:
    """
    Persist FAISS index and docstore to disk.

    Args:
        vector_store: FAISS vector store instance.
        path: Target folder path.
    """
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(folder))


def load_index(
    path: Union[str, Path],
    model_name: str = DEFAULT_EMBEDDING_MODEL
) -> FAISS:
    """
    Load a persisted FAISS index from disk.

    Args:
        path: Path to folder containing index.faiss and index.pkl.
        model_name: HuggingFace sentence transformer model name.

    Returns:
        FAISS vector store instance.
    """
    folder = Path(path)
    if not folder.exists():
        raise FileNotFoundError(f"Vector store directory not found: {path}")

    embeddings = get_embedding_model(model_name)
    return FAISS.load_local(
        str(folder),
        embeddings,
        allow_dangerous_deserialization=True
    )
