"""
Document loader module for parsing and chunking PDF documents and text files.
Includes robust handling for encrypted PDFs, empty files, and corrupted documents.
"""
import os
from typing import List, Union
from pathlib import Path
import pypdf.errors
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class PDFEncryptedError(Exception):
    """Raised when a PDF cannot be loaded because it is encrypted/password-protected."""
    pass


class EmptyDocumentError(Exception):
    """Raised when an ingested document is completely empty."""
    pass


def load_and_chunk_pdf(
    file_path: Union[str, Path],
    chunk_size: int = 750,
    chunk_overlap: int = 100
) -> List[Document]:
    """
    Load a PDF file and split it into chunks with preserved page metadata.

    Args:
        file_path: Path to the PDF document.
        chunk_size: Target size of each chunk in characters (default: 750).
        chunk_overlap: Overlap between consecutive chunks (default: 100).

    Returns:
        List of Document chunk objects with metadata (page number, source path, etc.).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {file_path}")

    # Check for empty / zero-byte file
    if path.stat().st_size == 0:
        raise EmptyDocumentError(f"PDF file is completely empty (0 bytes): {file_path}")

    try:
        loader = PyPDFLoader(str(path))
        raw_documents = loader.load()
    except pypdf.errors.FileNotDecryptedError as e:
        raise PDFEncryptedError(f"PDF is encrypted or password-protected: {file_path}") from e
    except (pypdf.errors.PdfReadError, Exception) as e:
        err_name = type(e).__name__.lower()
        err_msg = str(e).lower()
        if (
            "encrypted" in err_name
            or "decrypt" in err_name
            or "encrypted" in err_msg
            or "password" in err_msg
            or "not decrypted" in err_msg
            or "decrypt" in err_msg
        ):
            raise PDFEncryptedError(f"PDF is encrypted or password-protected: {file_path}") from e
        raise

    # Handle PDFs with 0 pages or whitespace-only content
    non_empty_docs = [doc for doc in raw_documents if doc.page_content and doc.page_content.strip()]
    if not non_empty_docs:
        raise EmptyDocumentError(f"PDF file contains no readable text content: {file_path}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(non_empty_docs)

    # Ensure metadata contains clean source and 1-based page numbers for display
    for chunk in chunks:
        page = chunk.metadata.get("page", 0)
        chunk.metadata["page_number"] = page + 1
        chunk.metadata["source_name"] = path.name

    return chunks


def load_and_chunk_text(
    file_path: Union[str, Path],
    chunk_size: int = 750,
    chunk_overlap: int = 100
) -> List[Document]:
    """
    Load a text/markdown file and split it into chunks.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.stat().st_size == 0:
        raise EmptyDocumentError(f"Text file is completely empty: {file_path}")

    loader = TextLoader(str(path), encoding="utf-8")
    raw_documents = loader.load()

    non_empty_docs = [doc for doc in raw_documents if doc.page_content and doc.page_content.strip()]
    if not non_empty_docs:
        raise EmptyDocumentError(f"Text file contains no readable content: {file_path}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(non_empty_docs)
    for chunk in chunks:
        chunk.metadata["page_number"] = 1
        chunk.metadata["source_name"] = path.name

    return chunks


def load_document_chunks(
    file_path: Union[str, Path],
    chunk_size: int = 750,
    chunk_overlap: int = 100
) -> List[Document]:
    """
    Generic loader dispatcher for PDF and text documents.
    """
    path = Path(file_path)
    if path.suffix.lower() == ".pdf":
        return load_and_chunk_pdf(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return load_and_chunk_text(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
