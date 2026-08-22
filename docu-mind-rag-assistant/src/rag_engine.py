"""
RAG Engine module utilizing LangChain, Groq API (llama-3.1-8b-instant), and FAISS retrieval.
"""
import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

FALLBACK_MESSAGE = "The requested information is not found in the uploaded document."
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-20b")

RAG_PROMPT_TEMPLATE = """You are DocuMind, an accurate and helpful document analysis assistant.
Your task is to answer the user's question STRICTLY using the provided context chunks below.

Rules:
1. Base your answer ONLY on the provided context. Do NOT extrapolate, speculate, or bring in external knowledge.
2. If the context does NOT contain enough information to answer the question, respond EXACTLY with:
"{fallback_message}"
3. Maintain factual accuracy and cite specific details (e.g. sections, figures, metrics) when available in the context.

Context:
{context}

Question:
{question}

Answer:"""


def sanitize_api_key(api_key: Optional[str] = None) -> str:
    """
    Sanitize and validate Groq API key string.
    Strips whitespace and surrounding quotes, and checks format prefix.
    """
    raw_key = api_key if api_key is not None else os.getenv("GROQ_API_KEY", "")
    clean_key = str(raw_key).strip().strip('"').strip("'")
    
    if not clean_key or clean_key in ["your_api_key_here", "your_groq_api_key_here", "your_key_here"]:
        raise ValueError(
            "GROQ_API_KEY is not configured. Please provide a valid API key in .env or the UI."
        )
    if not clean_key.startswith("gsk_"):
        raise ValueError(
            f"Invalid Groq API key format (received '{clean_key[:6]}...'). "
            "Groq API keys must begin with 'gsk_'. Generate a key at https://console.groq.com/keys."
        )
    return clean_key


def get_llm(
    model: str = DEFAULT_GROQ_MODEL,
    temperature: float = 0.0,
    api_key: Optional[str] = None
) -> ChatGroq:
    """
    Initialize and return ChatGroq instance with sanitized and validated API key.
    """
    clean_key = sanitize_api_key(api_key)
    return ChatGroq(
        model=model,
        temperature=temperature,
        groq_api_key=clean_key
    )


def format_context_docs(docs: List[Document]) -> str:
    """
    Format list of document chunks into structured context text with metadata citations.
    """
    formatted_chunks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_name", doc.metadata.get("source", "Document"))
        page = doc.metadata.get("page_number", doc.metadata.get("page", "?"))
        header = f"[Chunk {i} | Source: {source} | Page: {page}]"
        formatted_chunks.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n".join(formatted_chunks)


def build_rag_prompt() -> ChatPromptTemplate:
    """
    Create the bound RAG prompt template.
    """
    return ChatPromptTemplate.from_template(
        RAG_PROMPT_TEMPLATE.format(
            fallback_message=FALLBACK_MESSAGE,
            context="{context}",
            question="{question}"
        )
    )


def build_rag_chain(
    vector_store: FAISS,
    model: str = DEFAULT_GROQ_MODEL,
    temperature: float = 0.0,
    top_k: int = 4,
    api_key: Optional[str] = None
):
    """
    Build a retrieval QA chain that connects FAISS retriever with ChatGroq.
    """
    llm = get_llm(model=model, temperature=temperature, api_key=api_key)
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    prompt = build_rag_prompt()

    chain = (
        {
            "context": retriever | format_context_docs,
            "question": lambda x: x
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def query_rag(
    vector_store: FAISS,
    question: str,
    model: str = DEFAULT_GROQ_MODEL,
    temperature: float = 0.0,
    top_k: int = 4,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute a RAG query and return answer alongside retrieved context chunks.

    Returns:
        Dict with 'answer', 'source_documents', and 'formatted_context'.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(question)
    
    if not docs:
        return {
            "answer": FALLBACK_MESSAGE,
            "source_documents": [],
            "formatted_context": ""
        }

    formatted_context = format_context_docs(docs)
    prompt = build_rag_prompt()
    llm = get_llm(model=model, temperature=temperature, api_key=api_key)

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": formatted_context,
        "question": question
    })

    return {
        "answer": answer.strip(),
        "source_documents": docs,
        "formatted_context": formatted_context
    }
