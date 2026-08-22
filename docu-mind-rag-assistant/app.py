"""
DocuMind: Document Summarizer & QA Assistant
Production Streamlit frontend connecting to modular RAG & Summarization engines.
"""
import os
import hashlib
import tempfile
from pathlib import Path
from typing import Optional
import streamlit as st
from dotenv import load_dotenv

# Load local environment variables
load_dotenv()

# Import backend modules
from src.document_loader import (
    load_and_chunk_pdf,
    EmptyDocumentError,
    PDFEncryptedError
)
from src.vector_db import build_vector_store
from src.rag_engine import query_rag, FALLBACK_MESSAGE, DEFAULT_GROQ_MODEL
from src.summarizer import summarize_document

# ==============================================================================
# Page Configuration & Premium Aesthetics
# ==============================================================================
st.set_page_config(
    page_title="DocuMind: Document Summarizer & QA Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    /* Main container and font */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }
    
    /* Header Banner */
    .documind-header {
        background: linear-gradient(135deg, #1e1e38 0%, #2d1b4e 50%, #1e1e38 100%);
        border: 1px solid rgba(147, 51, 234, 0.3);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    .documind-header h1 {
        color: #ffffff;
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
        padding: 0;
    }
    .documind-header p {
        color: #c4b5fd;
        font-size: 1.05rem;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Citation Box */
    .citation-card {
        background: rgba(99, 102, 241, 0.05);
        border-left: 4px solid #6366f1;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 0.92rem;
    }
    .citation-meta {
        font-weight: 600;
        color: #4f46e5;
        margin-bottom: 4px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.05rem;
        font-weight: 600;
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
    }
    
    /* Metric pill */
    .status-pill {
        display: inline-block;
        background: rgba(34, 197, 94, 0.15);
        color: #15803d;
        border: 1px solid rgba(34, 197, 94, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# Session State Initialization
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "raw_chunks" not in st.session_state:
    st.session_state.raw_chunks = []
if "active_file_hash" not in st.session_state:
    st.session_state.active_file_hash = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "cached_summaries" not in st.session_state:
    st.session_state.cached_summaries = {}

# ==============================================================================
# Helper Functions
# ==============================================================================
def compute_file_hash(file_bytes: bytes) -> str:
    """Compute SHA256 checksum for uploaded file content."""
    return hashlib.sha256(file_bytes).hexdigest()

def reset_state():
    """Clear cached indices, documents, and conversation history."""
    st.session_state.messages = []
    st.session_state.vector_store = None
    st.session_state.raw_chunks = []
    st.session_state.active_file_hash = None
    st.session_state.uploaded_file_name = None
    st.session_state.cached_summaries = {}

# ==============================================================================
# Sidebar Configuration & Ingestion
# ==============================================================================
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    
    # 1. Groq API Key Input
    raw_env_key = os.getenv("GROQ_API_KEY", "")
    default_key_val = raw_env_key.strip().strip('"').strip("'") if raw_env_key not in ["your_api_key_here", "your_groq_api_key_here", ""] else ""
    
    api_key_input = st.text_input(
        "Groq API Key",
        value=default_key_val,
        type="password",
        placeholder="gsk_...",
        help="Provide your Groq Cloud API key. If omitted, defaults to the .env configuration."
    )
    
    # Sanitize and strip whitespace/quotes
    candidate_key = api_key_input.strip().strip('"').strip("'") if api_key_input.strip() else default_key_val
    effective_api_key = candidate_key if candidate_key and candidate_key.startswith("gsk_") else None
    
    if candidate_key and not candidate_key.startswith("gsk_"):
        st.warning("⚠️ Groq API key format invalid. It should start with `gsk_`.")

    st.divider()

    # 2. PDF File Uploader (Up to 25 MB)
    st.markdown("### 📄 Document Upload")
    uploaded_pdf = st.file_uploader(
        "Upload PDF Document",
        type=["pdf"],
        help="Upload standard research papers, business reports, or technical specifications (Max 25 MB)."
    )

    # 3. Dynamic Sliders
    st.markdown("### 🎛️ Pipeline Parameters")
    selected_model = st.selectbox(
        "Groq LLM Model",
        options=["openai/gpt-oss-20b", "groq/compound-mini", "openai/gpt-oss-120b", "qwen/qwen3.6-27b", "allam-2-7b"],
        index=0,
        help="Select the Groq LPU accelerated inference model."
    )
    chunk_size = st.slider("Chunk Size (characters)", min_value=300, max_value=2000, value=750, step=50, help="Target character size for each segmented passage.")
    chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=0, max_value=300, value=100, step=20, help="Character overlap between consecutive chunks to maintain context.")
    top_k_chunks = st.slider("Top-K Chunks", min_value=1, max_value=8, value=4, step=1, help="Number of most similar context chunks to retrieve.")

    st.divider()

    # 4. Status Indicator
    st.markdown("### 📊 Index Status")
    if st.session_state.vector_store is not None:
        st.markdown(f"**Indexed Document:** `{st.session_state.uploaded_file_name}`")
        st.markdown(f"**Total Chunks:** `{len(st.session_state.raw_chunks)}`")
        st.markdown('<span class="status-pill">● FAISS Vector Store Active</span>', unsafe_allow_html=True)
    else:
        st.info("No document indexed. Upload a PDF to begin.")

    st.divider()

    # 5. Reset / Clear Button
    if st.button("🗑️ Reset / Clear Index", use_container_width=True):
        reset_state()
        st.rerun()

# ==============================================================================
# Automatic Document Ingestion & Vector Indexing
# ==============================================================================
if uploaded_pdf is not None:
    pdf_bytes = uploaded_pdf.getvalue()
    current_hash = compute_file_hash(pdf_bytes)

    # Ingest and index only once per file change
    if st.session_state.active_file_hash != current_hash:
        with st.spinner("⏳ Parsing PDF, generating MiniLM embeddings, and building FAISS index..."):
            try:
                # Write to temp file for PyPDFLoader
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(pdf_bytes)
                    tmp_path = tmp.name

                chunks = load_and_chunk_pdf(
                    file_path=tmp_path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )

                # Set clean source name
                for chunk in chunks:
                    chunk.metadata["source_name"] = uploaded_pdf.name

                # Build FAISS vector database
                vector_store = build_vector_store(chunks)

                # Cache in session state
                st.session_state.vector_store = vector_store
                st.session_state.raw_chunks = chunks
                st.session_state.active_file_hash = current_hash
                st.session_state.uploaded_file_name = uploaded_pdf.name
                st.session_state.messages = []
                st.session_state.cached_summaries = {}

                # Clean temporary file
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

                st.sidebar.success("✅ Document indexed successfully!")
                st.rerun()

            except EmptyDocumentError:
                st.error("❌ The uploaded PDF appears to be empty or contains no readable text.")
            except PDFEncryptedError:
                st.error("🔒 The uploaded PDF is password-protected. Please provide an unencrypted version.")
            except Exception as e:
                st.error(f"⚠️ Ingestion error: {str(e)}")

# ==============================================================================
# Header Banner
# ==============================================================================
st.markdown("""
<div class="documind-header">
    <h1>📄 DocuMind: Document Summarizer & QA Assistant</h1>
    <p>High-Performance Document Intelligence & Retrieval-Augmented Generation powered by <strong>Groq Cloud</strong> (<code>llama-3.1-8b-instant</code>) and <strong>FAISS</strong>.</p>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# Tabs Navigation
# ==============================================================================
tab_summarize, tab_qa = st.tabs(["📝 Structured Summarization", "💬 Interactive Document Q&A"])

# ------------------------------------------------------------------------------
# Tab 1: 📝 Structured Summarization
# ------------------------------------------------------------------------------
with tab_summarize:
    if st.session_state.vector_store is None or not st.session_state.raw_chunks:
        st.info("👈 Please upload a PDF document in the sidebar to generate structured summaries.")
    else:
        st.subheader("📑 Document Summarization")
        st.caption(f"Synthesize key insights from **{st.session_state.uploaded_file_name}** ({len(st.session_state.raw_chunks)} chunks).")

        col1, col2 = st.columns([3, 1])
        with col1:
            summary_style = st.selectbox(
                "Select Summary Style:",
                options=[
                    "Executive Summary",
                    "Technical Key Points",
                    "Action Items",
                    "Comprehensive"
                ],
                index=0,
                help="Choose the analytical perspective for document synthesis."
            )

        # Style to internal mode mapping
        mode_mapping = {
            "Executive Summary": "executive",
            "Technical Key Points": "technical",
            "Action Items": "action_items",
            "Comprehensive": "comprehensive"
        }
        selected_mode = mode_mapping[summary_style]

        with col2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            generate_clicked = st.button("⚡ Generate Summary", type="primary", use_container_width=True)

        if generate_clicked:
            if not effective_api_key:
                st.error("⚠️ Groq API Key required. Please enter your API key in the sidebar or configure `.env`.")
            else:
                with st.spinner(f"Synthesizing {summary_style} via Groq LPUs..."):
                    try:
                        summary_result = summarize_document(
                            documents=st.session_state.raw_chunks,
                            mode=selected_mode,
                            model=selected_model,
                            api_key=effective_api_key
                        )
                        st.session_state.cached_summaries[selected_mode] = summary_result
                    except Exception as e:
                        st.error(f"Summarization failed: {str(e)}")

        # Display cached summary if available
        if selected_mode in st.session_state.cached_summaries:
            summary_text = st.session_state.cached_summaries[selected_mode]
            st.markdown("---")
            st.markdown(summary_text)

            # Copy-friendly block and download option
            col_d1, col_d2 = st.columns([2, 1])
            with col_d1:
                with st.expander("📋 View Raw Copy-Friendly Markdown"):
                    st.code(summary_text, language="markdown")
            with col_d2:
                st.download_button(
                    label="📥 Download Summary (.md)",
                    data=summary_text,
                    file_name=f"{st.session_state.uploaded_file_name}_{selected_mode}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

# ------------------------------------------------------------------------------
# Tab 2: 💬 Interactive Document Q&A
# ------------------------------------------------------------------------------
with tab_qa:
    if st.session_state.vector_store is None:
        st.info("👈 Please upload a PDF document in the sidebar to start asking questions.")
    else:
        st.caption(f"Currently chatting with: **{st.session_state.uploaded_file_name}**")

        # Render conversation history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # If assistant message has citations, display expandable section
                if message.get("citations"):
                    with st.expander("🔍 Retrieved Source Chunks & Page Numbers", expanded=False):
                        for idx, doc in enumerate(message["citations"], 1):
                            src_name = doc.metadata.get("source_name", st.session_state.uploaded_file_name)
                            page_num = doc.metadata.get("page_number", doc.metadata.get("page", 1))
                            st.markdown(f"**Citation #{idx}** — Source: `{src_name}` (Page {page_num})")
                            st.markdown(f"> *{doc.page_content.strip()}*")
                            st.divider()

        # Chat Input
        if query := st.chat_input("Ask any question about the uploaded document..."):
            if not effective_api_key:
                st.error("⚠️ Groq API Key required. Please enter your API key in the sidebar or configure `.env`.")
            else:
                # Add user message to state
                st.session_state.messages.append({"role": "user", "content": query})
                with st.chat_message("user"):
                    st.markdown(query)

                # Process query via RAG engine
                with st.chat_message("assistant"):
                    with st.spinner("Retrieving relevant passages and formulating answer..."):
                        try:
                            result = query_rag(
                                vector_store=st.session_state.vector_store,
                                question=query,
                                model=selected_model,
                                temperature=0.0,
                                top_k=top_k_chunks,
                                api_key=effective_api_key
                            )

                            answer_text = result["answer"]
                            source_docs = result["source_documents"]

                            st.markdown(answer_text)

                            # Display expandable citation block
                            citations_to_store = []
                            if source_docs and answer_text != FALLBACK_MESSAGE:
                                citations_to_store = source_docs
                                with st.expander("🔍 Retrieved Source Chunks & Page Numbers", expanded=False):
                                    for idx, doc in enumerate(source_docs, 1):
                                        src_name = doc.metadata.get("source_name", st.session_state.uploaded_file_name)
                                        page_num = doc.metadata.get("page_number", doc.metadata.get("page", 1))
                                        st.markdown(f"**Citation #{idx}** — Source: `{src_name}` (Page {page_num})")
                                        st.markdown(f"> *{doc.page_content.strip()}*")
                                        st.divider()

                            # Append assistant message to state
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer_text,
                                "citations": citations_to_store
                            })

                        except Exception as e:
                            st.error(f"Error querying RAG pipeline: {str(e)}")
