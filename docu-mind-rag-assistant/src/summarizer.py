"""
Document summarization module with structured summary modes using Groq LLM.
"""
from typing import List, Optional, Dict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.rag_engine import get_llm, DEFAULT_GROQ_MODEL


PROMPT_EXECUTIVE_SUMMARY = """You are an expert executive analyst.
Provide a high-level, concise Executive Summary of the following document.
Structure your output with:
1. **Core Purpose / Objective**
2. **Key Takeaways** (3-5 high-impact bullet points)
3. **Strategic Implications / Conclusion**

Document Content:
{text}

Executive Summary:"""


PROMPT_TECHNICAL_POINTS = """You are a senior technical specialist.
Extract and organize the Technical Key Points from the following document.
Structure your output with:
1. **System Architecture / Methodology**
2. **Key Technical Specifications & Parameters**
3. **Technical Findings & Data Points**
4. **Constraints, Limitations, or Assumptions**

Document Content:
{text}

Technical Key Points:"""


PROMPT_ACTION_ITEMS = """You are an operations and project manager.
Identify and extract all Action Items, Next Steps, and Recommendations from the following document.
Structure your output with:
1. **Immediate Next Steps (High Priority)**
2. **Action Items & Ownership/Tasks**
3. **Recommendations & Future Work**

Document Content:
{text}

Action Items & Recommendations:"""


PROMPT_COMPREHENSIVE = """You are an expert research analyst.
Provide a structured, comprehensive summary of the following document incorporating:
- Executive Summary
- Technical Key Points
- Action Items & Next Steps

Document Content:
{text}

Comprehensive Summary:"""


SUMMARY_PROMPTS: Dict[str, str] = {
    "executive": PROMPT_EXECUTIVE_SUMMARY,
    "technical": PROMPT_TECHNICAL_POINTS,
    "action_items": PROMPT_ACTION_ITEMS,
    "comprehensive": PROMPT_COMPREHENSIVE,
}


def _combine_document_text(documents: List[Document], max_chars: int = 16000) -> str:
    """
    Concatenate document pages up to a character limit to fit Groq context windows cleanly.
    """
    full_text = "\n\n".join(doc.page_content.strip() for doc in documents if doc.page_content.strip())
    if len(full_text) > max_chars:
        return full_text[:max_chars] + "\n\n... [Text truncated for summarization context limit]"
    return full_text


def summarize_executive(
    documents: List[Document],
    model: str = DEFAULT_GROQ_MODEL,
    api_key: Optional[str] = None
) -> str:
    """Generate an Executive Summary for the documents."""
    return summarize_document(documents, mode="executive", model=model, api_key=api_key)


def summarize_technical_key_points(
    documents: List[Document],
    model: str = DEFAULT_GROQ_MODEL,
    api_key: Optional[str] = None
) -> str:
    """Generate Technical Key Points from the documents."""
    return summarize_document(documents, mode="technical", model=model, api_key=api_key)


def summarize_action_items(
    documents: List[Document],
    model: str = DEFAULT_GROQ_MODEL,
    api_key: Optional[str] = None
) -> str:
    """Generate Action Items and Next Steps from the documents."""
    return summarize_document(documents, mode="action_items", model=model, api_key=api_key)


def summarize_document(
    documents: List[Document],
    mode: str = "executive",
    model: str = DEFAULT_GROQ_MODEL,
    api_key: Optional[str] = None
) -> str:
    """
    Generate a structured summary based on the requested mode:
    - 'executive': Executive Summary
    - 'technical': Technical Key Points
    - 'action_items': Action Items and Next Steps
    - 'comprehensive': Full structured analysis

    Args:
        documents: List of Document objects or chunks.
        mode: Summarization mode ('executive', 'technical', 'action_items', 'comprehensive').
        model: Groq model name.
        api_key: Groq API key override.

    Returns:
        Structured summary markdown string.
    """
    if not documents:
        raise ValueError("Cannot summarize empty document list.")

    mode_key = mode.lower().strip()
    if mode_key not in SUMMARY_PROMPTS:
        mode_key = "executive"

    prompt_template = SUMMARY_PROMPTS[mode_key]
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = get_llm(model=model, temperature=0.1, api_key=api_key)

    chain = prompt | llm | StrOutputParser()
    text = _combine_document_text(documents)
    
    return chain.invoke({"text": text})
