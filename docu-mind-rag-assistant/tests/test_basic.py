import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.document_loader import load_and_chunk_text, load_document_chunks, load_and_chunk_pdf
from src.vector_db import build_vector_store, save_index, load_index
from src.rag_engine import (
    FALLBACK_MESSAGE,
    DEFAULT_GROQ_MODEL,
    format_context_docs,
    build_rag_prompt,
    query_rag
)
from src.summarizer import (
    summarize_document,
    summarize_executive,
    summarize_technical_key_points,
    summarize_action_items,
    _combine_document_text
)


class TestDocumentLoader(unittest.TestCase):
    def setUp(self):
        self.sample_txt = Path(__file__).parent.parent / "data" / "sample.txt"

    def test_load_and_chunk_text(self):
        chunks = load_and_chunk_text(self.sample_txt, chunk_size=200, chunk_overlap=20)
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIn("page_number", chunk.metadata)
            self.assertEqual(chunk.metadata["page_number"], 1)
            self.assertEqual(chunk.metadata["source_name"], "sample.txt")

    def test_load_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            load_and_chunk_text("nonexistent_file.txt")

    def test_pdf_extension_validation(self):
        with self.assertRaises(ValueError):
            load_and_chunk_pdf(self.sample_txt)


class TestRAGEngine(unittest.TestCase):
    def test_fallback_message_constant(self):
        self.assertEqual(
            FALLBACK_MESSAGE,
            "The requested information is not found in the uploaded document."
        )

    def test_format_context_docs(self):
        docs = [
            Document(page_content="Content 1", metadata={"source_name": "doc1.pdf", "page_number": 1}),
            Document(page_content="Content 2", metadata={"source_name": "doc2.pdf", "page_number": 2}),
        ]
        formatted = format_context_docs(docs)
        self.assertIn("[Chunk 1 | Source: doc1.pdf | Page: 1]", formatted)
        self.assertIn("Content 1", formatted)
        self.assertIn("[Chunk 2 | Source: doc2.pdf | Page: 2]", formatted)
        self.assertIn("Content 2", formatted)

    def test_rag_prompt_contains_fallback_instruction(self):
        prompt = build_rag_prompt()
        prompt_text = prompt.messages[0].prompt.template
        self.assertIn(FALLBACK_MESSAGE, prompt_text)
        self.assertIn("STRICTLY", prompt_text)


class TestSummarizer(unittest.TestCase):
    def setUp(self):
        self.sample_docs = [
            Document(page_content="Section 1: AI systems require high throughput.", metadata={}),
            Document(page_content="Section 2: Groq provides fast inference for Llama models.", metadata={})
        ]

    def test_combine_document_text(self):
        combined = _combine_document_text(self.sample_docs)
        self.assertIn("Section 1", combined)
        self.assertIn("Section 2", combined)

    @patch("src.summarizer.get_llm")
    def test_summarize_modes(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Mocked Summary Output"
        mock_get_llm.return_value = mock_llm

        # Check dispatcher for each mode
        for mode in ["executive", "technical", "action_items", "comprehensive"]:
            with patch("langchain_core.runnables.base.RunnableSequence.invoke", return_value="Mocked Output"):
                result = summarize_document(self.sample_docs, mode=mode, api_key="fake_key")
                self.assertEqual(result, "Mocked Output")


if __name__ == "__main__":
    unittest.main()
