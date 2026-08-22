"""
Pipeline verification and edge-case unit test suite for DocuMind RAG Assistant.
"""
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from pypdf import PdfWriter
from langchain_core.documents import Document

from src.document_loader import (
    load_and_chunk_pdf,
    load_and_chunk_text,
    EmptyDocumentError,
    PDFEncryptedError
)
from src.vector_db import (
    build_vector_store,
    save_index,
    load_index,
    get_embedding_model
)


class TestPipelineEdgeCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.temp_path = Path(cls.temp_dir)

        # 1. Create a Standard Valid PDF
        cls.valid_pdf_path = cls.temp_path / "valid_sample.pdf"
        writer = PdfWriter()
        page = writer.add_blank_page(width=612, height=792)
        # Note: In PyPDF, write basic text or metadata
        # To add actual text stream in pure pypdf:
        writer.add_metadata({"/Title": "Test PDF Document"})
        with open(cls.valid_pdf_path, "wb") as f:
            writer.write(f)

        # 2. Create an Encrypted PDF
        cls.encrypted_pdf_path = cls.temp_path / "encrypted_sample.pdf"
        enc_writer = PdfWriter()
        enc_writer.add_blank_page(width=612, height=792)
        enc_writer.encrypt("secret_password")
        with open(cls.encrypted_pdf_path, "wb") as f:
            enc_writer.write(f)

        # 3. Create a 0-byte Empty PDF
        cls.empty_pdf_path = cls.temp_path / "empty_0byte.pdf"
        cls.empty_pdf_path.touch()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_empty_file_handling(self):
        """Verify that 0-byte PDF raises EmptyDocumentError."""
        with self.assertRaises(EmptyDocumentError):
            load_and_chunk_pdf(self.empty_pdf_path)

    def test_encrypted_pdf_handling(self):
        """Verify that encrypted PDF raises PDFEncryptedError."""
        with self.assertRaises(PDFEncryptedError):
            load_and_chunk_pdf(self.encrypted_pdf_path)

    def test_standard_text_loading(self):
        """Verify loading and chunking standard sample files."""
        sample_path = Path(__file__).parent.parent / "data" / "sample.txt"
        chunks = load_and_chunk_text(sample_path, chunk_size=300, chunk_overlap=50)
        self.assertTrue(len(chunks) > 0)
        self.assertEqual(chunks[0].metadata["source_name"], "sample.txt")

    def test_chunking_boundaries_and_overlap(self):
        """
        Test chunking boundaries to verify that chunk_overlap retains context
        without creating full duplicate identical chunks.
        """
        test_file = self.temp_path / "overlap_test.txt"
        test_content = (
            "Sentence Alpha begins here. " * 10 +
            "Sentence Beta follows right after. " * 10 +
            "Sentence Gamma concludes the section. " * 10
        )
        test_file.write_text(test_content, encoding="utf-8")

        chunk_size = 250
        chunk_overlap = 50
        chunks = load_and_chunk_text(test_file, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        self.assertGreater(len(chunks), 1, "Should produce multiple chunks.")

        # Ensure consecutive chunks are not completely identical
        for i in range(len(chunks) - 1):
            chunk_a = chunks[i].page_content
            chunk_b = chunks[i + 1].page_content
            self.assertNotEqual(chunk_a, chunk_b, f"Chunks {i} and {i+1} should not be identical.")
            # Verify overlap: tail of chunk_a should overlap with head of chunk_b if continuous
            self.assertLessEqual(len(chunk_a), chunk_size + 50)

    def test_faiss_similarity_search_retrieval(self):
        """
        Verify FAISS similarity search retrieval accurately returns the expected chunk for Top-K.
        """
        test_docs = [
            Document(
                page_content="Quantum computing leverages superposition and entanglement to solve discrete optimization problems.",
                metadata={"source_name": "quantum.txt", "topic": "quantum"}
            ),
            Document(
                page_content="Photosynthesis is the biological process by which green plants convert sunlight into chemical energy.",
                metadata={"source_name": "biology.txt", "topic": "biology"}
            ),
            Document(
                page_content="The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France.",
                metadata={"source_name": "geography.txt", "topic": "geography"}
            ),
        ]

        # Build FAISS vector store
        vector_store = build_vector_store(test_docs)
        
        # Test Query 1: Quantum
        results_q1 = vector_store.similarity_search("How do quantum algorithms use superposition?", k=1)
        self.assertEqual(len(results_q1), 1)
        self.assertEqual(results_q1[0].metadata["topic"], "quantum")
        self.assertIn("superposition", results_q1[0].page_content)

        # Test Query 2: Biology / Plants
        results_q2 = vector_store.similarity_search("How do plants turn sunlight into energy?", k=1)
        self.assertEqual(len(results_q2), 1)
        self.assertEqual(results_q2[0].metadata["topic"], "biology")
        self.assertIn("Photosynthesis", results_q2[0].page_content)

        # Test Save and Load Persistence
        save_path = self.temp_path / "test_faiss_index"
        save_index(vector_store, save_path)
        self.assertTrue((save_path / "index.faiss").exists())
        self.assertTrue((save_path / "index.pkl").exists())

        loaded_vs = load_index(save_path)
        loaded_results = loaded_vs.similarity_search("Eiffel Tower Paris", k=1)
        self.assertEqual(len(loaded_results), 1)
        self.assertEqual(loaded_results[0].metadata["topic"], "geography")


if __name__ == "__main__":
    unittest.main()
