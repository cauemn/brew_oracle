import unittest
from unittest.mock import MagicMock, patch

from brew_oracle.knowledge.pdf_kb import build_pdf_kb, ingest_pdfs


class TestPDFKnowledgeBase(unittest.TestCase):
    @patch("brew_oracle.knowledge.pdf_kb.Settings")
    @patch("brew_oracle.knowledge.pdf_kb.PDFKnowledgeBase")
    @patch("brew_oracle.knowledge.pdf_kb.SentenceTransformerEmbedder")
    @patch("os.path.isdir")
    @patch("os.makedirs")
    def test_build_pdf_kb(
        self, mock_makedirs, mock_isdir, mock_embedder, mock_pdf_kb, mock_settings
    ):
        """Test that the PDFKnowledgeBase is built correctly."""
        mock_settings_instance = MagicMock()
        mock_settings_instance.PDF_PATH = "/fake/path"
        mock_settings_instance.EMBEDDER_ID = "fake_embedder_id"
        mock_settings_instance.EMBEDDER_DIM = 384
        mock_settings_instance.QDRANT_COLLECTION = "fake_collection"
        mock_settings_instance.QDRANT_URL = "fake_url"
        mock_settings_instance.DENSE_VECTOR_NAME = "fake_dense_vector_name"
        mock_settings_instance.SPARSE_VECTOR_NAME = "fake_sparse_vector_name"
        mock_settings_instance.CHUNK_SIZE = 2000
        mock_settings_instance.CHUNK_OVERLAP = 300
        mock_settings_instance.NUM_DOCUMENTS = 5
        mock_settings.return_value = mock_settings_instance

        mock_isdir.return_value = False
        mock_pdf_kb_instance = MagicMock()
        mock_pdf_kb.return_value = mock_pdf_kb_instance

        kb = build_pdf_kb()

        self.assertIsNotNone(kb)
        mock_embedder.assert_called_once()
        mock_pdf_kb.assert_called_once()
        mock_makedirs.assert_called_once_with("/fake/path", exist_ok=True)

    @patch("brew_oracle.knowledge.pdf_kb.build_pdf_kb")
    @patch("qdrant_client.QdrantClient")
    def test_ingest_pdfs(self, mock_qdrant_client, mock_build_pdf_kb):
        """Test that the PDF ingestion process is called correctly."""
        mock_kb = MagicMock()
        mock_build_pdf_kb.return_value = mock_kb
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client
        mock_client.count.return_value.count = 10

        ingest_pdfs()

        mock_build_pdf_kb.assert_called_once_with(hybrid=False)
        mock_kb.load.assert_called_once_with(upsert=True)
        mock_qdrant_client.assert_called_once()
        mock_client.count.assert_called_once()


if __name__ == "__main__":
    unittest.main()
