import unittest
from unittest.mock import MagicMock, patch

from brew_oracle.scripts.create_collections import main


class TestCreateCollections(unittest.TestCase):
    @patch("brew_oracle.scripts.create_collections.Settings")
    @patch("brew_oracle.scripts.create_collections.QdrantClient")
    def test_main_create_collection(self, mock_qdrant_client, mock_settings):
        """Test that a new collection is created when it doesn't exist."""
        mock_settings.return_value.QDRANT_COLLECTION = "test_collection"
        mock_settings.return_value.QDRANT_URL = "http://localhost:6333"
        mock_settings.return_value.EMBEDDER_DIM = 384
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client
        mock_client.collection_exists.return_value = False

        result = main()

        mock_client.create_collection.assert_called_once()
        self.assertIn("criada", result)

    @patch("brew_oracle.scripts.create_collections.Settings")
    @patch("brew_oracle.scripts.create_collections.QdrantClient")
    def test_main_collection_exists(self, mock_qdrant_client, mock_settings):
        """Test that the collection is not created when it already exists."""
        mock_settings.return_value.QDRANT_COLLECTION = "test_collection"
        mock_settings.return_value.QDRANT_URL = "http://localhost:6333"
        mock_settings.return_value.EMBEDDER_DIM = 384
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client
        mock_client.collection_exists.return_value = True

        result = main()

        mock_client.create_collection.assert_not_called()
        self.assertIn("já existe", result)

    @patch("brew_oracle.scripts.create_collections.Settings")
    @patch("brew_oracle.scripts.create_collections.QdrantClient")
    def test_main_force_recreate(self, mock_qdrant_client, mock_settings):
        """Test that the collection is recreated when force_recreate is True."""
        mock_settings.return_value.QDRANT_COLLECTION = "test_collection"
        mock_settings.return_value.QDRANT_URL = "http://localhost:6333"
        mock_settings.return_value.EMBEDDER_DIM = 384
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client
        mock_client.collection_exists.side_effect = [True, False]

        result = main(force_recreate=True, hybrid=True)

        mock_client.delete_collection.assert_called_once_with("test_collection")
        mock_client.create_collection.assert_called_once()
        self.assertIn("criada", result)

        self.assertIn("criada", result)

    @patch("brew_oracle.scripts.create_collections.Settings")
    @patch("brew_oracle.scripts.create_collections.QdrantClient")
    def test_main_hybrid_collection(self, mock_qdrant_client, mock_settings):
        """Test that a hybrid collection is created with correct configs."""
        mock_settings.return_value.QDRANT_COLLECTION = "test_hybrid_collection"
        mock_settings.return_value.QDRANT_URL = "http://localhost:6333"
        mock_settings.return_value.EMBEDDER_DIM = 384
        mock_settings.return_value.DENSE_VECTOR_NAME = "dense_test"
        mock_settings.return_value.SPARSE_VECTOR_NAME = "sparse_test"
        mock_settings.return_value.SPARSE_MODEL_ID = "Qdrant/bm25"

        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client
        mock_client.collection_exists.return_value = False

        result = main(hybrid=True)

        mock_client.create_collection.assert_called_once()
        call_args = mock_client.create_collection.call_args[1]
        self.assertIn("criada", result)
        self.assertEqual(call_args["collection_name"], "test_hybrid_collection")
        self.assertIsInstance(call_args["vectors_config"], dict)
        self.assertIn("dense_test", call_args["vectors_config"])
        self.assertIsInstance(call_args["sparse_vectors_config"], dict)
        self.assertIn("sparse_test", call_args["sparse_vectors_config"])

    @patch("brew_oracle.scripts.create_collections.Settings")
    @patch("brew_oracle.scripts.create_collections.QdrantClient")
    def test_main_custom_collection_name(self, mock_qdrant_client, mock_settings):
        """Test that a collection is created with a custom name."""
        mock_settings.return_value.QDRANT_COLLECTION = "default_collection"
        mock_settings.return_value.QDRANT_URL = "http://localhost:6333"
        mock_settings.return_value.EMBEDDER_DIM = 384

        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client
        mock_client.collection_exists.return_value = False

        custom_name = "my_custom_collection"
        result = main(collection_name=custom_name)

        mock_client.create_collection.assert_called_once()
        call_args = mock_client.create_collection.call_args[1]
        self.assertIn("criada", result)
        self.assertEqual(call_args["collection_name"], custom_name)


if __name__ == "__main__":
    unittest.main()
