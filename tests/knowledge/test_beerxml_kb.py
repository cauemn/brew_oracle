import unittest
from unittest.mock import MagicMock, patch

from agno.document import Document
from agno.vectordb.search import SearchType

from brew_oracle.knowledge.beerxml_kb import build_recipe_kb, ingest_recipes


class TestBeerXMLKnowledgeBase(unittest.TestCase):
    @patch("brew_oracle.knowledge.beerxml_kb.Settings")
    @patch("brew_oracle.knowledge.beerxml_kb.SentenceTransformerEmbedder")
    @patch("brew_oracle.knowledge.beerxml_kb.Qdrant")
    def test_build_recipe_kb(self, mock_qdrant, mock_embedder, mock_settings):
        """Test that the BeerXMLKnowledgeBase is built correctly."""
        mock_settings_instance = MagicMock()
        mock_settings_instance.EMBEDDER_ID = "fake_embedder"
        mock_settings_instance.EMBEDDER_DIM = 384
        mock_settings_instance.QDRANT_RECIPE_COLLECTION = "recipes"
        mock_settings_instance.QDRANT_URL = "http://localhost:6333"
        mock_settings_instance.DENSE_VECTOR_NAME = "dense"
        mock_settings_instance.SPARSE_VECTOR_NAME = "sparse"
        mock_settings_instance.SPARSE_MODEL_ID = "Qdrant/bm25"
        mock_settings.return_value = mock_settings_instance

        with patch("os.path.isdir", return_value=False):
            kb = build_recipe_kb()

        mock_embedder.assert_called_once_with(id="fake_embedder", dimensions=384)
        mock_qdrant.assert_called_once_with(
            collection="recipes",
            url="http://localhost:6333",
            embedder=mock_embedder.return_value,
            search_type=SearchType.vector,
            dense_vector_name="dense",
            sparse_vector_name="sparse",
            fastembed_kwargs={"model_name": "Qdrant/bm25"},
        )
        self.assertEqual(kb, mock_qdrant.return_value)

    @patch("brew_oracle.knowledge.beerxml_kb.Settings")
    @patch("brew_oracle.knowledge.beerxml_kb.SentenceTransformerEmbedder")
    @patch("brew_oracle.knowledge.beerxml_kb.Qdrant")
    def test_build_recipe_kb_hybrid(self, mock_qdrant, mock_embedder, mock_settings):
        """Test that the BeerXMLKnowledgeBase is built correctly with hybrid search."""
        mock_settings_instance = MagicMock()
        mock_settings_instance.EMBEDDER_ID = "fake_embedder"
        mock_settings_instance.EMBEDDER_DIM = 384
        mock_settings_instance.QDRANT_RECIPE_COLLECTION = "recipes_hybrid"
        mock_settings_instance.QDRANT_URL = "http://localhost:6333"
        mock_settings_instance.DENSE_VECTOR_NAME = "dense_hybrid"
        mock_settings_instance.SPARSE_VECTOR_NAME = "sparse_hybrid"
        mock_settings_instance.SPARSE_MODEL_ID = "Qdrant/bm25"
        mock_settings.return_value = mock_settings_instance

        with patch("os.path.isdir", return_value=False):
            kb = build_recipe_kb(hybrid=True)

        mock_embedder.assert_called_once_with(id="fake_embedder", dimensions=384)
        mock_qdrant.assert_called_once_with(
            collection="recipes_hybrid",
            url="http://localhost:6333",
            embedder=mock_embedder.return_value,
            search_type=SearchType.hybrid,
            dense_vector_name="dense_hybrid",
            sparse_vector_name="sparse_hybrid",
            fastembed_kwargs={"model_name": "Qdrant/bm25"},
        )
        self.assertEqual(kb, mock_qdrant.return_value)

    @patch("brew_oracle.knowledge.beerxml_kb.Settings")
    @patch("brew_oracle.knowledge.beerxml_kb.build_recipe_kb")
    @patch("brew_oracle.knowledge.beerxml_kb.Parser")
    @patch("os.listdir")
    @patch("os.path.join")
    @patch("qdrant_client.QdrantClient")
    def test_ingest_recipes_success(
        self, mock_qdrant_client, mock_join, mock_listdir, mock_parser, mock_build_kb, mock_settings
    ):
        """Test successful ingestion of recipes."""
        mock_settings_instance = MagicMock()
        mock_settings_instance.BEERXML_PATH = "/fake/recipes"
        mock_settings_instance.QDRANT_RECIPE_COLLECTION = "fake_collection"
        mock_settings.return_value = mock_settings_instance

        mock_kb = MagicMock()
        mock_build_kb.return_value = mock_kb

        mock_listdir.return_value = ["recipe1.xml"]
        mock_join.return_value = "/fake/recipes/recipe1.xml"

        mock_recipe = MagicMock()
        mock_recipe.name = "Test IPA"
        mock_recipe.brewer = "Test Brewer"
        mock_recipe.style.name = "American IPA"
        mock_recipe.og = 1.065
        mock_recipe.fg = 1.015
        mock_recipe.abv = 6.5
        mock_recipe.ibu = 70.0
        mock_recipe.srm = 6.0
        mock_recipe.color = 6.0
        mock_recipe.batch_size = 20.0
        mock_recipe.boil_size = 25.0
        mock_recipe.boil_time = 60.0
        mock_recipe.efficiency = 75.0
        mock_recipe.hops = [MagicMock(), MagicMock()]
        mock_recipe.hops[0].name = "Citra"
        mock_recipe.hops[1].name = "Mosaic"
        mock_recipe.fermentables = [MagicMock()]
        mock_recipe.fermentables[0].name = "2-row"
        mock_recipe.yeasts = [MagicMock()]
        mock_recipe.yeasts[0].name = "US-05"
        mock_recipe.miscs = []
        mock_recipe.notes = "Dry hop with Citra and Mosaic."

        # Correctly create the full_text content
        hops_list = [hop.name for hop in mock_recipe.hops]
        fermentables_list = [f.name for f in mock_recipe.fermentables]
        yeasts_list = [y.name for y in mock_recipe.yeasts]
        full_text = (
            f"{mock_recipe.name} by {mock_recipe.brewer}. Style: "
            f"{getattr(mock_recipe.style, 'name', 'N/A')}. OG: "
            f"{getattr(mock_recipe, 'og', 0.0):.3f}, FG: "
            f"{getattr(mock_recipe, 'fg', 0.0):.3f}, ABV: "
            f"{getattr(mock_recipe, 'abv', 0.0):.2f}%, IBU: "
            f"{getattr(mock_recipe, 'ibu', 0.0):.2f}. Hops: "
            f"{', '.join(hops_list)}. "
            f"Fermentables: {', '.join(fermentables_list)}. "
            f"Yeasts: {', '.join(yeasts_list)}. "
            f"Notes: {getattr(mock_recipe, 'notes', '')}"
        )
        mock_recipe.full_text = full_text

        mock_parser_instance = MagicMock()
        mock_parser_instance.parse.return_value = [mock_recipe]
        mock_parser.return_value = mock_parser_instance

        mock_qdrant_client_instance = MagicMock()
        mock_qdrant_client_instance.count.return_value.count = 1
        mock_qdrant_client.return_value = mock_qdrant_client_instance

        with patch("os.makedirs"):
            ingest_recipes()

        mock_build_kb.assert_called_once_with(hybrid=False)
        mock_listdir.assert_called_once_with("/fake/recipes")
        mock_parser_instance.parse.assert_called_once_with("/fake/recipes/recipe1.xml")

        self.assertEqual(mock_kb.upsert.call_count, 1)
        upserted_doc = mock_kb.upsert.call_args[0][0][0]
        self.assertIsInstance(upserted_doc, Document)
        self.assertEqual(upserted_doc.content, full_text)
        self.assertEqual(upserted_doc.meta_data["name"], "Test IPA")

    @patch("brew_oracle.knowledge.beerxml_kb.Settings")
    @patch("brew_oracle.knowledge.beerxml_kb.build_recipe_kb")
    @patch("brew_oracle.knowledge.beerxml_kb.Parser")
    @patch("os.listdir")
    @patch("os.path.join")
    def test_ingest_recipes_malformed_xml(
        self, mock_join, mock_listdir, mock_parser, mock_build_kb, mock_settings
    ):
        """Test that malformed XML files are handled gracefully."""
        mock_settings_instance = MagicMock()
        mock_settings_instance.BEERXML_PATH = "/fake/recipes"
        mock_settings.return_value = mock_settings_instance

        mock_kb = MagicMock()
        mock_build_kb.return_value = mock_kb

        mock_listdir.return_value = ["malformed.xml"]
        mock_join.return_value = "/fake/recipes/malformed.xml"

        mock_parser_instance = MagicMock()
        mock_parser_instance.parse.side_effect = Exception("Malformed XML")
        mock_parser.return_value = mock_parser_instance

        with patch("os.makedirs"), patch("qdrant_client.QdrantClient"):
            ingest_recipes()

        mock_kb.upsert.assert_not_called()


if __name__ == "__main__":
    unittest.main()
