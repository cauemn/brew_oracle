import unittest
from unittest.mock import MagicMock, patch

from brew_oracle.orchestrator.brewing_orchestrator import BrewingOrchestrator


class TestBrewingOrchestrator(unittest.TestCase):
    @patch("brew_oracle.orchestrator.brewing_orchestrator.build_pdf_kb")
    @patch("brew_oracle.orchestrator.brewing_orchestrator.build_recipe_kb")
    @patch("brew_oracle.orchestrator.brewing_orchestrator.Gemini")
    def test_initialization(self, mock_gemini, mock_build_recipe_kb, mock_build_pdf_kb):
        """Test that the BrewingOrchestrator is initialized correctly."""
        mock_pdf_kb = MagicMock()
        mock_build_pdf_kb.return_value = mock_pdf_kb
        mock_recipe_kb = MagicMock()
        mock_build_recipe_kb.return_value = mock_recipe_kb
        mock_model = MagicMock()
        mock_gemini.return_value = mock_model

        agent = BrewingOrchestrator()

        self.assertEqual(agent.pdf_kb, mock_pdf_kb)
        self.assertEqual(agent.recipe_kb, mock_recipe_kb)
        self.assertEqual(agent.model, mock_model)
        self.assertFalse(agent.rerank)
        mock_build_pdf_kb.assert_called_once_with(hybrid=False)
        mock_build_recipe_kb.assert_called_once_with(hybrid=False)
        mock_gemini.assert_called_once()

    @patch("brew_oracle.orchestrator.brewing_orchestrator.build_pdf_kb")
    @patch("brew_oracle.orchestrator.brewing_orchestrator.build_recipe_kb")
    @patch("brew_oracle.orchestrator.brewing_orchestrator.Gemini")
    def test_ask_with_refs(self, mock_gemini, mock_build_recipe_kb, mock_build_pdf_kb):
        """Test that the ask_with_refs method returns a tuple with a string and a list."""
        mock_pdf_kb = MagicMock()
        mock_build_pdf_kb.return_value = mock_pdf_kb
        mock_recipe_kb = MagicMock()
        mock_build_recipe_kb.return_value = mock_recipe_kb
        mock_model = MagicMock()
        mock_gemini.return_value = mock_model

        agent = BrewingOrchestrator()
        agent.agent.run = MagicMock(
            return_value=MagicMock(content="Test answer", references=["ref1", "ref2"])
        )

        text, refs = agent.ask_with_refs("Test question")

        self.assertEqual(text, "Test answer")
        self.assertEqual(refs, ["ref1", "ref2"])
        agent.agent.run.assert_called_once_with("Test question")

    @patch("brew_oracle.orchestrator.brewing_orchestrator.build_pdf_kb")
    @patch("brew_oracle.orchestrator.brewing_orchestrator.build_recipe_kb")
    @patch("brew_oracle.orchestrator.brewing_orchestrator.Gemini")
    @patch("sentence_transformers.CrossEncoder")
    def test_reranking(
        self, mock_cross_encoder, mock_gemini, mock_build_recipe_kb, mock_build_pdf_kb
    ):
        """Test that the reranking functionality is applied when rerank is True."""
        mock_pdf_kb = MagicMock()
        mock_build_pdf_kb.return_value = mock_pdf_kb
        mock_recipe_kb = MagicMock()
        mock_build_recipe_kb.return_value = mock_recipe_kb
        mock_model = MagicMock()
        mock_gemini.return_value = mock_model
        mock_encoder = MagicMock()
        mock_cross_encoder.return_value = mock_encoder

        agent = BrewingOrchestrator(rerank=True)

        self.assertTrue(agent.rerank)
        mock_cross_encoder.assert_called_once()
        self.assertIsNotNone(agent.agent.search_knowledge)

    @patch("brew_oracle.orchestrator.brewing_orchestrator.build_pdf_kb")
    @patch("brew_oracle.orchestrator.brewing_orchestrator.build_recipe_kb")
    @patch("brew_oracle.orchestrator.brewing_orchestrator.Gemini")
    def test_combined_search_calls_both_kbs(
        self, mock_gemini, mock_build_recipe_kb, mock_build_pdf_kb
    ):
        mock_pdf_kb = MagicMock()
        mock_recipe_kb = MagicMock()
        mock_build_pdf_kb.return_value = mock_pdf_kb
        mock_build_recipe_kb.return_value = mock_recipe_kb

        agent = BrewingOrchestrator()
        agent.agent.search_knowledge("test query")

        mock_pdf_kb.search.assert_called_once_with("test query")
        mock_recipe_kb.search.assert_called_once_with("test query")

    @patch("brew_oracle.orchestrator.brewing_orchestrator.build_pdf_kb")
    @patch("brew_oracle.orchestrator.brewing_orchestrator.build_recipe_kb")
    @patch("brew_oracle.orchestrator.brewing_orchestrator.Gemini")
    def test_combined_search_combines_results(
        self, mock_gemini, mock_build_recipe_kb, mock_build_pdf_kb
    ):
        mock_pdf_kb = MagicMock()
        mock_recipe_kb = MagicMock()
        mock_build_pdf_kb.return_value = mock_pdf_kb
        mock_build_recipe_kb.return_value = mock_recipe_kb

        mock_pdf_kb.search.return_value = [MagicMock(content="pdf_doc1")]
        mock_recipe_kb.search.return_value = [MagicMock(content="recipe_doc1")]

        agent = BrewingOrchestrator()
        combined_docs = agent.agent.search_knowledge("test query")

        self.assertEqual(len(combined_docs), 2)
        self.assertEqual(combined_docs[0].content, "pdf_doc1")
        self.assertEqual(combined_docs[1].content, "recipe_doc1")

    @patch("brew_oracle.orchestrator.brewing_orchestrator.build_pdf_kb")
    @patch("brew_oracle.orchestrator.brewing_orchestrator.build_recipe_kb")
    @patch("brew_oracle.orchestrator.brewing_orchestrator.Gemini")
    @patch("sentence_transformers.CrossEncoder")
    def test_combined_search_reranks_results(
        self, mock_cross_encoder, mock_gemini, mock_build_recipe_kb, mock_build_pdf_kb
    ):
        mock_pdf_kb = MagicMock()
        mock_recipe_kb = MagicMock()
        mock_build_pdf_kb.return_value = mock_pdf_kb
        mock_build_recipe_kb.return_value = mock_recipe_kb

        mock_pdf_kb.search.return_value = [MagicMock(content="pdf_doc1")]
        mock_recipe_kb.search.return_value = [MagicMock(content="recipe_doc1")]

        mock_encoder = MagicMock()
        mock_cross_encoder.return_value = mock_encoder
        mock_encoder.predict.return_value = [0.9, 0.1]  # Simulate reranking scores

        agent = BrewingOrchestrator(rerank=True)
        reranked_docs = agent.agent.search_knowledge("test query")

        self.assertEqual(len(reranked_docs), 2)
        self.assertEqual(reranked_docs[0].content, "pdf_doc1")
        self.assertEqual(reranked_docs[1].content, "recipe_doc1")
        mock_cross_encoder.assert_called_once()
        mock_encoder.predict.assert_called_once()

    @patch("brew_oracle.orchestrator.brewing_orchestrator.build_pdf_kb")
    @patch("brew_oracle.orchestrator.brewing_orchestrator.build_recipe_kb")
    @patch("brew_oracle.orchestrator.brewing_orchestrator.Gemini")
    def test_hybrid_parameter_passed_to_kbs(
        self, mock_gemini, mock_build_recipe_kb, mock_build_pdf_kb
    ):
        BrewingOrchestrator(hybrid=True)

        mock_build_pdf_kb.assert_called_once_with(hybrid=True)
        mock_build_recipe_kb.assert_called_once_with(hybrid=True)


if __name__ == "__main__":
    unittest.main()
