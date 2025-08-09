import logging
import os
from typing import Any

from agno.document import Document
from agno.embedder.sentence_transformer import SentenceTransformerEmbedder
from agno.vectordb.qdrant import Qdrant
from agno.vectordb.search import SearchType
from pybeerxml.parser import Parser
from tqdm import tqdm

from brew_oracle.utils.config import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_recipe_kb(hybrid: bool = False) -> Qdrant:
    """Create and configure the Qdrant knowledge base for recipes.

    Parameters
    ----------
    hybrid : bool, optional
        When ``True`` also generates sparse BM25 vectors and enables
        fusion scoring between dense and sparse results, by default ``False``.

    Returns
    -------
    Qdrant
        The configured Qdrant client for recipes.
    """
    s = Settings()
    embedder_id = s.EMBEDDER_ID
    if os.path.isdir(embedder_id):
        embedder_id = os.path.abspath(embedder_id)

    embedder = SentenceTransformerEmbedder(
        id=embedder_id,
        dimensions=s.EMBEDDER_DIM,
    )
    kb = Qdrant(
        collection=s.QDRANT_RECIPE_COLLECTION,
        url=s.QDRANT_URL,
        embedder=embedder,
        search_type=SearchType.hybrid if hybrid else SearchType.vector,
        dense_vector_name=s.DENSE_VECTOR_NAME,
        sparse_vector_name=s.SPARSE_VECTOR_NAME,
        fastembed_kwargs={"model_name": getattr(s, "SPARSE_MODEL_ID", "Qdrant/bm25")},
    )
    return kb


def ingest_recipes(upsert: bool = True, hybrid: bool = False) -> None:
    """Load BeerXML files into the Qdrant collection for recipes.

    Parameters
    ----------
    upsert : bool, optional
        If ``True`` (default), existing documents are updated during
        ingestion; otherwise, only new documents are added.
    hybrid : bool, optional
        Also create sparse BM25 vectors for hybrid search, by default ``False``.
    """
    s = Settings()
    kb = build_recipe_kb(hybrid=hybrid)
    os.makedirs(s.BEERXML_PATH, exist_ok=True)

    parser = Parser()
    recipes_to_upsert = []

    for filename in tqdm(os.listdir(s.BEERXML_PATH), desc="Ingesting BeerXML files"):
        if filename.endswith(".xml"):
            filepath = os.path.join(s.BEERXML_PATH, filename)
            try:
                recipes = parser.parse(filepath)
                for recipe in recipes:
                    recipe_data: dict[str, Any] = {
                        "name": recipe.name,
                        "brewer": recipe.brewer,
                        "style": getattr(recipe.style, "name", None),
                        "og": getattr(recipe, "og", None),
                        "fg": getattr(recipe, "fg", None),
                        "abv": getattr(recipe, "abv", None),
                        "ibu": getattr(recipe, "ibu", None),
                        "srm": getattr(recipe, "srm", None),
                        "color": getattr(recipe, "color", None),
                        "batch_size": getattr(recipe, "batch_size", None),
                        "boil_size": getattr(recipe, "boil_size", None),
                        "boil_time": getattr(recipe, "boil_time", None),
                        "efficiency": getattr(recipe, "efficiency", None),
                        "hops": [hop.name for hop in recipe.hops],
                        "fermentables": [f.name for f in recipe.fermentables],
                        "yeasts": [y.name for y in recipe.yeasts],
                        "miscs": [m.name for m in recipe.miscs],
                        "notes": getattr(recipe, "notes", None),
                        "full_text": (
                            f"{recipe.name} by {recipe.brewer}. Style: "
                            f"{getattr(recipe.style, 'name', 'N/A')}. OG: "
                            f"{getattr(recipe, 'og', 0.0):.3f}, FG: "
                            f"{getattr(recipe, 'fg', 0.0):.3f}, ABV: "
                            f"{getattr(recipe, 'abv', 0.0):.2f}%, IBU: "
                            f"{getattr(recipe, 'ibu', 0.0):.2f}. Hops: "
                            f"{', '.join([hop.name for hop in recipe.hops])}. "
                            f"Fermentables: {', '.join([f.name for f in recipe.fermentables])}. "
                            f"Yeasts: {', '.join([y.name for y in recipe.yeasts])}. "
                            f"Notes: {getattr(recipe, 'notes', '')}"
                        ),
                    }
                    recipes_to_upsert.append(
                        Document(content=recipe_data["full_text"], meta_data=recipe_data)
                    )
            except Exception as e:
                logger.error(f"Error parsing {filepath}: {e}")

    if recipes_to_upsert:
        kb.upsert(recipes_to_upsert)
        logger.info(
            "Ingested %d recipes into collection '%s'.",
            len(recipes_to_upsert),
            s.QDRANT_RECIPE_COLLECTION,
        )
    else:
        logger.info("No BeerXML files found or parsed successfully in '%s'.", s.BEERXML_PATH)

    from qdrant_client import QdrantClient

    c = QdrantClient(url=s.QDRANT_URL)
    count = c.count(s.QDRANT_RECIPE_COLLECTION, exact=True).count
    logger.info("OK: %d points in collection '%s'.", count, s.QDRANT_RECIPE_COLLECTION)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Also create sparse BM25 vectors for hybrid search",
    )
    args = parser.parse_args()
    ingest_recipes(hybrid=args.hybrid)
