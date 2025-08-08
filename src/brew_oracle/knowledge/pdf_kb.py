# src/brew_oracle/knowledge/pdf_kb.py
import logging
import os
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.vectordb.qdrant import Qdrant
from agno.embedder.sentence_transformer import SentenceTransformerEmbedder
from agno.document.chunking.recursive import RecursiveChunking
from brew_oracle.utils.config import Settings


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_pdf_kb() -> PDFKnowledgeBase:
    """Create and configure the PDF knowledge base.

    The knowledge base uses settings defined in :class:`Settings` to configure
    the embedder, vector database and PDF reader.

    Returns
    -------
    PDFKnowledgeBase
        The configured knowledge base ready to ingest documents.
    """

    s = Settings()
    os.makedirs(s.PDF_PATH, exist_ok=True)

    embedder = SentenceTransformerEmbedder(
        id=s.EMBEDDER_ID,
        dimensions=s.EMBEDDER_DIM,
    )

    kb = PDFKnowledgeBase(
        path=s.PDF_PATH,
        vector_db=Qdrant(
            collection=s.QDRANT_COLLECTION,
            url=s.QDRANT_URL,
            embedder=embedder,
        ),
        reader=PDFReader(
            chunk=True,
            chunk_size=s.CHUNK_SIZE,
            chunking_strategy=RecursiveChunking(
                chunk_size=s.CHUNK_SIZE,
                overlap=s.CHUNK_OVERLAP,
            ),
        ),
        num_documents=s.NUM_DOCUMENTS,
    )
    return kb

def ingest_pdfs(upsert: bool = True) -> None:
    """Load PDF files into the Qdrant collection.

    Parameters
    ----------
    upsert : bool, optional
        If ``True`` (default), existing documents are updated during
        ingestion; otherwise, only new documents are added.
    """

    s = Settings()
    kb = build_pdf_kb()
    logger.info("Iniciando ingestão dos arquivos - Pasta: '%s'.", s.PDF_PATH)
    kb.load(upsert=upsert)
    from qdrant_client import QdrantClient

    c = QdrantClient(url=s.QDRANT_URL)
    logger.info(
        "Conectei em '%s' irei incluir na collection '%s'.",
        s.QDRANT_URL,
        s.QDRANT_COLLECTION,
    )
    count = c.count(s.QDRANT_COLLECTION, exact=True).count
    logger.info("OK: %d pontos na coleção '%s'.", count, s.QDRANT_COLLECTION)
