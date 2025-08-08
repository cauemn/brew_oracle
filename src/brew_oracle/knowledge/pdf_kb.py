# src/brew_oracle/knowledge/pdf_kb.py
import logging
import os
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.vectordb.qdrant import Qdrant
from agno.embedder.sentence_transformer import SentenceTransformerEmbedder
from agno.document.chunking.recursive import RecursiveChunking
from brew_oracle.utils.config import Settings

try:  # Optional imports for sparse vectors / hybrid search
    from qdrant_client import models as qmodels
    from qdrant_client.fastembed import SparseEncoder
except Exception:  # pragma: no cover - modules are optional
    qmodels = None
    SparseEncoder = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_pdf_kb(hybrid: bool = False) -> PDFKnowledgeBase:
    """Create and configure the PDF knowledge base.

    Parameters
    ----------
    hybrid : bool, optional
        When ``True`` also generates sparse BM25 vectors and enables
        fusion scoring between dense and sparse results, by default ``False``.

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
    sparse_encoder = SparseEncoder() if hybrid and SparseEncoder else None

    kb = PDFKnowledgeBase(
        path=s.PDF_PATH,
        vector_db=Qdrant(
            collection=s.QDRANT_COLLECTION,
            url=s.QDRANT_URL,
            embedder=embedder,
            sparse=sparse_encoder,
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
    if hybrid and qmodels is not None:
        original_search = kb.search

        def _fusion_search(query: str, top_k: int | None = None, *args, **kwargs):
            top_k = top_k or s.TOP_K
            dense_docs = original_search(query, top_k=top_k, *args, **kwargs)
            if not SparseEncoder:
                return dense_docs
            sparse_query = SparseEncoder().encode_queries([query])[0]
            sparse_docs = original_search(
                query,
                top_k=top_k,
                sparse_vector=qmodels.SparseVector(
                    indices=sparse_query.indices, values=sparse_query.values
                ),
            )
            scores: dict[str, float] = {}
            for rank, doc in enumerate(dense_docs):
                doc_id = getattr(doc, "id", getattr(doc, "doc_id", str(rank)))
                scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (rank + 60)
            for rank, doc in enumerate(sparse_docs):
                doc_id = getattr(doc, "id", getattr(doc, "doc_id", str(rank)))
                scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (rank + 60)
            fused_docs = {getattr(doc, "id", getattr(doc, "doc_id", str(i))): doc for i, doc in enumerate(dense_docs + sparse_docs)}
            return [doc for doc, _ in sorted(
                [(fused_docs[i], sc) for i, sc in scores.items()],
                key=lambda x: x[1],
                reverse=True,
            )][:top_k]

        kb.search = _fusion_search  # type: ignore[assignment]

    return kb

def ingest_pdfs(upsert: bool = True, hybrid: bool = False) -> None:
    """Load PDF files into the Qdrant collection.

    Parameters
    ----------
    upsert : bool, optional
        If ``True`` (default), existing documents are updated during
        ingestion; otherwise, only new documents are added.
    hybrid : bool, optional
        Also create sparse BM25 vectors for hybrid search, by default ``False``.
    """

    s = Settings()
    kb = build_pdf_kb(hybrid=hybrid)
    logger.info("Iniciando ingestão dos arquivos - Pasta: '%s'.", s.PDF_PATH)
    load_kwargs = {"upsert": upsert}
    if hybrid:
        load_kwargs["sparse"] = True
    kb.load(**load_kwargs)
    from qdrant_client import QdrantClient

    c = QdrantClient(url=s.QDRANT_URL)
    logger.info(
        "Conectei em '%s' irei incluir na collection '%s'.",
        s.QDRANT_URL,
        s.QDRANT_COLLECTION,
    )
    count = c.count(s.QDRANT_COLLECTION, exact=True).count
    logger.info("OK: %d pontos na coleção '%s'.", count, s.QDRANT_COLLECTION)
