# src/brew_oracle/knowledge/pdf_kb.py
import os
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.vectordb.qdrant import Qdrant
from agno.embedder.sentence_transformer import SentenceTransformerEmbedder
from agno.document.chunking.recursive import RecursiveChunking
from brew_oracle.utils.config import Settings

def build_pdf_kb() -> PDFKnowledgeBase:
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
    s = Settings()
    kb = build_pdf_kb()
    print(f"Iniciando ingestão dos arquivos - Pasta:  '{s.PDF_PATH}'.")
    kb.load(upsert=upsert)
    from qdrant_client import QdrantClient
    c = QdrantClient(url=s.QDRANT_URL)
    print(f"Conectei em '{s.QDRANT_URL}' irei incluir na collection '{s.QDRANT_COLLECTION}'.")
    count = c.count(s.QDRANT_COLLECTION, exact=True).count
    print(f"OK: {count} pontos na coleção '{s.QDRANT_COLLECTION}'.")
