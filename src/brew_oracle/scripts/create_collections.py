import argparse

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams

from brew_oracle.utils.config import Settings


def main(force_recreate: bool = False, hybrid: bool = False, collection_name: str | None = None):
    s = Settings()
    client = QdrantClient(url=s.QDRANT_URL)

    target_collection = collection_name if collection_name else s.QDRANT_COLLECTION

    if force_recreate and client.collection_exists(target_collection):
        client.delete_collection(target_collection)

    if not client.collection_exists(target_collection):
        vectors_config: VectorParams | dict[str, VectorParams] = VectorParams(
            size=s.EMBEDDER_DIM, distance=Distance.COSINE
        )
        if hybrid:
            vectors_config = {
                s.DENSE_VECTOR_NAME: VectorParams(size=s.EMBEDDER_DIM, distance=Distance.COSINE)
            }

        sparse_vectors_config = None
        if hybrid:
            sparse_vectors_config = {s.SPARSE_VECTOR_NAME: SparseVectorParams()}

        client.create_collection(
            collection_name=target_collection,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )
        return f"Coleção '{target_collection}' criada em {s.QDRANT_URL}"
    else:
        return f"Coleção '{target_collection}' já existe."


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Recria a coleção se existir")
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Inclui configuração para vetores esparsos (BM25)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        help="Nome da coleção a ser criada (padrão: QDRANT_COLLECTION do .env)",
    )
    args = parser.parse_args()
    main(force_recreate=args.force, hybrid=args.hybrid, collection_name=args.collection)
