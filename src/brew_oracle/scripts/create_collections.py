import argparse
from typing import Dict, Union, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams

from brew_oracle.utils.config import Settings


def main(force_recreate: bool = False, hybrid: bool = False):
    s = Settings()
    client = QdrantClient(url=s.QDRANT_URL)

    if force_recreate and client.collection_exists(s.QDRANT_COLLECTION):
        client.delete_collection(s.QDRANT_COLLECTION)

    if not client.collection_exists(s.QDRANT_COLLECTION):
        vectors_config: Union[VectorParams, Dict[str, VectorParams]] = VectorParams(
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
            collection_name=s.QDRANT_COLLECTION,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )
        print(f"✅ Coleção '{s.QDRANT_COLLECTION}' criada em {s.QDRANT_URL}")
    else:
        print(f"ℹ️ Coleção '{s.QDRANT_COLLECTION}' já existe.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Recria a coleção se existir")
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Inclui configuração para vetores esparsos (BM25)",
    )
    args = parser.parse_args()
    main(force_recreate=args.force, hybrid=args.hybrid)
