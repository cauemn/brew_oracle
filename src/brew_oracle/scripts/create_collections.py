from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, SparseVectorParams, Distance
from brew_oracle.utils.config import Settings

def main(force_recreate: bool = False):
    s = Settings()
    client = QdrantClient(url=s.QDRANT_URL)

    if force_recreate and client.collection_exists(s.QDRANT_COLLECTION):
        client.delete_collection(s.QDRANT_COLLECTION)

    if not client.collection_exists(s.QDRANT_COLLECTION):
        client.create_collection(
            collection_name=s.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=s.EMBEDDER_DIM, distance=Distance.COSINE),
            sparse_vectors_config={
                "bm25_sparse": SparseVectorParams()
            },
        )
        print(f"✅ Coleção '{s.QDRANT_COLLECTION}' criada em {s.QDRANT_URL}")
    else:
        print(f"ℹ️ Coleção '{s.QDRANT_COLLECTION}' já existe.")

if __name__ == "__main__":
    main(force_recreate=False)
