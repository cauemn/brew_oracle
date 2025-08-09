from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    QDRANT_URL: str = Field(default="http://localhost:6333")
    QDRANT_COLLECTION: str = Field(default="brew_books")
    DENSE_VECTOR_NAME: str = Field(default="dense")
    SPARSE_VECTOR_NAME: str = Field(default="sparse")
    SPARSE_MODEL_ID: str = Field(default="Qdrant/bm25")

    PDF_PATH: str = Field(default="knowledge/pdfs")
    BEERXML_PATH: str = Field(default="knowledge/recipes")

    QDRANT_RECIPE_COLLECTION: str = Field(default="brew_recipes")

    EMBEDDER_ID: str = Field(default="./models/all-MiniLM-L6-v2")
    EMBEDDER_DIM: int = Field(default=384)

    TOP_K: int = Field(default=20)

    CHUNK_SIZE: int = Field(default=2000)
    CHUNK_OVERLAP: int = Field(default=300)
    NUM_DOCUMENTS: int = Field(default=5)

    GOOGLE_API_KEY: str | None = Field(default=None)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
