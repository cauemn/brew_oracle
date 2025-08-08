from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    QDRANT_URL: str = Field(default="http://localhost:6333")
    QDRANT_COLLECTION: str = Field(default="brew_books")

    PDF_PATH: str = Field(default="knowledge/pdfs")

    EMBEDDER_ID: str = Field(default="./models/all-MiniLM-L6-v2")
    EMBEDDER_DIM: int = Field(default=384)

    TOP_K: int = Field(default=20)

    GOOGLE_API_KEY: str | None = Field(default=None)

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',   
    )
