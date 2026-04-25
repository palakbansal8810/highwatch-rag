from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
from typing import Literal


class Settings(BaseSettings):
    # Groq
    groq_api_key: str = Field("", env="GROQ_API_KEY")
    groq_model: str = Field("llama-3.1-8b-instant", env="GROQ_MODEL")

    # Google Drive
    google_client_id: str = Field("", env="GOOGLE_CLIENT_ID")
    google_client_secret: str = Field("", env="GOOGLE_CLIENT_SECRET")
    google_redirect_uri: str = Field("http://localhost:8000/auth/callback", env="GOOGLE_REDIRECT_URI")
    google_service_account_file: str = Field("service_account.json", env="GOOGLE_SERVICE_ACCOUNT_FILE")
    google_auth_mode: Literal["oauth", "service_account"] = Field("oauth", env="GOOGLE_AUTH_MODE")

    # Embedding
    embedding_model: str = Field("all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    embedding_dimension: int = Field(384, env="EMBEDDING_DIMENSION")
    embedding_batch_size: int = Field(32, env="EMBEDDING_BATCH_SIZE")

    # Chunking
    chunk_size: int = Field(512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(64, env="CHUNK_OVERLAP")
    min_chunk_length: int = Field(50, env="MIN_CHUNK_LENGTH")

    # FAISS
    faiss_index_path: str = Field("./storage/faiss_index", env="FAISS_INDEX_PATH")
    metadata_db_path: str = Field("./storage/metadata.json", env="METADATA_DB_PATH")

    # Cache
    cache_dir: str = Field("./storage/cache", env="CACHE_DIR")
    cache_ttl: int = Field(3600, env="CACHE_TTL")

    # API
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    debug: bool = Field(False, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    # RAG
    top_k_chunks: int = Field(5, env="TOP_K_CHUNKS")
    max_context_tokens: int = Field(4000, env="MAX_CONTEXT_TOKENS")
    answer_max_tokens: int = Field(1024, env="ANSWER_MAX_TOKENS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def ensure_dirs(self):
        for path in [
            self.faiss_index_path,
            Path(self.metadata_db_path).parent,
            self.cache_dir,
        ]:
            Path(path).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()