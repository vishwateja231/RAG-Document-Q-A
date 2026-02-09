"""Application configuration for the RAG Document QA system."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized environment-driven settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Document QA RAG"
    upload_dir: Path = Path("data/documents")
    vectorstore_dir: Path = Path("data/vectorstore")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str = ""
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4


settings = Settings()
