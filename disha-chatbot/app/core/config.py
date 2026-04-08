from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # App
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    ALLOWED_ORIGINS: List[str] = ["*"]

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    CHAT_MODEL: str = "phi3:14b"
    EMBED_MODEL: str = "mxbai-embed-large"

    # ChromaDB
    CHROMA_PERSIST_DIR: str = "./chromadb_data"
    CHROMA_COLLECTION: str = "nspcl_docs"

    # Retriever
    RETRIEVER_K: int = 3
    RETRIEVER_TYPE: str = "mmr"

    # Chat
    MAX_HISTORY: int = 10
    BOT_NAME: str = "Disha"
    COMPANY_NAME: str = "NSPCL"

    class Config:
        env_file = ".env"


settings = Settings()
