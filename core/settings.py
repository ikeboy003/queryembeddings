"""Configuration settings loaded from environment variables."""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""
    
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "embeddinggemma:300m")
    
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "query_embeddings")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "10"))


settings = Settings()

