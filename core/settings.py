"""Configuration settings loaded from environment variables."""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""
    
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "embeddinggemma:300m")
    
    TRANSFORMER_MODEL: str = os.getenv("TRANSFORMER_MODEL", "llama3.1:latest")
    
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "query_embeddings")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))
    
    HIGH_CONFIDENCE_THRESHOLD: float = float(os.getenv("HIGH_CONFIDENCE_THRESHOLD", "0.97"))


settings = Settings()

