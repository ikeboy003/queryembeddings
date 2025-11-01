"""Dependency injection container."""
from semantic_service.core.settings import settings
import sys
import os

# Add parent directory to path to import from root modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from providers.ollama_provider import OllamaEmbeddingProvider
from storage.chroma_store import ChromaStore
from services.semantic_service import SemanticService


class Container:
    """Dependency injection container for wiring up services."""
    
    def __init__(self):
        self._embedding_provider = None
        self._storage = None
        self._semantic_service = None
    
    @property
    def embedding_provider(self) -> OllamaEmbeddingProvider:
        """Get or create embedding provider."""
        if self._embedding_provider is None:
            self._embedding_provider = OllamaEmbeddingProvider(
                model=settings.EMBEDDING_MODEL
            )
        return self._embedding_provider
    
    @property
    def storage(self) -> ChromaStore:
        """Get or create storage instance."""
        if self._storage is None:
            self._storage = ChromaStore(
                collection_name=settings.CHROMA_COLLECTION_NAME,
                persist_directory=settings.CHROMA_PERSIST_DIR
            )
        return self._storage
    
    @property
    def semantic_service(self) -> SemanticService:
        """Get or create semantic service."""
        if self._semantic_service is None:
            self._semantic_service = SemanticService(
                embedding_provider=self.embedding_provider,
                storage=self.storage,
                similarity_threshold=settings.SIMILARITY_THRESHOLD
            )
        return self._semantic_service


# Global container instance
container = Container()

