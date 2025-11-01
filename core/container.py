"""Dependency injection container."""
import logging
from core.settings import settings
from providers.base import EmbeddingProvider
from providers.ollama_provider import OllamaEmbeddingProvider
from storage.base import VectorStore
from storage.chroma_store import ChromaStore
from transformer.base import QueryTransformer
from transformer.ollama_transformer import OllamaQueryTransformer
from services.semantic_service import SemanticService

logger = logging.getLogger(__name__)


class Container:
    """Dependency injection container for wiring up services."""
    
    def __init__(self):
        self._embedding_provider = None
        self._storage = None
        self._query_transformer = None
        self._semantic_service = None
    
    def _ping_service(self, service, service_name: str) -> None:
        """Helper method to ping a service and log the result."""
        try:
            service.ping()
            logger.info(f"{service_name} connection verified successfully")
        except Exception as e:
            logger.error(f"Failed to connect to {service_name}: {e}")
            raise
    
    @property
    def embedding_provider(self) -> EmbeddingProvider:
        """Get or create embedding provider."""
        if self._embedding_provider is None:
            self._embedding_provider = OllamaEmbeddingProvider(
                model=settings.EMBEDDING_MODEL
            )
            self._ping_service(self._embedding_provider, "Embedding provider")
        return self._embedding_provider
    
    @property
    def storage(self) -> VectorStore:
        """Get or create storage instance."""
        if self._storage is None:
            self._storage = ChromaStore(
                collection_name=settings.CHROMA_COLLECTION_NAME,
                persist_directory=settings.CHROMA_PERSIST_DIR
            )
            self._ping_service(self._storage, "Database")
        return self._storage
    
    @property
    def query_transformer(self) -> QueryTransformer:
        """Get or create query transformer."""
        if self._query_transformer is None:
            self._query_transformer = OllamaQueryTransformer(
                model=settings.TRANSFORMER_MODEL
            )
            self._ping_service(self._query_transformer, "Query transformer")
        return self._query_transformer
    
    @property
    def semantic_service(self) -> SemanticService:
        """Get or create semantic service."""
        if self._semantic_service is None:
            self._semantic_service = SemanticService(
                embedding_provider=self.embedding_provider,
                storage=self.storage,
                query_transformer=self.query_transformer,
                similarity_threshold=settings.SIMILARITY_THRESHOLD
            )
        return self._semantic_service


# Global container instance
container = Container()
