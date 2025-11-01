"""Semantic service - core orchestrator for query processing."""
import logging
from providers.base import EmbeddingProvider
from storage.base import VectorStore

logger = logging.getLogger(__name__)


class SemanticService:
    """Main service orchestrating embedding generation, caching, and query processing."""
    
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        storage: VectorStore,
        similarity_threshold: float = 0.85
    ):
        """
        Initialize semantic service.
        
        Args:
            embedding_provider: Provider for generating embeddings
            storage: Storage backend for embeddings
            similarity_threshold: Minimum similarity threshold for cache hits
        """
        self.embedding_provider = embedding_provider
        self.storage = storage
        self.similarity_threshold = similarity_threshold
    
    def process_query(self, text: str) -> str:
        """
        Process a user query through the semantic cache.
        
        Returns cached query if similar match found, otherwise stores and returns original.
        
        Args:
            text: User query text
            
        Returns:
            Query string (cached or original)
        """
        logger.info(f"Processing query: {text[:50]}...")
        
        embedding = self.embedding_provider.create(text)
        logger.debug(f"Generated embedding vector of length {len(embedding)}")
        
        similar_items = self.storage.find(
            embedding=embedding,
            threshold=self.similarity_threshold,
            top_k=1
        )
        
        if similar_items:
            cached_query = similar_items[0]["query"]
            similarity = similar_items[0]["similarity"]
            logger.info(f"Found cached query with similarity: {similarity:.3f}")
            return cached_query
        
        logger.info("No cached match found, storing new query")
        self.storage.put(query=text, embedding=embedding)
        logger.info(f"Stored new embedding for query: {text[:50]}...")
        
        return text
