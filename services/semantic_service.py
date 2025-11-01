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
        
        Flow:
        1. Generate embedding for query
        2. Check storage for similar embeddings
        3. If found above threshold, return cached query
        4. If not found, return original query and store it
        
        Args:
            text: User query text
            
        Returns:
            Query string (cached or original)
        """
        logger.info(f"Processing query: {text[:50]}...")
        
        # Step 1: Generate embedding
        try:
            embedding = self.embedding_provider.create(text)
            logger.debug(f"Generated embedding vector of length {len(embedding)}")
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
        
        # Step 2: Check for similar cached queries
        similar_items = self.storage.find(
            embedding=embedding,
            threshold=self.similarity_threshold,
            top_k=1  # We only need the best match
        )
        
        # Step 3: Return cached query if found
        if similar_items:
            best_match = similar_items[0]
            cached_query = best_match["query"]
            logger.info(f"Found cached query with similarity: {best_match['similarity']:.3f}")
            return cached_query
        
        # Step 4: Store the new embedding and return original query
        logger.info("No cached match found, storing new query")
        try:
            self.storage.put(
                query=text,
                embedding=embedding
            )
            logger.info(f"Stored new embedding for query: {text[:50]}...")
        except Exception as e:
            logger.error(f"Failed to store embedding: {e}")
        
        return text
