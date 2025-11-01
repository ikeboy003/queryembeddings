"""Semantic service - core orchestrator for query processing."""
import logging
from providers.base import EmbeddingProvider
from storage.base import VectorStore
from transformer.base import QueryTransformer

logger = logging.getLogger(__name__)


class SemanticService:
    """Main service orchestrating embedding generation, caching, and query processing."""
    
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        storage: VectorStore,
        query_transformer: QueryTransformer,
        similarity_threshold: float = 0.85
    ):
        """
        Initialize semantic service.
        
        Args:
            embedding_provider: Provider for generating embeddings
            storage: Storage backend for embeddings
            query_transformer: Transformer to normalize queries
            similarity_threshold: Minimum similarity threshold for cache hits
        """
        self.embedding_provider = embedding_provider
        self.storage = storage
        self.query_transformer = query_transformer
        self.similarity_threshold = similarity_threshold
    
    def process_query(self, text: str) -> str:
        """
        Process a user query through the semantic cache.
        
        Flow: User query -> Transformer -> Normalized query -> Embedding -> Check DB
        
        Args:
            text: User query text
            
        Returns:
            Query string (cached normalized query or normalized query)
        """
        logger.info(f"Processing query: {text[:50]}...")
        
        # Transform user query to normalized search query
        normalized_query = self.query_transformer.transform(text)
        logger.debug(f"Normalized query: '{normalized_query}'")
        
        # Create embedding from normalized query
        embedding = self.embedding_provider.create(normalized_query)
        logger.debug(f"Generated embedding vector of length {len(embedding)}")
        
        # Check for similar cached queries
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
        
        # Store normalized query (not original user query)
        logger.info("No cached match found, storing normalized query")
        self.storage.put(query=normalized_query, embedding=embedding)
        logger.info(f"Stored new embedding for query: {normalized_query[:50]}...")
        
        return normalized_query
