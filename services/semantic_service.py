"""Semantic service - core orchestrator for query processing."""
import logging
from providers.base import EmbeddingProvider
from storage.base import VectorStore
from transformer.base import QueryTransformer
from core.settings import settings

logger = logging.getLogger(__name__)


class SemanticService:
    """Main service orchestrating embedding generation, caching, and query processing."""
    
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        storage: VectorStore,
        query_transformer: QueryTransformer,
        similarity_threshold: float = 0.85,
        high_confidence_threshold: float = 0.97
    ):
        """
        Initialize semantic service.
        
        Args:
            embedding_provider: Provider for generating embeddings
            storage: Storage backend for embeddings
            query_transformer: Transformer to normalize queries
            similarity_threshold: Minimum similarity threshold for cache hits
            high_confidence_threshold: Threshold for exact match detection (skip transformer)
        """
        self.embedding_provider = embedding_provider
        self.storage = storage
        self.query_transformer = query_transformer
        self.similarity_threshold = similarity_threshold
        self.high_confidence_threshold = high_confidence_threshold
    
    def process_query(self, text: str) -> str:
        """
        Process a user query through the semantic cache.
        
        Flow:
        1. Check DB with original query first
        2. If similarity >= high_confidence_threshold: return cached query (skip transformer)
        3. Otherwise: transform query and check DB again
        
        Args:
            text: User query text
            
        Returns:
            Query string (cached query or normalized query)
        """
        logger.info(f"Processing query: {text[:50]}...")
        
        # Step 1: Check DB with original query first
        original_embedding = self.embedding_provider.create(text)
        logger.debug(f"Generated embedding vector of length {len(original_embedding)}")
        
        similar_items = self.storage.find(
            embedding=original_embedding,
            threshold=self.high_confidence_threshold,
            top_k=1
        )
        
        if similar_items:
            cached_query = similar_items[0]["query"]
            similarity = similar_items[0]["similarity"]
            logger.info(f"Found high-confidence cached query with similarity: {similarity:.3f}")
            return cached_query
        
        # Step 2: No high-confidence match, transform query and check again
        logger.debug("No high-confidence match found, transforming query")
        normalized_query = self.query_transformer.transform(text)
        logger.debug(f"Normalized query: '{normalized_query}'")
        
        # Create embedding from normalized query
        normalized_embedding = self.embedding_provider.create(normalized_query)
        
        # Check DB with normalized query
        similar_items = self.storage.find(
            embedding=normalized_embedding,
            threshold=self.similarity_threshold,
            top_k=1
        )
        
        if similar_items:
            cached_query = similar_items[0]["query"]
            similarity = similar_items[0]["similarity"]
            logger.info(f"Found cached query with similarity: {similarity:.3f}")
            return cached_query
        
        # Step 3: No match found, store normalized query
        logger.info("No cached match found, storing normalized query")
        self.storage.put(query=normalized_query, embedding=normalized_embedding)
        logger.info(f"Stored new embedding for query: {normalized_query[:50]}...")
        
        return normalized_query
