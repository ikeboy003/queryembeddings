"""Abstract base class for vector database implementations."""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class VectorStore(ABC):
    """Abstract base class for vector database implementations."""
    
    @abstractmethod
    def ping(self) -> bool:
        """
        Check if the vector database is accessible and healthy.
        
        Returns:
            True if the database is accessible, False otherwise
            
        Raises:
            Exception: If the database connection fails
        """
        pass
    
    @abstractmethod
    def put(
        self,
        query: str,
        embedding: List[float],
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Store embedding with metadata.
        
        Args:
            query: User query text
            embedding: Embedding vector
            metadata: Additional metadata dictionary
            
        Returns:
            Generated embedding ID
        """
        pass
    
    @abstractmethod
    def find(
        self,
        embedding: List[float],
        threshold: float = 0.85,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Find similar embeddings above threshold.
        
        Args:
            embedding: Query embedding vector
            threshold: Minimum similarity threshold (0.0-1.0)
            top_k: Maximum number of results to return
            
        Returns:
            List of dictionaries with id, distance, query, similarity, metadata
        """
        pass

