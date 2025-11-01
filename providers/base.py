"""Abstract base class for embedding provider implementations."""
from abc import ABC, abstractmethod
from typing import List


class EmbeddingProvider(ABC):
    """Abstract base class for embedding provider implementations."""
    
    @abstractmethod
    def create(self, text: str) -> List[float]:
        """
        Create embedding vector for given text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            Exception: If embedding generation fails
        """
        pass
    
    @abstractmethod
    def ping(self) -> bool:
        """
        Check if the embedding service is accessible and healthy.
        
        Returns:
            True if the service is accessible, False otherwise
            
        Raises:
            Exception: If the service connection fails
        """
        pass

