"""Abstract base class for query transformer implementations."""
from abc import ABC, abstractmethod


class QueryTransformer(ABC):
    """Abstract base class for query transformer implementations."""
    
    @abstractmethod
    def transform(self, query: str) -> str:
        """
        Transform user query into a normalized search query.
        
        Args:
            query: User query text
            
        Returns:
            Normalized search query string
            
        Raises:
            Exception: If transformation fails
        """
        pass
    
    @abstractmethod
    def ping(self) -> bool:
        """
        Check if the transformer service is accessible and healthy.
        
        Returns:
            True if the service is accessible, False otherwise
            
        Raises:
            Exception: If the service connection fails
        """
        pass

