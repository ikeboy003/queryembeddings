"""Ollama embedding provider."""
import ollama
from typing import List
import logging
from providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Provider for generating embeddings using Ollama API."""
    
    def __init__(self, model: str):
        """
        Initialize Ollama embedding provider.
        
        Args:
            model: The embedding model to use (e.g., 'embeddinggemma:300m')
        """
        self.model = model
    
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
        try:
            response = ollama.embeddings(model=self.model, prompt=text)
            embedding = response.get("embedding", [])
            if not embedding:
                raise ValueError("Ollama returned empty embedding")
            return embedding
        except Exception as e:
            logger.error(f"Failed to create embedding: {e}")
            raise
    
    def ping(self) -> bool:
        """
        Check if Ollama is accessible and healthy.
        
        Returns:
            True if Ollama is accessible, False otherwise
            
        Raises:
            Exception: If the service connection fails
        """
        try:
            response = ollama.embeddings(model=self.model, prompt="ping")
            if not response.get("embedding"):
                raise Exception("Ollama returned empty embedding")
            logger.info("Ollama ping successful")
            return True
        except Exception as e:
            logger.error(f"Ollama ping failed: {e}")
            raise

