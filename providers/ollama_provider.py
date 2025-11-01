"""Ollama embedding provider."""
import ollama
from typing import List
import logging

logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider:
    """Provider for generating embeddings using Ollama API."""
    
    def __init__(self, model: str):
        """
        Initialize Ollama embedding provider.
        
        Args:
            model: The embedding model to use (e.g., 'gemma3:270m')
        """
        self.model = model
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for given text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            Exception: If embedding generation fails
        """
        try:
            response = ollama.embeddings(model=self.model, prompt=text)
            return response.get("embedding", [])
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

