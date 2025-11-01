"""Ollama query transformer implementation."""
import ollama
import logging
from transformer.base import QueryTransformer

logger = logging.getLogger(__name__)


class OllamaQueryTransformer(QueryTransformer):
    """Transformer that uses Ollama LLM to normalize queries."""
    
    def __init__(self, model: str):
        """
        Initialize Ollama query transformer.
        
        Args:
            model: The LLM model to use for transformation (e.g., 'gemma2:2b')
        """
        self.model = model
    
    def transform(self, query: str) -> str:
        """
        Transform user query into a normalized search query using LLM.
        
        Args:
            query: User query text
            
        Returns:
            Normalized search query string
            
        Raises:
            Exception: If transformation fails
        """
        try:
            prompt = f"Transform this user query into a normalized search query. Return only the normalized query, nothing else:\n\n{query}"
            
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a query normalization assistant. Transform user queries into clean, normalized search queries."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            normalized_query = response["message"]["content"].strip()
            
            if not normalized_query:
                logger.warning("Transformer returned empty query, using original")
                return query
            
            logger.debug(f"Transformed query: '{query}' -> '{normalized_query}'")
            return normalized_query
            
        except Exception as e:
            logger.error(f"Failed to transform query: {e}")
            logger.warning("Using original query due to transformation failure")
            return query
    
    def ping(self) -> bool:
        """
        Check if Ollama is accessible and healthy.
        
        Returns:
            True if Ollama is accessible, False otherwise
            
        Raises:
            Exception: If the service connection fails
        """
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}]
            )
            if response.get("message", {}).get("content"):
                logger.info("Ollama transformer ping successful")
                return True
            else:
                raise Exception("Ollama returned empty response")
        except Exception as e:
            logger.error(f"Ollama transformer ping failed: {e}")
            raise

