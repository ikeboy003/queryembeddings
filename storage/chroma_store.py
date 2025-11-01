"""ChromaDB storage implementation."""
import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Optional
import uuid
from datetime import datetime
import logging
from storage.base import VectorStore

logger = logging.getLogger(__name__)


class ChromaStore(VectorStore):
    """ChromaDB storage for embeddings and metadata."""
    
    def __init__(self, collection_name: str, persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB client and collection.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist ChromaDB data
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(name=collection_name)
            logger.info(f"Created new collection: {collection_name}")
    
    def ping(self) -> bool:
        """
        Check if ChromaDB is accessible and healthy.
        
        Returns:
            True if ChromaDB is accessible, False otherwise
            
        Raises:
            Exception: If the database connection fails
        """
        try:
            _ = self.collection.count()
            logger.info("ChromaDB ping successful")
            return True
        except Exception as e:
            logger.error(f"ChromaDB ping failed: {e}")
            raise
    
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
        embedding_id = str(uuid.uuid4())
        doc_metadata = {
            "timestamp": datetime.utcnow().isoformat(),
            **(metadata or {})
        }
        
        try:
            self.collection.add(
                ids=[embedding_id],
                embeddings=[embedding],
                documents=[query],
                metadatas=[doc_metadata]
            )
            logger.info(f"Stored embedding with ID: {embedding_id}")
            return embedding_id
        except Exception as e:
            logger.error(f"Failed to store embedding: {e}")
            raise
    
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
        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k
            )
            
            if not results['ids'] or not results['ids'][0]:
                return []
            
            ids = results['ids'][0]
            distances = results['distances'][0]
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            
            similar_items = []
            for i, (id_val, distance) in enumerate(zip(ids, distances)):
                similarity = 1.0 - distance
                
                if similarity >= threshold:
                    similar_items.append({
                        "id": id_val,
                        "similarity": similarity,
                        "distance": distance,
                        "query": documents[i] if i < len(documents) else "",
                        "metadata": metadatas[i] if i < len(metadatas) else {}
                    })
            
            return similar_items
        except Exception as e:
            logger.error(f"Failed to find similar embeddings: {e}")
            return []

