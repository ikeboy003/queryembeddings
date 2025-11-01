"""Tests for storage layer."""
import pytest
import os
import shutil
from storage.chroma_store import ChromaStore


@pytest.fixture
def temp_storage():
    """Create temporary storage for testing."""
    test_dir = "./test_chroma_db"
    store = ChromaStore(
        collection_name="test_collection",
        persist_directory=test_dir
    )
    yield store
    # Cleanup
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


def test_add_embedding(temp_storage):
    """Test adding an embedding to storage."""
    embedding = [0.1] * 384  # Mock embedding vector
    query = "test query"
    
    embedding_id = temp_storage.add_embedding(
        query=query,
        embedding=embedding
    )
    
    assert embedding_id is not None
    assert len(embedding_id) > 0


def test_find_similar(temp_storage):
    """Test finding similar embeddings."""
    # Add a test embedding
    embedding1 = [0.1] * 384
    temp_storage.add_embedding(
        query="test query 1",
        embedding=embedding1
    )
    
    # Search with same embedding
    results = temp_storage.find_similar(
        embedding=embedding1,
        threshold=0.5
    )
    
    assert len(results) > 0
    assert results[0]["similarity"] >= 0.5


def test_find_similar_threshold(temp_storage):
    """Test similarity threshold filtering."""
    embedding1 = [0.1] * 384
    embedding2 = [0.9] * 384  # Very different embedding
    
    temp_storage.add_embedding(
        query="test query",
        embedding=embedding1
    )
    
    # Search with very different embedding
    results = temp_storage.find_similar(
        embedding=embedding2,
        threshold=0.95  # High threshold
    )
    
    # Should find nothing or very low similarity
    assert all(r["similarity"] < 0.95 for r in results) or len(results) == 0


if __name__ == "__main__":
    pytest.main([__file__])
