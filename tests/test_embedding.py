"""Tests for embedding service."""
import pytest
from services.similarity import cosine_similarity


def test_cosine_similarity():
    """Test cosine similarity calculation."""
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]
    
    similarity = cosine_similarity(vec1, vec2)
    assert similarity == 1.0
    
    vec3 = [0.0, 1.0, 0.0]
    similarity2 = cosine_similarity(vec1, vec3)
    assert similarity2 == 0.0


def test_cosine_similarity_different_vectors():
    """Test cosine similarity with different vectors."""
    vec1 = [1.0, 2.0, 3.0]
    vec2 = [4.0, 5.0, 6.0]
    
    similarity = cosine_similarity(vec1, vec2)
    assert 0.0 <= similarity <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])
