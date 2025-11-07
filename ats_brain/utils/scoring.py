"""Semantic similarity scoring using SBERT embeddings with TF-IDF fallback."""
from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
from loguru import logger
import base64

# Global model cache - loaded once at startup
_sbert_model: Optional[object] = None
_sbert_available: bool = False


def load_sbert_model() -> Optional[object]:
    """
    Load SBERT model 'all-MiniLM-L6-v2' and cache globally.
    Should be called at FastAPI startup event.
    Returns the model or None if loading fails.
    """
    global _sbert_model, _sbert_available
    
    if _sbert_model is not None:
        return _sbert_model
    
    try:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        _sbert_available = True
        logger.info("Loaded SBERT model: all-MiniLM-L6-v2")
        return _sbert_model
    except Exception as exc:
        logger.warning("SBERT model loading failed, will use TF-IDF fallback: {}", exc)
        _sbert_model = None
        _sbert_available = False
        return None


def is_sbert_available() -> bool:
    """Check if SBERT model is loaded and available."""
    return _sbert_available and _sbert_model is not None


def compute_embedding(text: str) -> Tuple[Optional[np.ndarray], str]:
    """
    Compute embedding for text using SBERT if available, otherwise returns None.
    
    Returns:
        Tuple of (embedding_array, method_name)
        - embedding_array: numpy array (384-dim for all-MiniLM-L6-v2) or None
        - method_name: "sbert" or "tfidf" (if fallback needed)
    """
    if is_sbert_available():
        try:
            emb = _sbert_model.encode([text], normalize_embeddings=True)[0]
            return np.array(emb, dtype=np.float32), "sbert"
        except Exception as e:
            logger.warning("SBERT encoding failed: {}, falling back to TF-IDF", e)
            return None, "tfidf"
    return None, "tfidf"


def serialize_embedding(embedding: np.ndarray) -> str:
    """
    Serialize numpy embedding to base64 string for storage.
    
    Args:
        embedding: numpy array of floats
        
    Returns:
        Base64-encoded string of the embedding bytes
    """
    if embedding is None:
        return ""
    # Convert to bytes and then base64
    emb_bytes = embedding.astype(np.float32).tobytes()
    return base64.b64encode(emb_bytes).decode('utf-8')


def deserialize_embedding(embedding_str: str) -> Optional[np.ndarray]:
    """
    Deserialize base64 string back to numpy embedding array.
    
    Args:
        embedding_str: Base64-encoded string
        
    Returns:
        numpy array or None if deserialization fails
    """
    if not embedding_str:
        return None
    try:
        emb_bytes = base64.b64decode(embedding_str)
        embedding = np.frombuffer(emb_bytes, dtype=np.float32)
        return embedding
    except Exception as e:
        logger.warning("Failed to deserialize embedding: {}", e)
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two normalized vectors.
    
    Args:
        a: numpy array (1D or 2D)
        b: numpy array (1D or 2D)
        
    Returns:
        Cosine similarity score (0.0 to 1.0)
    """
    # Ensure 1D arrays
    if a.ndim > 1:
        a = a.flatten()
    if b.ndim > 1:
        b = b.flatten()
    
    # Normalize
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    similarity = float(np.dot(a, b) / (norm_a * norm_b))
    # Clamp to [0, 1] range
    return max(0.0, min(1.0, similarity))


def score_texts_sbert(text_a: str, text_b: str, embedding_a: Optional[np.ndarray] = None, embedding_b: Optional[np.ndarray] = None) -> Tuple[float, str]:
    """
    Compute similarity using SBERT embeddings.
    
    Args:
        text_a: First text
        text_b: Second text
        embedding_a: Precomputed embedding for text_a (optional)
        embedding_b: Precomputed embedding for text_b (optional)
        
    Returns:
        Tuple of (similarity_score, "sbert")
    """
    if not is_sbert_available():
        return -1.0, "tfidf"
    
    try:
        if embedding_a is None:
            embedding_a, _ = compute_embedding(text_a)
        if embedding_b is None:
            embedding_b, _ = compute_embedding(text_b)
        
        if embedding_a is None or embedding_b is None:
            return -1.0, "tfidf"
        
        score = cosine_similarity(embedding_a, embedding_b)
        return score, "sbert"
    except Exception as e:
        logger.warning("SBERT scoring failed: {}, falling back to TF-IDF", e)
        return -1.0, "tfidf"


def score_texts_tfidf(text_a: str, text_b: str) -> Tuple[float, str]:
    """
    Compute similarity using TF-IDF cosine similarity (fallback method).
    
    Args:
        text_a: First text
        text_b: Second text
        
    Returns:
        Tuple of (similarity_score, "tfidf")
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as sk_cos
        
        if not text_a or not text_b:
            return 0.0, "tfidf"
        
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
        similarity = sk_cos(tfidf_matrix[0:1], tfidf_matrix[1:2])
        score = float(similarity[0, 0])
        return score, "tfidf"
    except Exception as e:
        logger.error("TF-IDF scoring failed: {}", e)
        return 0.0, "tfidf"


def score_texts(text_a: str, text_b: str, embedding_a: Optional[np.ndarray] = None, embedding_b: Optional[np.ndarray] = None) -> Tuple[float, str]:
    """
    Compute similarity between two texts using SBERT if available, otherwise TF-IDF.
    
    Args:
        text_a: First text
        text_b: Second text
        embedding_a: Precomputed embedding for text_a (optional, for efficiency)
        embedding_b: Precomputed embedding for text_b (optional, for efficiency)
        
    Returns:
        Tuple of (similarity_score, method_name)
        - similarity_score: float between 0.0 and 1.0
        - method_name: "sbert" or "tfidf"
    """
    # Try SBERT first
    score, method = score_texts_sbert(text_a, text_b, embedding_a, embedding_b)
    if score >= 0:
        return score, method
    
    # Fallback to TF-IDF
    return score_texts_tfidf(text_a, text_b)
