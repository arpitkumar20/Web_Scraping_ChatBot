import re
import logging
from app.core.config import EMBEDDING_MODEL, genai

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not isinstance(text, str):
        return str(text)
    text = re.sub(r'[\n\xa0]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def preprocess_text(text: str) -> str:
    """Full preprocessing pipeline."""
    return clean_text(text)

def generate_embedding(text: str) -> list[float]:
    """
    Generate embedding vector for the given text using GenAI.

    Raises:
        ValueError: If embedding response is invalid.
    """
    try:
        logger.info("Generating embedding...")
        text = preprocess_text(text)

        response = genai.embed_content(model=EMBEDDING_MODEL, content=text)

        if "embedding" in response:
            embedding_vector = response["embedding"]
        elif "embeddings" in response:
            embedding_vector = response["embeddings"][0]
        else:
            raise ValueError("Unexpected embedding response format.")

        if len(embedding_vector) != 768:
            raise ValueError(f"Embedding vector dimension mismatch: Expected 768, got {len(embedding_vector)}.")

        logger.info(f"Generated embedding of length {len(embedding_vector)}")
        return embedding_vector

    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise