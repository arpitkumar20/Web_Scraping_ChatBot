# import re
# from app.core.logging import get_logger
# from app.core.config import EMBEDDING_MODEL, genai

# logger = get_logger(__name__)

# def clean_text(text: str) -> str:
#     """Clean and normalize text."""
#     if not isinstance(text, str):
#         return str(text)
#     text = re.sub(r'[\n\xa0]+', ' ', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text.lower()

# def preprocess_text(text: str) -> str:
#     """Full preprocessing pipeline."""
#     return clean_text(text)

# def generate_embedding(text: str) -> list[float]:
#     """
#     Generate embedding vector for the given text using GenAI.

#     Raises:
#         ValueError: If embedding response is invalid.
#     """
#     try:
#         logger.info("Generating embedding...")
#         text = preprocess_text(text)

#         response = genai.embed_content(model=EMBEDDING_MODEL, content=text)

#         if "embedding" in response:
#             embedding_vector = response["embedding"]
#         elif "embeddings" in response:
#             embedding_vector = response["embeddings"][0]
#         else:
#             raise ValueError("Unexpected embedding response format.")

#         if len(embedding_vector) != 768:
#             raise ValueError(f"Embedding vector dimension mismatch: Expected 768, got {len(embedding_vector)}.")

#         logger.info(f"Generated embedding of length {len(embedding_vector)}")
#         return embedding_vector

#     except Exception as e:
#         logger.error(f"Embedding generation failed: {str(e)}")
#         raise

import os
import re
import openai
from app.core.logging import get_logger
from app.core.config import EMBEDDING_MODEL

logger = get_logger(__name__)

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not set in environment variables.")
openai.api_key = OPENAI_API_KEY


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
    Generate embedding vector for the given text using OpenAI embeddings.

    Raises:
        ValueError: If embedding response is invalid.
    """
    try:
        logger.info("Generating embedding with OpenAI...")
        text = preprocess_text(text)

        response = openai.embeddings.create(
            model=EMBEDDING_MODEL,  # e.g., "text-embedding-3-small" or "text-embedding-3-large"
            input=text
        )

        # Extract embedding from response
        embedding_vector = response.data[0].embedding

        logger.info(f"✅ Generated embedding of length {len(embedding_vector)}")
        return embedding_vector

    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {str(e)}")
        raise
