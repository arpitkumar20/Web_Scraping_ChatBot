import os
import re
import logging
import google.generativeai as genai
from pinecone import Pinecone

# Environment Variables
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
PINECONE_INDEX = os.getenv('PINECONE_INDEX')
NAMESPACE = os.getenv('NAMESPACE')

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL')          # Reserved for future use
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')  # High-quality embedding model

# Configure Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Configure GenAI
genai.configure(api_key=GOOGLE_API_KEY)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = re.sub(r'[\n\xa0]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def preprocess_text(text: str) -> str:
    text = clean_text(text)
    return text


def generate_embedding(text: str) -> list[float]:
    try:
        logger.info("Generating embedding using EMBEDDING_MODEL.")

        text = preprocess_text(text)

        response = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text
        )

        if "embedding" in response:
            embedding_vector = response["embedding"]
        elif "embeddings" in response:
            embedding_vector = response["embeddings"][0]
        else:
            raise ValueError("Unexpected embedding response format.")

        if len(embedding_vector) != 768:
            raise ValueError(f"Embedding vector dimension mismatch: Expected 768, got {len(embedding_vector)}.")

        logger.info(f"Generated embedding vector of length {len(embedding_vector)}.")
        return embedding_vector

    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise


def query_pinecone_index(query_text: str, top_k: int = 5, namespace: str = NAMESPACE) -> list[dict]:
    # Do NOT reformulate the query with any non-existent API.
    # Instead, rely on strong embeddings directly.
    enriched_query = preprocess_text(query_text)

    query_vector = generate_embedding(enriched_query)

    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    index = pc.Index(PINECONE_INDEX)

    logger.info(f"Querying Pinecone index (top_k={top_k})")

    query_response = index.query(
        vector=query_vector,
        top_k=top_k,
        namespace=namespace,
        include_values=False,
        include_metadata=True
    )

    results = []
    for match in query_response.get('matches', []):
        metadata = match.get('metadata', {})
        cleaned_metadata = {k: clean_text(str(v)) for k, v in metadata.items()}
        result_dict = {
            "id": clean_text(match['id']),
            "score": match['score'],
            **cleaned_metadata
        }
        results.append(result_dict)

    logger.info(f"Retrieved {len(results)} results from Pinecone.")
    return results
