# import os
# import re
# import logging
# import google.generativeai as genai
# from pinecone import Pinecone

# # Environment Variables
# PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
# PINECONE_ENV = os.getenv('PINECONE_ENV')
# PINECONE_INDEX = os.getenv('PINECONE_INDEX')
# NAMESPACE = os.getenv('NAMESPACE')

# GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
# GEMINI_MODEL = os.getenv('GEMINI_MODEL')          # Reserved for future use
# EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')  # High-quality embedding model

# # Configure Logger
# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# logger = logging.getLogger(__name__)

# # Configure GenAI
# genai.configure(api_key=GOOGLE_API_KEY)


# def clean_text(text: str) -> str:
#     if not isinstance(text, str):
#         return text
#     text = re.sub(r'[\n\xa0]+', ' ', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text.lower()


# def preprocess_text(text: str) -> str:
#     text = clean_text(text)
#     return text


# def generate_embedding(text: str) -> list[float]:
#     try:
#         logger.info("Generating embedding using EMBEDDING_MODEL.")

#         text = preprocess_text(text)

#         response = genai.embed_content(
#             model=EMBEDDING_MODEL,
#             content=text
#         )

#         if "embedding" in response:
#             embedding_vector = response["embedding"]
#         elif "embeddings" in response:
#             embedding_vector = response["embeddings"][0]
#         else:
#             raise ValueError("Unexpected embedding response format.")

#         if len(embedding_vector) != 768:
#             raise ValueError(f"Embedding vector dimension mismatch: Expected 768, got {len(embedding_vector)}.")

#         logger.info(f"Generated embedding vector of length {len(embedding_vector)}.")
#         return embedding_vector

#     except Exception as e:
#         logger.error(f"Embedding generation failed: {str(e)}")
#         raise


# def query_pinecone_index(query_text: str, top_k: int = 5, namespace: str = NAMESPACE) -> list[dict]:
#     # Do NOT reformulate the query with any non-existent API.
#     # Instead, rely on strong embeddings directly.
#     enriched_query = preprocess_text(query_text)

#     query_vector = generate_embedding(enriched_query)

#     pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
#     index = pc.Index(PINECONE_INDEX)

#     logger.info(f"Querying Pinecone index (top_k={top_k})")

#     query_response = index.query(
#         vector=query_vector,
#         top_k=top_k,
#         namespace=namespace,
#         include_values=False,
#         include_metadata=True
#     )

#     results = []
#     for match in query_response.get('matches', []):
#         metadata = match.get('metadata', {})
#         cleaned_metadata = {k: clean_text(str(v)) for k, v in metadata.items()}
#         result_dict = {
#             "id": clean_text(match['id']),
#             "score": match['score'],
#             **cleaned_metadata
#         }
#         results.append(result_dict)

#     logger.info(f"Retrieved {len(results)} results from Pinecone.")
#     return results






# def query_pinecone_index(query_text: str, top_k: int = 5, namespace: str = None) -> list[dict]:
#     """
#     Query Pinecone index using embedding vector of user query.

#     Args:
#         query_text (str): User query text.
#         top_k (int): Number of results to return.
#         namespace (str): Pinecone namespace to query.

#     Returns:
#         list[dict]: List of matches with metadata and scores.
#     """
#     enriched_query = preprocess_text(query_text)
#     query_vector = generate_embedding(enriched_query)

#     logger.info(f"Querying Pinecone index based on (top_k={top_k})...")
#     query_response = pinecone_index.query(
#         vector=query_vector,
#         top_k=top_k,
#         namespace=namespace,
#         include_values=False,
#         include_metadata=True
#     )
#     print(">>>>>>>>>>>>>>>>query_response>>>>>>>>>>>>>>>>",query_response)

#     results = []
#     for match in query_response.get('matches', []):
#         metadata = match.get('metadata', {})
#         cleaned_metadata = {k: clean_text(str(v)) for k, v in metadata.items()}
#         results.append({
#             "id": clean_text(match['id']),
#             "score": match['score'],
#             **cleaned_metadata
#         })

#     logger.info(f"Retrieved {len(results)} results from Pinecone.")
#     return results



# def query_pinecone_index(
#     query_text: str,
#     top_k: int = 5,
#     namespace: Optional[str] = None,
#     similarity_threshold: float = 0.8
# ) -> List[Dict]:
#     """
#     Query Pinecone index and return most relevant matches above a similarity threshold.

#     Args:
#         query_text (str): User query.
#         top_k (int): Maximum number of results to return.
#         namespace (str, optional): Pinecone namespace.
#         similarity_threshold (float): Only return matches above this score.

#     Returns:
#         List[Dict]: List of matches with metadata and scores.
#     """
#     enriched_query = preprocess_text(query_text)
#     query_vector = generate_embedding(enriched_query)

#     logger.info(f"Querying Pinecone index with top_k={top_k}, namespace={namespace}...")
#     query_response = pinecone_index.query(
#         vector=query_vector,
#         top_k=top_k,
#         namespace=namespace,
#         include_metadata=True,
#         include_values=False
#     )
#     print("=============query_response=============",query_response)
#     # Handle different Pinecone SDK versions
#     matches = getattr(query_response, "matches", None) or query_response.get("matches", [])

#     # Filter by similarity threshold and clean metadata
#     results = []
#     for match in matches:
#         score = match.get("score", 0)
#         if score < similarity_threshold:
#             continue  # Skip low-similarity matches

#         metadata = match.get("metadata", {})
#         cleaned_metadata = {k: clean_text(str(v)) for k, v in metadata.items()}

#         results.append({
#             "id": clean_text(match.get("id", "")),
#             "score": score,
#             **cleaned_metadata
#         })

#     # Sort results by descending similarity
#     results = sorted(results, key=lambda x: x["score"], reverse=True)

#     logger.info(f"✅ Found {len(results)} relevant matches for the query.")
#     return results


import numpy as np
from typing import List, Dict, Optional
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from app.core.config import pinecone_index
from app.core.embedding_utils import preprocess_text, generate_embedding, clean_text
from app.core.logging import get_logger
logger = get_logger(__name__)

def query_pinecone_index(
    query_text: str,
    top_k: int = 5,
    namespace: Optional[str] = None,
    similarity_threshold: float = 0.8
) -> List[Dict]:
    """
    Query Pinecone index, calculate cosine similarity explicitly, 
    and return most relevant matches above a similarity threshold.
    """
    # Step 1: Preprocess and generate embedding
    enriched_query = preprocess_text(query_text)
    query_vector = generate_embedding(enriched_query)
    logger.info(f"Query vector type: {type(query_vector)}, length: {len(query_vector)}")

    # Step 2: Query Pinecone for top_k results (scores returned might not be exact cosine)
    logger.info(f"Querying Pinecone index with top_k={top_k}, namespace={namespace}...")
    query_response = pinecone_index.query(
        vector=query_vector,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
        include_values=True  # include vector values for cosine similarity
    )
    matches = getattr(query_response, "matches", None) or query_response.get("matches", [])
    # Step 3: Compute cosine similarity manually for consistency
    results = []
    # --- Example query vector ---
    query_vector_np = np.array(query_vector).reshape(1, -1)
    query_vector_np = normalize(query_vector_np, axis=1)

    results = []

    similarity_threshold = 0.3  # lower threshold to catch more matches

    for match in matches:
        match_values = match.get("values")
        if not match_values:
            continue  # skip empty vectors

        # Convert to numpy array and normalize
        match_vector_np = np.array(match_values, dtype=np.float32).reshape(1, -1)
        match_vector_np = normalize(match_vector_np, axis=1)

        # Compute cosine similarity
        similarity = float(cosine_similarity(query_vector_np, match_vector_np)[0][0])

        if similarity < similarity_threshold:
            continue

        # Clean metadata
        metadata = match.get("metadata", {})
        cleaned_metadata = {k: clean_text(str(v)) for k, v in metadata.items()}

        results.append({
            "id": clean_text(match.get("id", "")),
            "score": similarity,
            **cleaned_metadata
        })

    # Sort results by similarity descending
    results.sort(key=lambda x: x["score"], reverse=True)

    logger.info(f"✅ Found {len(results)} relevant matches above threshold {similarity_threshold}")
    return results


def normalize_vector(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm



# import logging
# import numpy as np
# from typing import List, Dict, Optional
# from sklearn.preprocessing import normalize
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import CrossEncoder

# from app.core.config import pinecone_index
# from app.core.embedding_utils import preprocess_text, generate_embedding

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Initialize cross-encoder once
# cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# def query_pinecone_index(
#     query_text: str,
#     top_k: int = 5,
#     namespace: Optional[str] = None,
#     similarity_threshold: float = 0.3,
#     rerank_method: str = "cross_encoder"
# ) -> List[Dict]:

#     # Step 1: generate query embedding
#     enriched_query = preprocess_text(query_text)
#     query_vector = generate_embedding(enriched_query)
#     query_vector_np = normalize(np.array(query_vector).reshape(1, -1), axis=1)

#     # Step 2: fetch candidates
#     fetch_k = top_k * 3
#     query_response = pinecone_index.query(
#         vector=query_vector,
#         top_k=fetch_k,
#         namespace=namespace,
#         include_metadata=True,
#         include_values=True
#     )

#     matches = getattr(query_response, "matches", None) or query_response.get("matches", [])
#     results = []

#     # Step 3: compute cosine similarity
#     for match in matches:
#         vector = match.get("values")
#         if vector is None:
#             continue
#         sim = float(cosine_similarity(query_vector_np, normalize(np.array(vector).reshape(1, -1)))[0][0])
#         if sim < similarity_threshold:
#             continue

#         metadata = match.get("metadata", {})
#         # Use `source_text` if `text` is empty
#         text = metadata.get("text") or metadata.get("source_text", "")
#         results.append({
#             "id": match.get("id", ""),
#             "score": sim,
#             "text": text,
#             **metadata
#         })

#     if not results:
#         logger.warning("No matches found above similarity threshold.")
#         return []

#     # Print results before re-ranking
#     logger.info("\n--- Before Re-ranking ---")
#     # for r in results:
#     #     logger.info(f"ID: {r['id']}, Cosine Score: {r['score']:.4f}, Text: {r['text'][:150]}")

#     # Step 4: re-ranking using cross-encoder
#     if rerank_method == "cross_encoder":
#         pairs = [(query_text, r["text"]) for r in results if r["text"]]
#         if pairs:
#             rerank_scores = cross_encoder.predict(pairs)
#             for i, s in enumerate(rerank_scores):
#                 results[i]["rerank_score"] = float(s)
#             results.sort(key=lambda x: x.get("rerank_score", x["score"]), reverse=True)
#         else:
#             logger.warning("No text for cross-encoder; fallback to cosine.")
#             results.sort(key=lambda x: x["score"], reverse=True)
#     else:
#         results.sort(key=lambda x: x["score"], reverse=True)

#     # Print results after re-ranking
#     logger.info("\n--- After Re-ranking ---")
#     # for r in results[:top_k]:
#     #     score = r.get("rerank_score", r["score"])
#     #     logger.info(f"ID: {r['id']}, Score: {score:.4f}, Text: {r['text'][:150]}")

#     return results[:top_k]
